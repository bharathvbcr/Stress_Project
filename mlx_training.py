# mlx_training.py — MLX training loop for Apple Silicon
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation import find_best_threshold
from mlx_backend import (
    cast_mlx_model_parameters,
    configure_mlx_runtime,
    resolve_mlx_dtype,
    resolve_training_backend,
)
from mlx_models import batch_to_mlx, get_mlx_model, load_mlx_checkpoint, save_mlx_checkpoint
from utils import safe_get

log = logging.getLogger(__name__)


def _preload_split(dataset, max_bytes: float):
    """Materialize an entire HF split into MLX device arrays once.

    Arrow slicing of nested sequence columns is expensive (~tens of ms per
    batch) and the current loop paid that cost *every epoch*. Converting the
    whole split to on-device MLX arrays a single time turns per-epoch host->
    device conversion into an O(1) preload plus cheap on-device slicing.

    Returns a dict of mx.arrays, or ``None`` to signal the caller to stream
    per batch (split too large for the memory cap, or conversion failed).
    """
    import mlx.core as mx

    try:
        n = len(dataset)
        if n == 0:
            return None
        # Estimate device footprint from one row before materializing the
        # whole split, so a huge dataset never blows up host RAM here.
        sample = dataset[0]
        seq_bytes = np.asarray(sample["sequence"], dtype=np.float32).nbytes
        static_bytes = (
            np.asarray(sample["static"], dtype=np.float32).nbytes
            if sample.get("static") is not None
            else 0
        )
        est_bytes = n * (seq_bytes + static_bytes + 4)
        if est_bytes > max_bytes:
            log.info(
                "MLX preload skipped: split ~%.2f GB exceeds cap %.2f GB "
                "(set STRESS_MLX_PRELOAD_MAX_GB to raise it); streaming per batch.",
                est_bytes / 1e9,
                max_bytes / 1e9,
            )
            return None

        cols = dataset[:]  # single Arrow->host materialization for the split
        seq_np = np.asarray(cols["sequence"], dtype=np.float32)
        label_np = np.asarray(cols["label"], dtype=np.float32).reshape(-1, 1)
        static_np = None
        raw_static = cols["static"] if "static" in cols else None
        if raw_static is not None:
            static_np = np.asarray(raw_static, dtype=np.float32)
            if static_np.ndim == 1:
                static_np = static_np.reshape(n, -1)
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.warning("MLX preload failed (%s); falling back to per-batch conversion.", exc)
        return None

    dtype = resolve_mlx_dtype()
    split = {
        "seq": mx.array(seq_np, dtype=dtype),
        "static": mx.array(static_np, dtype=dtype) if static_np is not None else None,
        "labels": mx.array(label_np),
        "n": n,
    }
    to_eval = [split["seq"], split["labels"]]
    if split["static"] is not None:
        to_eval.append(split["static"])
    mx.eval(*to_eval)
    return split


def _iter_mlx_batches(split, batch_size: int, *, shuffle: bool = False, seed: int = 0, drop_last: bool = False):
    """Yield ``(seq, static, labels)`` MLX tuples from a preloaded split or raw HF dataset."""
    import mlx.core as mx

    if isinstance(split, dict):  # preloaded on device
        n = split["n"]
        seq, static, labels = split["seq"], split["static"], split["labels"]
        stop = n - (n % batch_size) if (drop_last and n >= batch_size) else n
        if shuffle:
            order = mx.array(np.random.default_rng(seed).permutation(n))
            for start in range(0, stop, batch_size):
                idx = order[start : start + batch_size]
                yield seq[idx], (static[idx] if static is not None else None), labels[idx]
        else:
            for start in range(0, stop, batch_size):
                end = min(start + batch_size, n)
                yield (
                    seq[start:end],
                    static[start:end] if static is not None else None,
                    labels[start:end],
                )
        return

    # Streaming fallback over a raw HF dataset (per-batch conversion).
    n = len(split)
    stop = n - (n % batch_size) if (drop_last and n >= batch_size) else n
    if shuffle:
        order = np.random.default_rng(seed).permutation(n).tolist()
        for start in range(0, stop, batch_size):
            yield batch_to_mlx(split[order[start : start + batch_size]])
    else:
        for start in range(0, stop, batch_size):
            end = min(start + batch_size, n)
            yield batch_to_mlx(split[start:end])


def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = float(np.mean(preds == labels))
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _evaluate_split(model, split, batch_size: int) -> Tuple[Dict[str, float], float]:
    import mlx.core as mx
    import mlx.nn as nn

    try:
        from torch.utils.data import DataLoader
    except ImportError:
        DataLoader = ()  # type: ignore

    model.eval()
    prob_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    total_loss = 0.0
    batches = 0

    if isinstance(split, DataLoader):
        batch_iter = (batch_loader_to_mlx(batch) for batch in split)
    else:
        batch_iter = _iter_mlx_batches(split, batch_size)

    for seq, static, y in batch_iter:
        # Cast logits up so loss/metrics stay full precision under bf16/fp16.
        logits = model(seq, static).astype(mx.float32)
        loss = nn.losses.binary_cross_entropy(
            logits, y, with_logits=True, reduction="mean"
        )
        probs = mx.sigmoid(logits)
        batch_probs = probs.reshape(-1)
        batch_labels = y.reshape(-1)
        mx.eval(loss, batch_probs, batch_labels)
        total_loss += float(loss.item())
        prob_chunks.append(np.asarray(batch_probs))
        label_chunks.append(np.asarray(batch_labels))
        batches += 1

    if batches == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, 0.0

    metrics = _binary_metrics(np.concatenate(prob_chunks), np.concatenate(label_chunks))
    return metrics, total_loss / batches


def _evaluate_model(model, split, batch_size: int) -> Tuple[Dict[str, float], float]:
    return _evaluate_split(model, split, batch_size)


def _collect_split_probs(model, split, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx

    try:
        from torch.utils.data import DataLoader
    except ImportError:
        DataLoader = ()  # type: ignore

    model.eval()
    prob_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    if isinstance(split, DataLoader):
        batch_iter = (batch_loader_to_mlx(batch) for batch in split)
    else:
        batch_iter = _iter_mlx_batches(split, batch_size)

    for seq, static, y in batch_iter:
        batch_probs = mx.sigmoid(model(seq, static).astype(mx.float32)).reshape(-1)
        batch_labels = y.reshape(-1)
        mx.eval(batch_probs, batch_labels)
        prob_chunks.append(np.asarray(batch_probs))
        label_chunks.append(np.asarray(batch_labels))
    if not prob_chunks:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return (
        np.concatenate(prob_chunks).astype(np.float64),
        np.concatenate(label_chunks).astype(np.float64),
    )


def batch_loader_to_mlx(batch) -> Tuple[Any, Optional[Any], Any]:
    """Convert a PyTorch DataLoader batch tuple to MLX arrays."""
    import mlx.core as mx

    dtype = resolve_mlx_dtype()
    seq = batch[0]
    static = batch[1] if len(batch) > 1 else None
    labels = batch[2]
    seq_mx = mx.array(seq.detach().cpu().numpy(), dtype=dtype)
    labels_mx = mx.array(labels.detach().cpu().numpy(), dtype=mx.float32).reshape(-1, 1)
    static_mx = None
    if static is not None:
        static_mx = mx.array(static.detach().cpu().numpy(), dtype=dtype)
    return seq_mx, static_mx, labels_mx


def _preload_from_loader(loader, max_bytes: float):
    """Materialize a PyTorch DataLoader into on-device MLX arrays."""
    import mlx.core as mx

    seq_chunks: List[np.ndarray] = []
    static_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    est_bytes = 0

    for batch in loader:
        seq_np = batch[0].detach().cpu().numpy().astype(np.float32)
        label_np = batch[2].detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        static_np = None
        if len(batch) > 1 and batch[1] is not None:
            static_np = batch[1].detach().cpu().numpy().astype(np.float32)
        est_bytes += seq_np.nbytes + label_np.nbytes + (static_np.nbytes if static_np is not None else 0)
        if est_bytes > max_bytes:
            return None
        seq_chunks.append(seq_np)
        label_chunks.append(label_np)
        if static_np is not None:
            static_chunks.append(static_np)

    if not seq_chunks:
        return None

    seq_np = np.concatenate(seq_chunks, axis=0)
    labels_np = np.concatenate(label_chunks, axis=0)
    static_np = np.concatenate(static_chunks, axis=0) if static_chunks else None
    dtype = resolve_mlx_dtype()
    split = {
        "seq": mx.array(seq_np, dtype=dtype),
        "static": mx.array(static_np, dtype=dtype) if static_np is not None else None,
        "labels": mx.array(labels_np),
        "n": int(seq_np.shape[0]),
    }
    to_eval = [split["seq"], split["labels"]]
    if split["static"] is not None:
        to_eval.append(split["static"])
    mx.eval(*to_eval)
    return split


def _iter_loader_batches(loader, batch_size: int, *, shuffle: bool = False, seed: int = 0):
    """Yield MLX batches from a preloaded split or a live PyTorch DataLoader."""
    if isinstance(loader, dict):
        yield from _iter_mlx_batches(loader, batch_size, shuffle=shuffle, seed=seed, drop_last=True)
        return

    batches = list(loader)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    for batch in batches:
        yield batch_loader_to_mlx(batch)


def run_mlx_training_loaders(
    train_loader,
    val_loader,
    test_loader,
    config: Dict[str, Any],
    model_type: str,
    input_dim_sequence: int,
    input_dim_static: int,
    output_dir: Optional[str] = None,
    pos_weight: Optional[float] = None,
    return_metrics: bool = False,
):
    """Train an MLX model from PyTorch DataLoaders (legacy run_pipeline path)."""

    class _LoaderShim:
        def __init__(self, train, val, test, seq_dim, static_dim, pw):
            self.train_ds = train
            self.val_ds = val
            self.test_ds = test
            self.input_dim_sequence = seq_dim
            self.input_dim_static = static_dim
            self.pos_weight = pw

    shim = _LoaderShim(
        train_loader,
        val_loader,
        test_loader,
        input_dim_sequence,
        input_dim_static,
        pos_weight,
    )
    return run_mlx_training(
        shim, config, model_type, output_dir=output_dir, return_metrics=return_metrics
    )


def run_mlx_training(
    datamodule,
    config: Dict[str, Any],
    model_type: str,
    output_dir: Optional[str] = None,
    return_metrics: bool = False,
):
    """Train an MLX-native model and return the best checkpoint path."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mlx_info = configure_mlx_runtime()
    log.info(
        "MLX training backend active (mlx=%s, device=%s, compile=%s, dtype=%s)",
        mlx_info.get("mlx_version"),
        mlx_info.get("mlx_device", "gpu"),
        mlx_info.get("mlx_compile"),
        mlx_info.get("mlx_dtype", "float32"),
    )

    train_cfg = config.get("training_config", {})
    batch_size = int(safe_get(train_cfg, ["batch_size"], 32))
    epochs = int(safe_get(train_cfg, ["epochs"], 50))
    # Dry-run / timing cap honored across every entrypoint (Hydra, loaders, CLI).
    env_max = os.environ.get("STRESS_MAX_EPOCHS", "").strip()
    if env_max:
        try:
            capped = min(epochs, max(1, int(env_max)))
            if capped != epochs:
                log.info("STRESS_MAX_EPOCHS=%s: capping epochs %s -> %s (dry run).", env_max, epochs, capped)
            epochs = capped
        except ValueError:
            log.warning("Ignoring non-integer STRESS_MAX_EPOCHS=%r.", env_max)
    learning_rate = float(safe_get(train_cfg, ["learning_rate"], 1e-4))
    weight_decay = float(safe_get(train_cfg, ["weight_decay"], 0.01))
    patience = int(safe_get(train_cfg, ["early_stopping", "patience"], 5))

    output_dir = output_dir or safe_get(config, ["save_paths", "models"], "outputs/models")
    os.makedirs(output_dir, exist_ok=True)
    model_tag = model_type.lower()
    best_path = os.path.join(output_dir, f"best_{model_tag}_mlx.npz")

    # Preload each split to on-device MLX arrays once (guarded by a memory cap)
    # so per-epoch Arrow slicing/host->device conversion is not repeated.
    preload_max_bytes = float(os.environ.get("STRESS_MLX_PRELOAD_MAX_GB", "4")) * 1e9

    def _prepare(dataset_or_loader, name):
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            DataLoader = ()  # type: ignore

        if isinstance(dataset_or_loader, DataLoader):
            split = _preload_from_loader(dataset_or_loader, preload_max_bytes)
            log.info(
                "MLX %s loader: %s",
                name,
                "preloaded on device" if split is not None else "streaming per batch",
            )
            return split if split is not None else dataset_or_loader

        split = _preload_split(dataset_or_loader, preload_max_bytes)
        log.info(
            "MLX %s split: %s",
            name,
            "preloaded on device" if split is not None else "streaming per batch",
        )
        return split if split is not None else dataset_or_loader

    train_split = _prepare(datamodule.train_ds, "train")
    val_split = _prepare(datamodule.val_ds, "val")
    test_split = _prepare(datamodule.test_ds, "test")

    model = get_mlx_model(config, datamodule.input_dim_sequence, datamodule.input_dim_static)
    # Opt-in reduced precision (STRESS_MLX_DTYPE=bfloat16|float16): params and
    # activations run in the low dtype; loss/metrics are computed in float32.
    cast_mlx_model_parameters(model, resolve_mlx_dtype())
    pw = getattr(datamodule, "pos_weight", None)
    if pw is None:
        pos_weight = 1.0
    elif hasattr(pw, "item"):
        pos_weight = float(pw.item())
    else:
        pos_weight = float(pw)

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def loss_fn(model, seq, static, labels):
        # Cast logits up so BCE + class weighting run in float32 even when the
        # model computes in bf16/fp16 (labels are always float32).
        logits = model(seq, static).astype(mx.float32)
        sample_weight = mx.where(labels > 0.5, pos_weight, 1.0)
        # Feed logits (with_logits=True) into MLX's fused, numerically stable
        # BCE and keep it per-sample (reduction="none"); the default reduction
        # is "mean", which collapses to a scalar and silently cancels the
        # pos_weight class weighting below.
        per_sample = nn.losses.binary_cross_entropy(
            logits, labels, with_logits=True, reduction="none"
        )
        return mx.sum(per_sample * sample_weight) / mx.sum(sample_weight)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # State updated in-place by the training step. mx.random.state MUST be
    # included so Dropout re-randomizes each call under mx.compile; without it
    # a compiled step reuses a single frozen dropout mask for all steps.
    step_state = [model.state, optimizer.state, mx.random.state]

    def _train_step(seq, static, labels):
        loss, grads = loss_and_grad(model, seq, static, labels)
        optimizer.update(model, grads)
        return loss

    compile_steps = bool(mlx_info.get("mlx_compile"))
    if compile_steps:
        _train_step = mx.compile(_train_step, inputs=step_state, outputs=step_state)

    best_val_f1 = -1.0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = None
        train_batches = 0

        # shuffle=True gives per-epoch batch reshuffling (matching the torch
        # DataLoader path); drop_last keeps the compiled step's input shape
        # constant so it does not retrace on a small trailing batch.
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            DataLoader = ()  # type: ignore

        if isinstance(train_split, DataLoader):
            batch_iter = (batch_loader_to_mlx(batch) for batch in train_split)
        else:
            batch_iter = _iter_mlx_batches(
                train_split, batch_size, shuffle=True, seed=epoch, drop_last=True
            )

        for seq, static, labels in batch_iter:
            loss = _train_step(seq, static, labels)
            # Force the step's compute but stay on-device: no float()/.item()
            # host round-trip per batch. The scalar loss is summed lazily and
            # materialized once at epoch end.
            mx.eval(step_state)
            running_loss = loss if running_loss is None else running_loss + loss
            train_batches += 1

        train_loss = float(running_loss.item()) if running_loss is not None else 0.0

        val_metrics, val_loss = _evaluate_model(model, val_split, batch_size)
        log.info(
            "MLX epoch %s/%s train_loss=%.4f val_loss=%.4f val_f1=%.4f val_acc=%.4f",
            epoch,
            epochs,
            train_loss / max(train_batches, 1),
            val_loss,
            val_metrics["f1"],
            val_metrics["accuracy"],
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            stale_epochs = 0
            save_mlx_checkpoint(model, best_path)
            log.info("Saved MLX checkpoint: %s (val_f1=%.4f)", best_path, best_val_f1)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log.info("MLX early stopping after %s epochs without improvement.", patience)
                break

    if os.path.exists(best_path):
        load_mlx_checkpoint(model, best_path)

    test_metrics, test_loss = _evaluate_model(model, test_split, batch_size)
    log.info(
        "MLX test metrics: loss=%.4f f1=%.4f acc=%.4f precision=%.4f recall=%.4f",
        test_loss,
        test_metrics["f1"],
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
    )

    if os.path.exists(best_path):
        val_probs, val_labels = _collect_split_probs(model, val_split, batch_size)
        test_probs, test_labels = _collect_split_probs(model, test_split, batch_size)
        best_threshold, best_val_f1 = find_best_threshold(val_probs, val_labels, metric="f1")
        results_dir = safe_get(config, ["save_paths", "results"], "outputs/results")
        os.makedirs(results_dir, exist_ok=True)
        from evaluate_checkpoint import build_eval_payload, write_eval_payload

        payload = build_eval_payload(
            checkpoint_path=best_path,
            val_probs=val_probs,
            val_labels=val_labels,
            test_probs=test_probs,
            test_labels=test_labels,
            best_threshold=best_threshold,
            best_val_f1=best_val_f1,
        )
        eval_path = write_eval_payload(payload, os.path.join(results_dir, "eval_pass_mlx.json"))
        log.info("MLX evaluation artifact written to %s", eval_path)

    if return_metrics:
        return {"checkpoint": best_path, "best_val_f1": best_val_f1}
    return best_path


def should_use_mlx_training(model_type: str, runtime_caps: Dict[str, Any]) -> bool:
    return (
        resolve_training_backend(
            model_type,
            mlx_available=bool(runtime_caps.get("mlx_available")),
            cuda_available=bool(runtime_caps.get("cuda_available")),
            mps_available=bool(runtime_caps.get("mps_available")),
            requested=runtime_caps.get("backend_request"),
        )
        == "mlx"
    )
