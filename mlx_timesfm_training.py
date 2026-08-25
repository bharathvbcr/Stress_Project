# mlx_timesfm_training.py — TimesFM Phase 1 MLX head training (frozen PyTorch backbone)
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation import find_best_threshold
from mlx_backend import configure_mlx_runtime, resolve_timesfm_hybrid_enabled
from mlx_convert import apply_mlx_head_to_torch_model
from mlx_models import MlxTimesFMClassifier, load_mlx_checkpoint, save_mlx_checkpoint
from utils import safe_get

log = logging.getLogger(__name__)


def _iter_batches(dataset, batch_size: int):
    size = len(dataset)
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        yield dataset[start:end]


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


def should_use_mlx_timesfm_phase1(model_type: str, runtime_caps: Dict[str, Any]) -> bool:
    """Use MLX for TimesFM Phase 1 (frozen backbone, MLX classifier head)."""
    if model_type.upper() != "TIMESFM":
        return False
    if os.environ.get("TIMESFM_MLX_PHASE1", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return resolve_timesfm_hybrid_enabled(runtime_caps.get("backend_request"))


def _torch_batch_tensors(batch) -> Tuple[Any, Any, Any]:
    import torch

    seq = batch["sequence"]
    static = batch.get("static")
    labels = batch["label"]
    if not isinstance(seq, torch.Tensor):
        seq = torch.tensor(seq, dtype=torch.float32)
    if static is not None and not isinstance(static, torch.Tensor):
        static = torch.tensor(static, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
    return seq, static, labels


def _extract_fused_features(
    torch_model,
    seq,
    static,
    device,
) -> np.ndarray:
    """Mirror StressTimesFM.forward embedding path (backbone + static fusion)."""
    import torch

    torch_model.eval()
    if hasattr(torch_model, "_ensure_attached"):
        torch_model._ensure_attached()

    seq = seq.to(device)
    if static is not None:
        static = static.to(device)

    n, t, f = seq.shape
    x_flat = seq.permute(0, 2, 1).reshape(n * f, t)

    with torch.no_grad():
        channel_embeds = torch_model.extractor(x_flat)
        channel_embeds = channel_embeds.view(n, f, torch_model.embed_dim)
        pooled_embed = channel_embeds.mean(dim=1)

        if torch_model.input_dim_static > 0:
            if static is None:
                static = torch.zeros(n, torch_model.input_dim_static, device=device)
            if static.ndim == 1:
                static = static.unsqueeze(0).expand(n, -1)
            fused = torch.cat([pooled_embed, static], dim=1)
        else:
            fused = pooled_embed

    return fused.cpu().numpy().astype(np.float32)


def precompute_feature_matrix(
    torch_model,
    dataset,
    batch_size: int,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute fused backbone features and labels for MLX head training."""
    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    for batch in _iter_batches(dataset, batch_size):
        seq, static, labels = _torch_batch_tensors(batch)
        fused = _extract_fused_features(torch_model, seq, static, device)
        feature_chunks.append(fused)
        label_chunks.append(labels.reshape(-1).cpu().numpy().astype(np.float32))

    if not feature_chunks:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(feature_chunks, axis=0), np.concatenate(label_chunks, axis=0)


def precompute_from_dataloader(
    torch_model,
    dataloader,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute fused features from a PyTorch DataLoader (SequenceWindowDataset)."""
    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    for batch in dataloader:
        seq = batch[0]
        static = batch[1] if len(batch) > 1 else None
        labels = batch[2]
        fused = _extract_fused_features(torch_model, seq, static, device)
        feature_chunks.append(fused)
        label_chunks.append(labels.reshape(-1).cpu().numpy().astype(np.float32))

    if not feature_chunks:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(feature_chunks, axis=0), np.concatenate(label_chunks, axis=0)


def _build_mlx_head(config: Dict[str, Any], feature_dim: int) -> MlxTimesFMClassifier:
    model_config = safe_get(config, ["model_config"], {})
    head_hidden = int(safe_get(model_config, ["timesfm_head_hidden_dim"], 128))
    dropout = float(safe_get(model_config, ["dropout"], 0.2))
    return MlxTimesFMClassifier(feature_dim, head_hidden, dropout)


def _evaluate_head_on_features(
    mlx_head,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    pos_weight: float,
) -> Tuple[Dict[str, float], float]:
    import mlx.core as mx
    import mlx.nn as nn

    mlx_head.eval()
    total_loss = 0.0
    batches = 0
    prob_chunks: List[np.ndarray] = []

    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        x = mx.array(features[start:end])
        y = mx.array(labels[start:end].reshape(-1, 1))
        logits = mlx_head(x)
        probs = mx.sigmoid(logits)
        sample_weight = mx.where(y > 0.5, pos_weight, 1.0)
        # logits + with_logits=True for the fused, stable BCE; reduction="none"
        # so the pos_weight weighting is not cancelled by a mean reduction.
        per_sample = nn.losses.binary_cross_entropy(
            logits, y, with_logits=True, reduction="none"
        )
        loss = mx.sum(per_sample * sample_weight) / mx.sum(sample_weight)
        mx.eval(loss, probs)
        total_loss += float(loss.item())
        batches += 1
        prob_chunks.append(np.asarray(probs).reshape(-1))

    probs_np = np.concatenate(prob_chunks) if prob_chunks else np.zeros(0, dtype=np.float64)
    metrics = _binary_metrics(probs_np, labels)
    return metrics, total_loss / max(batches, 1)


def run_timesfm_mlx_phase1(
    datamodule,
    config: Dict[str, Any],
    torch_model,
    device,
    *,
    epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    pos_weight: Optional[float] = None,
    input_dim_static: Optional[int] = None,
) -> str:
    """
    Train TimesFM classification head in MLX with a frozen PyTorch backbone.

    Precomputes fused embeddings once, trains the MLX head, syncs weights back
    into the PyTorch classifier, and returns the MLX head checkpoint path.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mlx_info = configure_mlx_runtime()
    log.info(
        "TimesFM MLX Phase 1 (mlx=%s, device=%s, compile=%s)",
        mlx_info.get("mlx_version"),
        mlx_info.get("mlx_device", "gpu"),
        mlx_info.get("mlx_compile"),
    )

    if hasattr(torch_model, "freeze_backbone"):
        torch_model.freeze_backbone()

    train_cfg = config.get("training_config", {})
    batch_size = int(safe_get(train_cfg, ["batch_size"], 32))
    epochs = int(epochs if epochs is not None else safe_get(train_cfg, ["epochs"], 15))
    learning_rate = float(safe_get(train_cfg, ["learning_rate"], 1e-4))
    weight_decay = float(safe_get(train_cfg, ["weight_decay"], 0.01))
    patience = int(safe_get(train_cfg, ["early_stopping", "patience"], 5))

    output_dir = output_dir or safe_get(config, ["save_paths", "models"], "outputs/models")
    os.makedirs(output_dir, exist_ok=True)
    head_path = os.path.join(output_dir, "best_timesfm_head_mlx.npz")

    log.info("Precomputing TimesFM fused features for MLX head training...")
    if datamodule is not None:
        train_x, train_y = precompute_feature_matrix(
            torch_model, datamodule.train_ds, batch_size, device
        )
        val_x, val_y = precompute_feature_matrix(
            torch_model, datamodule.val_ds, batch_size, device
        )
        static_dim = datamodule.input_dim_static
        if pos_weight is None:
            pw = getattr(datamodule, "pos_weight", None)
            if pw is not None:
                pos_weight = float(pw.item()) if hasattr(pw, "item") else float(pw)
    elif train_loader is not None and val_loader is not None:
        train_x, train_y = precompute_from_dataloader(torch_model, train_loader, device)
        val_x, val_y = precompute_from_dataloader(torch_model, val_loader, device)
        static_dim = input_dim_static or 0
    else:
        raise ValueError("run_timesfm_mlx_phase1 requires datamodule or train/val loaders.")
    log.info(
        "Feature cache ready: train=%s val=%s (dim=%s)",
        train_x.shape,
        val_x.shape,
        train_x.shape[1] if train_x.size else 0,
    )

    feature_dim = train_x.shape[1] if train_x.size else (
        int(getattr(torch_model, "embed_dim", 1280)) + static_dim
    )
    mlx_head = _build_mlx_head(config, feature_dim)

    if pos_weight is None:
        pos_weight = 1.0

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def loss_fn(head, features, labels):
        logits = head(features)
        sample_weight = mx.where(labels > 0.5, pos_weight, 1.0)
        per_sample = nn.losses.binary_cross_entropy(
            logits, labels, with_logits=True, reduction="none"
        )
        return mx.sum(per_sample * sample_weight) / mx.sum(sample_weight)

    loss_and_grad = nn.value_and_grad(mlx_head, loss_fn)
    step_state = [mlx_head.state, optimizer.state, mx.random.state]

    def _train_step(features, labels):
        loss, grads = loss_and_grad(mlx_head, features, labels)
        optimizer.update(mlx_head, grads)
        return loss

    compile_steps = bool(mlx_info.get("mlx_compile"))
    if compile_steps:
        _train_step = mx.compile(_train_step, inputs=step_state, outputs=step_state)

    best_val_f1 = -1.0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        mlx_head.train()
        running_loss = None
        train_batches = 0

        indices = np.random.permutation(len(train_y))
        for start in range(0, len(train_y), batch_size):
            idx = indices[start : start + batch_size]
            features = mx.array(train_x[idx])
            labels = mx.array(train_y[idx].reshape(-1, 1))
            loss = _train_step(features, labels)
            mx.eval(step_state)
            running_loss = loss if running_loss is None else running_loss + loss
            train_batches += 1

        train_loss = float(running_loss.item()) if running_loss is not None else 0.0
        val_metrics, val_loss = _evaluate_head_on_features(
            mlx_head, val_x, val_y, batch_size, pos_weight
        )
        log.info(
            "TimesFM MLX Phase 1 epoch %s/%s train_loss=%.4f val_loss=%.4f val_f1=%.4f val_acc=%.4f",
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
            save_mlx_checkpoint(mlx_head, head_path)
            log.info("Saved TimesFM MLX head: %s (val_f1=%.4f)", head_path, best_val_f1)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log.info("Early stopping TimesFM MLX Phase 1 at epoch %s", epoch)
                break

    load_mlx_checkpoint(mlx_head, head_path)
    apply_mlx_head_to_torch_model(torch_model, mlx_head)
    log.info("Synced MLX head weights into PyTorch TimesFM classifier.")

    results_dir = safe_get(config, ["save_paths", "results"], "outputs/results")
    os.makedirs(results_dir, exist_ok=True)
    val_probs_arr = _collect_head_probs(mlx_head, val_x, batch_size)
    best_threshold, best_val_f1 = find_best_threshold(val_probs_arr, val_y, metric="f1")
    test_x, test_y = (
        precompute_from_dataloader(torch_model, test_loader, device)
        if test_loader is not None
        else (
            precompute_feature_matrix(torch_model, datamodule.test_ds, batch_size, device)
            if datamodule is not None
            else (np.zeros((0, feature_dim), dtype=np.float32), np.zeros(0, dtype=np.float32))
        )
    )
    test_probs = _collect_head_probs(mlx_head, test_x, batch_size)

    from evaluate_checkpoint import build_eval_payload, write_eval_payload

    payload = build_eval_payload(
        checkpoint_path=head_path,
        val_probs=val_probs_arr,
        val_labels=val_y,
        test_probs=test_probs,
        test_labels=test_y,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
    )
    write_eval_payload(payload, os.path.join(results_dir, "eval_pass_mlx_timesfm_phase1.json"))
    return head_path


def _collect_head_probs(mlx_head, features: np.ndarray, batch_size: int) -> np.ndarray:
    import mlx.core as mx

    mlx_head.eval()
    prob_chunks: List[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        probs = mx.sigmoid(mlx_head(mx.array(features[start:end])))
        mx.eval(probs)
        prob_chunks.append(np.asarray(probs).reshape(-1))
    if not prob_chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(prob_chunks).astype(np.float64)
