# evaluate_checkpoint.py (checkpoint evaluation and calibration artifact writer)
import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation import find_best_threshold
from lightning_data import StressDataModule
from mlx_backend import is_mlx_available, resolve_inference_backend, resolve_timesfm_hybrid_enabled
from mlx_convert import ensure_mlx_checkpoint
from mlx_models import batch_to_mlx, get_mlx_model, load_mlx_checkpoint
from models import get_model
from utils import configure_torch_runtime, extract_model_state_dict, load_config, safe_get, setup_logging

log = logging.getLogger("evaluate_checkpoint")

MIN_FULL_CHECKPOINT_BYTES = 50_000_000
LOSS_RE = re.compile(r"val_loss=([0-9]+(?:\.[0-9]+)?)")


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge legacy JSON keys to the keys used by the current runtime path."""
    dataset_cfg = config.get("dataset")
    if isinstance(dataset_cfg, dict):
        config.setdefault("datasets", dataset_cfg)
        config.setdefault("sampling_rates", dataset_cfg.get("sampling_rates", {}))
        config.setdefault("label_mapping", dataset_cfg.get("label_mapping", {}))

    if "model_config" not in config:
        config["model_config"] = config.get("model", {})
    if "training_config" not in config and "training" in config:
        config["training_config"] = config["training"]
    if "processing_config" not in config and "processing" in config:
        config["processing_config"] = config["processing"]
    return config


def _val_loss(path: Path) -> float:
    match = LOSS_RE.search(path.name)
    return float(match.group(1)) if match else float("inf")


def _has_val_loss(path: Path) -> bool:
    return _val_loss(path) != float("inf")


def select_best_checkpoint(
    model_dir: Path | str,
    min_full_size_bytes: int = MIN_FULL_CHECKPOINT_BYTES,
    prefer_mlx: bool = False,
) -> Optional[Path]:
    """Select the checkpoint that should be evaluated and served."""
    model_dir = Path(model_dir)

    if prefer_mlx:
        mlx_ckpts = list(model_dir.glob("*_mlx.npz"))
        if mlx_ckpts:
            return max(mlx_ckpts, key=lambda path: path.stat().st_mtime)

    ckpts = list(model_dir.glob("*.ckpt"))
    if not ckpts:
        return None

    full_phase2 = [
        path
        for path in ckpts
        if "best_model_phase2" in path.name
        and _has_val_loss(path)
        and path.stat().st_size >= min_full_size_bytes
    ]
    if full_phase2:
        return min(full_phase2, key=_val_loss)

    phase2 = [path for path in ckpts if "best_model_phase2" in path.name and _has_val_loss(path)]
    if phase2:
        return min(phase2, key=_val_loss)

    loss_ckpts = [path for path in ckpts if _has_val_loss(path)]
    if loss_ckpts:
        return min(loss_ckpts, key=_val_loss)

    return max(ckpts, key=lambda path: path.stat().st_mtime)


def resolve_repo_path(path_value: str | os.PathLike[str], repo_root: Path = ROOT_DIR) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA evaluation, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    config: Optional[Dict] = None,
) -> Tuple[Iterable[str], Iterable[str]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)

    if hasattr(model, "_ensure_attached"):
        model._ensure_attached()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if hasattr(model, "lora_adapters"):
        from timesfm_lora import restore_timesfm_lora_adapters

        restore_timesfm_lora_adapters(model, state_dict, config=config)

    return missing, unexpected


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if isinstance(batch, dict):
        sequence = batch.get("sequence")
        if sequence is None:
            sequence = batch.get("x", batch.get("input", batch.get("features")))
        static = batch.get("static")
        labels = batch.get("label", batch.get("labels"))
    elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
        sequence, static, labels = batch[:3]
    else:
        raise ValueError(f"Unsupported batch format: {type(batch).__name__}")

    if sequence is None or labels is None:
        raise ValueError("Evaluation batch must include sequence and label tensors.")
    return sequence, static, labels


def collect_probabilities(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    probs_out = []
    labels_out = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            sequence, static, labels = _unpack_batch(batch)
            sequence = sequence.to(device)
            static = static.to(device) if static is not None else None

            logits = model(sequence, static)
            probs = torch.sigmoid(logits).detach().cpu().reshape(-1).numpy()
            labels_np = labels.detach().cpu().reshape(-1).numpy().astype(int)

            probs_out.extend(probs.tolist())
            labels_out.extend(labels_np.tolist())

    return np.asarray(probs_out, dtype=float), np.asarray(labels_out, dtype=int)


def label_distribution(labels: np.ndarray) -> list[int]:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return []
    return [int(value) for value in np.bincount(labels)]


def probability_stats(probs: np.ndarray) -> Dict[str, Optional[float]]:
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return {"prob_min": None, "prob_max": None, "prob_mean": None}
    return {
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "prob_mean": float(np.mean(probs)),
    }


def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs > threshold).astype(int)

    roc_auc = None
    if len(np.unique(labels)) > 1:
        try:
            roc_auc = float(roc_auc_score(labels, probs))
        except ValueError:
            roc_auc = None

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def build_eval_payload(
    checkpoint_path: Path | str,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    best_threshold: float,
    best_val_f1: float,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()

    validation = {
        "n_samples": int(len(val_labels)),
        "label_distribution": label_distribution(val_labels),
        **probability_stats(val_probs),
        "best_threshold": float(best_threshold),
        "val_f1_at_best_threshold": float(best_val_f1),
    }

    return {
        "generated_at": generated_at,
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
        "validation": validation,
        "test_default_threshold_0.5": metrics_at_threshold(test_labels, test_probs, 0.5),
        "test_calibrated_threshold": metrics_at_threshold(test_labels, test_probs, best_threshold),
        "notes": "Generated by evaluate_checkpoint.py after selecting the best available checkpoint.",
    }


def write_eval_payload(payload: Dict[str, Any], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def collect_probabilities_mlx(model, dataset, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx

    probs: list[float] = []
    labels: list[float] = []
    size = len(dataset)
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        seq, static, y = batch_to_mlx(dataset[start:end])
        logits = model(seq, static)
        batch_probs = mx.sigmoid(logits).reshape(-1)
        probs.extend(np.asarray(batch_probs).tolist())
        labels.extend(np.asarray(y.reshape(-1)).tolist())
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def run_checkpoint_evaluation(
    config_path: Path,
    checkpoint_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    device_request: str = "auto",
    model_type_override: Optional[str] = None,
    backend_request: str = "auto",
) -> Path:
    config = load_config(str(config_path))
    if not config:
        raise RuntimeError(f"Could not load config: {config_path}")
    config = normalize_config(config)

    if model_type_override:
        config.setdefault("model_config", {})["type"] = model_type_override.upper()

    model_type = safe_get(config, ["model_config", "type"], "LSTM").upper()
    configure_torch_runtime()

    model_dir = model_dir or resolve_repo_path(safe_get(config, ["save_paths", "models"], "./outputs/models"))
    results_dir = results_dir or resolve_repo_path(safe_get(config, ["save_paths", "results"], "./outputs/results"))

    timesfm_hybrid = model_type == "TIMESFM" and resolve_timesfm_hybrid_enabled(backend_request)
    use_mlx = timesfm_hybrid or (
        resolve_inference_backend(
            model_type,
            str(checkpoint_path) if checkpoint_path else None,
            mlx_available=is_mlx_available(),
            requested=backend_request,
        )
        == "mlx"
    )
    checkpoint_path = checkpoint_path or select_best_checkpoint(model_dir, prefer_mlx=use_mlx)
    if checkpoint_path is None and use_mlx:
        checkpoint_path = select_best_checkpoint(model_dir, prefer_mlx=False)
        use_mlx = False
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".npz" and checkpoint_path.name.endswith("_mlx.npz"):
        use_mlx = True

    log.info("Loading DataModule from cached Arrow datasets...")
    datamodule = StressDataModule({}, {}, config)
    datamodule.setup()

    batch_size = int(safe_get(config, ["training_config", "batch_size"], 32))

    if use_mlx and not timesfm_hybrid:
        log.info("Evaluating MLX checkpoint: %s", checkpoint_path)
        mlx_ckpt = str(checkpoint_path)
        if not mlx_ckpt.endswith(".npz"):
            mlx_ckpt = ensure_mlx_checkpoint(
                config,
                mlx_ckpt,
                datamodule.input_dim_sequence,
                datamodule.input_dim_static,
            )
        model = get_mlx_model(config, datamodule.input_dim_sequence, datamodule.input_dim_static)
        load_mlx_checkpoint(model, mlx_ckpt)
        model.eval()  # disable Dropout/train-mode BatchNorm so scored probs are deterministic
        val_probs, val_labels = collect_probabilities_mlx(model, datamodule.val_ds, batch_size)
        test_probs, test_labels = collect_probabilities_mlx(model, datamodule.test_ds, batch_size)
    elif timesfm_hybrid:
        from mlx_hybrid import build_timesfm_hybrid, collect_timesfm_hybrid_probabilities

        device = _select_device(device_request)
        log.info("Evaluating TimesFM hybrid (PyTorch backbone + MLX head): %s", checkpoint_path)
        torch_model, mlx_head, _ = build_timesfm_hybrid(
            config,
            str(checkpoint_path),
            datamodule.input_dim_sequence,
            datamodule.input_dim_static,
            device=device,
        )
        val_probs, val_labels = collect_timesfm_hybrid_probabilities(
            torch_model, mlx_head, datamodule.val_ds, batch_size, device
        )
        test_probs, test_labels = collect_timesfm_hybrid_probabilities(
            torch_model, mlx_head, datamodule.test_ds, batch_size, device
        )
    else:
        device = _select_device(device_request)
        log.info("Using evaluation device: %s", device)
        log.info("Building model for checkpoint evaluation...")
        model = get_model(config, datamodule.input_dim_sequence, datamodule.input_dim_static)
        missing, unexpected = load_weights_into_model(model, checkpoint_path, config=config)
        if missing:
            log.warning("Missing keys while loading checkpoint: %s", list(missing))
        if unexpected:
            log.warning("Unexpected keys while loading checkpoint: %s", list(unexpected))
        log.info("Collecting validation probabilities for threshold calibration...")
        val_probs, val_labels = collect_probabilities(model, datamodule.val_dataloader(), device)
        log.info("Collecting test probabilities for held-out evaluation...")
        test_probs, test_labels = collect_probabilities(model, datamodule.test_dataloader(), device)

    best_threshold, best_val_f1 = find_best_threshold(val_probs, val_labels, metric="f1")
    log.info("Best validation threshold: %.4f (F1 %.4f)", best_threshold, best_val_f1)

    payload = build_eval_payload(
        checkpoint_path=checkpoint_path,
        val_probs=val_probs,
        val_labels=val_labels,
        test_probs=test_probs,
        test_labels=test_labels,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
    )

    if output_path is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = results_dir / f"eval_pass_{stamp}.json"

    output_path = write_eval_payload(payload, output_path)
    log.info("Evaluation artifact written to %s", output_path)
    log.info("Calibrated test metrics: %s", payload["test_calibrated_threshold"])
    return output_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and write an eval_pass JSON artifact.")
    parser.add_argument("--config", default=str(ROOT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Defaults to best checkpoint.")
    parser.add_argument("--model-dir", default=None, help="Directory to search for checkpoints.")
    parser.add_argument("--results-dir", default=None, help="Directory for eval_pass JSON output.")
    parser.add_argument("--output", default=None, help="Explicit eval_pass JSON output path.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Evaluation device.")
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx", "mps", "cuda", "cpu", "torch"],
        default="auto",
        help="Compute backend for evaluation (mlx uses native MLX checkpoints).",
    )
    parser.add_argument("--model-type", default=None, help="Override config model type, e.g. TIMESFM.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    run_checkpoint_evaluation(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        output_path=Path(args.output) if args.output else None,
        device_request=args.device,
        model_type_override=args.model_type,
        backend_request=args.backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
