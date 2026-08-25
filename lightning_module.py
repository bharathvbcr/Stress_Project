# lightning_module.py — PyTorch Lightning wrapper for stress-detection models
import logging
import json
import os
import tempfile
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
import lightning as L
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
)

from utils import safe_get

log = logging.getLogger(__name__)


def _diagnostics_config(config: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics_cfg = safe_get(config, ["training_config", "diagnostics"], None)
    if not isinstance(diagnostics_cfg, dict):
        diagnostics_cfg = safe_get(config, ["diagnostics"], None)
    return diagnostics_cfg if isinstance(diagnostics_cfg, dict) else {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "disabled", "none"}:
            return False
    return default


def _as_positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _as_nonnegative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _step_detail_enabled(config: Dict[str, Any], batch_idx: int) -> bool:
    diagnostics_cfg = _diagnostics_config(config)
    if not _coerce_bool(diagnostics_cfg.get("enabled", True), True):
        return False
    if not _coerce_bool(diagnostics_cfg.get("step_detail", True), True):
        return False
    interval = _as_positive_int(
        diagnostics_cfg.get("step_detail_interval", diagnostics_cfg.get("log_every_n_steps", 1)),
        1,
    )
    return batch_idx == 0 or (batch_idx + 1) % interval == 0


def _step_cuda_sync_enabled(config: Dict[str, Any], batch_idx: int, device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    diagnostics_cfg = _diagnostics_config(config)
    interval = _as_nonnegative_int(
        diagnostics_cfg.get("step_detail_cuda_sync_interval", diagnostics_cfg.get("cuda_sync_interval", 0)),
        0,
    )
    return interval > 0 and (batch_idx == 0 or (batch_idx + 1) % interval == 0)


def _maybe_cuda_synchronize(device: torch.device, enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        torch.cuda.synchronize(device)
        return True
    except Exception as exc:
        log.warning("CUDA synchronize failed during step diagnostics: %s", exc)
        return False


def _shape_summary(value: Optional[torch.Tensor]) -> str:
    if value is None:
        return "None"
    if hasattr(value, "shape"):
        return f"shape={list(value.shape)} dtype={value.dtype} device={value.device}"
    return type(value).__name__


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu())


class StressLightningModule(L.LightningModule):
    """
    PyTorch Lightning module wrapping any stress-detection model.
    Handles training/validation steps, optimizer/scheduler configuration,
    metric logging via torchmetrics, and optional Hugging Face Hub upload.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config

        # Save config as hyperparameters (skip large non-serialisable objects)
        self.save_hyperparameters(ignore=["model", "pos_weight"])
        self.hparams["model_input_dim_sequence"] = getattr(model, "input_dim_sequence", None)
        self.hparams["model_input_dim_static"] = getattr(model, "input_dim_static", None)

        # Build criterion — lazy import to break circular dependency with training.py
        from training import _get_criterion
        self.criterion = _get_criterion(config, pos_weight)

        # Metric the LR scheduler should monitor; train_model overrides this to
        # "train_loss" when no validation loader is supplied.
        self._monitor_metric = "val_loss"

        # Performance: Internal CUDA Events for nanosecond timing
        self.enable_cuda_events = torch.cuda.is_available()
        if self.enable_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        # torchmetrics — all binary classification
        self.train_acc  = BinaryAccuracy()
        self.val_acc    = BinaryAccuracy()
        self.val_f1     = BinaryF1Score()
        self.val_prec   = BinaryPrecision()
        self.val_rec    = BinaryRecall()
        self.val_auroc  = BinaryAUROC()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _static_for_model(self, static: torch.Tensor) -> Optional[torch.Tensor]:
        """Return static features only when the wrapped model actually uses them."""
        if hasattr(self.model, "input_dim_static") and self.model.input_dim_static > 0:
            return static
        return None

    def _unpack(self, batch):
        """Support both Arrow dict batches and the legacy 5-tuple loader output."""
        if isinstance(batch, dict):
            seq_features = batch["sequence"]
            static_features = batch.get("static")
            labels = batch["label"]
        else:
            seq_features, static_features, labels, *_metadata = batch
        static = self._static_for_model(static_features)
        return seq_features, static, labels

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, seq: torch.Tensor, static: Optional[torch.Tensor] = None):
        out = self.model(seq, static)

        # Stability: Check for NaNs/Infs in activations
        if self.config.get('debug_activations', False):
            if torch.isnan(out).any() or torch.isinf(out).any():
                log.error(f"NaN/Inf detected in model output! Shapes: seq={seq.shape}, static={static.shape if static is not None else 'None'}")

        return out

    def freeze_backbone(self) -> None:
        """Expose backbone freeze controls to training entrypoints."""
        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

    def unfreeze_backbone(self, last_n_blocks: int = 4) -> None:
        """Expose backbone unfreeze controls to training entrypoints."""
        if hasattr(self.model, "unfreeze_backbone"):
            self.model.unfreeze_backbone(last_n_blocks=last_n_blocks)

    def apply_backbone_lora(self, last_n_blocks: int = 4) -> None:
        if hasattr(self.model, "apply_backbone_lora"):
            self.model.apply_backbone_lora(last_n_blocks=last_n_blocks)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        step_detail = _step_detail_enabled(self.config, batch_idx)
        step_start = time.perf_counter()

        seq, static, labels = self._unpack(batch)
        unpack_done = time.perf_counter()
        cuda_sync_step = _step_cuda_sync_enabled(self.config, batch_idx, seq.device)
        # CUDA events only matter when internal GPU profiling is on; recording them
        # otherwise adds per-step overhead for numbers that are never read.
        cuda_events_step = (
            self.config.get('internal_gpu_profiling', False)
            and self.enable_cuda_events
            and seq.device.type == "cuda"
        )
        if cuda_events_step:
            self.start_event.record()
        _maybe_cuda_synchronize(seq.device, cuda_sync_step)

        logits = self(seq, static).squeeze(-1)
        _maybe_cuda_synchronize(seq.device, cuda_sync_step)
        forward_done = time.perf_counter()

        loss = self.criterion(logits, labels.float())
        _maybe_cuda_synchronize(seq.device, cuda_sync_step)
        loss_done = time.perf_counter()

        if cuda_events_step:
            self.end_event.record()
            torch.cuda.synchronize()
            ms = self.start_event.elapsed_time(self.end_event)
            self.log("perf/gpu_kernel_ms", ms, on_step=True, on_epoch=False)

        probs = torch.sigmoid(logits.detach())
        preds = (probs > 0.5).int()

        self.train_acc.update(preds, labels.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=False)

        metrics_done = time.perf_counter()
        if step_detail:
            unpack_ms = (unpack_done - step_start) * 1000
            forward_ms = (forward_done - unpack_done) * 1000
            loss_ms = (loss_done - forward_done) * 1000
            metric_ms = (metrics_done - loss_done) * 1000
            total_ms = (metrics_done - step_start) * 1000

            self.log("time/train_unpack_ms", unpack_ms, on_step=True, on_epoch=False)
            self.log("time/train_forward_ms", forward_ms, on_step=True, on_epoch=False)
            self.log("time/train_loss_ms", loss_ms, on_step=True, on_epoch=False)
            self.log("time/train_metrics_ms", metric_ms, on_step=True, on_epoch=False)
            self.log("time/train_step_internal_ms", total_ms, on_step=True, on_epoch=False)
            self.log("diag/train_logit_abs_max", logits.detach().abs().max(), on_step=True, on_epoch=False)
            self.log("diag/train_label_mean", labels.detach().float().mean(), on_step=True, on_epoch=False)

            log.info(
                "Train step detail batch=%s unpack=%.3fms forward=%.3fms loss=%.3fms metrics=%.3fms total=%.3fms loss_value=%.6f logit_abs_max=%.6f label_mean=%.6f cuda_sync=%s seq[%s] static[%s] labels[%s]",
                batch_idx + 1,
                unpack_ms,
                forward_ms,
                loss_ms,
                metric_ms,
                total_ms,
                _scalar(loss),
                _scalar(logits.detach().abs().max()),
                _scalar(labels.detach().float()),
                cuda_sync_step,
                _shape_summary(seq),
                _shape_summary(static),
                _shape_summary(labels),
            )
        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        step_detail = _step_detail_enabled(self.config, batch_idx)
        step_start = time.perf_counter()

        seq, static, labels = self._unpack(batch)
        unpack_done = time.perf_counter()
        cuda_sync_step = _step_cuda_sync_enabled(self.config, batch_idx, seq.device)
        _maybe_cuda_synchronize(seq.device, cuda_sync_step)

        logits = self(seq, static).squeeze(-1)          # shape: (N,)
        _maybe_cuda_synchronize(seq.device, cuda_sync_step)
        forward_done = time.perf_counter()

        loss = self.criterion(logits, labels.float())
        _maybe_cuda_synchronize(seq.device, cuda_sync_step)
        loss_done = time.perf_counter()

        probs = torch.sigmoid(logits.detach())
        preds = (probs > 0.5).int()
        labels_int = labels.int()

        self.val_acc.update(preds,    labels_int)
        self.val_f1.update(preds,     labels_int)
        self.val_prec.update(preds,   labels_int)
        self.val_rec.update(preds,    labels_int)
        self.val_auroc.update(probs,  labels_int)

        self.log("val_loss",  loss,            on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc",   self.val_acc,    on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1",    self.val_f1,     on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_prec",  self.val_prec,   on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_rec",   self.val_rec,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_auroc", self.val_auroc,  on_step=False, on_epoch=True, prog_bar=False)

        metrics_done = time.perf_counter()
        if step_detail:
            unpack_ms = (unpack_done - step_start) * 1000
            forward_ms = (forward_done - unpack_done) * 1000
            loss_ms = (loss_done - forward_done) * 1000
            metric_ms = (metrics_done - loss_done) * 1000
            total_ms = (metrics_done - step_start) * 1000

            self.log("time/val_unpack_ms", unpack_ms, on_step=True, on_epoch=False)
            self.log("time/val_forward_ms", forward_ms, on_step=True, on_epoch=False)
            self.log("time/val_loss_ms", loss_ms, on_step=True, on_epoch=False)
            self.log("time/val_metrics_ms", metric_ms, on_step=True, on_epoch=False)
            self.log("time/val_step_internal_ms", total_ms, on_step=True, on_epoch=False)
            self.log("diag/val_logit_abs_max", logits.detach().abs().max(), on_step=True, on_epoch=False)
            self.log("diag/val_label_mean", labels.detach().float().mean(), on_step=True, on_epoch=False)

            log.info(
                "Validation step detail batch=%s unpack=%.3fms forward=%.3fms loss=%.3fms metrics=%.3fms total=%.3fms loss_value=%.6f logit_abs_max=%.6f label_mean=%.6f cuda_sync=%s seq[%s] static[%s] labels[%s]",
                batch_idx + 1,
                unpack_ms,
                forward_ms,
                loss_ms,
                metric_ms,
                total_ms,
                _scalar(loss),
                _scalar(logits.detach().abs().max()),
                _scalar(labels.detach().float()),
                cuda_sync_step,
                _shape_summary(seq),
                _shape_summary(static),
                _shape_summary(labels),
            )

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Lazy import to break circular dependency with training.py
        from training import _get_optimizer

        optimizer = _get_optimizer(self.model, self.config, self.device)

        lr_cfg = safe_get(self.config, ["training_config", "lr_scheduler"], {})
        if not lr_cfg.get("enabled", True):
            return optimizer

        scheduler_type = lr_cfg.get("type", "plateau").lower()

        if scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=lr_cfg.get("factor", 0.5),
                patience=lr_cfg.get("patience", 3),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor":   getattr(self, "_monitor_metric", "val_loss"),
                    "interval":  "epoch",
                    "frequency": 1,
                },
            }

        elif scheduler_type == "onecycle":
            base_lr    = safe_get(self.config, ["training_config", "learning_rate"], 1e-3)
            max_lr     = lr_cfg.get("max_lr", base_lr * 10)
            total_steps = self.trainer.estimated_stepping_batches
            scheduler  = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=lr_cfg.get("pct_start", 0.3),
                div_factor=lr_cfg.get("div_factor", 25.0),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval":  "step",
                    "frequency": 1,
                },
            }

        elif scheduler_type == "cosine":
            epochs    = safe_get(self.config, ["training_config", "epochs"], 50)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval":  "epoch",
                    "frequency": 1,
                },
            }

        # Unknown scheduler type — return optimizer only
        log.warning(
            f"Unknown lr_scheduler type '{scheduler_type}'. Returning optimizer only."
        )
        return optimizer

    # ------------------------------------------------------------------
    # Hugging Face Hub upload
    # ------------------------------------------------------------------

    def push_to_hub(self, repo_id: str, metrics: Optional[Dict[str, Any]] = None):
        """Upload model weights, config, and a model card README to Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            log.warning(
                "huggingface_hub is not installed. "
                "Run `pip install huggingface-hub` to enable push_to_hub."
            )
            return

        api     = HfApi()
        metrics = metrics or {}

        with tempfile.TemporaryDirectory() as tmpdir:

            # 1. Model weights
            weights_path = os.path.join(tmpdir, "pytorch_model.bin")
            torch.save(self.model.state_dict(), weights_path)

            # 2. Config JSON
            config_path = os.path.join(tmpdir, "stress_config.json")
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(self.config, fh, indent=2, default=str)

            # 3. README / model card
            metrics_md = "\n".join(
                f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}"
                for k, v in metrics.items()
            )
            model_config_json = json.dumps(
                self.config.get("model_config", {}), indent=2, default=str
            )
            readme_content = f"""---
library_name: pytorch
tags:
  - stress-detection
  - time-series
  - binary-classification
---

# Stress Detection Model

Trained with PyTorch Lightning on physiological time-series data.

## Metrics

{metrics_md if metrics_md else "_No metrics provided._"}

## Model Configuration

```json
{model_config_json}
```
"""
            readme_path = os.path.join(tmpdir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as fh:
                fh.write(readme_content)

            # 4. Upload
            api.create_repo(repo_id=repo_id, exist_ok=True)
            for filename in ("pytorch_model.bin", "stress_config.json", "README.md"):
                api.upload_file(
                    path_or_fileobj=os.path.join(tmpdir, filename),
                    path_in_repo=filename,
                    repo_id=repo_id,
                )

        hub_url = f"https://huggingface.co/{repo_id}"
        log.info(f"Model uploaded to Hugging Face Hub: {hub_url}")
