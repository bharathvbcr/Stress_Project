# lightning_module.py — PyTorch Lightning wrapper for stress-detection models
import logging
import json
import os
import tempfile
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

        # Build criterion — lazy import to break circular dependency with training.py
        from training import _get_criterion
        self.criterion = _get_criterion(config, pos_weight)

        # Register pos_weight as buffer to avoid redundant device transfers
        if pos_weight is not None:
            self.register_buffer("pos_weight_buffer", pos_weight)
        else:
            self.register_buffer("pos_weight_buffer", torch.tensor([1.0]))

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
        """Unpack a 5-tuple DataLoader batch; conditionally keep static features."""
        seq_features, static_features, labels, _subject_ids, _window_starts = batch
        static = self._static_for_model(static_features)
        return seq_features, static, labels

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, seq: torch.Tensor, static: Optional[torch.Tensor] = None):
        return self.model(seq, static)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        seq, static, labels = self._unpack(batch)
        
        # Ensure criterion uses the buffer-based weight if applicable
        # Note: If using custom FocalLoss, we may need to pass the weight explicitly
        # In this implementation, _get_criterion already built the module with the weight.
        # However, for true buffer tracking, we re-ensure device affinity.
        logits = self(seq, static).squeeze(-1)          # shape: (N,)
        loss = self.criterion(logits, labels.float())

        probs = torch.sigmoid(logits.detach())
        preds = (probs > 0.5).int()

        self.train_acc.update(preds, labels.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        seq, static, labels = self._unpack(batch)
        logits = self(seq, static).squeeze(-1)          # shape: (N,)
        loss = self.criterion(logits, labels.float())

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
                    "monitor":   "val_loss",
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
