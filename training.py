# training.py — Lightning-based training loop for stress-detection models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import collections

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from losses import FocalLoss
from utils import safe_get

# Global GPU precision hint — enables TF32 on Ampere+ GPUs (free perf, negligible accuracy loss)
torch.set_float32_matmul_precision("high")

# ==============================================================================
# == SOTA Optimizers (LION) ==
# ==============================================================================

class Lion(optim.Optimizer):
    """
    Lion: EvolVed Sign Momentum.
    Google Research implementation (sign-based momentum).
    Significantly more memory efficient than AdamW and often faster.
    Ref: https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad  = p.grad
                state = self.state[p]

                # State initialisation
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg      = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

                # Update via sign of interpolated momentum
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


log = logging.getLogger(__name__)

# ==============================================================================
# == Helper Functions for Training ==
# ==============================================================================

def _calculate_pos_weight(train_loader: DataLoader, device: torch.device) -> Optional[torch.Tensor]:
    """
    Calculates the positive class weight for BCEWithLogitsLoss based on the
    distribution of labels in the training data. Helps mitigate class imbalance.

    Optimized to access dataset labels directly if available, avoiding full loader iteration.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        device (torch.device): Device where the weight tensor should reside.

    Returns:
        Optional[torch.Tensor]: A tensor containing the positive class weight,
                                or None if calculation fails or data is balanced/single-class.
    """
    log.info("Calculating class weights for training set (for potential BCE use)...")
    train_label_counts = collections.Counter()
    num_train_samples  = 0
    pos_weight         = None

    if not train_loader:
        log.error("Train loader is None, cannot calculate weights.")
        return None

    try:
        # Optimisation: access labels directly from the dataset when possible
        if hasattr(train_loader.dataset, 'labels_np'):
            log.info("Accessing labels directly from train_loader.dataset (Optimized)...")
            labels_np          = train_loader.dataset.labels_np.astype(int)
            train_label_counts = collections.Counter(labels_np)
            num_train_samples  = len(labels_np)
        else:
            log.info("Dataset labels not directly accessible. Iterating loader (Slower)...")
            for batch_data in train_loader:
                if len(batch_data) != 5:
                    log.warning(
                        "Train loader batch did not contain 5 items. "
                        "Skipping weight calculation for this batch."
                    )
                    continue
                _, _, labels, _, _ = batch_data
                batch_counts = collections.Counter(labels.cpu().numpy().astype(int))
                train_label_counts.update(batch_counts)
                num_train_samples += len(labels)

        if num_train_samples == 0:
            log.error("Train loader has 0 samples. Cannot calculate weights.")
            return None

        count_0 = train_label_counts.get(0, 0)
        count_1 = train_label_counts.get(1, 0)
        log.info(f"Training label counts — 0: {count_0}, 1: {count_1}")

        if count_0 > 0 and count_1 > 0:
            weight_for_1 = count_0 / count_1
            pos_weight   = torch.tensor([weight_for_1], device=device, dtype=torch.float32)
            log.info(f"Calculated pos_weight for BCE: {pos_weight.item():.4f}")
        else:
            log.warning(
                "Training data has only 1 class present or one class has zero samples. "
                "Weighting disabled for BCE."
            )

    except Exception as e:
        log.error(f"Error calculating weights: {e}. Weighting disabled for BCE.", exc_info=True)
        pos_weight = None

    return pos_weight


def _get_criterion(config: Dict[str, Any], pos_weight: Optional[torch.Tensor]) -> nn.Module:
    """
    Initializes the loss function based on the configuration.
    Supports BCEWithLogitsLoss (with optional weighting) and FocalLoss.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        pos_weight (Optional[torch.Tensor]): Pre-calculated positive class weight for BCE.

    Returns:
        nn.Module: The initialized loss function module.

    Raises:
        ImportError: If FocalLoss is selected but the class is not available.
    """
    loss_function_type = safe_get(config, ['training_config', 'loss_function'], 'bce').lower()
    criterion          = None

    # --- Focal Loss ---
    if loss_function_type == 'focal':
        try:
            alpha     = safe_get(config, ['training_config', 'focal_loss_alpha'], 0.25)
            gamma     = safe_get(config, ['training_config', 'focal_loss_gamma'], 2.0)
            criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
            log.info(f"Using FocalLoss (alpha={alpha}, gamma={gamma})")
        except NameError:
            log.critical("FocalLoss selected but class definition not found! Aborting.")
            raise ImportError("FocalLoss class not found.")
        except Exception as e:
            log.error(f"Error initializing FocalLoss: {e}. Falling back to BCE.")
            loss_function_type = 'bce'

    # --- BCEWithLogitsLoss (default or fallback) ---
    if criterion is None:
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            log.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.4f}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            log.info("Using standard BCEWithLogitsLoss (no weighting).")

    return criterion


def _get_optimizer(model: nn.Module, config: Dict[str, Any], device: torch.device) -> optim.Optimizer:
    """
    Initializes the optimizer based on configuration.
    Only optimizes parameters with requires_grad=True, making it compatible
    with frozen-backbone models such as StressTimesFM in Phase-1 training.

    Args:
        model: The model parameters to optimize.
        config: Configuration dictionary.
        device: The current device (used for fused kernel availability check).

    Returns:
        optim.Optimizer: The initialized optimizer.
    """
    lr             = safe_get(config, ['training_config', 'learning_rate'], 0.001)
    optimizer_type = safe_get(config, ['training_config', 'optimizer'], 'adam').lower()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable      = sum(p.numel() for p in trainable_params)
    n_total          = sum(p.numel() for p in model.parameters())
    log.info(
        f"Optimizer will train {n_trainable:,} / {n_total:,} parameters "
        f"({100 * n_trainable / max(n_total, 1):.1f}% trainable)."
    )

    if not trainable_params:
        log.error(
            "No trainable parameters found! All parameters are frozen. "
            "The optimizer will be created but no weights will be updated."
        )
        trainable_params = list(model.parameters())[:1]

    if optimizer_type == 'adam':
        is_cuda   = device.type == 'cuda'
        optimizer = optim.Adam(trainable_params, lr=lr, fused=is_cuda)
        log.info(f"Using Adam optimizer (LR={lr}, Fused={is_cuda})")
    elif optimizer_type == 'adamw':
        weight_decay = safe_get(config, ['training_config', 'weight_decay'], 0.01)
        is_cuda      = device.type == 'cuda'
        optimizer    = optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay, fused=is_cuda
        )
        log.info(f"Using AdamW optimizer (LR={lr}, Weight Decay={weight_decay}, Fused={is_cuda})")
    elif optimizer_type == 'lion':
        weight_decay = safe_get(config, ['training_config', 'weight_decay'], 0.01)
        optimizer    = Lion(trainable_params, lr=lr, weight_decay=weight_decay)
        log.info(f"Using LION optimizer (Sign-Based Momentum, LR={lr}, WD={weight_decay})")
    elif optimizer_type == 'sgd':
        momentum  = safe_get(config, ['training_config', 'momentum'], 0.9)
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum)
        log.info(f"Using SGD optimizer (LR={lr}, Momentum={momentum})")
    else:
        log.warning(f"Unknown optimizer type '{optimizer_type}'. Defaulting to Adam.")
        optimizer = optim.Adam(trainable_params, lr=lr)

    return optimizer


# ==============================================================================
# == History Callback ==
# ==============================================================================

class _HistoryCallback(L.Callback):
    """Accumulates per-epoch metrics into a history dict compatible with the legacy API."""

    def __init__(self):
        self.history = {
            "train_loss":    [],
            "val_loss":      [],
            "val_accuracy":  [],
            "val_f1":        [],
        }

    def _scalar(self, v):
        if v is None:
            return float("nan")
        return v.item() if hasattr(v, "item") else float(v)

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.history["train_loss"].append(self._scalar(m.get("train_loss")))

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.history["val_loss"].append(self._scalar(m.get("val_loss")))
        self.history["val_accuracy"].append(self._scalar(m.get("val_acc")))
        self.history["val_f1"].append(self._scalar(m.get("val_f1")))


# ==============================================================================
# == Main Training Function ==
# ==============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Optional[str],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Trains the model using PyTorch Lightning.

    Args:
        model:        The PyTorch model to train (already instantiated).
        train_loader: DataLoader for the training set.
        val_loader:   DataLoader for the validation set (optional).
        config:       Configuration dictionary.
        device:       The device (CPU or CUDA) to train on.
        output_dir:   Directory to save the best model checkpoint (optional).

    Returns:
        Tuple:
            - best_state_dict (Optional[Dict]): State dict of the best model by val_loss.
            - history (Optional[Dict]):         Per-epoch metrics dict.
    """
    log.info("--- Starting Model Training (Lightning) ---")

    # Lazy import to avoid circular dependency with lightning_module.py
    from lightning_module import StressLightningModule

    # --- Read config values ---
    epochs     = safe_get(config, ['training_config', 'epochs'], 50)
    patience   = safe_get(config, ['early_stopping', 'patience'], 10)
    min_delta  = safe_get(config, ['early_stopping', 'min_delta'], 0.001)
    accum      = safe_get(config, ['training_config', 'accumulation_steps'], 1)
    use_amp    = safe_get(config, ['training_config', 'mixed_precision'], True)
    is_cuda    = (device.type == "cuda")

    # Enable cudnn autotuner for fixed-size inputs (free throughput on GPU)
    if is_cuda:
        torch.backends.cudnn.benchmark = True

    # Determine precision string for Lightning Trainer
    if use_amp and is_cuda:
        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    else:
        precision = "32"
    log.info(f"Trainer precision: {precision}")

    # --- Build Lightning module ---
    pos_weight = _calculate_pos_weight(train_loader, device)
    lit_model  = StressLightningModule(model, config, pos_weight)

    # Optional: torch.compile for graph fusion (PyTorch 2.0+, skip on CPU or non-CUDA)
    use_compile = safe_get(config, ['training_config', 'torch_compile'], False)
    if use_compile and is_cuda:
        try:
            lit_model.model = torch.compile(lit_model.model)
            log.info("torch.compile applied to model (graph fusion enabled).")
        except Exception as compile_err:
            log.warning(f"torch.compile failed (non-fatal): {compile_err}")

    # --- Callbacks ---
    history_cb = _HistoryCallback()
    callbacks  = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        history_cb,
    ]

    # Estimate steps per epoch for log_every_n_steps (avoid Lightning's default=50 missing small datasets)
    steps_per_epoch = max(1, len(train_loader) // accum)
    log_every_n_steps = max(1, min(50, steps_per_epoch))

    ckpt_callback = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ckpt_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename="best_model-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_weights_only=True,
        )
        callbacks.append(ckpt_callback)

    # --- Logger ---
    csv_logger = CSVLogger(
        save_dir=output_dir if output_dir else "outputs/logs",
        name="stress_training",
    )

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if is_cuda else "cpu",
        devices=1,
        precision=precision,
        accumulate_grad_batches=accum,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=csv_logger,
        num_sanity_val_steps=0,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=False,
        enable_checkpointing=(output_dir is not None),
        log_every_n_steps=log_every_n_steps,
    )

    # --- Fit ---
    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    history = history_cb.history

    # --- Load best checkpoint weights (not last-epoch weights) ---
    best_state_dict = None
    if ckpt_callback is not None and ckpt_callback.best_model_path:
        try:
            ckpt = torch.load(ckpt_callback.best_model_path, map_location=device, weights_only=True)
            # ModelCheckpoint with save_weights_only=True stores under "state_dict" key
            raw = ckpt.get("state_dict", ckpt)
            # Strip Lightning "model." prefix if present
            best_state_dict = {
                k[len("model."):] if k.startswith("model.") else k: v
                for k, v in raw.items()
            }
            log.info(f"Loaded best checkpoint from: {ckpt_callback.best_model_path}")
        except Exception as ckpt_err:
            log.warning(f"Could not load best checkpoint (non-fatal): {ckpt_err}. Returning last-epoch weights.")

    if best_state_dict is None:
        best_state_dict = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in lit_model.state_dict().items()
            if k.startswith("model.")
        } or lit_model.model.state_dict()

    # --- Optional HF Hub push ---
    hf_repo_id = safe_get(config, ['save_paths', 'hf_hub_repo_id'], None)
    if hf_repo_id:
        try:
            val_f1_vals = [v for v in history.get("val_f1", []) if not np.isnan(v)]
            metrics = {"best_val_f1": max(val_f1_vals)} if val_f1_vals else {}
            lit_model.push_to_hub(hf_repo_id, metrics=metrics)
        except Exception as hub_err:
            log.warning(f"HF Hub push failed (non-fatal): {hub_err}")

    log.info("--- Model Training Finished (Lightning) ---")
    return best_state_dict, history
