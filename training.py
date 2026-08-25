# training.py — Lightning-based training loop for stress-detection models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import collections

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler
from lightning.pytorch.loggers import CSVLogger

from callbacks import (
    HistoryCallback,
    BatchTimeLoggingCallback,
    GradientNormLoggingCallback,
    NumericalStabilityCallback,
    SystemDiagnosticsCallback,
    ModelComplexityCallback,
    log_gpu_diagnostics as _callback_log_gpu_diagnostics,
    _maybe_cuda_synchronize,
    _cuda_memory_snapshot,
    _format_cuda_snapshot,
)

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
                # Support both Arrow dict batches and the legacy 5-tuple loader output.
                if isinstance(batch_data, dict):
                    labels = batch_data.get("label")
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                    labels = batch_data[2]
                else:
                    labels = None
                if labels is None:
                    log.warning(
                        "Train loader batch had no recognizable labels "
                        "(expected dict['label'] or tuple index 2). Skipping batch."
                    )
                    continue
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
# == Callbacks & Utilities ==
# ==============================================================================

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


def _get_diagnostics_config(config: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics_cfg = safe_get(config, ["training_config", "diagnostics"], None)
    if not isinstance(diagnostics_cfg, dict):
        diagnostics_cfg = safe_get(config, ["diagnostics"], None)
    return diagnostics_cfg if isinstance(diagnostics_cfg, dict) else {}


def _config_value(config: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    for path in paths:
        value = safe_get(config, path, None)
        if value is not None:
            return value
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


def _safe_len(value: Any) -> str:
    try:
        return str(len(value))
    except (TypeError, AttributeError):
        return "unknown"


def _log_trainer_metrics(trainer: Any, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    loggers = getattr(trainer, "loggers", None)
    if not loggers:
        logger_obj = getattr(trainer, "logger", None)
        loggers = [logger_obj] if logger_obj else []
    for logger_obj in loggers:
        if logger_obj is None:
            continue
        try:
            logger_obj.log_metrics(metrics, step=step)
        except Exception as exc:
            log.debug("Trainer metric logging failed for %s: %s", metrics.keys(), exc)


def log_gpu_diagnostics(log_obj: logging.Logger, device: torch.device) -> None:
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        log_obj.info("CUDA is not active for this run; skipping GPU diagnostics.")
        return
    _callback_log_gpu_diagnostics(log_obj, device)


@contextmanager
def training_stage(
    log_obj: logging.Logger,
    stage_name: str,
    device: Optional[torch.device] = None,
    **context: Any,
):
    start = time.perf_counter()
    context_text = " ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    context_suffix = f" {context_text}" if context_text else ""
    start_snapshot = _cuda_memory_snapshot(device) if device is not None else None
    log_obj.info(
        "[Training Stage] START %s%s%s",
        stage_name,
        context_suffix,
        _format_cuda_snapshot(start_snapshot),
    )
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        end_snapshot = _cuda_memory_snapshot(device) if device is not None else None
        log_obj.exception(
            "[Training Stage] FAILED %s elapsed=%.2fs%s",
            stage_name,
            elapsed,
            _format_cuda_snapshot(end_snapshot),
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        end_snapshot = _cuda_memory_snapshot(device) if device is not None else None
        log_obj.info(
            "[Training Stage] END %s elapsed=%.2fs%s",
            stage_name,
            elapsed,
            _format_cuda_snapshot(end_snapshot),
        )


def build_training_profiler(config: Dict[str, Any], run_name: str = "training"):
    profiler_cfg = _config_value(
        config,
        [
            ["training_config", "diagnostics", "profiler"],
            ["training_config", "diagnostics", "profile_training"],
            ["diagnostics", "profiler"],
            ["diagnostics", "profile_training"],
            ["training_config", "profile_training"],
            ["training_config", "profiler"],
            ["profile_training"],
            ["profiler"],
        ],
        False,
    )
    env_enabled = _coerce_bool(os.environ.get("STRESS_PROFILE_TRAINING"), False)
    env_type = os.environ.get("STRESS_TRAINING_PROFILER")
    env_dir = os.environ.get("STRESS_PROFILER_DIR")

    if isinstance(profiler_cfg, dict):
        enabled = _coerce_bool(profiler_cfg.get("enabled"), False) or env_enabled
        profiler_type = str(env_type or profiler_cfg.get("type", "pytorch")).lower()
        dirpath = env_dir or profiler_cfg.get("dirpath") or "outputs/profiler_results"
    elif isinstance(profiler_cfg, str):
        profiler_type = str(env_type or profiler_cfg).lower()
        enabled = profiler_type not in {"0", "false", "no", "n", "off", "disabled", "none"} or env_enabled
        if profiler_type in {"1", "true", "yes", "y", "on", "enabled"}:
            profiler_type = "pytorch"
        dirpath = env_dir or "outputs/profiler_results"
    else:
        enabled = _coerce_bool(profiler_cfg, False) or env_enabled
        profiler_type = str(env_type or "pytorch").lower()
        dirpath = env_dir or "outputs/profiler_results"

    if not enabled:
        return None

    try:
        os.makedirs(dirpath, exist_ok=True)
        filename = f"{run_name}_{profiler_type}_profiler"
        if profiler_type in {"advanced", "lightning"}:
            return AdvancedProfiler(dirpath=dirpath, filename=filename)
        return PyTorchProfiler(dirpath=dirpath, filename=filename)
    except Exception as exc:
        log.warning("Could not initialize training profiler '%s': %s", profiler_type, exc)
        return None


def get_standard_callbacks(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    is_cuda: bool = False,
    enable_lr_monitor: bool = True,
    monitor: str = "val_loss",
) -> List[L.Callback]:
    """
    Builds a standard list of callbacks including performance monitoring.

    Args:
        monitor: Metric to drive EarlyStopping and ModelCheckpoint. Defaults to
                 "val_loss"; callers without a validation loader should pass
                 "train_loss" to avoid monitoring a metric that is never logged.
    """
    patience = _config_value(
        config,
        [["early_stopping", "patience"], ["training_config", "early_stopping", "patience"]],
        10,
    )
    min_delta = _config_value(
        config,
        [["early_stopping", "min_delta"], ["training_config", "early_stopping", "min_delta"]],
        0.001,
    )
    diagnostics_cfg = _get_diagnostics_config(config)
    diagnostics_enabled = _coerce_bool(diagnostics_cfg.get("enabled", True), True)
    step_log_interval = _as_positive_int(diagnostics_cfg.get("log_every_n_steps", 50), 50)
    cuda_sync_interval = _as_nonnegative_int(diagnostics_cfg.get("cuda_sync_interval", 0), 0)
    memory_log_interval = _as_positive_int(diagnostics_cfg.get("memory_log_interval", 100), 100)
    epoch_memory_summary = _coerce_bool(diagnostics_cfg.get("epoch_memory_summary", True), True)
    weight_stats = _coerce_bool(diagnostics_cfg.get("weight_stats", True), True)
    stability_check_interval = _as_positive_int(diagnostics_cfg.get("stability_check_interval", 50), 50)

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            mode="min",
        ),
    ]

    if enable_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if diagnostics_enabled:
        callbacks.extend([
            BatchTimeLoggingCallback(
                log_interval=step_log_interval,
                cuda_sync_interval=cuda_sync_interval if is_cuda else 0,
                memory_log_interval=memory_log_interval,
                epoch_memory_summary=epoch_memory_summary,
            ),
            GradientNormLoggingCallback(
                log_interval=step_log_interval,
                weight_stats=weight_stats,
            ),
            NumericalStabilityCallback(check_every_n_steps=stability_check_interval),
            ModelComplexityCallback(),
            SystemDiagnosticsCallback(),
        ])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath=output_dir,
            filename="best_model-{epoch:02d}-{" + monitor + ":.4f}",
            monitor=monitor,
            save_top_k=1,
            mode="min",
            save_weights_only=True,
        ))

    return callbacks


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
) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
    """
    Trains the model using PyTorch Lightning.

    Args:
        model:        The PyTorch model to train (already instantiated).
        train_loader: DataLoader for the training set.
        val_loader:   DataLoader for the validation set (optional). When None,
                      early stopping / checkpointing / the LR scheduler fall back
                      to monitoring train_loss instead of val_loss.
        config:       Configuration dictionary.
        device:       The device (CPU or CUDA) to train on.
        output_dir:   Directory to save the best model checkpoint (optional).

    Returns:
        Tuple:
            - best_state_dict (Optional[Dict]):    State dict of the best model.
            - history (Optional[List[Dict]]):      List of per-epoch metrics dicts.
    """
    log.info("--- Starting Model Training (Lightning) ---")

    # Lazy import to avoid circular dependency with lightning_module.py
    from lightning_module import StressLightningModule

    # --- Read config values ---
    epochs     = safe_get(config, ['training_config', 'epochs'], 50)
    accum      = safe_get(config, ['training_config', 'accumulation_steps'], 1)
    use_amp    = safe_get(config, ['training_config', 'mixed_precision'], True)
    is_cuda    = (device.type == "cuda")
    is_mps     = (device.type == "mps")

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
    log.info(
        "Training config: epochs=%s accumulation_steps=%s mixed_precision=%s precision=%s train_batches=%s val_batches=%s",
        epochs,
        accum,
        use_amp,
        precision,
        _safe_len(train_loader),
        _safe_len(val_loader),
    )

    # --- Sync Point Debugging ---
    debug_cuda_sync = _config_value(config, [["debug_cuda_sync"], ["training_config", "debug_cuda_sync"]], False)
    if is_cuda and _coerce_bool(debug_cuda_sync, False):
        try:
            log.info("CUDA sync debugging enabled. PyTorch will warn on CPU-GPU sync points.")
            torch.cuda.set_sync_debug_mode(1)
        except Exception as sync_debug_err:
            log.warning("Could not enable CUDA sync debugging: %s", sync_debug_err)

    # --- GPU Diagnostics ---
    with training_stage(log, "gpu_diagnostics", device):
        log_gpu_diagnostics(log, device)

    # Determine which metric to monitor — val_loss requires a validation loader.
    has_validation = val_loader is not None
    monitor_metric = "val_loss" if has_validation else "train_loss"
    if not has_validation:
        log.warning(
            "No validation loader provided. Early stopping, checkpointing, and the "
            "LR scheduler will monitor 'train_loss' instead of 'val_loss'."
        )

    # --- Build Lightning module ---
    with training_stage(log, "build_lightning_module", device):
        # pos_weight only feeds BCE; focal loss ignores it, so skip the (potentially
        # full-epoch) computation when focal loss is configured.
        loss_function_type = safe_get(config, ['training_config', 'loss_function'], 'bce').lower()
        if loss_function_type == 'focal':
            pos_weight = None
            log.info("Focal loss configured; skipping pos_weight computation.")
        else:
            pos_weight = _calculate_pos_weight(train_loader, device)
        lit_model = StressLightningModule(model, config, pos_weight)
        # Tell the module which metric the LR scheduler should monitor.
        lit_model._monitor_metric = monitor_metric

    # Optional: torch.compile for graph fusion (PyTorch 2.0+, skip on CPU or non-CUDA).
    # Never compile TimesFM: torch.compile strips forward hooks used for embeddings.
    use_compile = safe_get(config, ['training_config', 'torch_compile'], False)
    model_type = safe_get(config, ['model_config', 'type'], '').upper()
    if use_compile and model_type == "TIMESFM":
        log.warning(
            "Ignoring training.torch_compile for TIMESFM: compilation breaks "
            "forward-hook embedding extraction (flat predictions)."
        )
        use_compile = False
    if use_compile and is_cuda:
        with training_stage(log, "torch_compile", device):
            try:
                lit_model.model = torch.compile(lit_model.model)
                log.info("torch.compile applied to model (graph fusion enabled).")
            except Exception as compile_err:
                log.warning("torch.compile failed (non-fatal): %s", compile_err)

    # --- Callbacks ---
    with training_stage(log, "build_callbacks", device):
        history_cb = HistoryCallback()
        callbacks = get_standard_callbacks(
            config=config,
            output_dir=output_dir,
            is_cuda=is_cuda,
            monitor=monitor_metric,
        )
        callbacks.append(history_cb)

    # Estimate steps per epoch for log_every_n_steps (avoid Lightning's default=50 missing small datasets)
    steps_per_epoch = max(1, len(train_loader) // accum)
    log_every_n_steps = max(1, min(50, steps_per_epoch))

    # --- Logger ---
    with training_stage(log, "build_logger_profiler", device):
        csv_logger = CSVLogger(
            save_dir=output_dir if output_dir else "outputs/logs",
            name="stress_training",
        )
        profiler = build_training_profiler(config, run_name="train_model")

    # --- Trainer ---
    with training_stage(log, "initialize_trainer", device):
        enable_progress_bar = _coerce_bool(
            _config_value(config, [["training_config", "enable_progress_bar"], ["enable_progress_bar"]], False),
            False,
        )
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if is_cuda else ("mps" if is_mps else "cpu"),
            devices=1,
            precision=precision,
            accumulate_grad_batches=accum,
            gradient_clip_val=1.0,
            callbacks=callbacks,
            logger=csv_logger,
            profiler=profiler,
            num_sanity_val_steps=0,
            enable_model_summary=True,
            enable_progress_bar=enable_progress_bar,
            deterministic=False,
            enable_checkpointing=(output_dir is not None),
            log_every_n_steps=log_every_n_steps,
        )

    # --- Fit ---
    with training_stage(log, "trainer_fit", device, epochs=epochs, steps_per_epoch=steps_per_epoch):
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    history = history_cb.history

    # --- Load best checkpoint weights (not last-epoch weights) ---
    best_state_dict = None
    with training_stage(log, "load_best_checkpoint", device):
        ckpt_callback = next((c for c in callbacks if isinstance(c, ModelCheckpoint)), None)
        if ckpt_callback is not None and ckpt_callback.best_model_path:
            try:
                ckpt = torch.load(ckpt_callback.best_model_path, map_location=device, weights_only=True)
                raw = ckpt.get("state_dict", ckpt)
                # Keep only the wrapped model's weights. Dropping non-"model." keys
                # (e.g. pos_weight_buffer, criterion.pos_weight) avoids strict-load
                # failures when this dict is loaded onto the bare model downstream.
                best_state_dict = {
                    k[len("model."):]: v
                    for k, v in raw.items()
                    if k.startswith("model.")
                } or lit_model.model.state_dict()
                log.info("Loaded best checkpoint from: %s", ckpt_callback.best_model_path)
            except Exception as ckpt_err:
                log.warning("Could not load best checkpoint (non-fatal): %s. Returning last-epoch weights.", ckpt_err)

    if best_state_dict is None:
        best_state_dict = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in lit_model.state_dict().items()
            if k.startswith("model.")
        } or lit_model.model.state_dict()

    # --- Optional HF Hub push ---
    hf_repo_id = safe_get(config, ['save_paths', 'hf_hub_repo_id'], None)
    if hf_repo_id:
        with training_stage(log, "push_to_hf_hub", device, repo=hf_repo_id):
            try:
                # history is a list of per-epoch metric dicts (see HistoryCallback).
                val_f1_vals = [
                    epoch_metrics["val_f1"]
                    for epoch_metrics in (history or [])
                    if isinstance(epoch_metrics, dict)
                    and "val_f1" in epoch_metrics
                    and not np.isnan(epoch_metrics["val_f1"])
                ]
                metrics = {"best_val_f1": max(val_f1_vals)} if val_f1_vals else {}
                lit_model.push_to_hub(hf_repo_id, metrics=metrics)
            except Exception as hub_err:
                log.warning("HF Hub push failed (non-fatal): %s", hub_err)

    log.info("--- Model Training Finished (Lightning) ---")
    return best_state_dict, history
