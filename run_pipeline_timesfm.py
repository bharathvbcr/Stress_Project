"""
run_pipeline_timesfm.py
=======================
Dedicated entry point for the StressProject pipeline using
TimesFM 2.5 as the feature-extraction backbone.

Two-Phase Training Strategy
----------------------------
Phase 1  [epochs 1 .. phase1_epochs]
    • TimesFM backbone is frozen.
    • Only the MLP classification head trains.
    • High LR (from config) for fast head convergence.

Phase 2  [epochs phase1_epochs+1 .. total_epochs]
    • The last `timesfm_finetune_last_n_blocks` transformer blocks
      in TimesFM are unfrozen.
    • LR is reduced by `phase2_lr_factor` for stable fine-tuning.
    • Best model from Phase 1 is the starting point.

Usage
-----
    python run_pipeline_timesfm.py

Optional overrides via environment variables
--------------------------------------------
    TIMESFM_FREEZE=1           Force frozen backbone (skip Phase 2)
    TIMESFM_PHASE1_EPOCHS=10   Override number of Phase-1 epochs (default 15)
    TIMESFM_PHASE2_LR_FACTOR=0.05   LR multiplier for Phase-2 (default 0.1)
    TIMESFM_BATCH_SIZE=16      Override batch size (useful if OOM on GPU)
"""

from __future__ import annotations

import copy
import logging
import os
import sys

import torch

# Standard workaround for multiple OpenMP runtime initialization conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils import load_config, setup_logging
from data_loader import load_all_datasets
from preprocessing import preprocess_all_subjects
from data_pipeline import prepare_dataloaders
from models import get_model
from training import train_model
from evaluation import evaluate_model

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Defaults (overridable via env vars)
# ------------------------------------------------------------------
_DEFAULT_PHASE1_EPOCHS: int = 15
_DEFAULT_PHASE2_LR_FACTOR: float = 0.1


# ==================================================================
# Helpers
# ==================================================================

def _get_env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            log.warning(f"Env var {key}='{val}' is not an integer; using default {default}.")
    return default


def _get_env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            log.warning(f"Env var {key}='{val}' is not a float; using default {default}.")
    return default


def _patch_config_for_timesfm(config: dict) -> dict:
    """
    Ensure the config targets TIMESFM.  Applies optional env-var overrides.
    """
    config = copy.deepcopy(config)

    # Force model type
    config["model_config"]["type"] = "TIMESFM"

    # Optional batch size override (helpful to avoid OOM with TimesFM + many channels)
    env_bs = _get_env_int("TIMESFM_BATCH_SIZE", 0)
    if env_bs > 0:
        old_bs = config.get("training_config", {}).get("batch_size", 64)
        config["training_config"]["batch_size"] = env_bs
        log.info(f"[env] Batch size overridden: {old_bs} → {env_bs}")

    return config


def _print_gpu_info(device: torch.device) -> None:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")
        log.info(
            f"SOTA Optimization Plan: Optimizer=LION, Scheduler=OneCycle, "
            f"AMP=Enabled, Accumulation={safe_get(config, ['training_config', 'accumulation_steps'], 4)}"
        )
    else:
        log.info("Running on CPU — Phase-1 will be slow.  Consider using a GPU.")


def _build_phase2_config(config: dict, phase2_lr_factor: float) -> dict:
    """Return a config copy with a scaled-down learning rate for Phase 2."""
    cfg = copy.deepcopy(config)
    base_lr = cfg.get("training_config", {}).get("learning_rate", 1e-4)
    phase2_lr = base_lr * phase2_lr_factor
    cfg["training_config"]["learning_rate"] = phase2_lr
    log.info(
        f"Phase-2 LR: {base_lr:.2e} × {phase2_lr_factor} = {phase2_lr:.2e}"
    )
    return cfg


# ==================================================================
# Main pipeline
# ==================================================================

def main() -> None:
    # Optimize matmul on CUDA GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    log.info("=" * 70)
    log.info("  StressProject — TimesFM 2.5 Foundation Model Pipeline")
    log.info("=" * 70)

    # --- Check TimesFM availability before spending time preprocessing ---
    try:
        from timesfm_wrapper import is_available as timesfm_ok
        if not timesfm_ok():
            log.error(
                "TimesFM package is not installed!\n"
                "Install with:  pip install 'timesfm[torch]>=2.5.0'\n"
                "Aborting."
            )
            sys.exit(1)
        log.info("✓ timesfm package is available.")
    except ImportError:
        log.error(
            "timesfm_wrapper not found.  Make sure timesfm_wrapper.py is in the "
            "project directory and that 'timesfm' is installed."
        )
        sys.exit(1)

    # --- Phase settings ---
    force_frozen = os.environ.get("TIMESFM_FREEZE", "0") == "1"
    phase1_epochs = _get_env_int("TIMESFM_PHASE1_EPOCHS", _DEFAULT_PHASE1_EPOCHS)
    phase2_lr_factor = _get_env_float("TIMESFM_PHASE2_LR_FACTOR", _DEFAULT_PHASE2_LR_FACTOR)

    # ----------------------------------------------------------------
    # Stage 1: Config
    # ----------------------------------------------------------------
    log.info("\n[Stage 1] Loading Configuration")
    config = load_config("config.json")
    if not config:
        log.error("Failed to load configuration. Exiting.")
        return
    config = _patch_config_for_timesfm(config)

    total_epochs: int = config.get("training_config", {}).get("epochs", 50)
    log.info(
        f"Training plan: {total_epochs} total epochs  "
        f"| Phase 1 (frozen backbone): {phase1_epochs} epochs  "
        f"| Phase 2 (fine-tune last "
        f"{config['model_config'].get('timesfm_finetune_last_n_blocks', 4)} blocks): "
        f"{max(0, total_epochs - phase1_epochs)} epochs"
    )

    # ----------------------------------------------------------------
    # Stage 2: Data Loading (generator — lazy)
    # ----------------------------------------------------------------
    log.info("\n[Stage 2] Data Loading (Generator Setup)")
    raw_data_placeholder = {}
    loaded_ids_placeholder = []

    # ----------------------------------------------------------------
    # Stage 3: Preprocessing
    # ----------------------------------------------------------------
    log.info("\n[Stage 3] Preprocessing (Resampling, Alignment, Feature Extraction)")
    processed_data_paths, static_features, r_peaks = preprocess_all_subjects(
        raw_data_placeholder, loaded_ids_placeholder, config
    )
    if not processed_data_paths:
        log.error("Preprocessing failed (no subjects processed). Exiting.")
        return

    # ----------------------------------------------------------------
    # Stage 4: DataLoaders
    # ----------------------------------------------------------------
    log.info("\n[Stage 4] Preparing DataLoaders (Windows, Splits, Sampling)")
    train_loader, val_loader, test_loader, seq_dim, static_dim = prepare_dataloaders(
        processed_data_paths, static_features, config
    )
    if not train_loader:
        log.error("Failed to create DataLoaders. Exiting.")
        return

    # ----------------------------------------------------------------
    # Stage 5: Build Model
    # ----------------------------------------------------------------
    log.info("\n[Stage 5] Building StressTimesFM Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    _print_gpu_info(device)

    try:
        model = get_model(config, seq_dim, static_dim)
        if not hasattr(model, "freeze_backbone"):
            log.error(
                "Built model does not have freeze_backbone(). "
                "TimesFM model was not instantiated (likely fell back to CNN-LSTM). "
                "Check logs above."
            )
        model.to(device)
    except Exception as exc:
        log.error(f"Failed to build model: {exc}", exc_info=True)
        return

    # ----------------------------------------------------------------
    # Stage 6a: Phase 1 — Train Classification Head (backbone frozen)
    # ----------------------------------------------------------------
    log.info("\n[Stage 6a] Phase 1 Training — Frozen backbone, head only")

    # Ensure backbone is frozen for Phase 1
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    phase1_config = copy.deepcopy(config)
    phase1_config["training_config"]["epochs"] = phase1_epochs
    # Slightly higher LR for head-only training is fine
    phase1_lr = phase1_config["training_config"].get("learning_rate", 1e-4)
    log.info(f"Phase-1 LR: {phase1_lr:.2e}  |  epochs: {phase1_epochs}")

    try:
        best_state_p1, history_p1 = train_model(
            model,
            train_loader,
            val_loader,
            phase1_config,
            device,
            config["save_paths"].get("models"),
        )
    except Exception as exc:
        log.error(f"Phase-1 training failed: {exc}", exc_info=True)
        return

    # Load best Phase-1 state as starting point for Phase 2
    if best_state_p1:
        model.load_state_dict(best_state_p1)
        log.info("Loaded best Phase-1 model state.")

    # ----------------------------------------------------------------
    # Stage 6b: Phase 2 — Fine-tune backbone (optional)
    # ----------------------------------------------------------------
    phase2_epochs = total_epochs - phase1_epochs

    if force_frozen or phase2_epochs <= 0:
        log.info(
            "\n[Stage 6b] Skipping Phase 2 "
            f"({'TIMESFM_FREEZE=1' if force_frozen else f'0 epochs left after Phase 1'})."
        )
        best_state_final = best_state_p1
        history_final = history_p1
    else:
        log.info(
            f"\n[Stage 6b] Phase 2 Training — Unfreezing "
            f"{config['model_config'].get('timesfm_finetune_last_n_blocks', 4)} "
            f"backbone blocks for {phase2_epochs} epochs"
        )

        n_blocks = config["model_config"].get("timesfm_finetune_last_n_blocks", 4)
        if hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone(last_n_blocks=n_blocks)

        phase2_config = _build_phase2_config(config, phase2_lr_factor)
        phase2_config["training_config"]["epochs"] = phase2_epochs

        try:
            best_state_p2, history_p2 = train_model(
                model,
                train_loader,
                val_loader,
                phase2_config,
                device,
                config["save_paths"].get("models"),
            )
            best_state_final = best_state_p2 if best_state_p2 else best_state_p1
            history_final = history_p2
        except Exception as exc:
            log.error(f"Phase-2 training failed: {exc}", exc_info=True)
            log.warning("Falling back to best Phase-1 model for evaluation.")
            best_state_final = best_state_p1
            history_final = history_p1

    # ----------------------------------------------------------------
    # Stage 7: Evaluate
    # ----------------------------------------------------------------
    log.info("\n[Stage 7] Evaluating on Test Set")

    if best_state_final:
        log.info("Loading best model state for evaluation...")
        model.load_state_dict(best_state_final)
    else:
        log.warning("No best state available; using final model state.")

    try:
        results = evaluate_model(
            model,
            test_loader,
            None,
            device,
            config,
            config["save_paths"].get("results"),
            set_name="Test Set (TimesFM)",
        )
        if results:
            log.info("Pipeline completed successfully.")
            print("\n" + "=" * 50)
            print("  TimesFM Pipeline — Final Test Metrics")
            print("=" * 50)
            print(f"  Accuracy  : {results['accuracy']:.4f}")
            print(f"  F1 Score  : {results['f1_score']:.4f}")
            print(f"  Precision : {results['precision']:.4f}")
            print(f"  Recall    : {results['recall']:.4f}")
            print("=" * 50)
    except Exception as exc:
        log.error(f"Evaluation failed: {exc}", exc_info=True)


# ==================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        log.critical(f"Unhandled exception in TimesFM pipeline: {exc}", exc_info=True)
        sys.exit(1)
