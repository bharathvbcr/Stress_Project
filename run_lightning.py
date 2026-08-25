import os
import sys
import logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils import (
    load_config,
    setup_logging,
    safe_get,
    configure_torch_runtime,
    log_runtime_capabilities,
    extract_model_state_dict,
)
from preprocessing import preprocess_all_subjects
from lightning_data import StressDataModule

log = logging.getLogger(__name__)

def main():
    # 1. Setup Logging and Config
    import argparse

    parser = argparse.ArgumentParser(description="Run the Lightning/MLX training pipeline.")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap training to at most N epochs (quick dry run for timing/smoke tests).",
    )
    args, _ = parser.parse_known_args()
    if args.max_epochs is not None:
        # Universal cap honored by run_mlx_training regardless of entrypoint.
        os.environ["STRESS_MAX_EPOCHS"] = str(max(1, args.max_epochs))

    setup_logging()
    config = load_config("config.json")
    if not config:
        log.error("Failed to load configuration. Exiting.")
        return
    if args.max_epochs is not None:
        # Also cap the torch/Lightning path so the flag works for every backend.
        config.setdefault("training_config", {})["epochs"] = max(1, args.max_epochs)
        log.info("Dry run: --max-epochs=%s", max(1, args.max_epochs))

    runtime_caps = configure_torch_runtime()
    log_runtime_capabilities(log, runtime_caps, prefix="Training Runtime")

    # 2. Preprocessing (Generator-based, memory efficient)
    log.info("\n[Stage 1] Preprocessing and Feature Extraction...")
    processed_data_paths, static_features, _ = preprocess_all_subjects({}, [], config)

    # 3. Setup DataModule
    log.info("\n[Stage 2] Initializing LightningDataModule...")
    datamodule = StressDataModule(processed_data_paths, static_features, config)
    datamodule.setup()

    model_type = safe_get(config, ['model_config', 'type'], 'CNN-LSTM').upper()
    from mlx_training import run_mlx_training, should_use_mlx_training

    if should_use_mlx_training(model_type, runtime_caps):
        log.info("\n[MLX] Using Apple MLX training backend for model=%s", model_type)
        ckpt_path = run_mlx_training(datamodule, config, model_type, output_dir="outputs/models")
        log.info("MLX training complete. Best checkpoint: %s", ckpt_path)
        return

    # Import heavy torch/Lightning dependencies after dataset-backed module is loaded.
    import torch
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from models import get_model
    from training import build_training_profiler, get_standard_callbacks, training_stage
    from lightning_module import StressLightningModule

    # 4. Initialize Model
    log.info("\n[Stage 3] Building Model Architecture...")
    base_model = get_model(config, datamodule.input_dim_sequence, datamodule.input_dim_static)

    # 5. Initialize Lightning Module
    # Pass the pre-calculated pos_weight from the DataModule
    lightning_model = StressLightningModule(
        model=base_model, 
        config=config, 
        pos_weight=datamodule.pos_weight
    )

    # ------------------------------------------------------------------
    # Step 6: Trainer Configuration
    # ------------------------------------------------------------------
    train_cfg = config.get('training_config', {})
    acc_steps = train_cfg.get('accumulation_steps', 1)
    
    # Check for BF16 support (RTX 30-series+)
    precision = runtime_caps["precision"]
    is_cuda = (runtime_caps["accelerator"] == "gpu")
    trainer_device = torch.device("cuda" if is_cuda else runtime_caps["device_type"])
    profiler = build_training_profiler(config, run_name="run_lightning")
    enable_progress_bar = str(
        safe_get(config, ['training_config', 'enable_progress_bar'], False)
    ).strip().lower() in {"1", "true", "yes", "on"}
    
    trainer_defaults = {
        "accelerator": runtime_caps["accelerator"],
        "devices": runtime_caps["devices"],
        "strategy": runtime_caps["strategy"],
        "precision": precision,
        "accumulate_grad_batches": acc_steps,
        "gradient_clip_val": 1.0,
        "logger": CSVLogger("outputs/logs", name=f"stress_{model_type.lower()}"),
        "profiler": profiler,
        "enable_progress_bar": enable_progress_bar,
        "num_sanity_val_steps": 0,
        "callbacks": get_standard_callbacks(
            config=config,
            output_dir="outputs/models",
            is_cuda=is_cuda
        )
    }

    # ------------------------------------------------------------------
    # Step 7: Execution (Two-Phase for TimesFM, Single for others)
    # ------------------------------------------------------------------
    is_timesfm = (model_type == 'TIMESFM')
    phase1_epochs = 15 if is_timesfm else train_cfg.get('epochs', 50)
    trainer = None

    # Phase 1: Training (Frozen backbone if TimesFM)
    log.info(f"\n[Stage 4] Starting Training Phase 1 ({phase1_epochs} epochs)...")
    from mlx_timesfm_training import run_timesfm_mlx_phase1, should_use_mlx_timesfm_phase1

    use_mlx_phase1 = is_timesfm and should_use_mlx_timesfm_phase1(runtime_caps)
    if use_mlx_phase1:
        log.info("TimesFM Phase 1: MLX head training (frozen PyTorch backbone)")
        with training_stage(log, "timesfm_mlx_phase1", trainer_device, epochs=phase1_epochs):
            run_timesfm_mlx_phase1(
                datamodule,
                config,
                lightning_model.model,
                trainer_device,
                epochs=phase1_epochs,
                output_dir=safe_get(config, ["save_paths", "models"], "outputs/models"),
            )
    else:
        trainer = L.Trainer(**trainer_defaults, max_epochs=phase1_epochs)
        with training_stage(log, "trainer_fit_phase1", trainer_device, epochs=phase1_epochs):
            trainer.fit(lightning_model, datamodule=datamodule)

    # Phase 2: Fine-tuning (Only for TimesFM)
    if is_timesfm and not os.environ.get("TIMESFM_FREEZE", "0") == "1":
        total_epochs = train_cfg.get('epochs', 50)
        phase2_epochs = max(0, total_epochs - phase1_epochs)
        
        if phase2_epochs > 0:
            log.info(f"\n[Stage 5] Starting Training Phase 2 (Fine-tuning last blocks, {phase2_epochs} epochs)...")
            
            # 1. Load the best Phase 1 weights explicitly. Checkpoints are saved
            # weights-only, so they must not be passed to Lightning as ckpt_path.
            best_p1_ckpt = None
            if trainer is not None and trainer.checkpoint_callback is not None:
                best_p1_ckpt = trainer.checkpoint_callback.best_model_path
            if best_p1_ckpt and os.path.exists(best_p1_ckpt):
                log.info("Loading Phase 1 best checkpoint from %s...", best_p1_ckpt)
                try:
                    checkpoint = torch.load(best_p1_ckpt, map_location=trainer_device, weights_only=False)
                    if hasattr(lightning_model.model, "_ensure_attached"):
                        lightning_model.model._ensure_attached()
                    state_dict = extract_model_state_dict(checkpoint)
                    missing, unexpected = lightning_model.model.load_state_dict(state_dict, strict=False)
                    if missing:
                        log.warning("Checkpoint load missing %d keys (showing 5): %s", len(missing), missing[:5])
                    if unexpected:
                        log.warning("Checkpoint load ignored %d unexpected keys (showing 5): %s", len(unexpected), unexpected[:5])
                except Exception as exc:
                    log.warning("Failed to load Phase 1 best checkpoint: %s. Continuing with current weights.", exc)

            # 2. Unfreeze backbone or attach LoRA adapters
            from timesfm_lora import should_use_timesfm_lora_phase2

            fine_tune_blocks = config['model_config'].get('timesfm_finetune_last_n_blocks', 4)
            if should_use_timesfm_lora_phase2(runtime_caps):
                log.info("TimesFM Phase 2: LoRA adapters on last %s block(s)", fine_tune_blocks)
                lightning_model.apply_backbone_lora(last_n_blocks=fine_tune_blocks)
            else:
                lightning_model.unfreeze_backbone(last_n_blocks=fine_tune_blocks)
            
            # 3. Lower Learning Rate for fine-tuning
            lr_factor = 0.1 # Could be in config
            lightning_model.config['training_config']['learning_rate'] *= lr_factor
            # OneCycle drives the effective LR from max_lr (overriding the optimizer's
            # base lr), so scale it too — otherwise Phase-2 fine-tuning runs at the
            # Phase-1 peak LR despite the reduced learning_rate above.
            _p2_lr_sched = lightning_model.config['training_config'].get('lr_scheduler')
            if isinstance(_p2_lr_sched, dict) and _p2_lr_sched.get('max_lr') is not None:
                _p2_lr_sched['max_lr'] *= lr_factor
                log.info("Phase-2 scheduler max_lr scaled by %.3g → %.2e", lr_factor, _p2_lr_sched['max_lr'])
            
            # 4. New Trainer and callbacks for Phase 2
            from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

            p2_callbacks = get_standard_callbacks(
                config=config,
                output_dir="outputs/models",
                is_cuda=is_cuda,
            )
            for cb in p2_callbacks:
                if isinstance(cb, ModelCheckpoint):
                    cb.filename = "best_model_phase2-{epoch:02d}-{val_loss:.4f}"
                elif isinstance(cb, EarlyStopping):
                    cb.patience = 10

            trainer_defaults_p2 = trainer_defaults.copy()
            trainer_defaults_p2["callbacks"] = p2_callbacks
            trainer_p2 = L.Trainer(**trainer_defaults_p2, max_epochs=phase2_epochs)
            with training_stage(log, "trainer_fit_phase2", trainer_device, epochs=phase2_epochs):
                trainer_p2.fit(lightning_model, datamodule=datamodule)

            from mlx_hybrid import export_timesfm_mlx_artifacts

            best_p2 = None
            if trainer_p2.checkpoint_callback is not None:
                best_p2 = trainer_p2.checkpoint_callback.best_model_path
            export_timesfm_mlx_artifacts(
                config,
                lightning_model.model,
                checkpoint_path=best_p2,
                input_dim_static=datamodule.input_dim_static,
                output_dir=safe_get(config, ["save_paths", "models"], "outputs/models"),
            )

    # 8. Evaluation
    log.info("\n[Stage 6] Training Complete. Best model saved in outputs/models.")
    # You can now call trainer.test() or existing evaluation.py logic

if __name__ == "__main__":
    main()
