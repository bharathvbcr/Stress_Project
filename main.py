import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("WANDB_CONSOLE", "off")
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from preprocessing import preprocess_all_subjects
from utils import (
    setup_logging,
    safe_get,
    configure_torch_runtime,
    log_runtime_capabilities,
    extract_model_state_dict,
)
from lightning_data import StressDataModule


# Workaround for OpenMP on Windows (apply before any torch/Lightning imports)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup
    setup_logging()
    
    # Convert OmegaConf to standard dict for compatibility with existing bio-pipeline
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Bridge: Hydra uses 'dataset', 'training', 'processing' groups but code 
    # expects specific keys like 'datasets', 'sampling_rates', 'label_mapping', etc.
    if 'dataset' in config:
        if 'datasets' not in config: config['datasets'] = config['dataset']
        if 'sampling_rates' not in config: config['sampling_rates'] = config['dataset'].get('sampling_rates', {})
        if 'label_mapping' not in config: config['label_mapping'] = config['dataset'].get('label_mapping', {})
    
    # Bridge model and training config names
    if 'model' in config and 'model_config' not in config:
        config['model_config'] = config['model']
    if 'training' in config and 'training_config' not in config:
        config['training_config'] = config['training']
    if 'processing' in config and 'processing_config' not in config:
        config['processing_config'] = config['processing']
    
    # Print config for visibility
    log.info("\n" + "="*50 + "\nPHYSIOPULSE SOTA: ADVANCED STRESS INTELLIGENCE\n" + "="*50)
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    runtime_caps = configure_torch_runtime()
    log_runtime_capabilities(log, runtime_caps, prefix="Training Runtime")

    # 2. Preprocessing (Smart Cache)
    log.info("\n[Stage 1] Preprocessing & Cache Check...")
    hf_path = safe_get(config, ['hf_path'], "./outputs/processed_data_hf")
    force_preprocess = cfg.get('force_preprocess', False)
    
    # Only run preprocessing if cache is missing or forced
    if force_preprocess or not os.path.exists(hf_path) or not os.listdir(hf_path):
        log.info("Arrow cache not found or refresh forced. Starting preprocessing...")
        processed_data_paths, static_features, _ = preprocess_all_subjects({}, [], config)
    else:
        log.info(f"Found existing Arrow cache at {hf_path}. Skipping preprocessing.")
        # We still need static_features map for the DataModule if we want to bypass full logic
        # However, StressDataModule.setup() is already designed to load from disk if hf_path exists.
        # We just need to ensure the DataModule initialization is safe with empty inputs if cache exists.
        processed_data_paths, static_features = {}, {}

    # 3. DataModule
    log.info("\n[Stage 2] Initializing DataModule...")
    datamodule = StressDataModule(processed_data_paths, static_features, config)
    datamodule.setup()

    model_type = safe_get(config, ['model_config', 'type'], 'CNN-LSTM').upper()
    from mlx_training import run_mlx_training, should_use_mlx_training

    if should_use_mlx_training(model_type, runtime_caps):
        log.info("\n[MLX] Using Apple MLX training backend for model=%s", model_type)
        ckpt_path = run_mlx_training(
            datamodule,
            config,
            model_type,
            output_dir=safe_get(config, ['save_paths', 'models'], cfg.save_paths.models),
        )
        log.info("MLX training complete. Best checkpoint: %s", ckpt_path)
        return

    # 3b. Import heavy training/runtime dependencies after dataset module import to
    # avoid the torch/Lightning + datasets import-order crash.
    import torch
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger, CSVLogger
    from models import get_model
    from training import build_training_profiler, get_standard_callbacks, training_stage
    from lightning_module import StressLightningModule
    from validation import run_integrity_check
    
    # --- Expert Step: Integrity Check ---
    if cfg.get('run_integrity', False):
        log.info("\n[Expert Stage] Running Signal Integrity Suite...")
        run_integrity_check()

    # 4. Model Architecture
    log.info("\n[Stage 3] Building Model...")
    base_model = get_model(config, datamodule.input_dim_sequence, datamodule.input_dim_static)

    # 5. Lightning Module
    lightning_model = StressLightningModule(
        model=base_model, 
        config=config, 
        pos_weight=datamodule.pos_weight
    )

    # --- SOTA Optimization: Graph Compilation ---
    # torch.compile provides significant speedups by fusing kernels.
    # Never compile TimesFM: torch.compile strips forward hooks used for embeddings.
    use_compile = safe_get(config, ['training_config', 'torch_compile'], False)
    if use_compile and model_type == "TIMESFM":
        log.warning(
            "Ignoring training.torch_compile for TIMESFM: compilation breaks "
            "forward-hook embedding extraction (flat predictions)."
        )
        use_compile = False
    if use_compile and hasattr(torch, "compile") and runtime_caps["device_type"] in ("cuda", "mps"):
        try:
            log.info("Compiling model (mode='reduce-overhead') for maximum throughput...")
            # 'reduce-overhead' is excellent for non-transformer classification heads
            # Use 'max-autotune' for transformer backbones if on Linux/Ampere
            compile_mode = "max-autotune" if cfg.model.type.upper() == "TIMESFM" and os.name != "nt" else "reduce-overhead"
            lightning_model.model = torch.compile(lightning_model.model, mode=compile_mode)
        except Exception as e:
            log.warning(f"torch.compile failed (common on Windows without MSVC): {e}")

    # 6. Weights & Biases Logger
    # Uses anonymous-mode by default. Fall back to CSV logger when W&B is unavailable.
    try:
        logger = WandbLogger(
            project="PhysioPulse_SOTA",
            name=f"{model_type.lower()}_{cfg.training.optimizer}",
            save_dir="outputs/logs",
            anonymous="allow"
        )
        logger.experiment.config.update(config)
    except Exception as e:
        log.warning(f"W&B logger unavailable, using CSVLogger fallback: {e}")
        logger = CSVLogger("outputs/logs", name=f"stress_{model_type.lower()}")

    # 7. Trainer Setup
    precision = runtime_caps["precision"]
    is_cuda = (runtime_caps["accelerator"] == "gpu")
    trainer_device = torch.device("cuda" if is_cuda else runtime_caps["device_type"])
    profiler = build_training_profiler(config, run_name="hydra_main")
    enable_progress_bar = str(
        safe_get(config, ['training_config', 'enable_progress_bar'], False)
    ).strip().lower() in {"1", "true", "yes", "on"}

    trainer_defaults = {
        "accelerator": runtime_caps["accelerator"],
        "devices": runtime_caps["devices"],
        "strategy": runtime_caps["strategy"],
        "precision": precision,
        "accumulate_grad_batches": cfg.training.accumulation_steps,
        "gradient_clip_val": 1.0,
        "logger": logger,
        "profiler": profiler,
        "enable_progress_bar": enable_progress_bar,
        "num_sanity_val_steps": 0,
        "callbacks": get_standard_callbacks(
            config=config,
            output_dir=cfg.save_paths.models,
            is_cuda=is_cuda
        )
    }

    # 8. Training (Phase 1)
    is_timesfm = (model_type == 'TIMESFM')
    phase1_epochs = 15 if is_timesfm else cfg.training.epochs
    resume_ckpt = os.environ.get("TIMESFM_RESUME_CKPT", "").strip()
    skip_phase1 = bool(resume_ckpt and is_timesfm and os.path.isfile(resume_ckpt))
    trainer = None

    if skip_phase1:
        log.info(
            "TIMESFM_RESUME_CKPT set — skipping Phase 1 and resuming fine-tuning from: %s",
            resume_ckpt,
        )
    else:
        from mlx_timesfm_training import run_timesfm_mlx_phase1, should_use_mlx_timesfm_phase1

        use_mlx_phase1 = is_timesfm and should_use_mlx_timesfm_phase1(runtime_caps)
        log.info(f"\n[Stage 4] Starting Training Phase 1...")
        if use_mlx_phase1:
            log.info("TimesFM Phase 1: MLX head training (frozen PyTorch backbone)")
            with training_stage(log, "timesfm_mlx_phase1", trainer_device, epochs=phase1_epochs):
                run_timesfm_mlx_phase1(
                    datamodule,
                    config,
                    lightning_model.model,
                    trainer_device,
                    epochs=phase1_epochs,
                    output_dir=cfg.save_paths.models,
                )
        else:
            trainer = L.Trainer(**trainer_defaults, max_epochs=phase1_epochs)
            with training_stage(log, "trainer_fit_phase1", trainer_device, epochs=phase1_epochs):
                trainer.fit(lightning_model, datamodule=datamodule)

    # 9. Training (Phase 2 Fine-tuning for FM)
    if is_timesfm and not os.environ.get("TIMESFM_FREEZE", "0") == "1":
        total_epochs = cfg.training.epochs
        phase2_epochs = max(0, total_epochs - phase1_epochs)
        
        if phase2_epochs > 0:
            log.info(f"\n[Stage 5] Starting Training Phase 2 (Fine-tuning)...")
            
            # 1. Load resume checkpoint (Phase 2 resume) or best Phase 1 checkpoint
            best_p1_ckpt = resume_ckpt if skip_phase1 else None
            if best_p1_ckpt is None and trainer is not None and trainer.checkpoint_callback is not None:
                best_p1_ckpt = trainer.checkpoint_callback.best_model_path

            load_ckpt = best_p1_ckpt if best_p1_ckpt and os.path.exists(best_p1_ckpt) else None
            if load_ckpt:
                label = "Phase 2 resume" if skip_phase1 else "Phase 1 best"
                log.info(f"Loading {label} checkpoint from {load_ckpt}...")
                try:
                    checkpoint = torch.load(
                        load_ckpt, map_location=trainer_device, weights_only=False
                    )
                    if hasattr(lightning_model.model, "_ensure_attached"):
                        lightning_model.model._ensure_attached()
                    state_dict = extract_model_state_dict(checkpoint)
                    missing, unexpected = lightning_model.model.load_state_dict(
                        state_dict, strict=False
                    )
                    if missing:
                        log.warning(
                            "Checkpoint load missing %d keys (showing 5): %s",
                            len(missing),
                            missing[:5],
                        )
                    if unexpected:
                        log.warning(
                            "Checkpoint load ignored %d unexpected keys (showing 5): %s",
                            len(unexpected),
                            unexpected[:5],
                        )
                except Exception as e:
                    log.warning(f"Failed to load checkpoint: {e}. Proceeding with current weights.")

            # 2. Unfreeze backbone or attach LoRA (and scale LR only when starting Phase 2 fresh)
            from timesfm_lora import should_use_timesfm_lora_phase2

            n_blocks = cfg.model.timesfm_finetune_last_n_blocks
            if should_use_timesfm_lora_phase2(runtime_caps) and hasattr(
                lightning_model.model, "apply_backbone_lora"
            ):
                log.info("TimesFM Phase 2: LoRA adapters on last %s block(s)", n_blocks)
                lightning_model.model.apply_backbone_lora(last_n_blocks=n_blocks)
            else:
                lightning_model.unfreeze_backbone(last_n_blocks=n_blocks)
            if not skip_phase1:
                lightning_model.config['training_config']['learning_rate'] *= 0.1
                # OneCycle drives the effective LR from max_lr (overriding base lr),
                # so scale it too or Phase-2 fine-tuning runs at the Phase-1 peak LR.
                _p2_lr_sched = lightning_model.config['training_config'].get('lr_scheduler')
                if isinstance(_p2_lr_sched, dict) and _p2_lr_sched.get('max_lr') is not None:
                    _p2_lr_sched['max_lr'] *= 0.1
                    log.info("Phase-2 scheduler max_lr scaled by 0.1 → %.2e", _p2_lr_sched['max_lr'])
            
            # 3. Re-create standard callbacks to get fresh, un-mutated instances
            from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
            p2_callbacks = get_standard_callbacks(
                config=config,
                output_dir=cfg.save_paths.models,
                is_cuda=is_cuda
            )
            
            # 4. Configure callbacks for Phase 2: custom prefix & increased early stopping patience
            for cb in p2_callbacks:
                if isinstance(cb, ModelCheckpoint):
                    cb.filename = "best_model_phase2-{epoch:02d}-{val_loss:.4f}"
                elif isinstance(cb, EarlyStopping):
                    cb.patience = 10  # Increase patience for backbone fine-tuning
            
            # 5. Construct trainer defaults for Phase 2
            trainer_defaults_p2 = trainer_defaults.copy()
            trainer_defaults_p2["callbacks"] = p2_callbacks
            
            # 6. Fit model in Phase 2 (resume from ckpt when TIMESFM_RESUME_CKPT is set)
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
                output_dir=cfg.save_paths.models,
            )

    log.info("\n[Stage 6] Complete. Track and visualize your results on the Weights & Biases dashboard.")

if __name__ == "__main__":
    main()
