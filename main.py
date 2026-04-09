import os
import logging
import torch
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from utils import setup_logging
from preprocessing import preprocess_all_subjects
from models import get_model
from lightning_data import StressDataModule
from lightning_module import StressLightningModule
from validation import run_integrity_check

# Workaround for OpenMP on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup
    setup_logging()
    
    # Convert OmegaConf to standard dict for compatibility with existing bio-pipeline
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Print config for visibility
    log.info("\n" + "="*50 + "\nPHYSIOPULSE SOTA: ADVANCED STRESS INTELLIGENCE\n" + "="*50)
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if torch.cuda.is_available():
        # Global TF32 Enablement (Speedup for Ampere/Ada Lovelace GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Aggressive Kernel Tuning (SOTA Warmup)
        torch.backends.cudnn.benchmark = True
        
        # Enable persistent kernel caching to avoid re-warming up in future sessions
        # Note: This uses the default PyTorch cache directory
        os.environ["TORCH_COMPILE_DEBUG"] = "0"
        
        log.info("SOTA Aggressive Autotuning Engaged: TF32=ON, CUDNN_BENCHMARK=ON")

    # 2. Preprocessing (Smart Cache)
    log.info("\n[Stage 1] Preprocessing & Cache Check...")
    hf_path = cfg.get('hf_path', "./outputs/processed_data_hf")
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
    
    # --- Expert Step: Integrity Check ---
    if cfg.get('run_integrity', False):
        log.info("\n[Expert Stage] Running Signal Integrity Suite...")
        run_integrity_check()

    # 4. Model Architecture
    log.info("\n[Stage 3] Building Model...")
    model_type = cfg.model.type.upper()
    base_model = get_model(
        model_type=model_type,
        input_dim_sequence=datamodule.input_dim_sequence,
        input_dim_static=datamodule.input_dim_static,
        model_config=config['model_config'],
        device=torch.device('cpu') 
    )

    # 5. Lightning Module
    lightning_model = StressLightningModule(
        model=base_model, 
        config=config, 
        pos_weight=datamodule.pos_weight
    )

    # --- SOTA Optimization: Graph Compilation ---
    # torch.compile provides significant speedups by fusing kernels
    if hasattr(torch, "compile") and torch.cuda.is_available():
        try:
            log.info("Compiling model (mode='reduce-overhead') for maximum throughput...")
            # 'reduce-overhead' is excellent for non-transformer classification heads
            # Use 'max-autotune' for transformer backbones if on Linux/Ampere
            compile_mode = "max-autotune" if cfg.model.type.upper() == "TIMESFM" and os.name != "nt" else "reduce-overhead"
            lightning_model.model = torch.compile(lightning_model.model, mode=compile_mode)
        except Exception as e:
            log.warning(f"torch.compile failed (common on Windows without MSVC): {e}")

    # 6. Weights & Biases Logger
    # Uses anonymous-mode by default. Set WANDB_API_KEY env var to log to your account.
    wandb_logger = WandbLogger(
        project="PhysioPulse_SOTA",
        name=f"{model_type.lower()}_{cfg.training.optimizer}",
        save_dir="outputs/logs",
        anonymous="allow"
    )
    # Log the full config to W&B
    wandb_logger.experiment.config.update(config)

    # 7. Trainer Setup
    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"
    
    trainer_defaults = {
        "accelerator": "auto",
        "devices": "auto", # Scale to all available GPUs
        "strategy": "ddp_find_unused_parameters_false" if torch.cuda.device_count() > 1 else "auto",
        "precision": precision,
        "accumulate_grad_batches": cfg.training.accumulation_steps,
        "gradient_clip_val": 1.0,
        "logger": wandb_logger,
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=cfg.training.early_stopping.patience, mode="min"),
            ModelCheckpoint(
                dirpath=cfg.save_paths.models,
                filename=f"best_{model_type.lower()}_" + "{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )
        ]
    }

    # 8. Training (Phase 1)
    is_timesfm = (model_type == 'TIMESFM')
    phase1_epochs = 15 if is_timesfm else cfg.training.epochs

    log.info(f"\n[Stage 4] Starting Training Phase 1...")
    trainer = L.Trainer(**trainer_defaults, max_epochs=phase1_epochs)
    trainer.fit(lightning_model, datamodule=datamodule)

    # 9. Training (Phase 2 Fine-tuning for FM)
    if is_timesfm and not os.environ.get("TIMESFM_FREEZE", "0") == "1":
        total_epochs = cfg.training.epochs
        phase2_epochs = max(0, total_epochs - phase1_epochs)
        
        if phase2_epochs > 0:
            log.info(f"\n[Stage 5] Starting Training Phase 2 (Fine-tuning)...")
            lightning_model.unfreeze_backbone(last_n_blocks=cfg.model.timesfm_finetune_last_n_blocks)
            lightning_model.config['training_config']['learning_rate'] *= 0.1
            
            trainer_p2 = L.Trainer(**trainer_defaults, max_epochs=total_epochs)
            trainer_p2.fit(lightning_model, datamodule=datamodule, ckpt_path="last")

    log.info("\n[Stage 6] Complete. Track and visualize your results on the Weights & Biases dashboard.")

if __name__ == "__main__":
    main()
