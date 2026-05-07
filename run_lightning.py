import os
import sys
import logging
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from utils import (
    load_config,
    setup_logging,
    safe_get,
    configure_torch_runtime,
    log_runtime_capabilities,
)
from preprocessing import preprocess_all_subjects
from models import get_model
from lightning_data import StressDataModule
from lightning_module import StressLightningModule

# Standard workaround for OpenMP conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

log = logging.getLogger(__name__)

def main():
    # 1. Setup Logging and Config
    setup_logging()
    config = load_config("config.json")
    if not config:
        log.error("Failed to load configuration. Exiting.")
        return

    runtime_caps = configure_torch_runtime()
    log_runtime_capabilities(log, runtime_caps, prefix="Training Runtime")

    # 2. Preprocessing (Generator-based, memory efficient)
    log.info("\n[Stage 1] Preprocessing and Feature Extraction...")
    processed_data_paths, static_features, _ = preprocess_all_subjects({}, [], config)

    # 3. Setup DataModule
    log.info("\n[Stage 2] Initializing LightningDataModule...")
    datamodule = StressDataModule(processed_data_paths, static_features, config)
    datamodule.setup() # Pre-calculate dimensions and splits

    # 4. Initialize Model
    log.info("\n[Stage 3] Building Model Architecture...")
    model_type = safe_get(config, ['model_config', 'type'], 'CNN-LSTM').upper()
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
    
    trainer_defaults = {
        "accelerator": "auto",
        "devices": "auto" if runtime_caps["cuda_available"] else 1,
        "precision": precision,
        "accumulate_grad_batches": acc_steps,
        "gradient_clip_val": 1.0,
        "logger": CSVLogger("outputs/logs", name=f"stress_{model_type.lower()}"),
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(
                dirpath="outputs/models",
                filename="best_lightning_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )
        ]
    }

    # ------------------------------------------------------------------
    # Step 7: Execution (Two-Phase for TimesFM, Single for others)
    # ------------------------------------------------------------------
    is_timesfm = (model_type == 'TIMESFM')
    phase1_epochs = 15 if is_timesfm else train_cfg.get('epochs', 50)

    # Phase 1: Training (Frozen backbone if TimesFM)
    log.info(f"\n[Stage 4] Starting Training Phase 1 ({phase1_epochs} epochs)...")
    trainer = L.Trainer(**trainer_defaults, max_epochs=phase1_epochs)
    trainer.fit(lightning_model, datamodule=datamodule)

    # Phase 2: Fine-tuning (Only for TimesFM)
    if is_timesfm and not os.environ.get("TIMESFM_FREEZE", "0") == "1":
        total_epochs = train_cfg.get('epochs', 50)
        phase2_epochs = max(0, total_epochs - phase1_epochs)
        
        if phase2_epochs > 0:
            log.info(f"\n[Stage 5] Starting Training Phase 2 (Fine-tuning last blocks, {phase2_epochs} epochs)...")
            
            # 1. Unfreeze backbone
            fine_tune_blocks = config['model_config'].get('timesfm_finetune_last_n_blocks', 4)
            lightning_model.unfreeze_backbone(last_n_blocks=fine_tune_blocks)
            
            # 2. Lower Learning Rate for fine-tuning
            lr_factor = 0.1 # Could be in config
            lightning_model.config['training_config']['learning_rate'] *= lr_factor
            
            # 3. New Trainer for Phase 2
            trainer_p2 = L.Trainer(**trainer_defaults, max_epochs=total_epochs) # Total epochs includes P1
            trainer_p2.fit(lightning_model, datamodule=datamodule, ckpt_path="last")

    # 8. Evaluation
    log.info("\n[Stage 6] Training Complete. Best model saved in outputs/models.")
    # You can now call trainer.test() or existing evaluation.py logic

if __name__ == "__main__":
    main()
