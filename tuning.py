# tuning.py — Hyperparameter optimisation using Optuna + PyTorch Lightning
import optuna
import torch
import numpy as np
import torch.nn as nn
import logging
import os
import json
from typing import Dict, Any, Optional

# --- Import necessary components from the project ---
try:
    from utils import load_config, safe_get, load_preprocessed_data, setup_logging
    from data_pipeline import get_data_splits, create_pytorch_dataloaders
    from models import get_model
    from training import _calculate_pos_weight
except ImportError as e:
    logging.critical(f"Failed to import project modules needed for tuning: {e}")
    raise

log = logging.getLogger(__name__)
setup_logging()

# ==============================================================================
# == Global Data Loading (Load Once) ==
# ==============================================================================

CONFIG_PATH = 'config.json'
config = load_config(CONFIG_PATH)
if config is None:
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

# Global variables to hold data splits — reused across all trials
train_data_sampled_global  = None
val_data_global            = None
test_data_global           = None
input_dim_sequence_global  = None
input_dim_static_global    = None


def initialize_tuning_data():
    """Initializes global data splits to be reused across trials."""
    global config
    global train_data_sampled_global, val_data_global, test_data_global
    global input_dim_sequence_global, input_dim_static_global

    log.info("Loading preprocessed data for tuning...")
    processed_data, static_features_results, _ = load_preprocessed_data(config)
    if not processed_data:
        raise ValueError("Failed to load processed data. Cannot run tuning.")
    log.info("Preprocessed data loaded successfully.")

    log.info("Preparing data splits (Windowing/Splitting) ONCE for all trials...")
    (
        train_data_sampled_global,
        val_data_global,
        test_data_global,
        input_dim_sequence_global,
        input_dim_static_global,
    ) = get_data_splits(processed_data, static_features_results, config)

    if train_data_sampled_global is None:
        raise RuntimeError("Failed to prepare global data splits.")
    log.info("Global data splits prepared.")


# Initialise data immediately on module load
initialize_tuning_data()


# ==============================================================================
# == Optuna Objective Function ==
# ==============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimisation.
    Returns best val_f1 (maximise).
    """
    global config
    global train_data_sampled_global, val_data_global, test_data_global
    global input_dim_sequence_global, input_dim_static_global

    log.info(f"\n--- Starting Optuna Trial {trial.number} ---")

    # --- Deep-copy config for this trial ---
    try:
        trial_config = json.loads(json.dumps(config))
    except Exception as e:
        log.error(f"Trial {trial.number}: Failed to deep copy config: {e}")
        raise optuna.TrialPruned("Config copy failed.")

    # ===========================================================
    # Hyperparameter Search Space
    # ===========================================================

    # General training
    lr       = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout  = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    trial_config['training_config']['learning_rate'] = lr
    trial_config['model_config']['dropout']          = dropout
    trial_config['training_config']['batch_size']    = batch_size

    # Model architecture
    model_type = trial.suggest_categorical("model_type", ["LSTM", "CNN-LSTM"])
    trial_config['model_config']['type'] = model_type

    # LSTM-specific
    num_lstm_layers  = trial.suggest_int("num_lstm_layers", 1, 3)
    lstm_hidden_base = trial.suggest_categorical("lstm_hidden_base", [32, 64, 128, 256])
    lstm_layers      = [max(16, lstm_hidden_base // (2 ** i)) for i in range(num_lstm_layers)]
    trial_config['model_config']['lstm_layers']     = lstm_layers
    trial_config['model_config']['bidirectional']   = trial.suggest_categorical(
        "lstm_bidirectional", [True, False]
    )

    # CNN-specific
    if model_type == "CNN-LSTM":
        num_cnn_layers   = trial.suggest_int("num_cnn_layers", 1, 3)
        cnn_filters_base = trial.suggest_categorical("cnn_filters_base", [16, 32, 64])
        cnn_filters      = [cnn_filters_base * (2 ** i) for i in range(num_cnn_layers)]
        cnn_kernels      = [
            trial.suggest_int(f"cnn_kernel_{i}", 3, 9, step=2)
            for i in range(num_cnn_layers)
        ]
        trial_config['model_config']['cnn_filters'] = cnn_filters
        trial_config['model_config']['cnn_kernels'] = cnn_kernels

        use_attention = trial.suggest_categorical("use_attention", [True, False])
        if use_attention:
            attn_heads = trial.suggest_categorical("attn_heads", [2, 4, 8])
            trial_config['model_config']['attn_heads'] = attn_heads
        else:
            trial_config['model_config']['attn_heads'] = 0

    log.info(f"Trial {trial.number} Parameters: {trial.params}")

    # --- DataLoaders ---
    try:
        train_loader, val_loader, _ = create_pytorch_dataloaders(
            train_data_sampled_global,
            val_data_global,
            test_data_global,
            batch_size,
            trial_config,
        )
        if not train_loader or not val_loader:
            raise optuna.TrialPruned("DataLoader creation failed.")
    except optuna.TrialPruned:
        raise
    except Exception as e:
        log.error(f"Trial {trial.number}: Error creating DataLoaders: {e}", exc_info=True)
        raise optuna.TrialPruned("DataLoader creation error.")

    # --- Device & Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        model = get_model(trial_config, input_dim_sequence_global, input_dim_static_global)
        model.to(device)
    except Exception as e:
        log.error(f"Trial {trial.number}: Failed to build model: {e}", exc_info=True)
        raise optuna.TrialPruned("Model building failed.")

    # --- Tuning epoch / patience override ---
    tuning_epochs   = safe_get(trial_config, ['tuning_config', 'epochs'], 30)
    trial_config['training_config']['epochs'] = tuning_epochs
    tuning_patience = safe_get(trial_config, ['tuning_config', 'patience'], 5)
    trial_config['early_stopping']['patience'] = tuning_patience

    # --- Precision ---
    use_amp   = safe_get(trial_config, ['training_config', 'mixed_precision'], True)
    is_cuda   = (device.type == "cuda")
    if use_amp and is_cuda:
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    else:
        precision = "32"

    # --- Lightning imports (lazy — keep tuning.py importable without lightning) ---
    try:
        import lightning as L
        from lightning_module import StressLightningModule
        from lightning.pytorch.callbacks import EarlyStopping
        from optuna.integration import PyTorchLightningPruningCallback
    except ImportError as imp_err:
        log.error(f"Trial {trial.number}: Missing dependency: {imp_err}")
        return 0.0

    # --- Build Lightning module ---
    pos_weight = _calculate_pos_weight(train_loader, device)
    lit_model  = StressLightningModule(model, trial_config, pos_weight)

    # --- Callbacks ---
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=tuning_patience,
        mode="min",
    )
    pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_f1")

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=tuning_epochs,
        accelerator="gpu" if is_cuda else "cpu",
        devices=1,
        precision=precision,
        accumulate_grad_batches=safe_get(
            trial_config, ['training_config', 'accumulation_steps'], 1
        ),
        gradient_clip_val=1.0,
        callbacks=[early_stop_cb, pruning_cb],
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    # --- Fit ---
    try:
        trainer.fit(lit_model, train_loader, val_loader)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        log.error(f"Trial {trial.number}: Training failed: {e}", exc_info=True)
        return 0.0

    val_f1 = trainer.callback_metrics.get("val_f1", torch.tensor(0.0)).item()
    log.info(f"Trial {trial.number}: val_f1 = {val_f1:.5f}")
    return val_f1


# ==============================================================================
# == Main Tuning Execution ==
# ==============================================================================

def run_tuning(n_trials: int = 50):
    """Runs the Optuna hyperparameter search study (maximises val_f1)."""
    global config
    study_name  = safe_get(config, ['tuning_config', 'study_name'], "stress_detection_study")
    results_dir = safe_get(config, ['save_paths', 'results'], './outputs/results')

    log.info(f"Starting Optuna study '{study_name}' for {n_trials} trials.")
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=5, interval_steps=1
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",       # Maximise val_f1
        storage=None,
        pruner=pruner,
    )

    try:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        log.warning("Tuning interrupted by user.")
    except Exception as e:
        log.error(f"Error during optimization: {e}", exc_info=True)

    log.info("\n--- Optuna Study Finished ---")
    log.info(f"Trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        log.info(f"Best val_f1: {best_trial.value:.5f}")
        log.info("Best Parameters:")
        for k, v in best_trial.params.items():
            log.info(f"  {k}: {v}")

        best_params_file = os.path.join(results_dir, "best_hyperparameters.json")
        os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
        with open(best_params_file, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
        log.info(f"Saved best params to: {best_params_file}")
    except ValueError:
        log.warning("No completed trials.")


if __name__ == "__main__":
    run_tuning(n_trials=50)
