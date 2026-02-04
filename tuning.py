# tuning.py (Handles hyperparameter optimization using Optuna)
import optuna # Import optuna library
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import logging
import os
import json
from typing import Dict, Any, Optional, Tuple

# --- Import necessary components from your project ---
# Adjust paths as needed if tuning.py is not in the root directory
try:
    from utils import load_config, safe_get, load_preprocessed_data, setup_logging
    # Import the newly refactored functions for data splits and loaders
    from data_pipeline import get_data_splits, create_pytorch_dataloaders
    from models import get_model # Use the get_model factory function
    from training import train_model # Import the main training function
except ImportError as e:
    logging.critical(f"Failed to import project modules needed for tuning: {e}")
    raise

log = logging.getLogger(__name__)
setup_logging()

# ==============================================================================
# == Global Data Loading (Load Once) ==
# ==============================================================================
# Load configuration and preprocessed data once to avoid reloading in each trial

CONFIG_PATH = 'config.json' # Adjust path if needed
config = load_config(CONFIG_PATH)
if config is None:
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

# Global variables to hold data splits
train_data_sampled_global = None
val_data_global = None
test_data_global = None
input_dim_sequence_global = None
input_dim_static_global = None

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
    # Generate splits based on the initial config (window size, features, etc.)
    # Note: If you want to tune window size, this logic needs to be inside the objective function.
    train_data_sampled_global, val_data_global, test_data_global, input_dim_sequence_global, input_dim_static_global = get_data_splits(
        processed_data, static_features_results, config
    )
    if train_data_sampled_global is None:
        raise RuntimeError("Failed to prepare global data splits.")
    log.info("Global data splits prepared.")

# Initialize data immediately
initialize_tuning_data()

# ==============================================================================
# == Optuna Objective Function ==
# ==============================================================================
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    """
    global config
    global train_data_sampled_global, val_data_global, test_data_global
    global input_dim_sequence_global, input_dim_static_global

    log.info(f"\n--- Starting Optuna Trial {trial.number} ---")

    # --- Hyperparameter Search Space Definition ---
    try:
        trial_config = json.loads(json.dumps(config))
    except Exception as e:
        log.error(f"Trial {trial.number}: Failed to deep copy config: {e}")
        raise optuna.TrialPruned("Config copy failed.")

    # === Suggest Hyperparameters ===
    # General Training
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # Tuning batch size

    trial_config['training_config']['learning_rate'] = lr
    trial_config['model_config']['dropout'] = dropout
    trial_config['training_config']['batch_size'] = batch_size

    # Model Architecture
    model_type = trial.suggest_categorical("model_type", ["LSTM", "CNN-LSTM"])
    trial_config['model_config']['type'] = model_type

    # LSTM Specific
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
    lstm_hidden_base = trial.suggest_categorical("lstm_hidden_base", [32, 64, 128, 256])
    lstm_layers = [max(16, lstm_hidden_base // (2**i)) for i in range(num_lstm_layers)]
    trial_config['model_config']['lstm_layers'] = lstm_layers
    trial_config['model_config']['bidirectional'] = trial.suggest_categorical("lstm_bidirectional", [True, False])

    # CNN Specific
    if model_type == "CNN-LSTM":
        num_cnn_layers = trial.suggest_int("num_cnn_layers", 1, 3)
        cnn_filters_base = trial.suggest_categorical("cnn_filters_base", [16, 32, 64])
        cnn_filters = [cnn_filters_base * (2**i) for i in range(num_cnn_layers)]
        cnn_kernels = [trial.suggest_int(f"cnn_kernel_{i}", 3, 9, step=2) for i in range(num_cnn_layers)]

        trial_config['model_config']['cnn_filters'] = cnn_filters
        trial_config['model_config']['cnn_kernels'] = cnn_kernels
        
        use_attention = trial.suggest_categorical("use_attention", [True, False])
        if use_attention:
            attn_heads = trial.suggest_categorical("attn_heads", [2, 4, 8])
            trial_config['model_config']['attn_heads'] = attn_heads
        else:
            trial_config['model_config']['attn_heads'] = 0

    log.info(f"Trial {trial.number} Parameters: {trial.params}")

    # --- Create DataLoaders with Trial Batch Size ---
    try:
        # Reuse the global splits, only creating new loaders with the suggested batch_size
        train_loader, val_loader, _ = create_pytorch_dataloaders(
            train_data_sampled_global, val_data_global, test_data_global, 
            batch_size, trial_config # Use batch_size from trial
        )
        if not train_loader or not val_loader:
             raise optuna.TrialPruned("DataLoader creation failed.")
    except Exception as e:
         log.error(f"Trial {trial.number}: Error creating DataLoaders: {e}", exc_info=True)
         raise optuna.TrialPruned("DataLoader creation error.")

    # --- Setup Device & Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"): torch.backends.cudnn.benchmark = True

    try:
        model = get_model(trial_config, input_dim_sequence_global, input_dim_static_global)
        model.to(device)
    except Exception as e:
        log.error(f"Trial {trial.number}: Failed to build model: {e}", exc_info=True)
        raise optuna.TrialPruned("Model building failed.")

    # --- Train Model ---
    tuning_epochs = safe_get(trial_config, ['tuning_config', 'epochs'], 30)
    trial_config['training_config']['epochs'] = tuning_epochs
    tuning_patience = safe_get(trial_config, ['tuning_config', 'patience'], 5)
    trial_config['early_stopping']['patience'] = tuning_patience

    try:
        best_model_state, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trial_config,
            device=device,
            output_dir=None
        )
        if best_model_state is None or history is None:
             raise optuna.TrialPruned("Training failed.")
    except optuna.TrialPruned as e: raise e
    except Exception as e:
         log.error(f"Trial {trial.number}: Training loop failed: {e}", exc_info=True)
         return float('inf')

    # --- Evaluate ---
    val_losses = [loss for loss in history.get('val_loss', []) if loss is not None and not np.isnan(loss)]
    if not val_losses:
        log.error(f"Trial {trial.number}: No valid validation losses.")
        return float('inf')
    
    metric_to_optimize = min(val_losses)
    log.info(f"Trial {trial.number}: Best Validation Loss = {metric_to_optimize:.5f}")

    # --- Pruning Reporting ---
    for epoch, val_loss in enumerate(val_losses):
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            log.warning(f"Trial {trial.number}: Pruned at epoch {epoch+1}.")
            raise optuna.TrialPruned()

    return metric_to_optimize

# ==============================================================================
# == Main Tuning Execution ==
# ==============================================================================
def run_tuning(n_trials: int = 50):
    """Runs the Optuna hyperparameter search study."""
    global config
    study_name = safe_get(config, ['tuning_config', 'study_name'], "stress_detection_study")
    results_dir = safe_get(config, ['save_paths', 'results'], './outputs/results')

    log.info(f"Starting Optuna study '{study_name}' for {n_trials} trials.")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=None,
        pruner=pruner
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
        log.info(f"Best Value: {best_trial.value:.5f}")
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