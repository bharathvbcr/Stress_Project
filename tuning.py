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
    from utils import load_config, safe_get, load_preprocessed_data
    # Import the main orchestrator function for DataLoaders
    from data_pipeline import prepare_dataloaders
    from models import get_model # Use the get_model factory function
    from training import train_model # Import the main training function
    # Evaluation might be needed if optimizing based on final eval metric, but usually validation metric is used
    # from evaluation import evaluate_model
except ImportError as e:
    logging.critical(f"Failed to import project modules needed for tuning: {e}")
    raise

log = logging.getLogger(__name__)
# Configure logging for tuning process if not done elsewhere
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

# ==============================================================================
# == Global Data Loading (Load Once) ==
# ==============================================================================
# Load configuration and preprocessed data once to avoid reloading in each trial

CONFIG_PATH = 'config.json' # Adjust path if needed
config = load_config(CONFIG_PATH)
if config is None:
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

# --- Load Data ONCE ---
log.info("Loading preprocessed data for tuning...")
# Load only necessary data (processed signals/labels and static features)
# R-peaks are not typically needed for tuning model hyperparameters
processed_data, static_features_results, _ = load_preprocessed_data(config)
if not processed_data:
     raise ValueError("Failed to load processed data. Cannot run tuning.")
log.info("Preprocessed data loaded successfully for tuning.")

# ==============================================================================
# == Optuna Objective Function ==
# ==============================================================================
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    This function defines the search space, trains a model with suggested
    hyperparameters, and returns a metric (e.g., validation loss) for Optuna to minimize/maximize.

    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.

    Returns:
        float: The value of the metric to be optimized (e.g., best validation loss).
               Optuna aims to minimize this value by default. Return float('inf') or
               raise optuna.TrialPruned on failure.
    """
    global config, processed_data, static_features_results # Access globally loaded data

    log.info(f"\n--- Starting Optuna Trial {trial.number} ---")

    # --- Hyperparameter Search Space Definition ---
    # Create a deep copy of the original config for this trial to avoid modifying the global one
    # Using json loads/dumps is a simple way to deep copy nested dictionaries
    try:
        trial_config = json.loads(json.dumps(config))
    except Exception as e:
        log.error(f"Trial {trial.number}: Failed to deep copy config: {e}")
        raise optuna.TrialPruned("Config copy failed.")

    # === Suggest Hyperparameters using trial.suggest_... ===
    # Override values in trial_config with suggested ones.

    # --- General Training Hyperparameters ---
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True) # Log scale for learning rate
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1) # Dropout rate
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # Optional: Tune batch size
    # loss_function = trial.suggest_categorical("loss_function", ["bce", "focal"]) # Optional: Tune loss
    # sampling_strategy = trial.suggest_categorical("sampling_strategy", ["random", "smote"]) # Optional: Tune sampling

    trial_config['training_config']['learning_rate'] = lr
    trial_config['model_config']['dropout'] = dropout # Assuming dropout is primarily a model config
    # trial_config['training_config']['batch_size'] = batch_size
    # trial_config['training_config']['loss_function'] = loss_function
    # trial_config['training_config']['sampling_strategy'] = sampling_strategy

    # --- Model Architecture Hyperparameters ---
    model_type = trial.suggest_categorical("model_type", ["LSTM", "CNN-LSTM"])
    trial_config['model_config']['type'] = model_type

    # --- LSTM Specific (if model_type is LSTM or CNN-LSTM) ---
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
    lstm_hidden_base = trial.suggest_categorical("lstm_hidden_base", [32, 64, 128, 256])
    # Example: Create layers like [128, 64] or [256]
    lstm_layers = [lstm_hidden_base // (2**i) for i in range(num_lstm_layers)]
    lstm_layers = [max(16, h) for h in lstm_layers] # Ensure minimum hidden size
    trial_config['model_config']['lstm_layers'] = lstm_layers
    trial_config['model_config']['bidirectional'] = trial.suggest_categorical("lstm_bidirectional", [True, False])

    # --- CNN Specific (only if model_type is CNN-LSTM) ---
    if model_type == "CNN-LSTM":
        num_cnn_layers = trial.suggest_int("num_cnn_layers", 1, 3)
        cnn_filters_base = trial.suggest_categorical("cnn_filters_base", [16, 32, 64])
        # Example: Create filter sizes like [32, 64] or [16]
        cnn_filters = [cnn_filters_base * (2**i) for i in range(num_cnn_layers)]
        # Suggest odd kernel sizes (common practice)
        cnn_kernels = [trial.suggest_int(f"cnn_kernel_{i}", 3, 9, step=2) for i in range(num_cnn_layers)] # Kernels: 3, 5, 7, 9

        trial_config['model_config']['cnn_filters'] = cnn_filters
        trial_config['model_config']['cnn_kernels'] = cnn_kernels
        # Keep stride/padding/activation fixed for simplicity, or tune them:
        # trial_config['model_config']['cnn_stride'] = trial.suggest_int("cnn_stride", 1, 2)
        # trial_config['model_config']['cnn_activation'] = trial.suggest_categorical("cnn_activation", ["relu", "tanh"])

        # Attention Heads (only for CNN-LSTM)
        use_attention = trial.suggest_categorical("use_attention", [True, False])
        if use_attention:
            attn_heads = trial.suggest_categorical("attn_heads", [2, 4, 8])
            trial_config['model_config']['attn_heads'] = attn_heads
        else:
            trial_config['model_config']['attn_heads'] = 0 # Indicate no attention

    log.info(f"Trial {trial.number} Parameters: {trial.params}")

    # --- Prepare DataLoaders with Trial Config ---
    # Use the globally loaded data but potentially modified batch size from trial_config
    try:
        # Note: input_dims are calculated based on the original config's feature settings
        # If feature selection were part of tuning, this would need adjustment.
        train_loader, val_loader, _, input_dim_sequence, input_dim_static = prepare_dataloaders(
            processed_data, static_features_results, trial_config # Pass the modified config for this trial
        )
        # Check if loaders were created successfully
        if not train_loader or not val_loader:
            log.error(f"Trial {trial.number}: Failed to create DataLoaders (Train or Val is None). Pruning.")
            raise optuna.TrialPruned("DataLoader creation failed.")
        # Check if dimensions were calculated
        if input_dim_sequence is None or input_dim_static is None:
             log.error(f"Trial {trial.number}: Failed to get input dimensions. Pruning.")
             raise optuna.TrialPruned("Input dimension calculation failed.")

    except optuna.TrialPruned as e:
        raise e # Re-raise prune exceptions
    except Exception as e:
         log.error(f"Trial {trial.number}: Error during data preparation: {e}", exc_info=True)
         raise optuna.TrialPruned("Data preparation error.") # Prune on other data prep errors


    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Trial {trial.number}: Using device: {device}")
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True # Can speed up if input sizes are fixed

    # --- Build Model ---
    try:
        # Use the get_model factory function with the trial's config
        model = get_model(trial_config, input_dim_sequence, input_dim_static)
        model.to(device) # Move model to the selected device
    except Exception as e:
        log.error(f"Trial {trial.number}: Failed to build model with suggested params: {e}", exc_info=True)
        raise optuna.TrialPruned("Model building failed.") # Prune if model build fails

    # --- Train Model ---
    # Use the imported train_model function
    # We don't necessarily need to save the model during tuning, just get the metric.
    # Set a potentially reduced number of epochs and shorter patience for tuning trials.
    tuning_epochs = safe_get(trial_config, ['tuning_config', 'epochs'], 30) # Get max epochs for tuning
    trial_config['training_config']['epochs'] = tuning_epochs # Override general epochs
    tuning_patience = safe_get(trial_config, ['tuning_config', 'patience'], 5) # Get patience for tuning
    trial_config['early_stopping']['patience'] = tuning_patience # Override general patience

    try:
        # Pass the trial-specific config to train_model
        # output_dir=None prevents saving models during tuning trials
        best_model_state, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trial_config, # Pass the trial's config
            device=device,
            output_dir=None # Don't save models during individual trials
        )
        # Check if training failed (returned None)
        if best_model_state is None or history is None:
             log.error(f"Trial {trial.number}: Training function returned None. Pruning.")
             raise optuna.TrialPruned("Training failed.")

    except optuna.TrialPruned as e:
        raise e # Re-raise prune exceptions
    except Exception as e:
         log.error(f"Trial {trial.number}: Training loop failed: {e}", exc_info=True)
         # Return a high loss value to indicate failure to Optuna
         return float('inf')


    # --- Evaluate and Return Metric ---
    # Optimize based on the best validation metric achieved during training.
    # We typically use validation loss for minimization.
    val_losses = [loss for loss in history.get('val_loss', []) if loss is not None and not np.isnan(loss)]
    # val_f1s = [f1 for f1 in history.get('val_f1', []) if f1 is not None and not np.isnan(f1)] # Alternative metric

    if not val_losses:
        log.error(f"Trial {trial.number}: No valid validation losses recorded. Returning high loss.")
        metric_to_optimize = float('inf')
    else:
        # --- Optimization Metric ---
        # Option 1: Minimize best validation loss (most common)
        metric_to_optimize = min(val_losses)
        log.info(f"Trial {trial.number}: Best Validation Loss = {metric_to_optimize:.5f}")

        # Option 2: Maximize best validation F1-score (better for imbalance)
        # if val_f1s:
        #     best_f1 = max(val_f1s)
        #     metric_to_optimize = -best_f1 # Optuna minimizes, so return negative F1
        #     log.info(f"Trial {trial.number}: Best Validation F1 = {best_f1:.5f}")
        # else:
        #     log.error(f"Trial {trial.number}: No valid validation F1 scores recorded. Returning high loss.")
        #     metric_to_optimize = float('inf') # Or 0 if maximizing -F1

    # --- Optuna Pruning ---
    # Report intermediate results (e.g., validation loss at each epoch) to Optuna
    # This allows Optuna to prune unpromising trials early.
    for epoch, val_loss in enumerate(val_losses):
        trial.report(val_loss, step=epoch)
        # Check if the trial should be pruned based on intermediate results
        if trial.should_prune():
            log.warning(f"Trial {trial.number}: Pruned at epoch {epoch+1} based on intermediate value: {val_loss:.5f}.")
            raise optuna.TrialPruned()

    # --- Clean up GPU memory (important when running many trials) ---
    del model, train_loader, val_loader, history, best_model_state
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()

    # Return the metric Optuna should minimize/maximize
    return metric_to_optimize


# ==============================================================================
# == Main Tuning Execution ==
# ==============================================================================
def run_tuning(n_trials: int = 50):
    """
    Runs the Optuna hyperparameter search study.

    Args:
        n_trials (int): The number of optimization trials to run.
    """
    global config # Access global config for study naming/saving paths

    # Get study configuration from main config
    study_name = safe_get(config, ['tuning_config', 'study_name'], "stress_detection_study")
    results_dir = safe_get(config, ['save_paths', 'results'], './outputs/results')
    models_dir = safe_get(config, ['save_paths', 'models'], './outputs/models') # For potential study db

    # --- Optional: Configure Storage for Resuming Studies ---
    # Using SQLite allows resuming interrupted studies.
    # storage_path = os.path.join(models_dir, f"{study_name}.db")
    # storage_name = f"sqlite:///{storage_path}"
    # log.info(f"Using Optuna storage: {storage_name}")
    # study = optuna.create_study(
    #     study_name=study_name,
    #     storage=storage_name,
    #     load_if_exists=True, # Resume if database exists
    #     direction="minimize", # Optimize for minimum validation loss
    #     pruner=optuna.pruners.MedianPruner() # Example pruner
    # )
    # --- End Optional Storage ---

    # --- Create Study (In-memory by default) ---
    storage_name = None # Use in-memory storage
    direction = "minimize" # Optimize for minimum validation loss
    log.info(f"Starting Optuna study '{study_name}' for {n_trials} trials (in-memory).")
    log.info(f"Optimization direction: {direction} (lower is better)")

    # Example Pruner: MedianPruner stops trials performing worse than median of previous trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    log.info(f"Using Pruner: {pruner.__class__.__name__}")

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_name, # None for in-memory
        pruner=pruner
    )

    # --- Run Optimization ---
    try:
        study.optimize(
            objective, # The function to optimize
            n_trials=n_trials, # Maximum number of trials
            timeout=None, # Optional timeout in seconds
            gc_after_trial=True # Enable garbage collection after each trial
        )
    except KeyboardInterrupt:
        log.warning("Tuning interrupted by user.")
    except Exception as e:
        log.error(f"An error occurred during optimization: {e}", exc_info=True)

    # --- Print Results ---
    log.info("\n--- Optuna Study Finished ---")
    log.info(f"Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        log.info(f"Best trial number: {best_trial.number}")
        log.info(f"  Best Value (Validation Loss): {best_trial.value:.5f}")
        log.info("  Best Parameters Found:")
        for key, value in best_trial.params.items():
            log.info(f"    {key}: {value}")

        # --- Save Best Parameters ---
        # Save the best hyperparameters found to a JSON file
        best_params_file = os.path.join(results_dir, "best_hyperparameters.json")
        try:
            os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
            with open(best_params_file, 'w') as f:
                json.dump(best_trial.params, f, indent=4)
            log.info(f"Best hyperparameters saved to: {best_params_file}")

            # --- Optional: Update main config with best params ---
            # Be cautious with automatically overwriting the main config
            # update_main_config = False # Set to True to enable update
            # if update_main_config:
            #     log.warning(f"Updating main config file '{CONFIG_PATH}' with best parameters...")
            #     main_config_copy = load_config(CONFIG_PATH) # Reload original
            #     if main_config_copy:
            #         # Update relevant sections (model_config, training_config)
            #         for key, value in best_trial.params.items():
            #             # This logic needs refinement based on how params map to config sections
            #             if key in main_config_copy.get('training_config', {}): main_config_copy['training_config'][key] = value
            #             if key in main_config_copy.get('model_config', {}): main_config_copy['model_config'][key] = value
            #             # Add specific handling for nested params like layers, model_type etc.
            #             if key == 'model_type': main_config_copy['model_config']['type'] = value
            #             # ... add more specific updates ...
            #         try:
            #             with open(CONFIG_PATH, 'w') as f: json.dump(main_config_copy, f, indent=4)
            #             log.info(f"Successfully updated '{CONFIG_PATH}'.")
            #         except Exception as write_e: log.error(f"Failed to write updated config: {write_e}")

        except Exception as e:
            log.error(f"Failed to save best hyperparameters: {e}")

    except ValueError:
        log.warning("No completed trials found in the study. Cannot determine best parameters.")


if __name__ == "__main__":
    # This block runs only when the script is executed directly
    log.info("Running Optuna tuning script...")

    # --- Configuration for running tuning ---
    NUM_TRIALS = 50 # Set the number of trials to run

    # Ensure 'tuning_config' section exists in config.json or provide defaults here
    # Example: Add to config.json:
    # "tuning_config": {
    #     "study_name": "stress_study_v1",
    #     "epochs": 30,  // Max epochs per trial
    #     "patience": 5   // Early stopping patience per trial
    # }

    run_tuning(n_trials=NUM_TRIALS)
    log.info("Optuna tuning script finished.")
