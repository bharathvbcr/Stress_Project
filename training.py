# training.py (Handles model training loop, validation, and early stopping)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
import os
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import collections

# --- Import custom loss and utils ---
try:
    from losses import FocalLoss # Import custom Focal Loss if used
    from utils import safe_get
except ImportError:
    logging.error("Could not import FocalLoss or safe_get. Ensure losses.py and utils.py are accessible.")
    # Define dummy fallbacks if necessary for the script to load
    class FocalLoss(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); logging.critical("Dummy FocalLoss used!"); raise ImportError("FocalLoss not defined")
        def forward(self, inputs, targets): raise NotImplementedError
    def safe_get(data_dict, keys, default=None): temp = data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default

log = logging.getLogger(__name__)

# ==============================================================================
# == Helper Functions for Training ==
# ==============================================================================

def _calculate_pos_weight(train_loader: DataLoader, device: torch.device) -> Optional[torch.Tensor]:
    """
    Calculates the positive class weight for BCEWithLogitsLoss based on the
    distribution of labels in the training data. Helps mitigate class imbalance.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        device (torch.device): Device where the weight tensor should reside.

    Returns:
        Optional[torch.Tensor]: A tensor containing the positive class weight,
                                or None if calculation fails or data is balanced/single-class.
    """
    log.info("Calculating class weights for training set (for potential BCE use)...")
    train_label_counts = collections.Counter()
    num_train_samples = 0
    pos_weight = None

    if not train_loader:
        log.error("Train loader is None, cannot calculate weights.")
        return None

    try:
        # Iterate through the training loader to count labels
        for batch_data in train_loader:
            # Expecting (seq, static, labels, subj_id, start_idx) from SequenceWindowDataset
            if len(batch_data) != 5:
                 log.warning("Train loader batch did not contain 5 items. Skipping weight calculation for this batch.")
                 continue
            _, _, labels, _, _ = batch_data # Unpack labels
            # Ensure labels are on CPU and converted to numpy for Counter
            batch_counts = collections.Counter(labels.cpu().numpy().astype(int))
            train_label_counts.update(batch_counts)
            num_train_samples += len(labels)

        if num_train_samples == 0:
            log.error("Train loader has 0 samples after iteration. Cannot calculate weights.")
            return None

        # Calculate counts for class 0 (negative) and class 1 (positive)
        count_0 = train_label_counts.get(0, 0)
        count_1 = train_label_counts.get(1, 0)
        log.info(f"Training label counts - 0: {count_0}, 1: {count_1}")

        # Calculate weight only if both classes are present
        if count_0 > 0 and count_1 > 0:
             # Weight for positive class = (number of negatives) / (number of positives)
             weight_for_1 = count_0 / count_1
             # Create a tensor on the specified device
             pos_weight = torch.tensor([weight_for_1], device=device, dtype=torch.float32)
             log.info(f"Calculated pos_weight for BCE: {pos_weight.item():.4f}")
        else:
             log.warning("Training data has only 1 class present or one class has zero samples. Weighting disabled for BCE.")

    except Exception as e:
        log.error(f"Error calculating weights: {e}. Weighting disabled for BCE.", exc_info=True)
        pos_weight = None # Ensure None on error

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
    # Get loss function type from config, default to 'bce'
    loss_function_type = safe_get(config, ['training_config', 'loss_function'], 'bce').lower()
    criterion = None

    # --- Focal Loss ---
    if loss_function_type == 'focal':
        try:
            # Get alpha and gamma parameters from config
            alpha = safe_get(config, ['training_config', 'focal_loss_alpha'], 0.25)
            gamma = safe_get(config, ['training_config', 'focal_loss_gamma'], 2.0)
            # Initialize FocalLoss (ensure FocalLoss class is imported)
            criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
            log.info(f"Using FocalLoss (alpha={alpha}, gamma={gamma})")
        except NameError:
             # This happens if FocalLoss class wasn't imported successfully
             log.critical("FocalLoss selected but class definition not found! Aborting.")
             raise ImportError("FocalLoss class not found.")
        except Exception as e:
             log.error(f"Error initializing FocalLoss: {e}. Falling back to BCE.")
             loss_function_type = 'bce' # Fallback to BCE

    # --- BCEWithLogitsLoss (Default or Fallback) ---
    # Use BCE if 'bce' is specified or if FocalLoss initialization failed
    if criterion is None: # Checks if criterion is still None
         if pos_weight is not None:
              # Use weighted BCE if pos_weight was calculated successfully
              criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
              log.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.4f}")
         else:
              # Use standard unweighted BCE
              criterion = nn.BCEWithLogitsLoss()
              log.info("Using standard BCEWithLogitsLoss (no weighting).")

    return criterion


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int, # Current epoch number (0-based)
    total_epochs: int # Total number of epochs
) -> float:
    """
    Runs a single training epoch, calculating loss and updating model weights.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device to run training on (CPU or CUDA).
        epoch_num (int): The current epoch number (for logging).
        total_epochs (int): The total number of epochs (for logging).

    Returns:
        float: The average training loss for the epoch. Returns np.nan if epoch fails.
    """
    model.train() # Set model to training mode
    running_loss = 0.0
    num_samples_processed = 0

    # Iterate over batches in the training dataloader
    for i, batch_data in enumerate(dataloader):
        # Basic check for expected data format
        if len(batch_data) != 5:
            log.warning(f"Epoch [{epoch_num+1}/{total_epochs}] Batch {i}: Skipping, incorrect data format from loader.")
            continue

        # Unpack data and move to the target device
        seq_features, static_features, labels, _, _ = batch_data
        seq_features = seq_features.to(device)
        # Only move static features if the model uses them (input_dim_static > 0)
        static_features = static_features.to(device) if hasattr(model, 'input_dim_static') and model.input_dim_static > 0 else None
        labels = labels.to(device)

        batch_size = seq_features.size(0)
        if batch_size == 0: continue # Skip empty batches

        # --- Forward and Backward Pass ---
        optimizer.zero_grad() # Reset gradients

        # Forward pass: Get model predictions (logits)
        try:
            outputs = model(seq_features, static_features) # Pass both sequence and static features
        except Exception as model_e:
            log.error(f"Epoch [{epoch_num+1}/{total_epochs}] Batch {i}: Model forward pass error: {model_e}", exc_info=True)
            log.error(f"  Input Shapes - Seq: {seq_features.shape}, Static: {static_features.shape if static_features is not None else 'None'}")
            continue # Skip this batch if model fails

        # Loss calculation
        try:
            # Squeeze model output if necessary (e.g., if output is [N, 1])
            # Ensure labels are float type for BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), labels.float())
        except Exception as loss_e:
             log.error(f"Epoch [{epoch_num+1}/{total_epochs}] Batch {i}: Loss calculation error: {loss_e}", exc_info=True)
             log.error(f"  Output shape: {outputs.shape}, Label shape: {labels.shape}, Label dtype: {labels.dtype}")
             continue # Skip this batch if loss calculation fails

        # Backward pass: Calculate gradients
        try:
            loss.backward()
        except Exception as backward_e:
            log.error(f"Epoch [{epoch_num+1}/{total_epochs}] Batch {i}: Backward pass error: {backward_e}", exc_info=True)
            continue # Skip optimizer step if backward pass fails

        # Optimizer step: Update model weights
        try:
            optimizer.step()
        except Exception as step_e:
            log.error(f"Epoch [{epoch_num+1}/{total_epochs}] Batch {i}: Optimizer step error: {step_e}", exc_info=True)
            # Continue to next batch even if step fails for one batch? Or stop? Continuing for now.

        # Accumulate loss (weighted by batch size for accurate averaging)
        running_loss += loss.item() * batch_size
        num_samples_processed += batch_size

    # --- Epoch End ---
    if num_samples_processed == 0:
        log.error(f"Epoch [{epoch_num+1}/{total_epochs}]: No samples processed in training epoch.")
        return np.nan # Return NaN if no samples were processed

    # Calculate average loss for the epoch
    avg_train_loss = running_loss / num_samples_processed
    return avg_train_loss


def _validate_one_epoch(
    model: nn.Module,
    dataloader: Optional[DataLoader], # Validation loader is optional
    criterion: nn.Module,
    device: torch.device
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Runs a single validation epoch, calculating loss and metrics (accuracy, F1).

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (Optional[DataLoader]): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run validation on.

    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]:
            - Average validation loss.
            - Validation accuracy.
            - Validation F1-score (binary, pos_label=1).
            Returns (None, None, None) if validation loader is None or epoch fails.
    """
    # If no validation loader is provided, return None for all metrics
    if not dataloader:
        return None, None, None

    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    num_samples_processed = 0
    all_preds_val, all_labels_val = [], [] # Lists to store predictions and labels

    with torch.no_grad(): # Disable gradient calculations for validation
        for batch_data in dataloader:
            if len(batch_data) != 5:
                log.warning("Validation Batch: Skipping, incorrect data format from loader.")
                continue

            # Unpack data and move to device
            seq_features, static_features, labels, _, _ = batch_data
            seq_features = seq_features.to(device)
            static_features = static_features.to(device) if hasattr(model, 'input_dim_static') and model.input_dim_static > 0 else None
            labels = labels.to(device)

            batch_size = seq_features.size(0)
            if batch_size == 0: continue

            # Forward pass
            try:
                outputs = model(seq_features, static_features)
            except Exception as model_e:
                log.error(f"Validation Batch: Model forward pass error: {model_e}", exc_info=True)
                continue # Skip batch

            # Loss calculation (optional, but good for monitoring)
            try:
                # Use view(-1) to flatten both tensors reliably, handling batch size 1
                loss = criterion(outputs.view(-1), labels.view(-1).float())
                val_loss += loss.item() * batch_size
                num_samples_processed += batch_size
            except Exception as loss_e:
                 log.warning(f"Validation Batch: Loss calculation error: {loss_e}. Skipping loss for this batch.")
                 # Still try to get predictions even if loss fails

            # Get predictions (using threshold 0.5 for validation metrics during training)
            try:
                # Apply sigmoid to get probabilities, then threshold
                probs = torch.sigmoid(outputs.view(-1))
                preds = (probs > 0.5).int()
                # Ensure preds is iterable before extending (it should be 1D after view(-1))
                if preds.ndim > 0:
                    all_preds_val.extend(preds.cpu().numpy().astype(int))
                else: # Handle the unlikely case it's still 0-d
                    all_preds_val.append(preds.cpu().item()) # Append the single scalar value
                all_labels_val.extend(labels.cpu().numpy().astype(int))
            except Exception as pred_e:
                log.warning(f"Validation Batch: Prediction processing error: {pred_e}")

    # --- Epoch End ---
    if num_samples_processed == 0:
        log.warning("Validation epoch: No samples processed.")
        return None, None, None

    # Calculate average loss and metrics
    avg_val_loss = val_loss / num_samples_processed
    val_accuracy = None
    val_f1 = None

    # Calculate metrics only if predictions/labels were collected
    if all_labels_val:
        try:
            all_preds_val_np = np.array(all_preds_val)
            all_labels_val_np = np.array(all_labels_val)
            val_accuracy = accuracy_score(all_labels_val_np, all_preds_val_np)
            # Calculate F1 score for the positive class (stress=1)
            # zero_division=0 prevents warnings if precision/recall are zero for a class
            val_f1 = f1_score(all_labels_val_np, all_preds_val_np, pos_label=1, average='binary', zero_division=0)
        except Exception as metric_e:
            log.error(f"Error calculating validation metrics: {metric_e}", exc_info=True)
            # Set metrics to None if calculation fails
            val_accuracy = None
            val_f1 = None

    return avg_val_loss, val_accuracy, val_f1

# ==============================================================================
# == Main Training Function ==
# ==============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any], # Can be trial-specific config during tuning
    device: torch.device,
    output_dir: Optional[str] # Directory to save best model (optional)
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Trains the model with validation, early stopping, and selectable loss function.
    Can be called directly or from a hyperparameter tuning loop.

    Args:
        model: The PyTorch model to train (already instantiated and moved to device).
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set (optional, enables early stopping).
        config: Configuration dictionary.
        device: The device (CPU or CUDA) to train on.
        output_dir: Directory to save the best model state dictionary (optional).

    Returns:
        Tuple containing:
            - best_model_state (Optional[Dict]): State dictionary of the best model
                                                 based on validation loss, or the final
                                                 model state if no validation/improvement.
                                                 Returns None if training fails critically.
            - history (Optional[Dict]): Dictionary containing training and validation
                                        loss/metrics per epoch. Returns None if training fails.
    """
    log.info("--- Starting Model Training ---")

    # --- Basic Setup and Validation ---
    if not train_loader:
        log.critical("Training loader is None. Cannot start training.")
        return None, None
    if not isinstance(model, nn.Module):
         log.critical("Invalid model provided.")
         return None, None

    # --- Get Training Parameters from Config ---
    epochs = safe_get(config, ['training_config', 'epochs'], 50)
    lr = safe_get(config, ['training_config', 'learning_rate'], 0.001)
    patience = safe_get(config, ['early_stopping', 'patience'], 10)
    min_delta = safe_get(config, ['early_stopping', 'min_delta'], 0.001)
    monitor_metric = 'val_loss' # Metric to monitor for early stopping improvement

    # --- Setup Loss Function and Optimizer ---
    pos_weight = _calculate_pos_weight(train_loader, device) # Calculate weight for BCE
    try:
        criterion = _get_criterion(config, pos_weight) # Get loss function
    except ImportError: # Catch if FocalLoss wasn't found/imported
        log.critical("Failed to get criterion due to missing loss definition.")
        return None, None # Critical failure

    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam optimizer

    # --- Early Stopping Initialization ---
    best_monitor_value = float('inf') # Initialize for minimizing validation loss
    epochs_no_improve = 0
    best_model_state = None # Stores the state_dict of the best model found so far
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}

    # --- Hardware Optimization ---
    # Set benchmark flag if using CUDA (can speed up if input sizes don't vary much)
    if device == torch.device("cuda"):
        try:
            torch.backends.cudnn.benchmark = True
            log.info("torch.backends.cudnn.benchmark set to True")
        except Exception as e:
            log.warning(f"Could not set cudnn.benchmark: {e}")
    # --- End Hardware Optimization ---

    # --- Prepare Output Directory and Model Save Path ---
    best_model_path = None
    if output_dir: # Only proceed if an output directory is specified
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir); log.info(f"Created output directory: {output_dir}")
            except OSError as e:
                log.error(f"Could not create output directory {output_dir}: {e}. Model saving disabled.")
                output_dir = None # Disable saving if directory creation fails
        if output_dir: # Check again if creation succeeded or dir already existed
             best_model_path = os.path.join(output_dir, "best_model.pth")

    # --- Log Training Setup ---
    model.to(device) # Ensure model is on the correct device before training
    log.info(f"Training Configuration:")
    log.info(f"  Device: {device}")
    log.info(f"  Model Type: {model.__class__.__name__}")
    log.info(f"  Loss function: {type(criterion).__name__}")
    log.info(f"  Optimizer: Adam, LR: {lr}")
    log.info(f"  Max epochs: {epochs}")
    if val_loader:
        log.info(f"  Early stopping: Patience={patience}, Min Delta={min_delta}, Metric={monitor_metric}")
    else:
        log.info("  Early stopping: Disabled (no validation loader provided).")
    if best_model_path:
        log.info(f"  Best model will be saved to: {best_model_path}")
    else:
        log.info("  Model saving: Disabled (no output_dir specified or creation failed).")

    # --- Training Loop ---
    training_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Train one epoch
        avg_train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        history['train_loss'].append(avg_train_loss)

        # Validate one epoch (if val_loader exists)
        avg_val_loss, val_accuracy, val_f1 = _validate_one_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)

        epoch_duration = time.time() - epoch_start_time

        # Log epoch results
        log_msg = f"Epoch [{epoch+1}/{epochs}] | Dur: {epoch_duration:.2f}s | Train Loss: {avg_train_loss:.4f}"
        if avg_val_loss is not None: log_msg += f" | Val Loss: {avg_val_loss:.4f}"
        if val_accuracy is not None: log_msg += f" | Val Acc: {val_accuracy:.4f}"
        if val_f1 is not None: log_msg += f" | Val F1: {val_f1:.4f}"
        log.info(log_msg)

        # --- Early Stopping Check ---
        # Only perform check if validation loader exists and validation loss was calculated
        if val_loader and avg_val_loss is not None and not np.isnan(avg_val_loss):
             current_monitor_value = avg_val_loss # Monitor validation loss
             # Check if current value is better than best value found so far (minus delta)
             improved = current_monitor_value < best_monitor_value - min_delta
             if improved:
                 log.info(f"  Validation loss improved from {best_monitor_value:.4f} to {current_monitor_value:.4f}.")
                 best_monitor_value = current_monitor_value
                 epochs_no_improve = 0 # Reset counter
                 # Save the model state dictionary (in memory)
                 best_model_state = model.state_dict()
                 # Save to file if path is valid
                 if best_model_path:
                     try:
                         torch.save(best_model_state, best_model_path)
                         log.info(f"  Saved best model state to {best_model_path}")
                     except Exception as e:
                         log.error(f"  Error saving model to {best_model_path}: {e}")
             else:
                 epochs_no_improve += 1
                 log.debug(f"  Validation loss did not improve for {epochs_no_improve}/{patience} epochs. Best: {best_monitor_value:.4f}")
                 # Check if patience limit is reached
                 if epochs_no_improve >= patience:
                     log.info(f"Early stopping triggered after {epoch + 1} epochs.")
                     break # Exit training loop
        # --- End Early Stopping ---

        # Handle case where training loss becomes NaN (e.g., exploding gradients)
        if np.isnan(avg_train_loss):
            log.error("Training loss is NaN. Stopping training.")
            # Return None to indicate failure (useful for hyperparameter tuning)
            return None, history # Return history up to failure point

    # --- Training Loop Finished ---
    training_duration = time.time() - training_start_time
    log.info(f"--- Model Training Finished (Duration: {training_duration:.2f}s) ---")

    # If early stopping never triggered saving (e.g., ran for full epochs or no validation),
    # and training completed without NaN loss, use the final model state.
    if best_model_state is None and model is not None and (not np.isnan(history['train_loss'][-1]) if history['train_loss'] else True):
         log.warning("Early stopping did not save a best model state (or validation failed/absent). Using final model state.")
         best_model_state = model.state_dict()
         # Optionally save the final state if not already saved and path exists
         if best_model_path and (not os.path.exists(best_model_path) or epochs_no_improve < patience):
             try:
                 torch.save(best_model_state, best_model_path)
                 log.info(f"Saved final model state to {best_model_path}.")
             except Exception as e:
                 log.error(f"Error saving final model state to {best_model_path}: {e}")

    # Return the best model state found and the training history
    return best_model_state, history
