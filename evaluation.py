# evaluation.py (Handles model evaluation, metrics, and plots)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score,
                             precision_recall_curve, roc_curve, auc, average_precision_score)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
import os
import collections # For counting labels/predictions

# --- Import SHAP safely ---
try:
    import shap
    # Optional: Initialize SHAP JS visualization if running in a notebook context
    # This line should typically be run directly in the notebook cell:
    # shap.initjs()
except ImportError:
    shap = None
    logging.error("`shap` library not found. pip install shap")
    logging.warning("Feature importance calculation using SHAP will not be available.")

from visualization import plot_roc_curve, plot_precision_recall_curve
from utils import safe_get

log = logging.getLogger(__name__)

# --- Find Best Threshold ---
def find_best_threshold(
    all_probs_np: np.ndarray,
    all_labels_np: np.ndarray,
    metric: str = 'f1' # Metric to optimize ('f1', 'accuracy', etc.)
) -> Tuple[float, float]:
    """
    Finds the best probability threshold on the validation set (or any set)
    to maximize a given binary classification metric.

    Args:
        all_probs_np (np.ndarray): Array of predicted probabilities for the positive class.
        all_labels_np (np.ndarray): Array of true binary labels (0 or 1).
        metric (str, optional): The metric to optimize ('f1' or 'accuracy'). Defaults to 'f1'.

    Returns:
        Tuple[float, float]:
            - best_threshold: The probability threshold maximizing the chosen metric.
            - best_metric_value: The value of the metric at the best threshold.
    """
    log.info(f"Finding best threshold to optimize metric: {metric}...")
    best_threshold = 0.5 # Default threshold
    best_metric_value = -1.0 # Initialize with a value that will be beaten

    if len(all_probs_np) == 0 or len(all_labels_np) == 0 or len(all_probs_np) != len(all_labels_np):
        log.error("Invalid input arrays for threshold finding (empty or mismatched lengths). Using default 0.5.")
        # Calculate metric at default 0.5 threshold if possible
        try:
            default_preds = (all_probs_np > 0.5).astype(int)
            if metric == 'f1': default_metric = f1_score(all_labels_np, default_preds, pos_label=1, zero_division=0)
            else: default_metric = accuracy_score(all_labels_np, default_preds)
        except: default_metric = 0.0
        return 0.5, default_metric

    if len(np.unique(all_labels_np)) < 2:
        log.warning("Only one class present in labels. Threshold finding might not be meaningful. Using default 0.5.")
        try:
            default_preds = (all_probs_np > 0.5).astype(int)
            if metric == 'f1': default_metric = f1_score(all_labels_np, default_preds, pos_label=1, zero_division=0)
            else: default_metric = accuracy_score(all_labels_np, default_preds)
        except: default_metric = 0.0
        return 0.5, default_metric

    # --- Optimize for F1-Score ---
    if metric == 'f1':
        try:
            precisions, recalls, thresholds = precision_recall_curve(all_labels_np, all_probs_np, pos_label=1)
            # Calculate F1 score for each threshold (handle division by zero)
            # Note: thresholds array is one element shorter than precisions/recalls
            f1_scores = np.divide(2 * precisions * recalls, precisions + recalls,
                                  out=np.zeros_like(precisions), where=(precisions + recalls) != 0)

            # Align arrays: remove the last F1 score corresponding to recall=0
            if len(f1_scores) > len(thresholds):
                f1_scores = f1_scores[:-1]

            if len(thresholds) == 0: # Handle cases where PR curve is degenerate
                log.warning("Precision-recall curve returned no thresholds. Using default 0.5.")
                best_threshold = 0.5
                best_metric_value = f1_score(all_labels_np, (all_probs_np > 0.5).astype(int), pos_label=1, zero_division=0)
            else:
                # Find the threshold that yields the maximum F1 score
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                best_metric_value = f1_scores[best_idx]
            log.info(f"Best threshold for F1: {best_threshold:.4f} (F1 = {best_metric_value:.4f})")
        except Exception as e:
            log.error(f"Error finding best threshold for F1: {e}. Using default 0.5.")
            best_threshold = 0.5
            best_metric_value = f1_score(all_labels_np, (all_probs_np > 0.5).astype(int), pos_label=1, zero_division=0)

    # --- Optimize for Accuracy ---
    elif metric == 'accuracy':
        # Iterate through a range of possible thresholds
        thresholds_to_check = np.linspace(0.01, 0.99, 100) # Check 100 thresholds
        best_acc = -1.0
        best_thresh_acc = 0.5
        for thresh in thresholds_to_check:
            preds = (all_probs_np > thresh).astype(int)
            acc = accuracy_score(all_labels_np, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh_acc = thresh
        best_threshold = best_thresh_acc
        best_metric_value = best_acc
        log.info(f"Best threshold for Accuracy: {best_threshold:.4f} (Accuracy = {best_metric_value:.4f})")

    # --- Unsupported Metric ---
    else:
        log.warning(f"Unsupported metric '{metric}' for threshold tuning. Using default 0.5.")
        best_threshold = 0.5
        # Calculate the specified metric at the default threshold as the 'best' value
        try:
             preds_default = (all_probs_np > 0.5).astype(int)
             if metric == 'f1': best_metric_value = f1_score(all_labels_np, preds_default, pos_label=1, zero_division=0)
             elif metric == 'precision': best_metric_value = precision_score(all_labels_np, preds_default, pos_label=1, zero_division=0)
             elif metric == 'recall': best_metric_value = recall_score(all_labels_np, preds_default, pos_label=1, zero_division=0)
             else: best_metric_value = accuracy_score(all_labels_np, preds_default) # Default to accuracy
        except Exception as e:
             log.error(f"Could not calculate metric '{metric}' at threshold 0.5: {e}")
             best_metric_value = 0.0 # Fallback metric value

    # Ensure threshold is within [0, 1] range, handle potential edge cases from PR curve
    best_threshold = np.clip(best_threshold, 0.0, 1.0)

    return best_threshold, best_metric_value


# --- Evaluate Model ---
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module, # Loss function (optional for evaluation, but can calculate loss)
    device: torch.device,
    config: Dict[str, Any], # Main config for label names etc.
    output_dir: str, # Directory to save plots
    set_name: str = "Evaluation", # Name for logging/plotting (e.g., "Validation", "Test")
    threshold: Optional[float] = 0.5 # Probability threshold for classification; None = auto-optimize via F1
) -> Optional[Dict[str, Any]]:
    """
    Evaluates the trained model on a given dataset (e.g., validation or test).
    Calculates loss, metrics (accuracy, F1, precision, recall, AUC, AP),
    generates plots (Confusion Matrix, ROC, PR), and returns results.

    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        criterion (nn.Module): Loss function (used to calculate evaluation loss).
        device (torch.device): Device to run evaluation on.
        config (Dict[str, Any]): Main configuration dictionary.
        output_dir (str): Directory to save generated plots.
        set_name (str, optional): Name of the dataset being evaluated. Defaults to "Evaluation".
        threshold (float, optional): Probability threshold for converting probabilities
            to binary predictions. Defaults to 0.5. Pass None to auto-select the
            threshold that maximizes F1 score.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing evaluation metrics and results,
                                  or None if evaluation fails. Keys include: 'loss',
                                  'accuracy', 'f1_score', 'precision', 'recall', 'roc_auc',
                                  'avg_precision', 'classification_report', 'confusion_matrix',
                                  'threshold_used', 'probabilities', 'labels'.
    """
    auto_threshold = threshold is None
    if threshold is None:
        threshold = 0.5  # temporary default; will be replaced after collecting probs
    log.info(f"--- Starting Model Evaluation on {set_name} Set (Threshold: {threshold:.4f}{', auto-optimize enabled' if auto_threshold else ''}) ---")
    if not dataloader:
        log.error(f"{set_name} DataLoader is None. Cannot evaluate.")
        return None
    if not criterion:
        log.warning(f"Criterion (loss function) not provided for {set_name} evaluation. Loss will not be calculated.")

    model.to(device) # Ensure model is on the correct device
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_samples_processed = 0
    all_probs_list = [] # Store raw probabilities
    all_preds_list = [] # Store binary predictions based on threshold
    all_labels_list = [] # Store true labels

    with torch.no_grad(): # Disable gradient calculations during evaluation
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) != 5:
                log.warning(f"{set_name} Batch {batch_idx}: Skipping, incorrect data format from loader.")
                continue

            # Unpack data and move to device
            seq_features, static_features, labels, _, _ = batch_data
            seq_features = seq_features.to(device)
            static_features = static_features.to(device) if hasattr(model, 'input_dim_static') and model.input_dim_static > 0 else None
            labels = labels.to(device)

            batch_size = seq_features.size(0)
            if batch_size == 0: continue # Skip empty batches

            # --- Forward Pass ---
            try:
                outputs = model(seq_features, static_features) # Get model logits
            except Exception as model_e:
                log.error(f"{set_name} Batch {batch_idx}: Model forward pass error: {model_e}", exc_info=True)
                continue # Skip batch on error

            # --- Calculate Loss (Optional) ---
            if criterion:
                try:
                    loss = criterion(outputs.squeeze(), labels.float())
                    total_loss += loss.item() * batch_size # Accumulate loss weighted by batch size
                except Exception as loss_e:
                     log.warning(f"{set_name} Batch {batch_idx}: Loss calculation error: {loss_e}. Skipping loss for this batch.")
                     # Continue processing predictions even if loss fails

            # --- Get Probabilities and Predictions ---
            try:
                # Apply sigmoid to logits to get probabilities
                probs = torch.sigmoid(outputs.squeeze())
                # Apply threshold to get binary predictions
                preds = (probs > threshold).int()
                # Collect results (move to CPU, convert to numpy)
                all_probs_list.extend(probs.cpu().numpy())
                all_preds_list.extend(preds.cpu().numpy().astype(int))
                all_labels_list.extend(labels.cpu().numpy().astype(int))
                num_samples_processed += batch_size # Count samples only if prediction succeeded
            except Exception as pred_e:
                 log.warning(f"{set_name} Batch {batch_idx}: Prediction processing error: {pred_e}")

    # --- Post-Loop Checks ---
    log.info(f"{set_name} evaluation loop finished. Processed {num_samples_processed} samples.")
    if num_samples_processed == 0:
        log.error(f"No samples were successfully processed during {set_name} evaluation.")
        return None
    # Ensure collected lists have consistent lengths
    if len(all_labels_list) != len(all_preds_list) or len(all_labels_list) != len(all_probs_list):
        log.error(f"{set_name}: Label/Pred/Prob count mismatch after loop. Metrics might be inaccurate. "
                  f"(Labels: {len(all_labels_list)}, Preds: {len(all_preds_list)}, Probs: {len(all_probs_list)})")
        # Decide whether to proceed or return None. Proceeding for now.

    # Convert collected lists to numpy arrays
    all_preds_np = np.array(all_preds_list)
    all_labels_np = np.array(all_labels_list)
    all_probs_np = np.array(all_probs_list)

    # Auto-optimize threshold if requested
    if auto_threshold:
        threshold, best_f1 = find_best_threshold(all_probs_np, all_labels_np, metric='f1')
        log.info(f"{set_name}: Auto-optimized threshold = {threshold:.4f} (F1 = {best_f1:.4f})")
        # Recompute predictions with the optimized threshold
        all_preds_np = (all_probs_np > threshold).astype(int)

    # Calculate average loss if criterion was provided and loss was calculated
    avg_loss = (total_loss / num_samples_processed) if criterion and num_samples_processed > 0 else np.nan

    # Debugging output for label/prediction distribution
    unique_labels, label_counts = np.unique(all_labels_np, return_counts=True)
    unique_preds, pred_counts = np.unique(all_preds_np, return_counts=True)
    log.info(f"{set_name} DEBUG: True Labels Distribution: {dict(zip(unique_labels, label_counts))}")
    log.info(f"{set_name} DEBUG: Predicted Labels Distribution (Thresh={threshold:.2f}): {dict(zip(unique_preds, pred_counts))}")

    # --- Calculate Metrics ---
    try:
        # Define expected labels and names for reports/matrices
        expected_labels = [0, 1] # Assuming binary: 0=non-stress, 1=stress
        target_names = ['non-stress', 'stress'] # Corresponding names

        # Basic metrics
        accuracy = accuracy_score(all_labels_np, all_preds_np)
        # Calculate metrics for the positive class (stress=1)
        f1 = f1_score(all_labels_np, all_preds_np, labels=expected_labels, pos_label=1, average='binary', zero_division=0)
        precision = precision_score(all_labels_np, all_preds_np, labels=expected_labels, pos_label=1, average='binary', zero_division=0)
        recall = recall_score(all_labels_np, all_preds_np, labels=expected_labels, pos_label=1, average='binary', zero_division=0)

        # Area under curve metrics (require probabilities)
        avg_precision_score_val = average_precision_score(all_labels_np, all_probs_np, pos_label=1) # AP for PR curve
        roc_auc_score_val = None
        if len(unique_labels) > 1: # AUC requires at least two classes present in true labels
            try:
                fpr, tpr, _ = roc_curve(all_labels_np, all_probs_np, pos_label=1)
                roc_auc_score_val = auc(fpr, tpr)
            except Exception as auc_e:
                log.error(f"Error calculating ROC AUC score for {set_name}: {auc_e}")
                roc_auc_score_val = None # Set to None if calculation fails
        else:
            log.warning(f"Only one class present in true labels for {set_name}. ROC AUC is not defined.")

        # Classification Report
        report = "Classification report generation failed."
        try:
            # Generate report only if both classes are present in true labels
            if len(unique_labels) >= 2:
                report = classification_report(all_labels_np, all_preds_np, labels=expected_labels,
                                               target_names=target_names, zero_division=0, digits=4)
            else:
                report = f"Classification report skipped: Only one true class present ({unique_labels})."
        except Exception as report_e:
            log.error(f"Error generating classification_report for {set_name}: {report_e}", exc_info=True)

        # Confusion Matrix
        cm = confusion_matrix(all_labels_np, all_preds_np, labels=expected_labels)

        # --- Log Metrics ---
        log.info(f"--- {set_name} Evaluation Results (Threshold: {threshold:.4f}) ---")
        log.info(f"Average Loss: {avg_loss:.4f}")
        log.info(f"Accuracy: {accuracy:.4f}")
        log.info(f"F1-Score (Stress): {f1:.4f}")
        log.info(f"Precision (Stress): {precision:.4f}")
        log.info(f"Recall (Stress): {recall:.4f}")
        log.info(f"Avg Precision Score (AP): {avg_precision_score_val:.4f}") # Log AP
        if roc_auc_score_val is not None: log.info(f"ROC AUC Score: {roc_auc_score_val:.4f}") # Log AUC if calculated
        log.info("Classification Report:\n" + report)
        log.info("Confusion Matrix (Rows: True, Cols: Pred):\n" + str(cm))

        # --- Generate and Save Plots ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

            # Confusion Matrix Plot
            try:
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_names, yticklabels=target_names)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'{set_name} Confusion Matrix (Thresh={threshold:.2f})')
                # Include threshold in filename for clarity
                plot_filename_cm = f"confusion_matrix_{set_name.lower().replace(' ', '_')}_thresh{threshold:.2f}.png"
                plot_path_cm = os.path.join(output_dir, plot_filename_cm)
                plt.savefig(plot_path_cm)
                log.info(f"{set_name} confusion matrix plot saved to {plot_path_cm}")
                plt.close() # Close the plot figure context
            except Exception as plot_e:
                log.error(f"Failed to save {set_name} confusion matrix plot: {plot_e}")

            # ROC Curve Plot (call imported function)
            try:
                plot_roc_curve(all_labels_np, all_probs_np, output_dir, set_name)
            except Exception as plot_e:
                log.error(f"Failed to generate ROC curve plot for {set_name}: {plot_e}")

            # Precision-Recall Curve Plot (call imported function)
            try:
                plot_precision_recall_curve(all_labels_np, all_probs_np, output_dir, set_name)
            except Exception as plot_e:
                log.error(f"Failed to generate PR curve plot for {set_name}: {plot_e}")
        else:
            log.warning("Output directory not specified. Skipping plot saving.")

        # --- Prepare Results Dictionary ---
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'avg_precision': avg_precision_score_val, # Include AP
            'roc_auc': roc_auc_score_val, # Include AUC (can be None)
            'classification_report': report,
            'confusion_matrix': cm.tolist(), # Convert numpy array to list for JSON serialization
            'threshold_used': threshold,
            'probabilities': all_probs_np.tolist(), # Include raw probabilities
            'labels': all_labels_np.tolist() # Include true labels
        }
        return results

    except Exception as e:
        log.error(f"Error calculating metrics or generating plots for {set_name}: {e}", exc_info=True)
        return None


# --- Calculate SHAP Importance ---
def calculate_shap_importance(
    model: nn.Module,
    dataloader: DataLoader, # Use a representative subset (e.g., validation or test loader)
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    num_background_samples: int = 50, # Number of samples for background distribution
    num_explain_samples: int = 100 # Number of samples to explain
    ):
    """
    Calculates and plots feature importance using SHAP GradientExplainer.
    Requires the `shap` library to be installed.

    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader providing samples for background and explanation.
                                 Should yield data in the format (seq, static, ...).
        config (Dict[str, Any]): Main configuration dictionary (for feature names).
        device (torch.device): Device the model is on.
        output_dir (str): Directory to save the SHAP summary plot.
        num_background_samples (int, optional): Number of samples for the background dataset. Defaults to 50.
        num_explain_samples (int, optional): Number of samples to calculate SHAP values for. Defaults to 100.
    """
    # Check if SHAP library is available
    if shap is None:
        log.error("SHAP library not installed. Cannot calculate feature importance.")
        return
    log.info("--- Calculating SHAP Feature Importance ---")
    model.eval() # Ensure model is in evaluation mode

    # --- 1. Prepare Data Subsets (Background and Explanation) ---
    background_seq_list = []
    background_static_list = []
    explain_seq_list = []
    explain_static_list = []
    log.info(f"Collecting {num_background_samples} background and {num_explain_samples} explanation samples from DataLoader...")

    # Iterate through dataloader to collect samples
    samples_collected = 0
    background_collected = 0
    explain_collected = 0
    has_static = hasattr(model, 'input_dim_static') and model.input_dim_static > 0

    for batch_data in dataloader:
        if len(batch_data) != 5: continue # Skip incorrectly formatted batches
        seq_features, static_features, _, _, _ = batch_data
        batch_size = seq_features.shape[0]

        # Determine how many samples to take for background from this batch
        take_bg = 0
        if background_collected < num_background_samples:
            take_bg = min(batch_size, num_background_samples - background_collected)
            background_seq_list.append(seq_features[:take_bg].cpu())
            if has_static: background_static_list.append(static_features[:take_bg].cpu())
            else: background_static_list.append(torch.empty((take_bg, 0))) # Add empty tensor if no static
            background_collected += take_bg

        # Determine how many samples to take for explanation from this batch
        # Start taking explanation samples after background samples from the *same batch* if necessary
        start_idx_expl = take_bg
        take_expl = 0
        if explain_collected < num_explain_samples and batch_size > start_idx_expl:
            take_expl = min(batch_size - start_idx_expl, num_explain_samples - explain_collected)
            explain_seq_list.append(seq_features[start_idx_expl : start_idx_expl+take_expl].cpu())
            if has_static: explain_static_list.append(static_features[start_idx_expl : start_idx_expl+take_expl].cpu())
            else: explain_static_list.append(torch.empty((take_expl, 0))) # Add empty tensor if no static
            explain_collected += take_expl

        # Stop if enough samples are collected
        if background_collected >= num_background_samples and explain_collected >= num_explain_samples:
            break

    # Check if enough samples were collected
    if background_collected < num_background_samples or explain_collected < num_explain_samples:
        log.error(f"Could not collect enough samples (Need BG:{num_background_samples}, Expl:{num_explain_samples}. Got BG:{background_collected}, Expl:{explain_collected}). Aborting SHAP calculation.")
        return

    # Concatenate collected samples and move to the correct device
    background_seq = torch.cat(background_seq_list, dim=0).to(device)
    background_static = torch.cat(background_static_list, dim=0).to(device) if has_static and background_static_list else None
    explain_seq = torch.cat(explain_seq_list, dim=0).to(device)
    explain_static = torch.cat(explain_static_list, dim=0).to(device) if has_static and explain_static_list else None

    log.info(f"Background sequence shape: {background_seq.shape}")
    if background_static is not None: log.info(f"Background static shape: {background_static.shape}")
    log.info(f"Explain sequence shape: {explain_seq.shape}")
    if explain_static is not None: log.info(f"Explain static shape: {explain_static.shape}")

    # --- 2. Create SHAP Explainer ---
    # GradientExplainer is suitable for differentiable models like neural networks
    # It requires a background dataset to approximate expected values
    try:
        # Pass background data as a tuple if static features exist
        background_data = (background_seq, background_static) if background_static is not None else background_seq
        explainer = shap.GradientExplainer(model, background_data)
        log.info("SHAP GradientExplainer created.")
    except Exception as e:
        log.error(f"Failed to create SHAP GradientExplainer: {e}", exc_info=True)
        return

    # --- 3. Calculate SHAP Values ---
    log.info(f"Calculating SHAP values for {explain_seq.shape[0]} samples...")
    shap_values_seq = None
    shap_values_static = None
    try:
        # Pass explanation data as a tuple if static features exist
        explain_data = (explain_seq, explain_static) if explain_static is not None else explain_seq
        shap_values = explainer.shap_values(explain_data)
        log.info("SHAP values calculated.")

        # Unpack SHAP values based on whether static features were included
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Assumes order: [shap_seq, shap_static]
            shap_values_seq, shap_values_static = shap_values[0], shap_values[1]
        elif isinstance(shap_values, np.ndarray) and explain_static is None:
            # Only sequence features were explained
            shap_values_seq = shap_values
        else:
            log.error(f"Unexpected SHAP values format: {type(shap_values)}. Expected list of 2 arrays or single array.")
            return

    except Exception as e:
        log.error(f"SHAP value calculation failed: {e}", exc_info=True)
        # --- Fallback: Try explaining only sequence data if model allows ---
        # This might work if the model can handle static_features=None
        if explain_static is not None: # Only try fallback if static features were initially included
            try:
                log.warning("Attempting SHAP calculation using only sequence data as fallback...")
                # Re-create explainer and calculate SHAP values with sequence data only
                explainer_seq_only = shap.GradientExplainer(model, background_seq) # Background only sequence
                shap_values_seq_only = explainer_seq_only.shap_values(explain_seq) # Explain only sequence
                log.info("SHAP values calculated (sequence only fallback).")
                shap_values_seq = shap_values_seq_only # Assign sequence shap values
                shap_values_static = None # Ensure static shap values are None
            except Exception as e2:
                log.error(f"SHAP value calculation failed even with sequence-only fallback: {e2}", exc_info=True)
                return # Abort if fallback also fails
        else:
             return # Abort if initial calculation failed without static features

    # --- 4. Process & Plot SHAP Values ---
    if shap_values_seq is not None: log.info(f"SHAP Sequence values shape: {shap_values_seq.shape}") # (N, L, F_seq)
    if shap_values_static is not None: log.info(f"SHAP Static values shape: {shap_values_static.shape}") # (N, F_static)

    # Aggregate Sequence Importance: Calculate mean absolute SHAP value across the time dimension (L)
    # This gives an overall importance score for each sequence feature per sample.
    mean_abs_shap_seq = None
    if shap_values_seq is not None:
        mean_abs_shap_seq = np.mean(np.abs(shap_values_seq), axis=1) # Avg over time -> (N, F_seq)

    # --- Get Feature Names ---
    # Sequence features: Use generic names if specific names aren't easily available
    num_seq_features = model.input_dim_sequence if hasattr(model, 'input_dim_sequence') else (mean_abs_shap_seq.shape[1] if mean_abs_shap_seq is not None else 0)
    seq_feature_names = [f"SeqFeat_{i}" for i in range(num_seq_features)] # Placeholder names like SeqFeat_0, SeqFeat_1...
    # Static features: Get names from config
    static_feature_names = safe_get(config, ['static_features_to_use'], [])

    # Validate static feature names against SHAP output dimension
    if shap_values_static is not None and len(static_feature_names) != shap_values_static.shape[1]:
        log.warning(f"Mismatch between static feature names in config ({len(static_feature_names)}) and SHAP static values dimension ({shap_values_static.shape[1]}). Using generic static feature names.")
        static_feature_names = [f"StaticFeat_{i}" for i in range(shap_values_static.shape[1])]
    elif shap_values_static is None and static_feature_names:
         log.warning("Static feature names found in config, but no static SHAP values were calculated.")
         static_feature_names = [] # Clear static names if no static SHAP values exist

    # --- Combine SHAP values and Feature Values for Plotting ---
    # SHAP summary plot requires SHAP values and corresponding feature values
    combined_shap_values_list = []
    combined_features_list = []
    feature_names = []

    # Add sequence features if available
    if mean_abs_shap_seq is not None:
        combined_shap_values_list.append(mean_abs_shap_seq) # Append aggregated seq SHAP values (N, F_seq)
        # Use mean sequence feature value over time as the feature value for plotting
        explain_seq_np = explain_seq.cpu().numpy() # (N, L, F_seq)
        mean_explain_seq = np.mean(explain_seq_np, axis=1) # -> (N, F_seq)
        combined_features_list.append(mean_explain_seq)
        feature_names.extend(seq_feature_names)

    # Add static features if available
    if shap_values_static is not None:
        combined_shap_values_list.append(shap_values_static) # Append static SHAP values (N, F_static)
        explain_static_np = explain_static.cpu().numpy() # (N, F_static)
        combined_features_list.append(explain_static_np)
        feature_names.extend(static_feature_names)

    # Check if any SHAP values were processed
    if not combined_shap_values_list:
        log.error("No valid SHAP values (sequence or static) to plot.")
        return

    # Concatenate along the feature axis if both sequence and static are present
    final_shap_values = np.concatenate(combined_shap_values_list, axis=1) if len(combined_shap_values_list) > 1 else combined_shap_values_list[0]
    final_features = np.concatenate(combined_features_list, axis=1) if len(combined_features_list) > 1 else combined_features_list[0]

    # --- Generate SHAP Summary Plot ---
    log.info("Generating SHAP summary plot...")
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "shap_summary_plot.png")

        # Create a new figure context for saving the plot
        plt.figure()
        shap.summary_plot(
            final_shap_values,
            features=final_features,
            feature_names=feature_names,
            show=False # Prevent showing plot inline automatically
        )
        plt.tight_layout() # Adjust layout
        plt.savefig(save_path) # Save the plot
        plt.close() # Close the figure context to free memory
        log.info(f"SHAP summary plot saved to {save_path}")

    except Exception as plot_e:
        log.error(f"Failed to generate or save SHAP summary plot: {plot_e}", exc_info=True)

