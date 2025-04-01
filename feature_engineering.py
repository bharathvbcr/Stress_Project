# feature_engineering.py (Creates full subject feature matrices - potentially redundant)
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional, Union

# Assuming utils.py is available
try:
    from utils import safe_get
except ImportError:
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    logging.warning("Could not import 'safe_get' from 'utils'. Using basic fallback in feature_engineering.py.")

log = logging.getLogger(__name__)

def create_subject_feature_matrices(
    processed_data: Dict[Union[int, str], Dict[str, Any]],
    config: Dict[str, Any]
) -> Dict[Union[int, str], Dict[str, Any]]:
    """
    Combines selected processed signals into feature matrices for each subject.
    This assumes signals within a subject are already aligned in length
    by the preprocessing step (specifically, the functions in signal_processing.py).
    It primarily handles concatenation based on the feature order defined in the config.

    NOTE: This function creates a single large feature matrix per subject.
          This might be useful for certain types of models or analyses, but it's
          different from the windowing approach used in windowing.py and
          data_pipeline.py, which prepares data for sequence models like LSTMs.
          This module might be redundant if the main goal is sequence modeling.

    Args:
        processed_data (Dict): Dictionary with processed (resampled & aligned) signals and labels
                               (output from the signal_processing step).
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[Union[int, str], Dict[str, Any]]:
            Dictionary mapping subject ID to a dictionary containing:
                - 'features': Combined numpy array (num_samples, num_total_features).
                - 'labels': Corresponding label array (num_samples,).
                - 'feature_names': List of names for the columns in the 'features' matrix.
            Returns an empty dictionary if prerequisites are missing or no matrices are created.
    """
    log.info("--- Starting Feature Matrix Creation (Full Subject) ---")
    # Basic validation
    if not processed_data:
        log.error("Processed data dictionary is empty. Cannot create feature matrices.")
        return {}

    subject_feature_matrices = {} # Dictionary to store results

    # --- Get Feature Order from Config ---
    # This defines the order in which signals will be concatenated
    chest_features_to_use = safe_get(config, ['features_to_use', 'chest'], [])
    wrist_features_to_use = safe_get(config, ['features_to_use', 'wrist'], [])
    # Combine features, maintaining the order specified (chest first, then wrist)
    # Create unique keys like 'chest_ECG', 'wrist_EDA'
    feature_order_list = [f"chest_{k}" for k in chest_features_to_use] + \
                         [f"wrist_{k}" for k in wrist_features_to_use]

    if not feature_order_list:
        log.warning("No features specified in config['features_to_use']. Cannot create feature matrices.")
        return {}

    log.info(f"Target feature order for matrices: {feature_order_list}")

    # --- Iterate Through Subjects ---
    for subject_id, subj_data in processed_data.items():
        log.info(f"[S{subject_id}] Creating feature matrix...")
        subj_signals_list = [] # List to hold signal arrays for concatenation
        feature_names_list = [] # List to hold corresponding feature names
        expected_len = None # Length determined by the first valid signal found (usually labels)

        # --- Get Labels First & Determine Expected Length ---
        labels = safe_get(subj_data, ['label'])
        if labels is None or not isinstance(labels, np.ndarray) or labels.size == 0:
            log.error(f"[S{subject_id}] Labels missing or empty in processed_data. Skipping subject matrix creation.")
            continue
        labels = labels.flatten() # Ensure 1D
        expected_len = labels.shape[0]
        if expected_len == 0:
            log.error(f"[S{subject_id}] Labels array is empty (length 0). Skipping subject matrix creation.")
            continue
        log.info(f"[S{subject_id}] Reference length determined from labels: {expected_len}")

        # --- Gather Signals Based on feature_order_list ---
        valid_subject = True # Flag to track if all required signals are valid
        for unique_key in feature_order_list:
            device, key = unique_key.split('_', 1) # e.g., 'chest_ECG' -> 'chest', 'ECG'
            signal = safe_get(subj_data, ['signal', device, key])

            # Validate the signal
            if signal is None or not isinstance(signal, np.ndarray) or signal.size == 0:
                log.warning(f"[S{subject_id}] Signal {unique_key} missing or empty in processed data. Skipping subject matrix creation.")
                # If any required feature is missing, we cannot create a consistent matrix
                valid_subject = False
                break
            elif signal.shape[0] != expected_len:
                # This check ensures alignment from the previous processing step worked
                log.error(f"[S{subject_id}] Length mismatch for {unique_key}. Expected {expected_len}, got {signal.shape[0]}. Skipping subject matrix creation.")
                valid_subject = False
                break

            # Ensure signal is 2D (Time, Channels) before appending
            signal = signal.reshape(signal.shape[0], -1)
            subj_signals_list.append(signal)

            # --- Generate Feature Names ---
            # If signal has multiple channels (e.g., ACC), create names like 'chest_ACC_0', 'chest_ACC_1'
            if signal.shape[1] > 1:
                feature_names_list.extend([f"{unique_key}_{i}" for i in range(signal.shape[1])])
            else:
                # If single channel, just use the unique key
                feature_names_list.append(unique_key)

        # If any signal was invalid, skip to the next subject
        if not valid_subject:
            log.warning(f"[S{subject_id}] Skipping matrix creation due to missing or mismatched signals.")
            continue

        # --- Concatenate Features ---
        if not subj_signals_list:
             # This should not happen if valid_subject is True and feature_order_list is not empty
             log.error(f"[S{subject_id}] No signals collected for concatenation despite passing checks? Skipping subject.")
             continue

        try:
            # Concatenate all collected signal arrays horizontally (along axis=1)
            feature_matrix = np.concatenate(subj_signals_list, axis=1)
            log.info(f"[S{subject_id}] Feature matrix created with shape {feature_matrix.shape}. Labels shape: {labels.shape}.")

            # Ensure final matrix length matches label length (redundant check, but safe)
            if feature_matrix.shape[0] != labels.shape[0]:
                 log.error(f"[S{subject_id}] Final matrix length ({feature_matrix.shape[0]}) does not match label length ({labels.shape[0]}). Skipping.")
                 continue

            # Store the results for this subject
            subject_feature_matrices[subject_id] = {
                'features': feature_matrix,
                'labels': labels,
                'feature_names': feature_names_list # Store the generated feature names
            }
        except ValueError as e:
             # Handle errors during concatenation (e.g., if shapes somehow became inconsistent)
             log.error(f"[S{subject_id}] Error concatenating feature arrays: {e}")
             # Log shapes for debugging
             for i, sig in enumerate(subj_signals_list):
                 log.error(f"  Signal {i} ({feature_order_list[i]}) shape: {sig.shape}")
             continue # Skip this subject

    # --- Log Final Summary ---
    num_subjects_processed = len(subject_feature_matrices)
    log.info(f"--- Feature Matrix Creation Finished. Created matrices for {num_subjects_processed} subjects. ---")
    if num_subjects_processed == 0 and processed_data: # Check if input was not empty
        log.critical("Feature matrix creation resulted in NO valid subject data, possibly due to missing signals or length mismatches.")

    return subject_feature_matrices
