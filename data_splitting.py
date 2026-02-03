# data_splitting.py (Handles train/validation/test splitting)

import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Union
from sklearn.model_selection import GroupShuffleSplit

# Assuming utils.py is available
try:
    from utils import safe_get
except ImportError:
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    logging.warning("Could not import 'safe_get' from 'utils'. Using basic fallback in data_splitting.py.")

log = logging.getLogger(__name__)

def perform_group_split(
    all_data_lists: Tuple[List, ...], # Output from create_all_subject_windows
    config: Dict[str, Any],
    random_seed: int = 42
) -> Tuple[Tuple[List,...], Tuple[List,...], Tuple[List,...]]:
    """
    Performs group-stratified train/validation/test split on the windowed data.
    Ensures that all windows from a single subject remain in the same split.

    Args:
        all_data_lists (Tuple[List, ...]): A tuple containing the lists generated
                                           by create_all_subject_windows:
                                           (seq_features, static_features, labels,
                                            groups, subject_ids, start_indices).
        config (Dict[str, Any]): Configuration dictionary containing split ratios.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Tuple[List,...], Tuple[List,...], Tuple[List,...]]:
            A tuple containing three tuples, one for each split (train, val, test).
            Each inner tuple contains the corresponding subset of lists:
            (seq_features, static_features, labels, subject_ids, start_indices).
            Note: Group list is not returned as it's only needed for splitting.

    Raises:
        ValueError: If split ratios are invalid or splitting fails critically.
        TypeError: If group IDs are not hashable.
    """
    log.info("Performing Group-Stratified Split (Train/Val/Test)...")

    # Unpack the input lists
    (all_windows_seq_features_list, all_windows_static_features_list, all_windows_labels_list,
     all_windows_groups_list, all_windows_subject_ids_list, all_windows_start_indices_list) = all_data_lists

    # --- Prepare Data for Splitting ---
    # Ensure groups are hashable (convert potential numpy types or numbers to strings)
    # GroupShuffleSplit requires hashable group identifiers.
    try:
        groups_np = np.array([str(g) for g in all_windows_groups_list])
    except Exception as e:
        log.error(f"Could not convert group IDs to hashable type (string): {e}")
        raise TypeError("Group IDs must be convertible to a hashable type for GroupShuffleSplit.") from e

    # Convert labels to numpy array for splitting
    y_np = np.array(all_windows_labels_list, dtype=np.int32)
    n_total_windows = len(all_windows_seq_features_list)
    all_indices = np.arange(n_total_windows) # Indices [0, 1, ..., N-1]

    # --- Get Split Ratios from Config ---
    data_splits_config = safe_get(config, ['data_splits'], {})
    test_ratio = safe_get(data_splits_config, ['test_ratio'], 0.25) # Default 25%
    val_ratio = safe_get(data_splits_config, ['validation_ratio'], 0.25) # Default 25%

    # Validate ratios
    if not (0.0 < test_ratio < 1.0):
        test_ratio = 0.25; log.warning(f"Invalid test_ratio ({test_ratio}), using default {test_ratio}")
    if not (0.0 < val_ratio < 1.0):
        val_ratio = 0.25; log.warning(f"Invalid validation_ratio ({val_ratio}), using default {val_ratio}")
    if test_ratio + val_ratio >= 1.0:
        log.error(f"Sum of test_ratio ({test_ratio}) and validation_ratio ({val_ratio}) must be less than 1.0.")
        raise ValueError("Invalid split ratios: test + validation >= 1.0")

    # Log initial distribution
    unique_labels_overall, counts_overall = np.unique(y_np, return_counts=True)
    log.info(f"Overall window label distribution before split: {dict(zip(unique_labels_overall, counts_overall))}")
    if len(unique_labels_overall) < 2:
        log.warning("Only one class found in overall data. Stratification by label might not be possible or effective.")

    # --- First Split: Separate Test Set ---
    # Create a placeholder for features (GroupShuffleSplit doesn't need actual features)
    X_placeholder = np.zeros((n_total_windows, 1))
    # Initialize GroupShuffleSplit for the test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)

    try:
        # Perform the split based on groups
        # next() gets the first (and only) split from the generator
        train_val_idx_overall, test_idx_overall = next(gss_test.split(X_placeholder, y_np, groups=groups_np))
        log.info(f"Test split created: {len(test_idx_overall)} windows.")
    except ValueError as e:
        # This can happen if test_size is too large/small for the number of groups
        log.error(f"GroupShuffleSplit failed for test set: {e}. Check ratios and group distribution. Aborting.")
        raise e

    # --- Second Split: Separate Validation Set from Train+Val ---
    # Get the indices, labels, and groups remaining after the test split
    train_val_indices = all_indices[train_val_idx_overall]
    y_train_val_np = y_np[train_val_idx_overall]
    groups_train_val_np = groups_np[train_val_idx_overall]
    current_train_val_size = len(train_val_indices)

    train_final_indices = np.array([], dtype=int) # Initialize empty arrays
    val_final_indices = np.array([], dtype=int)

    if current_train_val_size > 0:
        # Calculate the validation ratio relative to the size of the train+val set
        # val_ratio_relative = val_ratio / (1.0 - test_ratio) # Equivalent calculation
        val_ratio_relative = (n_total_windows * val_ratio) / current_train_val_size

        # Ensure the relative ratio is valid (between 0 and 1)
        if val_ratio_relative <= 0 or val_ratio_relative >= 1.0:
             log.warning(f"Relative validation ratio invalid ({val_ratio_relative:.3f}) after test split. Assigning all remaining {current_train_val_size} samples to train set.")
             train_final_indices = train_val_indices # Assign all remaining to train
             val_final_indices = np.array([], dtype=int) # Validation set will be empty
        else:
            # Initialize GroupShuffleSplit for the validation set
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_relative, random_state=random_seed)
            X_tv_placeholder = np.zeros((current_train_val_size, 1)) # Placeholder

            try:
                # Perform the split on the train+val data
                train_idx_rel, val_idx_rel = next(gss_val.split(X_tv_placeholder, y_train_val_np, groups=groups_train_val_np))
                # Convert relative indices back to original indices
                train_final_indices = train_val_indices[train_idx_rel]
                val_final_indices = train_val_indices[val_idx_rel]
                log.info(f"Validation split created: {len(val_final_indices)} windows.")
            except ValueError as e:
                 # Handle potential errors during the validation split
                 log.error(f"GroupShuffleSplit failed for validation set: {e}. Assigning all remaining {current_train_val_size} samples to train set.")
                 train_final_indices = train_val_indices # Assign all to train if split fails
                 val_final_indices = np.array([], dtype=int) # Val set is empty
    else:
        # This case should be rare if test_ratio < 1.0
        log.warning("Train+Validation set is empty after test split. Train and Val sets will be empty.")

    # --- Extract Data Subsets Based on Indices ---
    def _extract_subset(indices: np.ndarray) -> Tuple[List,...]:
        """Helper function to extract data for a given set of indices."""
        if len(indices) == 0:
            # Return empty lists for all components if indices are empty
            return ([], [], [], [], [])
        # Use list comprehensions to efficiently select items based on indices
        seq = [all_windows_seq_features_list[i] for i in indices]
        static = [all_windows_static_features_list[i] for i in indices]
        labels = [all_windows_labels_list[i] for i in indices]
        sids = [all_windows_subject_ids_list[i] for i in indices] # Use original subject IDs
        starts = [all_windows_start_indices_list[i] for i in indices]
        return seq, static, labels, sids, starts

    # Extract data for train, validation, and test sets
    train_data = _extract_subset(train_final_indices)
    val_data = _extract_subset(val_final_indices)
    test_data = _extract_subset(test_idx_overall)

    # --- Log Final Split Information ---
    log.info("--- Split Sizes (Before Oversampling) ---")
    log.info(f"Train: {len(train_data[0])} windows")
    log.info(f"Validation: {len(val_data[0])} windows")
    log.info(f"Test: {len(test_data[0])} windows")

    # Verify subject distribution across splits
    train_subjects_final = sorted(list(set(train_data[3]))) if train_data[3] else []
    val_subjects_final = sorted(list(set(val_data[3]))) if val_data[3] else []
    test_subjects_final = sorted(list(set(test_data[3]))) if test_data[3] else []
    log.info(f"Final Subject Distribution:")
    log.info(f"  Train Subjects ({len(train_subjects_final)}): {train_subjects_final}")
    log.info(f"  Validation Subjects ({len(val_subjects_final)}): {val_subjects_final}")
    log.info(f"  Test Subjects ({len(test_subjects_final)}): {test_subjects_final}")

    # Check for subject overlap between sets (should not happen with GroupShuffleSplit)
    overlap_tr_val = set(train_subjects_final) & set(val_subjects_final)
    overlap_tr_test = set(train_subjects_final) & set(test_subjects_final)
    overlap_val_test = set(val_subjects_final) & set(test_subjects_final)
    if overlap_tr_val: log.error(f"CRITICAL: Subject overlap detected between Train and Validation sets: {overlap_tr_val}")
    if overlap_tr_test: log.error(f"CRITICAL: Subject overlap detected between Train and Test sets: {overlap_tr_test}")
    if overlap_val_test: log.error(f"CRITICAL: Subject overlap detected between Validation and Test sets: {overlap_val_test}")
    if not overlap_tr_val and not overlap_tr_test and not overlap_val_test:
        log.info("Group split successful: No subject overlap detected between Train/Val/Test sets.")

    return train_data, val_data, test_data

