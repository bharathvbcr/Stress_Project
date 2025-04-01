# sampling.py (Handles oversampling techniques for training data)

import numpy as np
import logging
import collections
import random
from typing import Tuple, List, Dict, Any

# Assuming utils.py is available
try:
    from utils import safe_get
except ImportError:
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    logging.warning("Could not import 'safe_get' from 'utils'. Using basic fallback in sampling.py.")

# Attempt to import SMOTE from imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
    logging.error("`imbalanced-learn` library not found. pip install imbalanced-learn")
    logging.warning("SMOTE sampling will not be available. Falling back to random oversampling if SMOTE is selected.")

log = logging.getLogger(__name__)

def apply_sampling(
    train_data: Tuple[List, ...], # Output from data_splitting.py
    config: Dict[str, Any]
) -> Tuple[List, ...]:
    """
    Applies the selected sampling method (Random Oversampling or SMOTE)
    to the training data to handle class imbalance.

    Args:
        train_data (Tuple[List, ...]): A tuple containing the training data lists:
                                       (seq_features, static_features, labels,
                                        subject_ids, start_indices).
        config (Dict[str, Any]): Configuration dictionary containing the sampling strategy.

    Returns:
        Tuple[List, ...]: A tuple containing the potentially resampled training data lists.
                          Returns the original data if sampling is skipped or fails.
    """
    # Get sampling method from config, default to 'random'
    sampling_method = safe_get(config, ['training_config', 'sampling_strategy'], 'random').lower() # Use sampling_strategy key
    log.info(f"Applying '{sampling_method}' sampling to the Training Set...")

    # Unpack training data
    X_train_seq_list, X_train_static_list, y_train_list, subj_ids_train_list, starts_train_list = train_data

    # --- Pre-checks ---
    if not X_train_seq_list: # Check if training data is empty
        log.warning("Training set is empty before sampling. Skipping.")
        return train_data

    # Check class distribution
    y_train_np_pre = np.array(y_train_list)
    train_counts = collections.Counter(y_train_np_pre)
    log.info(f"Training set before sampling - Class 0: {train_counts.get(0, 0)}, Class 1: {train_counts.get(1, 0)}")

    if len(train_counts) < 2:
        log.info("Training set has only one class. Sampling skipped.")
        return train_data

    # Identify majority and minority classes
    # Assumes binary classification (0 and 1)
    majority_class_label = max(train_counts, key=train_counts.get)
    minority_class_label = min(train_counts, key=train_counts.get)
    n_majority = train_counts.get(majority_class_label, 0)
    n_minority = train_counts.get(minority_class_label, 0)

    if n_minority <= 0 or n_majority <= 0:
        log.error("Cannot perform sampling: count for minority or majority class is zero.")
        return train_data
    if n_minority == n_majority:
        log.info("Training set is already balanced. Sampling skipped.")
        return train_data

    # --- Apply Selected Sampling Method ---
    X_train_seq_resampled_list = list(X_train_seq_list) # Create copies to modify
    X_train_static_resampled_list = list(X_train_static_list)
    y_train_resampled_list = list(y_train_list)
    subj_ids_train_resampled_list = list(subj_ids_train_list)
    starts_train_resampled_list = list(starts_train_list)

    # --- Random Oversampling ---
    if sampling_method == 'random':
        log.info("Using Random Oversampling.")
        # Find indices of minority class samples
        minority_indices = np.where(y_train_np_pre == minority_class_label)[0]
        n_samples_to_add = n_majority - n_minority
        log.info(f"Random Oversampling: Adding {n_samples_to_add} samples from minority class ({minority_class_label}).")

        if len(minority_indices) > 0:
            # Randomly choose minority samples with replacement
            oversample_indices = np.random.choice(minority_indices, size=n_samples_to_add, replace=True)
            # Append the selected samples to the lists
            X_train_seq_resampled_list.extend([X_train_seq_list[i] for i in oversample_indices])
            X_train_static_resampled_list.extend([X_train_static_list[i] for i in oversample_indices])
            y_train_resampled_list.extend([y_train_list[i] for i in oversample_indices])
            # Assign a specific ID or keep original for oversampled data? Keep original for now.
            subj_ids_train_resampled_list.extend([subj_ids_train_list[i] for i in oversample_indices])
            starts_train_resampled_list.extend([starts_train_list[i] for i in oversample_indices])
        else:
            # This should not happen if n_minority > 0
            log.error("Minority indices array is empty, cannot perform random oversampling.")

    # --- SMOTE (Synthetic Minority Over-sampling Technique) ---
    elif sampling_method == 'smote':
        if SMOTE is None:
            log.error("SMOTE selected but imbalanced-learn library not found. Falling back to random oversampling.")
            # --- Fallback to Random Oversampling Logic ---
            minority_indices = np.where(y_train_np_pre == minority_class_label)[0]
            n_samples_to_add = n_majority - n_minority
            log.info(f"Random Oversampling (Fallback): Adding {n_samples_to_add} samples from minority class ({minority_class_label}).")
            if len(minority_indices) > 0:
                oversample_indices = np.random.choice(minority_indices, size=n_samples_to_add, replace=True)
                X_train_seq_resampled_list.extend([X_train_seq_list[i] for i in oversample_indices])
                X_train_static_resampled_list.extend([X_train_static_list[i] for i in oversample_indices])
                y_train_resampled_list.extend([y_train_list[i] for i in oversample_indices])
                subj_ids_train_resampled_list.extend([subj_ids_train_list[i] for i in oversample_indices])
                starts_train_resampled_list.extend([starts_train_list[i] for i in oversample_indices])
            else: log.error("Minority indices array is empty, cannot perform random oversampling fallback.")
            # --- End Fallback ---
        else:
            log.info("Using SMOTE.")
            try:
                # 1. Prepare data for SMOTE: Stack sequence features and flatten them.
                #    Combine with static features if they exist.
                X_seq_np = np.stack(X_train_seq_list, axis=0)
                n_samples, window_len, n_seq_features = X_seq_np.shape
                X_seq_flat = X_seq_np.reshape(n_samples, -1) # Flatten sequence windows: (N, L*F)

                # Check if static features exist and are non-empty
                has_static = bool(X_train_static_list) and X_train_static_list[0].size > 0
                if has_static:
                    X_static_np = np.stack(X_train_static_list, axis=0)
                    # Ensure static features are 2D (N, S) even if S=1
                    if X_static_np.ndim == 1: X_static_np = X_static_np[:, np.newaxis]
                    # Combine flattened sequence and static features
                    X_combined_flat = np.concatenate((X_seq_flat, X_static_np), axis=1)
                    log.info(f"SMOTE input shape (Seq+Static): {X_combined_flat.shape}")
                else:
                    X_combined_flat = X_seq_flat # Use only sequence features
                    log.info(f"SMOTE input shape (Sequence only): {X_combined_flat.shape}")

                # 2. Apply SMOTE
                # Adjust k_neighbors based on the number of minority samples
                # k_neighbors must be less than the number of minority samples
                k_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1
                if k_neighbors < 1:
                     log.warning(f"SMOTE k_neighbors calculated as {k_neighbors}. Setting to 1.")
                     k_neighbors = 1

                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                # Fit and resample the combined flattened data
                X_resampled_flat, y_train_resampled_list = smote.fit_resample(X_combined_flat, y_train_list)
                n_resampled = len(y_train_resampled_list)
                log.info(f"SMOTE generated {n_resampled - n_samples} new synthetic samples.")

                # 3. Unpack resampled data
                # Separate sequence and static features from the resampled array
                X_seq_resampled_flat = X_resampled_flat[:, :window_len * n_seq_features]
                X_static_resampled_np = X_resampled_flat[:, window_len * n_seq_features:] if has_static else None

                # Reshape sequence features back: (N_resampled, L*F) -> (N_resampled, L, F)
                X_train_seq_resampled_np = X_seq_resampled_flat.reshape(n_resampled, window_len, n_seq_features)
                # Convert back to list of arrays
                X_train_seq_resampled_list = [X_train_seq_resampled_np[i] for i in range(n_resampled)]

                # Convert static features back to list of arrays
                if X_static_resampled_np is not None:
                    X_train_static_resampled_list = [X_static_resampled_np[i] for i in range(n_resampled)]
                else: # Handle case where static features were not used
                    X_train_static_resampled_list = [np.array([], dtype=np.float32)] * n_resampled

                # Handle metadata (subject IDs, start indices) for synthetic samples
                original_n = len(subj_ids_train_list)
                # Mark synthetic samples with a special ID
                subj_ids_train_resampled_list = list(subj_ids_train_list) + ['SMOTE_SYNTHETIC'] * (n_resampled - original_n)
                # Assign a placeholder start index (e.g., -1) for synthetic samples
                starts_train_resampled_list = list(starts_train_list) + [-1] * (n_resampled - original_n)

            except ValueError as smote_e:
                 log.error(f"SMOTE failed: {smote_e}. This often happens if minority samples < k_neighbors+1.")
                 log.error("Falling back to random oversampling.")
                 # --- Fallback to Random Oversampling Logic ---
                 minority_indices = np.where(y_train_np_pre == minority_class_label)[0]
                 n_samples_to_add = n_majority - n_minority
                 log.info(f"Random Oversampling (Fallback): Adding {n_samples_to_add} samples from minority class ({minority_class_label}).")
                 if len(minority_indices) > 0:
                     oversample_indices = np.random.choice(minority_indices, size=n_samples_to_add, replace=True)
                     X_train_seq_resampled_list.extend([X_train_seq_list[i] for i in oversample_indices])
                     X_train_static_resampled_list.extend([X_train_static_list[i] for i in oversample_indices])
                     y_train_resampled_list.extend([y_train_list[i] for i in oversample_indices])
                     subj_ids_train_resampled_list.extend([subj_ids_train_list[i] for i in oversample_indices])
                     starts_train_resampled_list.extend([starts_train_list[i] for i in oversample_indices])
                 else: log.error("Minority indices array is empty, cannot perform random oversampling fallback.")
                 # --- End Fallback ---

            except Exception as e:
                log.error(f"Unexpected error during SMOTE: {e}", exc_info=True)
                log.error("Falling back to random oversampling.")
                # --- Fallback to Random Oversampling Logic ---
                # (Duplicate code - consider refactoring into a separate helper if used frequently)
                minority_indices = np.where(y_train_np_pre == minority_class_label)[0]
                n_samples_to_add = n_majority - n_minority
                log.info(f"Random Oversampling (Fallback): Adding {n_samples_to_add} samples from minority class ({minority_class_label}).")
                if len(minority_indices) > 0:
                    oversample_indices = np.random.choice(minority_indices, size=n_samples_to_add, replace=True)
                    X_train_seq_resampled_list.extend([X_train_seq_list[i] for i in oversample_indices])
                    X_train_static_resampled_list.extend([X_train_static_list[i] for i in oversample_indices])
                    y_train_resampled_list.extend([y_train_list[i] for i in oversample_indices])
                    subj_ids_train_resampled_list.extend([subj_ids_train_list[i] for i in oversample_indices])
                    starts_train_resampled_list.extend([starts_train_list[i] for i in oversample_indices])
                else: log.error("Minority indices array is empty, cannot perform random oversampling fallback.")
                # --- End Fallback ---

    # --- Unknown Sampling Method ---
    else:
        log.error(f"Unknown sampling method '{sampling_method}' specified in config. Using random oversampling as fallback.")
        # --- Fallback to Random Oversampling Logic ---
        # (Duplicate code)
        minority_indices = np.where(y_train_np_pre == minority_class_label)[0]
        n_samples_to_add = n_majority - n_minority
        log.info(f"Random Oversampling (Fallback): Adding {n_samples_to_add} samples from minority class ({minority_class_label}).")
        if len(minority_indices) > 0:
            oversample_indices = np.random.choice(minority_indices, size=n_samples_to_add, replace=True)
            X_train_seq_resampled_list.extend([X_train_seq_list[i] for i in oversample_indices])
            X_train_static_resampled_list.extend([X_train_static_list[i] for i in oversample_indices])
            y_train_resampled_list.extend([y_train_list[i] for i in oversample_indices])
            subj_ids_train_resampled_list.extend([subj_ids_train_list[i] for i in oversample_indices])
            starts_train_resampled_list.extend([starts_train_list[i] for i in oversample_indices])
        else: log.error("Minority indices array is empty, cannot perform random oversampling fallback.")
        # --- End Fallback ---

    # --- Log Final Distribution ---
    log.info(f"Training set AFTER sampling ({sampling_method}): {len(y_train_resampled_list)} windows")
    final_train_counts = collections.Counter(y_train_resampled_list)
    log.info(f"  New distribution - Class 0: {final_train_counts.get(0, 0)}, Class 1: {final_train_counts.get(1, 0)}")

    # Return the resampled data lists
    return (
        X_train_seq_resampled_list,
        X_train_static_resampled_list,
        y_train_resampled_list,
        subj_ids_train_resampled_list,
        starts_train_resampled_list
    )
