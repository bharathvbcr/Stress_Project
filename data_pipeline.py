# data_pipeline.py (Orchestrates windowing, splitting, sampling, and DataLoader creation)

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, Tuple, List, Optional, Any, Union
from torch.utils.data import DataLoader

from windowing import create_all_subject_windows
from data_splitting import perform_group_split
from sampling import apply_sampling
from pytorch_datasets import create_pytorch_dataloaders
from utils import safe_get

log = logging.getLogger(__name__)

# ==============================================================================
# == Helper Function: Calculate Input Dimensions ==
# ==============================================================================
def _calculate_input_dims(
    processed_data: Dict[Union[int, str], Dict[str, Any]],
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]],
    config: Dict[str, Any]
) -> Tuple[Optional[int], Optional[int], Optional[List[str]], Optional[Dict[str, int]]]:
    """
    Calculates sequence and static input dimensions based on config and processed data.
    Also determines the order of sequence features and their channel counts.

    Args:
        processed_data: Dictionary with processed (resampled & aligned) signals and labels.
        static_features_results: Dictionary mapping subject ID to their static features DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple containing:
            - input_dim_sequence (Optional[int]): Total number of sequence feature channels.
            - input_dim_static (Optional[int]): Total number of static features used.
            - feature_order_list (Optional[List[str]]): Ordered list of unique sequence feature keys.
            - feature_channel_map (Optional[Dict[str, int]]): Map from unique feature key to its channel count.
            Returns (None, None, None, None) if calculation fails.
    """
    log.info("Calculating input dimensions for combined datasets...")
    input_dim_sequence: Optional[int] = None
    input_dim_static: Optional[int] = None
    feature_order_list: Optional[List[str]] = None
    feature_channel_map: Dict[str, int] = {} # Map feature key to its number of channels

    # --- Sequence Dimension (Based on features_to_use in config) ---
    chest_features_to_use = safe_get(config, ['features_to_use', 'chest'], [])
    wrist_features_to_use = safe_get(config, ['features_to_use', 'wrist'], [])
    temp_feature_order_list = [] # Build the order based on config
    expected_input_dim_sequence = 0

    # Find the first subject with actual signal data to infer dimensions
    first_subj_data = None
    first_subj_id = None
    if processed_data:
        for subj_id_candidate in processed_data:
            # Check if 'signal' key exists and is not empty
            if safe_get(processed_data[subj_id_candidate], ['signal']):
                first_subj_id = subj_id_candidate
                first_subj_data = processed_data[first_subj_id]
                break

    if not first_subj_data:
        log.error("Cannot calculate sequence dimension: processed_data is empty or no subjects have signal data.")
        return None, None, None, None

    log.debug(f"Using S{first_subj_id} to determine sequence dimensions and channel counts.")

    # Helper function to get channel count for a feature and update the map
    def get_channels(subj_data, device, key, unique_key):
        signal = safe_get(subj_data, ['signal', device, key])
        num_channels = 0
        if signal is not None and isinstance(signal, np.ndarray) and signal.size > 0:
            # Infer channels from shape (assume Time, Channels)
            num_channels = signal.shape[1] if signal.ndim > 1 else 1
            # Special handling for ACC: Warn if not 3 channels (unless it's magnitude)
            if key.upper() == 'ACC' and num_channels != 3 and num_channels != 1:
                 log.warning(f"S{first_subj_id}: ACC signal '{unique_key}' has {num_channels} channel(s). Expected 3 (XYZ) or 1 (magnitude). Assuming {num_channels}.")
            elif key.upper() == 'ACC' and num_channels == 3:
                 log.debug(f"S{first_subj_id}: Found 3 channels for ACC signal '{unique_key}'.")
            elif key.upper() == 'ACC' and num_channels == 1:
                 log.debug(f"S{first_subj_id}: Found 1 channel for ACC signal '{unique_key}' (likely magnitude).")

        elif signal is None:
            # If missing in the sample subject, try to infer standard channels
            if key.upper() == 'ACC': num_channels = 3 # Assume ACC is 3 channels if missing
            else: num_channels = 1 # Assume 1 channel for others if missing
            log.warning(f"S{first_subj_id} (used for dim calc) missing expected feature '{unique_key}'. Assuming {num_channels} channel(s) based on key type.")
        else:
             # Handle cases like empty arrays if not caught by signal.size > 0
             log.warning(f"Unexpected type {type(signal)} or empty array for feature '{unique_key}' in S{first_subj_id}. Assuming 1 channel.")
             num_channels = 1

        feature_channel_map[unique_key] = num_channels # Store channel count in the map
        return num_channels

    # Iterate through configured features to build order and count channels
    # Chest features
    for key in chest_features_to_use:
        unique_key = f"chest_{key}"
        # Ensure feature is added only once even if listed multiple times (unlikely)
        if unique_key not in temp_feature_order_list:
             num_ch = get_channels(first_subj_data, 'chest', key, unique_key)
             temp_feature_order_list.append(unique_key)
             expected_input_dim_sequence += num_ch

    # Wrist features
    for key in wrist_features_to_use:
        unique_key = f"wrist_{key}"
        if unique_key not in temp_feature_order_list:
             num_ch = get_channels(first_subj_data, 'wrist', key, unique_key)
             temp_feature_order_list.append(unique_key)
             expected_input_dim_sequence += num_ch

    # Validate sequence dimension
    if expected_input_dim_sequence <= 0:
        log.error("Failed sequence input dimension calculation (no valid features found/specified or zero channels).")
        return None, None, None, None

    input_dim_sequence = expected_input_dim_sequence
    feature_order_list = temp_feature_order_list # Assign the final calculated order
    log.info(f"Final Sequence Feature order (combined): {feature_order_list}")
    log.info(f"Feature Channel Map: {feature_channel_map}")
    log.info(f"Calculated Combined Sequence input dimension: {input_dim_sequence}")

    # --- Static Dimension ---
    # Calculate based on the length of 'static_features_to_use' in config (excluding comments)
    static_features_to_use_config = [
        f for f in safe_get(config, ['static_features_to_use'], [])
    ]
    input_dim_static = len(static_features_to_use_config)
    log.info(f"Combined Static Features specified ({input_dim_static}): {static_features_to_use_config}")
    log.info(f"Calculated Combined Static input dimension: {input_dim_static}")

    return input_dim_sequence, input_dim_static, feature_order_list, feature_channel_map


# ==============================================================================
# == Main Data Pipeline Function ==
# ==============================================================================

def get_data_splits(
    processed_data: Dict[Union[int, str], Dict[str, Any]],
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]],
    config: Dict[str, Any]
) -> Tuple[Tuple[List, ...], Tuple[List, ...], Tuple[List, ...], Optional[int], Optional[int]]:
    """
    Orchestrates the creation of train/val/test data splits from processed data.
    Performs dimension calculation, windowing, splitting, and sampling.

    Args:
        processed_data: Dictionary with processed (resampled & aligned) signals and labels.
        static_features_results: Dictionary mapping subject ID to their static features DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple containing:
            - train_data (Tuple[List, ...]): Sampled training data lists.
            - val_data (Tuple[List, ...]): Validation data lists.
            - test_data (Tuple[List, ...]): Test data lists.
            - input_dim_sequence (Optional[int]): Calculated sequence dimension.
            - input_dim_static (Optional[int]): Calculated static dimension.
            Returns (None, None, None, None, None) if any critical step fails.
    """
    log.info("--- Preparing Data Splits (Windowing -> Splitting -> Sampling) ---")
    if not processed_data:
        log.error("get_data_splits: processed_data dictionary is empty.")
        return None, None, None, None, None

    # --- Step 1: Calculate Input Dimensions ---
    input_dim_sequence, input_dim_static, feature_order_list, feature_channel_map = _calculate_input_dims(
        processed_data, static_features_results, config
    )
    if input_dim_sequence is None or input_dim_static is None or feature_order_list is None or feature_channel_map is None:
        log.error("get_data_splits: Failed to determine input dimensions or feature map.")
        return None, None, None, None, None

    # --- Step 2: Create Windows ---
    try:
        # Pass the calculated dimensions and feature map to the windowing function
        all_data_lists = create_all_subject_windows(
            processed_data, static_features_results, config,
            input_dim_sequence, input_dim_static, feature_order_list, feature_channel_map
        )
    except ValueError as e:
        log.error(f"get_data_splits: Error during window creation: {e}. Aborting.")
        return None, None, None, input_dim_sequence, input_dim_static
    except Exception as e:
        log.error(f"get_data_splits: Unexpected error during window creation: {e}", exc_info=True)
        return None, None, None, input_dim_sequence, input_dim_static

    # --- Step 3: Split Data ---
    try:
        train_data, val_data, test_data = perform_group_split(all_data_lists, config)
        log.info(f"Split results - Train: {len(train_data[2])} windows, Val: {len(val_data[2])} windows, Test: {len(test_data[2])} windows")
    except ValueError as e:
        log.error(f"get_data_splits: Error during data splitting: {e}. Aborting.")
        return None, None, None, input_dim_sequence, input_dim_static
    except Exception as e:
        log.error(f"get_data_splits: Unexpected error during data splitting: {e}", exc_info=True)
        return None, None, None, input_dim_sequence, input_dim_static

    # --- Step 4: Apply Sampling (to Training Data Only) ---
    try:
        train_data_sampled = apply_sampling(train_data, config)
        log.info(f"Training set size after sampling: {len(train_data_sampled[2])} windows (Original: {len(train_data[2])})")
    except Exception as e:
        log.error(f"get_data_splits: Unexpected error during sampling: {e}", exc_info=True)
        return None, None, None, input_dim_sequence, input_dim_static

    log.info("--- Data Splits Prepared Successfully ---")
    return train_data_sampled, val_data, test_data, input_dim_sequence, input_dim_static


def prepare_dataloaders(
    processed_data: Dict[Union[int, str], Dict[str, Any]], # Output from preprocessing.py
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]], # Output from preprocessing.py
    config: Dict[str, Any]
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Optional[int], Optional[int]]:
    """
    Main wrapper function to create PyTorch DataLoaders.
    Calls `get_data_splits` and then `create_pytorch_dataloaders`.

    Args:
        processed_data: Dictionary with processed (resampled & aligned) signals and labels.
        static_features_results: Dictionary mapping subject ID to their static features DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple containing:
            - train_loader (Optional[DataLoader])
            - val_loader (Optional[DataLoader])
            - test_loader (Optional[DataLoader])
            - input_dim_sequence (Optional[int]): Calculated sequence dimension.
            - input_dim_static (Optional[int]): Calculated static dimension.
            Returns (None, None, None, None, None) if any critical step fails.
    """
    # --- Step 1: Get Data Splits ---
    train_data_sampled, val_data, test_data, input_dim_sequence, input_dim_static = get_data_splits(
        processed_data, static_features_results, config
    )

    if train_data_sampled is None:
        log.error("prepare_dataloaders: Failed to get data splits.")
        return None, None, None, input_dim_sequence, input_dim_static

    # --- Step 2: Create PyTorch DataLoaders ---
    batch_size = safe_get(config, ['training_config', 'batch_size'], 64)
    if not isinstance(batch_size, int) or batch_size <= 0:
        log.warning(f"Invalid batch_size ({batch_size}) in config. Using default 64.")
        batch_size = 64

    try:
        # Pass sampled training data and original val/test data
        train_loader, val_loader, test_loader = create_pytorch_dataloaders(
            train_data_sampled, val_data, test_data, batch_size, config
        )
    except Exception as e:
        log.error(f"prepare_dataloaders: Unexpected error creating DataLoaders: {e}", exc_info=True)
        return None, None, None, input_dim_sequence, input_dim_static

    # Final check: Ensure train_loader was created if training data existed
    if not train_loader and train_data_sampled[0]: # Check if train_data was non-empty but loader creation failed
         log.critical("Training loader creation failed despite having training data.")
         return None, None, None, input_dim_sequence, input_dim_static
    elif not train_loader:
         log.critical("Training loader creation failed (no training data).")
         return None, None, None, input_dim_sequence, input_dim_static

    log.info("--- DataLoaders Prepared Successfully ---")
    return train_loader, val_loader, test_loader, input_dim_sequence, input_dim_static
