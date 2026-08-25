# data_pipeline.py (Orchestrates windowing, splitting, sampling, and DataLoader creation)

import os
import json
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, Tuple, List, Optional, Any, Union
from joblib import load
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
    processed_data_paths: Dict[Union[int, str], str], # CHANGED: Dict of paths
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]],
    config: Dict[str, Any]
) -> Tuple[Optional[int], Optional[int], Optional[List[str]], Optional[Dict[str, int]]]:
    """
    Calculates sequence and static input dimensions based on config and processed data.
    Also determines the order of sequence features and their channel counts.
    Loads one subject lazily to infer dimensions.

    Args:
        processed_data_paths: Dictionary mapping subject ID to processed data file path.
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
    chest_features_to_use = safe_get(config, ['processing', 'features_to_use', 'chest'], safe_get(config, ['features_to_use', 'chest'], []))
    wrist_features_to_use = safe_get(config, ['processing', 'features_to_use', 'wrist'], safe_get(config, ['features_to_use', 'wrist'], []))
    temp_feature_order_list = [] # Build the order based on config
    expected_input_dim_sequence = 0

    # Find the first subject with valid path
    first_subj_data = None
    first_subj_id = None
    if processed_data_paths:
        for subj_id_candidate, path in processed_data_paths.items():
            try:
                first_subj_id = subj_id_candidate
                first_subj_data = load(path) # LAZY LOAD
                if safe_get(first_subj_data, ['signal']):
                    break
            except Exception as e:
                log.warning(f"Failed to load sample subject {subj_id_candidate} for dim calc: {e}")
                first_subj_data = None
                continue

    if not first_subj_data:
        log.error("Cannot calculate sequence dimension: processed_data_paths is empty or valid subject not found.")
        return None, None, None, None

    log.debug(f"Using S{first_subj_id} to determine sequence dimensions and channel counts.")

    # Helper function to get channel count for a feature and update the map
    def get_channels(subj_data, device, key, unique_key):
        signal = safe_get(subj_data, ['signal', device, key])
        num_channels = 0
        if signal is not None and isinstance(signal, np.ndarray) and signal.size > 0:
            # Infer channels from shape (assume Time, Channels)
            num_channels = signal.shape[1] if signal.ndim > 1 else 1
            if key.upper() == 'ACC' and num_channels != 3 and num_channels != 1:
                 log.warning(f"S{first_subj_id}: ACC signal '{unique_key}' has {num_channels} channel(s). Expected 3 or 1.")
            elif key.upper() == 'ACC' and num_channels == 1:
                 log.debug(f"S{first_subj_id}: Found 1 channel for ACC signal '{unique_key}' (likely magnitude).")

        elif signal is None:
            # If missing in the sample subject, try to infer standard channels
            if key.upper() == 'ACC': num_channels = 3 # Assume ACC is 3 channels if missing
            else: num_channels = 1 # Assume 1 channel for others if missing
            log.warning(f"S{first_subj_id} (used for dim calc) missing expected feature '{unique_key}'. Assuming {num_channels} channel(s).")
        else:
             log.warning(f"Unexpected type {type(signal)} or empty array for feature '{unique_key}' in S{first_subj_id}. Assuming 1 channel.")
             num_channels = 1

        feature_channel_map[unique_key] = num_channels # Store channel count in the map
        return num_channels

    # Iterate through configured features to build order and count channels
    # Chest features
    for key in chest_features_to_use:
        unique_key = f"chest_{key}"
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
        f for f in safe_get(config, ['processing', 'static_features_to_use'], safe_get(config, ['static_features_to_use'], []))
    ]
    input_dim_static = len(static_features_to_use_config)
    log.info(f"Combined Static Features specified ({input_dim_static}): {static_features_to_use_config}")
    log.info(f"Calculated Combined Static input dimension: {input_dim_static}")

    return input_dim_sequence, input_dim_static, feature_order_list, feature_channel_map


# ==============================================================================
# == Main Data Pipeline Function ==
# ==============================================================================

def _normalize_sequence_enabled(config: Dict[str, Any]) -> bool:
    """Resolve the processing.normalize_sequence_features flag (default True)."""
    flag = safe_get(config, ['processing', 'normalize_sequence_features'], True)
    if isinstance(flag, str):
        return flag.strip().lower() not in {'0', 'false', 'no', 'off', 'none', ''}
    return bool(flag)


def _compute_channel_stats(seq_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel mean/std over all timesteps of the given windows, computed in a
    streaming fashion to avoid stacking the whole training set in memory.

    Args:
        seq_list: list of (window_length, num_features) arrays.

    Returns:
        (mean, std) each shaped (num_features,), float32. Constant channels get
        std=1.0 to avoid divide-by-zero.
    """
    n_features = int(np.asarray(seq_list[0]).shape[1])
    ch_sum = np.zeros(n_features, dtype=np.float64)
    ch_sumsq = np.zeros(n_features, dtype=np.float64)
    count = 0
    for window in seq_list:
        w = np.asarray(window, dtype=np.float64)  # (L, F)
        ch_sum += w.sum(axis=0)
        ch_sumsq += (w ** 2).sum(axis=0)
        count += w.shape[0]
    if count == 0:
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
    mean = ch_sum / count
    var = np.maximum(ch_sumsq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0  # constant channel → leave values centered, scale by 1
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_channel_norm(
    seq_list: List[np.ndarray], mean: np.ndarray, std: np.ndarray
) -> List[np.ndarray]:
    """Standardize each window per channel: (x - mean) / std."""
    return [((np.asarray(w, dtype=np.float32) - mean) / std).astype(np.float32) for w in seq_list]


def _persist_norm_stats(
    mean: np.ndarray, std: np.ndarray, feature_order: Optional[List[str]], config: Dict[str, Any]
) -> None:
    """Persist normalization stats so inference can reproduce the same scaling."""
    out_dir = safe_get(config, ['save_paths', 'processed_data'], './outputs/processed_data')
    try:
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'sequence_norm_stats.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(
                {'mean': mean.tolist(), 'std': std.tolist(), 'feature_order': feature_order},
                fh,
                indent=2,
            )
        log.info("Saved sequence normalization stats to %s", path)
    except Exception as exc:
        log.warning("Could not persist normalization stats (non-fatal): %s", exc)


def _normalize_splits(
    train_data: Tuple[List, ...],
    val_data: Tuple[List, ...],
    test_data: Tuple[List, ...],
    feature_order: Optional[List[str]],
    config: Dict[str, Any],
) -> Tuple[Tuple[List, ...], Tuple[List, ...], Tuple[List, ...]]:
    """
    Apply per-channel standardization using stats fit ONLY on the training split,
    then applied to train/val/test. No-op (returns inputs unchanged) if disabled,
    if training sequences are empty, or on any error (logged, non-fatal).
    """
    if not _normalize_sequence_enabled(config):
        return train_data, val_data, test_data
    train_seqs = train_data[0]
    if not train_seqs:
        log.warning("Sequence normalization skipped: training sequences are empty.")
        return train_data, val_data, test_data
    try:
        mean, std = _compute_channel_stats(train_seqs)
        log.info(
            "Sequence normalization (train-only stats) applied across %d channels.",
            len(mean),
        )
        train_data = (_apply_channel_norm(train_data[0], mean, std), *train_data[1:])
        if val_data[0]:
            val_data = (_apply_channel_norm(val_data[0], mean, std), *val_data[1:])
        if test_data[0]:
            test_data = (_apply_channel_norm(test_data[0], mean, std), *test_data[1:])
        _persist_norm_stats(mean, std, feature_order, config)
    except Exception as exc:
        log.error(
            "Sequence normalization failed (non-fatal; continuing unnormalized): %s",
            exc,
            exc_info=True,
        )
    return train_data, val_data, test_data


def get_data_splits(
    processed_data_paths: Dict[Union[int, str], str], # CHANGED: Dict of paths
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]],
    config: Dict[str, Any]
) -> Tuple[Tuple[List, ...], Tuple[List, ...], Tuple[List, ...], Optional[int], Optional[int]]:
    """
    Orchestrates the creation of train/val/test data splits from processed data.
    Performs dimension calculation, windowing, splitting, and sampling.

    Args:
        processed_data_paths: Dictionary mapping subject ID to processed data file path.
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
    if not processed_data_paths:
        log.error("get_data_splits: processed_data_paths dictionary is empty.")
        return None, None, None, None, None

    # --- Step 1: Calculate Input Dimensions ---
    input_dim_sequence, input_dim_static, feature_order_list, feature_channel_map = _calculate_input_dims(
        processed_data_paths, static_features_results, config
    )
    if input_dim_sequence is None or input_dim_static is None or feature_order_list is None or feature_channel_map is None:
        log.error("get_data_splits: Failed to determine input dimensions or feature map.")
        return None, None, None, None, None

    # --- Step 2: Create Windows ---
    try:
        # Pass the calculated dimensions and feature map to the windowing function
        all_data_lists = create_all_subject_windows(
            processed_data_paths, static_features_results, config,
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

    # --- Step 3.5: Per-channel normalization (stats fit on TRAIN split only) ---
    # Done before sampling so synthetic/duplicated samples don't skew the stats,
    # and applied to all three splits with the same train-derived mean/std.
    train_data, val_data, test_data = _normalize_splits(
        train_data, val_data, test_data, feature_order_list, config
    )

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
    processed_data_paths: Dict[Union[int, str], str], # CHANGED: Dict of paths
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]], # Output from preprocessing.py
    config: Dict[str, Any]
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Optional[int], Optional[int]]:
    """
    Main wrapper function to create PyTorch DataLoaders.
    Calls `get_data_splits` and then `create_pytorch_dataloaders`.

    Args:
        processed_data_paths: Dictionary mapping subject ID to processed data file path.
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
        processed_data_paths, static_features_results, config
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