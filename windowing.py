# windowing.py (Handles creating sequence windows)

import numpy as np
import pandas as pd
import logging
import collections
from typing import Dict, Tuple, List, Optional, Any, Union

# Assuming utils.py is available
try:
    from utils import safe_get
except ImportError:
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    logging.warning("Could not import 'safe_get' from 'utils'. Using basic fallback in windowing.py.")

log = logging.getLogger(__name__)

# Default value for padding missing static features
DEFAULT_STATIC_VALUE = 0.0

def create_all_subject_windows(
    processed_data: Dict[Union[int, str], Dict[str, Any]],
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]],
    config: Dict[str, Any],
    input_dim_sequence: int, # Expected number of sequence features per time step
    input_dim_static: int,   # Expected number of static features
    feature_order_list: List[str], # The definitive order of sequence features
    feature_channel_map: Dict[str, int] # Map of feature key to expected channels
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[Union[int, str]], List[Union[int, str]], List[int]]:
    """
    Creates sequence/static feature windows and extracts metadata for ALL subjects.
    Ensures consistent feature dimensions using padding/defaults for missing features.
    Uses feature_channel_map for accurate padding of sequence features.

    Args:
        processed_data: Dictionary with processed (resampled & aligned) signals and labels.
        static_features_results: Dictionary mapping subject ID to their static features DataFrame.
        config: Configuration dictionary.
        input_dim_sequence: The total number of expected sequence features (channels).
        input_dim_static: The total number of expected static features.
        feature_order_list: Ordered list of unique feature keys (e.g., 'chest_ECG', 'wrist_EDA').
        feature_channel_map: Dictionary mapping unique feature keys to their channel count.

    Returns:
        Tuple containing lists for all windows across all subjects:
        - Sequence feature windows (List[np.ndarray])
        - Static feature vectors (List[np.ndarray])
        - Labels (List[int])
        - Group IDs (subject IDs for splitting) (List[Union[int, str]])
        - Subject IDs (original IDs for tracking) (List[Union[int, str]])
        - Window start indices (List[int])

    Raises:
        ValueError: If window creation fails critically (e.g., no windows created, config issues).
    """
    log.info("--- Creating Windows for All Subjects ---")

    # --- Config Parameters ---
    window_size_sec = safe_get(config, ['windowing', 'window_size_sec'], 60)
    overlap_ratio = safe_get(config, ['windowing', 'window_overlap'], 0.5)
    target_fs = safe_get(config, ['processing', 'target_sampling_rate'])
    target_label_name = safe_get(config, ['target_label'], 'stress')
    label_map = safe_get(config, ['label_mapping'], {})
    # Get the list of static features to actually use from the config
    static_features_to_use_config = [
        f for f in safe_get(config, ['static_features_to_use'], []) if not f.startswith("comment")
    ]
    # Ensure the number of static features to use matches the expected input_dim_static
    if len(static_features_to_use_config) != input_dim_static:
        log.error(f"Mismatch between static features in config ({len(static_features_to_use_config)}) and expected static dim ({input_dim_static}).")
        raise ValueError("Static feature configuration mismatch.")

    # --- Label Mapping ---
    try:
        # Find the integer ID corresponding to the target label name (e.g., 'stress' -> 2)
        stress_label_id = next((int(k) for k, v in label_map.items() if v == target_label_name and k.isdigit()), None)
        if stress_label_id is None: raise ValueError(f"Target label '{target_label_name}' not found in label_mapping.")
    except Exception as e:
        log.error(f"Config Error (label mapping): {e}")
        raise ValueError("Invalid label mapping configuration.")
    log.info(f"Binary Class Target: Map '{target_label_name}' (ID: {stress_label_id}) to 1, others to 0.")

    # --- Windowing Parameters ---
    if not target_fs or target_fs <= 0:
        log.error(f"Invalid target_sampling_rate ({target_fs}) in config. Cannot create windows.")
        raise ValueError("Invalid target sampling rate.")
    window_samples = int(window_size_sec * target_fs) # Window length in samples
    overlap_samples = int(window_samples * overlap_ratio) # Overlap in samples
    step = window_samples - overlap_samples # Step size in samples
    if step <= 0:
        log.error(f"Window step size non-positive ({step}). Check window_size_sec and window_overlap.")
        raise ValueError("Invalid window step size.")
    log.info(f"Window: {window_size_sec}s ({window_samples} samples), Overlap: {overlap_ratio*100:.1f}%, Step: {step} samples")

    # --- Initialize Lists to store results for all subjects ---
    all_windows_seq_features_list = []
    all_windows_static_features_list = []
    all_windows_labels_list = []
    all_windows_groups_list = [] # Subject ID used for group splitting
    all_windows_subject_ids_list = [] # Original subject ID for tracking
    all_windows_start_indices_list = [] # Absolute start sample index of the window

    subjects_with_windows = 0
    total_windows_processed = 0
    processed_subject_ids = sorted(list(processed_data.keys())) # Process subjects in a consistent order
    log.info(f"Windowing for {len(processed_subject_ids)} subjects...")

    # --- Default Vectors/Placeholders ---
    # Placeholder for a single missing sequence channel (will be tiled) - Shape (window_samples, 1)
    default_seq_channel_template = np.full((window_samples, 1), 0.0, dtype=np.float32)
    # Default for a fully missing static feature vector
    default_static_vector = np.full(input_dim_static, DEFAULT_STATIC_VALUE, dtype=np.float32) if input_dim_static > 0 else np.array([], dtype=np.float32)

    # --- Iterate Through Subjects ---
    for subj_id in processed_subject_ids:
        subj_data = processed_data[subj_id]
        subj_static_df = static_features_results.get(subj_id) if static_features_results else None
        log.info(f"--- Windowing S{subj_id} ---")

        # --- Prepare Subject's Static Feature Vector (Consistent Dimension) ---
        subj_static_vector = default_static_vector.copy() # Start with default
        if input_dim_static > 0:
             if isinstance(subj_static_df, pd.DataFrame) and not subj_static_df.empty:
                 try:
                     # Extract the first row (assuming one row per subject)
                     subj_static_series = subj_static_df.iloc[0]
                     # Reindex to match the order specified in config, filling missing with default
                     selected_subj_static = subj_static_series.reindex(static_features_to_use_config)
                     selected_subj_static = selected_subj_static.fillna(DEFAULT_STATIC_VALUE)

                     # Check if the length matches the expected dimension
                     if len(selected_subj_static) == input_dim_static:
                          subj_static_vector = selected_subj_static.values.astype(np.float32)
                          log.debug(f"S{subj_id}: Successfully prepared static vector (shape: {subj_static_vector.shape}).")
                     else:
                          log.error(f"S{subj_id}: Static vector length mismatch ({len(selected_subj_static)}) vs expected ({input_dim_static}). Using default.")
                 except IndexError:
                     log.error(f"S{subj_id}: IndexError accessing static features DataFrame row 0. Using default.")
                 except Exception as e:
                     log.error(f"S{subj_id}: Error preparing static features: {e}. Using default.")
             else:
                 log.warning(f"S{subj_id}: Static feature DataFrame missing/empty for subject. Using default static vector.")
        # else: log.debug(f"S{subj_id}: No static features configured (input_dim_static=0).")

        # --- Prepare Subject's Sequence Signals and Labels ---
        labels_orig = safe_get(subj_data, ['label'])
        if labels_orig is None or not isinstance(labels_orig, np.ndarray) or labels_orig.size == 0:
            log.error(f"[S{subj_id}] Labels missing or empty in processed_data. Skipping subject.")
            continue
        labels_orig = labels_orig.flatten()
        # Convert original labels to binary (1 if target_label, 0 otherwise)
        binary_labels = (labels_orig == stress_label_id).astype(np.int32)

        # Gather all required sequence signals for this subject
        subject_signals = {} # Dict: {unique_key: signal_array or None}
        min_signal_len = float('inf') # Track minimum length among available signals
        found_any_signal = False # Track if at least one sequence signal is found

        for unique_key in feature_order_list:
            device, key = unique_key.split('_', 1) # e.g., 'chest_ECG' -> 'chest', 'ECG'
            signal = safe_get(subj_data, ['signal', device, key])

            if signal is not None and isinstance(signal, np.ndarray) and signal.size > 0:
                signal = signal.reshape(signal.shape[0], -1) # Ensure 2D (Time, Channels)
                subject_signals[unique_key] = signal
                min_signal_len = min(min_signal_len, signal.shape[0]) # Update minimum length
                found_any_signal = True
            else:
                subject_signals[unique_key] = None # Store None if signal is missing/empty

        # If no sequence signals were found at all (shouldn't happen if preprocessing worked and features_to_use is valid)
        if not found_any_signal and feature_order_list:
             log.error(f"[S{subj_id}] No sequence signals found in processed data structure. Skipping subject.")
             continue

        # Determine the maximum possible length for windowing based on labels and available signals
        max_possible_len = min(min_signal_len if min_signal_len != float('inf') else len(binary_labels), len(binary_labels))

        # --- Perform Windowing ---
        if window_samples <= 0:
            log.error(f"[S{subj_id}] Window size in samples is zero or negative ({window_samples}). Skipping subject.")
            continue
        if window_samples > max_possible_len:
            log.warning(f"[S{subj_id}] Window size ({window_samples}) > max data length ({max_possible_len}). No windows possible for this subject.")
            continue

        # Calculate the number of windows that can be created
        num_windows = max(0, (max_possible_len - window_samples) // step + 1)
        log.info(f"  S{subj_id}: Expecting {num_windows} windows. Static vector shape: {subj_static_vector.shape}")

        subj_windows_created = 0
        subj_window_labels_dist = collections.Counter() # Track label distribution per subject

        # Iterate through potential window start points
        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + window_samples

            # Safety check (should not happen with correct num_windows calculation)
            if end_idx > max_possible_len:
                 log.warning(f"S{subj_id} Win {i}: Calculated end index {end_idx} exceeds max length {max_possible_len}. Stopping windowing for subject.")
                 break

            # --- Create Combined Sequence Window Vector ---
            window_seq_segments = [] # List to hold segments for concatenation
            possible_window = True # Flag to track if window creation is valid
            current_segment_channels = 0 # Track channels added in this window

            # Iterate through the defined feature order
            for unique_key in feature_order_list:
                signal_array = subject_signals[unique_key] # Get the signal (or None)
                num_channels_for_this_feature = feature_channel_map.get(unique_key, 0) # Get expected channels from map

                if signal_array is not None:
                    # Extract the segment for this window
                    segment = signal_array[start_idx:end_idx]

                    # Validate segment length
                    if segment.shape[0] != window_samples:
                         log.error(f"S{subj_id} Win {i}: Segment length mismatch for {unique_key} ({segment.shape[0]} vs {window_samples}). Skipping window.")
                         possible_window = False; break

                    # Validate segment channel count against the map
                    if segment.shape[1] != num_channels_for_this_feature:
                         log.error(f"S{subj_id} Win {i}: Segment channel mismatch for {unique_key}. Expected {num_channels_for_this_feature}, got {segment.shape[1]}. Skipping window.")
                         possible_window = False; break

                    window_seq_segments.append(segment)
                    current_segment_channels += segment.shape[1]
                else:
                    # --- Handle Missing Signal: Padding ---
                    if num_channels_for_this_feature <= 0:
                         # This should ideally not happen if feature_channel_map is correct
                         log.error(f"S{subj_id} Win {i}: Cannot pad missing signal '{unique_key}' - zero channels expected in map. Skipping window.")
                         possible_window = False; break

                    # Create padding by tiling the default channel template
                    padding_segment = np.tile(default_seq_channel_template, (1, num_channels_for_this_feature))

                    # Validate padding shape
                    if padding_segment.shape != (window_samples, num_channels_for_this_feature):
                         log.error(f"S{subj_id} Win {i}: Padding segment shape mismatch for {unique_key} ({padding_segment.shape} vs {(window_samples, num_channels_for_this_feature)}). Skipping window.")
                         possible_window = False; break

                    window_seq_segments.append(padding_segment)
                    current_segment_channels += padding_segment.shape[1]
                    # Log padding occurrence (optional, can be noisy)
                    # log.warning(f"S{subj_id} Win {i}: Padding missing signal {unique_key} with defaults ({num_channels_for_this_feature} channels).")

            # If any segment validation failed, skip this window
            if not possible_window: continue

            # --- Final Checks and Concatenation ---
            # Check if total channels accumulated match the expected sequence dimension
            if current_segment_channels != input_dim_sequence:
                 log.error(f"S{subj_id} Win {i}: Total channels in segments ({current_segment_channels}) != expected sequence dimension ({input_dim_sequence}). Skipping window.")
                 continue

            # Concatenate all segments along the feature axis (axis=1)
            try:
                window_seq_vector = np.concatenate(window_seq_segments, axis=1).astype(np.float32)
            except ValueError as concat_e:
                log.error(f"S{subj_id} Win {i}: Sequence concatenation error: {concat_e}. Seg shapes: {[s.shape for s in window_seq_segments]}. Skipping window.")
                continue

            # Final shape validation (redundant if channel check above works, but safe)
            if window_seq_vector.shape != (window_samples, input_dim_sequence):
                 log.error(f"S{subj_id} Win {i}: Final sequence vector shape {window_seq_vector.shape} != expected {(window_samples, input_dim_sequence)}. Skipping window.")
                 continue

            # --- Determine Window Label ---
            # Label is 1 if *any* sample within the window's binary label is 1 (stress)
            label_segment = binary_labels[start_idx:end_idx]
            if label_segment.size == 0: # Should not happen if window is possible
                log.error(f"S{subj_id} Win {i}: Label segment is empty for start {start_idx}, end {end_idx}. Skipping window.")
                continue
            assigned_label = np.int32(1) if np.any(label_segment == 1) else np.int32(0)

            # --- Append Data to Lists ---
            all_windows_seq_features_list.append(window_seq_vector)
            all_windows_static_features_list.append(subj_static_vector) # Append the subject's static vector
            all_windows_labels_list.append(assigned_label)
            all_windows_groups_list.append(subj_id) # Use subject ID for grouping in splits
            all_windows_subject_ids_list.append(subj_id) # Store original ID for tracking
            all_windows_start_indices_list.append(start_idx) # Store absolute start index

            subj_windows_created += 1
            subj_window_labels_dist[assigned_label] += 1

        # Log summary for the subject
        if subj_windows_created > 0:
            subjects_with_windows += 1
            log.info(f"  S{subj_id}: Finished. Created {subj_windows_created} windows. Label dist: {dict(subj_window_labels_dist)}")
            total_windows_processed += subj_windows_created
        else:
            log.warning(f"  S{subj_id}: Finished. No valid windows created for this subject.")

    # --- Final Summary and Validation ---
    log.info("--- Window Creation Finished ---")
    log.info(f"Total windows created across {subjects_with_windows} subjects: {total_windows_processed}")

    if not all_windows_seq_features_list:
        log.error("No windows were created for any subject. Cannot proceed.")
        raise ValueError("Window creation failed: No windows generated.")

    # Final check for list length consistency
    list_lengths = [len(lst) for lst in [
        all_windows_seq_features_list, all_windows_static_features_list, all_windows_labels_list,
        all_windows_groups_list, all_windows_subject_ids_list, all_windows_start_indices_list
    ]]
    if len(set(list_lengths)) != 1:
        log.error(f"Mismatch between collected list lengths after windowing: {list_lengths}")
        raise ValueError("List length mismatch after windowing process.")

    return (
        all_windows_seq_features_list,
        all_windows_static_features_list,
        all_windows_labels_list,
        all_windows_groups_list,
        all_windows_subject_ids_list,
        all_windows_start_indices_list
    )

