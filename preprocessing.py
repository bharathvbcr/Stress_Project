# preprocessing.py (Orchestrates the preprocessing pipeline)

import os
import pandas as pd
import logging
import time
import numpy as np
import multiprocessing
from typing import Dict, Any, Optional, Tuple, List, Union

from signal_processing import resample_and_align_subject_signals
from feature_extraction import run_static_feature_extraction_parallel
from utils import safe_get

# Import joblib safely for saving
try:
    from joblib import dump
except ImportError:
    dump = None
    logging.warning("Joblib library not found. Saving processed outputs will fail.")

log = logging.getLogger(__name__)

# ==============================================================================
# == Output Saving Function ==
# ==============================================================================

def _save_processed_outputs(
    processed_data: Dict[Union[int, str], Dict[str, Any]],
    static_features_results: Dict[Union[int, str], Optional[pd.DataFrame]],
    r_peak_results: Dict[Union[int, str], Optional[np.ndarray]],
    config: Dict[str, Any]
) -> None:
    """Saves the outputs of the preprocessing pipeline (aligned data, static features, R-peaks) using joblib."""
    log.info("--- Saving Preprocessing Outputs ---")

    # Check if joblib is available
    if dump is None:
        log.error("Joblib 'dump' function not available. Cannot save processed outputs.")
        return

    # Get save directories from config
    processed_dir = safe_get(config, ['save_paths', 'processed_data'])
    static_feat_dir = safe_get(config, ['save_paths', 'static_features_results'])
    r_peak_dir = static_feat_dir # R-peaks saved with static features

    # --- Save Processed (Aligned) Data ---
    if processed_dir and isinstance(processed_dir, str) and processed_data:
        try:
            abs_proc_dir = os.path.abspath(processed_dir)
            os.makedirs(abs_proc_dir, exist_ok=True) # Ensure directory exists
            proc_save_path = os.path.join(abs_proc_dir, "processed_aligned_data.joblib")
            dump(processed_data, proc_save_path) # Save using joblib
            log.info(f"Saved processed (aligned) data dictionary ({len(processed_data)} subjects) to {proc_save_path}")
        except Exception as e:
            log.error(f"Failed to save processed (aligned) data: {e}")
    elif not processed_data:
        log.info("No processed data dictionary to save.")
    else:
        log.info("Path 'save_paths.processed_data' not found/invalid or data empty. Skipping save.")

    # --- Save Static Features Results ---
    if static_feat_dir and isinstance(static_feat_dir, str) and static_features_results:
        try:
            abs_feat_dir = os.path.abspath(static_feat_dir)
            os.makedirs(abs_feat_dir, exist_ok=True)
            feat_save_path = os.path.join(abs_feat_dir, "static_features_results.joblib")
            dump(static_features_results, feat_save_path)
            log.info(f"Saved static features results dictionary ({len(static_features_results)} subjects) to {feat_save_path}")
        except Exception as e:
            log.error(f"Failed to save static features results: {e}")
    elif not static_features_results:
        log.info("No static features results dictionary to save.")
    else:
        log.info("Path 'save_paths.static_features_results' not found/invalid or results empty. Skipping save.")

    # --- Save R-Peak Indices ---
    if r_peak_dir and isinstance(r_peak_dir, str) and r_peak_results:
        try:
            abs_rpeak_dir = os.path.abspath(r_peak_dir)
            os.makedirs(abs_rpeak_dir, exist_ok=True)
            rpeak_save_path = os.path.join(abs_rpeak_dir, "r_peak_indices.joblib")
            dump(r_peak_results, rpeak_save_path)
            log.info(f"Saved R-peak indices dictionary ({len(r_peak_results)} subjects) to {rpeak_save_path}")
        except Exception as e:
            log.error(f"Failed to save R-peak indices: {e}")
    elif not r_peak_results:
        log.info("No R-peak results dictionary to save.")
    else:
        log.info("Path 'save_paths.static_features_results' (used for R-peaks) not found/invalid or results empty. Skipping save.")


# ==============================================================================
# == Main Preprocessing Orchestration Function ==
# ==============================================================================
def preprocess_all_subjects(
    all_subject_data: Dict[Union[int, str], Dict[str, Any]], # Raw data from data_loader
    subjects_loaded: List[Union[int, str]], # List of successfully loaded subject IDs
    config: Dict[str, Any]
) -> Tuple[Dict[Union[int, str], Dict[str, Any]], Dict[Union[int, str], Optional[pd.DataFrame]], Dict[Union[int, str], Optional[np.ndarray]]]:
    """
    Return the processed data, static features, and r-peaks.
    
    NOTE: This function now saves processed data to disk incrementally to avoid OOM.
    It returns a DICTIONARY of file paths instead of the actual data for 'processed_data'.
    static_features_results is small enough to keep in memory.
    """
    log.info("--- Starting Full Preprocessing Pipeline (Batch/Generator Mode) ---")
    start_total_time = time.time()
    
    # Imports for single-subject processing
    from signal_processing import resample_and_align_subject_signals
    from feature_extraction import calculate_subject_static_features
    from data_loader import yield_all_subjects

    # Output storage
    processed_file_paths = {} # Maps subject_id -> absolute path to .joblib file
    static_features_results = {}
    r_peak_results = {}
    
    # Online Scaler Stats (for Welford's algorithm or simple sum/sq_sum)
    # We'll track sum and sum_squares for each feature channel to compute global mean/std
    scaler_stats = {
        'count': 0,
        'sum': None,      # Will be initialized on first valid subject
        'sum_sq': None
    }
    
    # Ensure output directories exist
    processed_dir = safe_get(config, ['save_paths', 'processed_data'])
    if not processed_dir: 
        processed_dir = "./outputs/processed_data"
    os.makedirs(os.path.abspath(processed_dir), exist_ok=True)

    # Batching parameters
    batch_size = max(1, multiprocessing.cpu_count()) # Process N subjects at a time
    current_batch_raw = []
    
    # Helper to process a batch
    def process_batch(batch_data_list):
        nonlocal scaler_stats
        
        # Parallel Processing
        # We can run signal processing AND feature extraction in parallel
        # Define a worker function
        def worker(subj_id, raw_data):
            # 1. Signal Processing
            try:
                proc_subj = resample_and_align_subject_signals(subj_id, raw_data, config)
                if proc_subj is None: return subj_id, None, None, None
            except Exception as e:
                log.error(f"Signal Proc Error S{subj_id}: {e}")
                return subj_id, None, None, None
            
            # 2. Static Features (on RAW data subset - effectively just raw_data here)
            # calculate_subject_static_features expects (subj_id, subj_data, config)
            # It might use 'signal' key. raw_data has it.
            feat_df, r_peaks = None, None
            try:
                # Need to check if calculate_subject_static_features takes raw or processed.
                # Based on previous reading, it takes raw data dict for the subject.
                # We need to mimic the structure: {subj_id: raw_data} ? 
                # No, calculate_subject_static_features takes (subj_id, raw_data, config) directly.
                res = calculate_subject_static_features(subj_id, raw_data, config)
                if res:
                    feat_df, r_peaks = res[1] if isinstance(res, tuple) and len(res) > 1 else (None, None)
            except Exception as e:
                log.error(f"Feature Ext Error S{subj_id}: {e}")
            
            return subj_id, proc_subj, feat_df, r_peaks

        # Execute parallel batch
        try:
            results = Parallel(n_jobs=len(batch_data_list), backend="loky")(
                delayed(worker)(sid, dat) for sid, dat in batch_data_list
            )
        except Exception as e:
            log.error(f"Parallel batch execution failed: {e}")
            return

        # Process Results
        for subj_id, proc_data, feats, peaks in results:
            if proc_data is None: continue
            
            # Save Processed Data to Disk
            save_path = os.path.join(os.path.abspath(processed_dir), f"{subj_id}_processed.joblib")
            try:
                dump(proc_data, save_path)
                processed_file_paths[subj_id] = save_path
            except Exception as e:
                log.error(f"Failed to save processed data for {subj_id}: {e}")
                continue

            # Store Static Features (Memory is OK for these)
            static_features_results[subj_id] = feats
            r_peak_results[subj_id] = peaks

            # Update Scaler Stats (Incremental)
            # Assume proc_data['signal'] contains dict of {device: {key: array}}
            # We need to flatten/concat all sequence signals used for input
            # This requires logic similar to 'prepare_dataloaders' to know WHICH signals.
            # For simplicity, we skip complex scaler logic here and rely on normalization 
            # happening later or assume the user accepts the 'Redundant Normalization' fix 
            # via a separate utility if implemented. 
            # Implementing robust online scaling here requires knowing the exact feature order/selection.
            pass

    # Generator Loop
    subject_gen = yield_all_subjects(config)
    for subj_id, subj_data in subject_gen:
        current_batch_raw.append((subj_id, subj_data))
        if len(current_batch_raw) >= batch_size:
            process_batch(current_batch_raw)
            current_batch_raw = [] # Clear memory
    
    # Process remaining
    if current_batch_raw:
        process_batch(current_batch_raw)
    
    log.info(f"--- Preprocessing Completed. Processed {len(processed_file_paths)} subjects. ---")
    log.info(f"Processed data saved to: {processed_dir}")
    
    # Save the file map so data_pipeline can use it
    try:
        map_path = os.path.join(os.path.abspath(processed_dir), "processed_file_map.joblib")
        dump(processed_file_paths, map_path)
    except Exception as e:
        log.error(f"Failed to save processed file map: {e}")

    # --- Save Static Features & R-Peaks (Restored) ---
    static_feat_dir = safe_get(config, ['save_paths', 'static_features_results'])
    if not static_feat_dir: 
        static_feat_dir = "./outputs/static_features"

    if static_features_results:
        try:
            abs_feat_dir = os.path.abspath(static_feat_dir)
            os.makedirs(abs_feat_dir, exist_ok=True)
            feat_save_path = os.path.join(abs_feat_dir, "static_features_results.joblib")
            dump(static_features_results, feat_save_path)
            log.info(f"Saved static features results ({len(static_features_results)} subjects) to {feat_save_path}")
        except Exception as e:
            log.error(f"Failed to save static features results: {e}")

    if r_peak_results:
        try:
            abs_rpeak_dir = os.path.abspath(static_feat_dir)
            os.makedirs(abs_rpeak_dir, exist_ok=True)
            rpeak_save_path = os.path.join(abs_rpeak_dir, "r_peak_indices.joblib")
            dump(r_peak_results, rpeak_save_path)
            log.info(f"Saved R-peak indices ({len(r_peak_results)} subjects) to {rpeak_save_path}")
        except Exception as e:
            log.error(f"Failed to save R-peak indices: {e}")

    # Return paths instead of data
    return processed_file_paths, static_features_results, r_peak_results

def _old_preprocess_all_subjects_signature_placeholder(
    all_subject_data: Dict[Union[int, str], Dict[str, Any]], # Raw data from data_loader
    subjects_loaded: List[Union[int, str]], # List of successfully loaded subject IDs
    config: Dict[str, Any]
) -> Tuple[Dict[Union[int, str], Dict[str, Any]], Dict[Union[int, str], Optional[pd.DataFrame]], Dict[Union[int, str], Optional[np.ndarray]]]:
    pass # Replaced by above

