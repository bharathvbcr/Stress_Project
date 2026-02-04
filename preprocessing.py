# preprocessing.py (Orchestrates the preprocessing pipeline)

import os
import pandas as pd
import logging
import time
import numpy as np
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
    Main function to orchestrate the preprocessing pipeline:
    1. Calls signal processing (resampling/alignment) for each subject.
    2. Calls static feature extraction (parallel).
    3. Saves the results.

    Args:
        all_subject_data (Dict): Dictionary containing raw data for all loaded subjects.
        subjects_loaded (List): List of subject IDs that were successfully loaded.
        config (Dict): The main configuration dictionary.

    Returns:
        Tuple[Dict, Dict, Dict]: A tuple containing:
            - processed_data: Dictionary of processed (resampled/aligned) data.
            - static_features_results: Dictionary of static feature DataFrames.
            - r_peak_results: Dictionary of R-peak indices arrays.
    """
    log.info("--- Starting Full Preprocessing Pipeline (Modular) ---")
    start_total_time = time.time()
    processed_data = {} # To store resampled/aligned data
    subjects_failed_initial_processing = [] # Track subjects failing resampling/alignment

    # --- Step 1: Signal Processing (Resampling & Alignment) ---
    log.info(f"--- Step 1: Applying Signal Processing (Resampling & Alignment) on {len(subjects_loaded)} subjects ---")
    for subject_id in subjects_loaded:
        if subject_id not in all_subject_data:
            log.warning(f"[S{subject_id}] Not found in loaded raw data dictionary. Skipping preprocessing.")
            continue

        # Call the function from the signal_processing module
        try:
            processed_subj = resample_and_align_subject_signals(subject_id, all_subject_data[subject_id], config)
        except Exception as e:
            log.error(f"Error during signal processing for S{subject_id}: {e}", exc_info=True)
            processed_subj = None # Ensure it's None on error

        if processed_subj is not None:
            processed_data[subject_id] = processed_subj
        else:
            subjects_failed_initial_processing.append(subject_id)

    log.info("--- Signal Processing Finished for All Subjects ---")
    log.info(f"Successfully processed signals for: {len(processed_data)} subjects.")
    if subjects_failed_initial_processing:
        log.warning(f"Failed signal processing (resample/align) for {len(subjects_failed_initial_processing)} subjects: {sorted(subjects_failed_initial_processing)}")

    # --- Step 2: Static Feature Extraction (Parallel) ---
    # Call the function from the feature_extraction module
    # Run only on subjects that passed Step 1
    # IMPORTANT: We pass 'all_subject_data' (RAW data) instead of 'processed_data'
    # This allows features like HRV to be calculated on the original high-frequency signals.
    try:
        # Filter raw data to include only successfully loaded subjects
        raw_data_subset = {k: v for k, v in all_subject_data.items() if k in processed_data}
        static_features_results, r_peak_results = run_static_feature_extraction_parallel(raw_data_subset, config)
    except Exception as e:
        log.error(f"Critical error during parallel feature extraction setup/execution: {e}", exc_info=True)
        static_features_results, r_peak_results = {}, {} # Return empty dicts on major failure

    # --- Step 3: Save Outputs ---
    _save_processed_outputs(processed_data, static_features_results, r_peak_results, config)

    # --- Pipeline Completion ---
    end_total_time = time.time()
    log.info(f"--- Total Preprocessing Pipeline Finished ---")
    log.info(f"Total time: {end_total_time - start_total_time:.2f} seconds")

    return processed_data, static_features_results, r_peak_results
