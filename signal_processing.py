# signal_processing.py (Handles resampling and alignment)

import numpy as np
from scipy import signal as scisignal
import warnings
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

# Assuming utils.py is in the same directory or PYTHONPATH
try:
    from utils import get_sampling_rate, safe_get
except ImportError:
    # Fallback implementations if utils not found
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    def get_sampling_rate(*args): return None # Placeholder
    logging.warning("Could not import from 'utils'. Using basic fallbacks in signal_processing.py.")

log = logging.getLogger(__name__)

# --- Resample Function ---
def resample_signal(signal_data: Optional[np.ndarray], original_fs: float, target_fs: float) -> Optional[np.ndarray]:
    """
    Resamples a signal using scipy.signal.resample. Handles empty arrays, invalid inputs.

    Args:
        signal_data (Optional[np.ndarray]): The input signal array (1D or 2D).
        original_fs (float): The original sampling rate of the signal.
        target_fs (float): The desired target sampling rate.

    Returns:
        Optional[np.ndarray]: The resampled signal array, or None on failure.
    """
    # --- Input Validation ---
    if signal_data is None:
        return None
    if not isinstance(signal_data, np.ndarray):
        log.warning(f"resample_signal: Input not a numpy array (type: {type(signal_data)}).")
        return None
    if signal_data.size == 0:
        return np.array([], dtype=signal_data.dtype) # Return empty array of same type
    if not isinstance(original_fs, (int, float)) or not isinstance(target_fs, (int, float)) or original_fs <= 0 or target_fs <= 0:
        log.error(f"resample_signal: Invalid sampling rates provided (orig={original_fs}, target={target_fs}).")
        return None
    # If rates are already the same (within tolerance), no need to resample
    if np.isclose(original_fs, target_fs):
        return signal_data

    # --- Calculate Target Length ---
    original_length = signal_data.shape[0]
    # Calculate the number of samples in the resampled signal
    target_length = int(np.round(original_length * (target_fs / original_fs)))

    # Handle cases where target length becomes zero or negative
    if target_length <= 0:
        log.warning(f"resample_signal: Target length {target_length} <= 0 (from original length {original_length}). Returning empty array.")
        # Create an empty array with the correct number of dimensions (channels)
        empty_shape = list(signal_data.shape); empty_shape[0] = 0
        try: return np.array([], dtype=signal_data.dtype).reshape(tuple(empty_shape))
        except: return np.array([]) # Fallback

    # --- Perform Resampling ---
    try:
        # Use scipy.signal.resample (works along axis 0 by default)
        with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             resampled_signal = scisignal.resample(signal_data, target_length, axis=0)
        return resampled_signal
    except Exception as e:
        log.error(f"resample_signal: Scipy signal.resample failed (orig:{original_length} -> target:{target_length}): {e}", exc_info=True)
        return None

# --- Resample and Align Subject Data ---
def resample_and_align_subject_signals(
    subject_id: Union[int, str], # Allow string IDs (e.g., "NURSE_1")
    raw_subj_data: Dict[str, Any],
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Performs resampling and alignment for a single subject's data based on config.
    This function focuses only on the signal processing part.

    Args:
        subject_id (Union[int, str]): The identifier for the subject.
        raw_subj_data (Dict[str, Any]): The raw data dictionary for the subject
                                        (output from data_loader).
        config (Dict[str, Any]): The main configuration dictionary.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the resampled and aligned
                                  signals and labels for the subject, or None if
                                  critical processing fails (e.g., label handling).
                                  Output keys: 'signal', 'label', 'sampling_rates', 'dataset_id'.
    """
    log_prefix = f"[S{subject_id} Resample/Align]"
    log.info(f"{log_prefix} Starting signal resampling and alignment...")

    # --- Get Target Sampling Rate ---
    target_fs = safe_get(config, ['processing', 'target_sampling_rate'], 64) # Default 64Hz
    if not target_fs or target_fs <= 0:
        log.error(f"{log_prefix} Invalid target_sampling_rate ({target_fs}) in config. Cannot process.")
        return None

    # --- Determine Dataset ID and Original Label Rate ---
    dataset_id = safe_get(raw_subj_data, ['dataset_id'])
    if not dataset_id:
        if isinstance(subject_id, str) and '_' in subject_id:
            dataset_id = subject_id.split('_')[0].upper()
            log.warning(f"{log_prefix} 'dataset_id' key missing in raw data. Inferred '{dataset_id}' from subject ID.")
        else:
            log.error(f"{log_prefix} Cannot determine dataset_id for subject. Cannot get label sampling rate accurately.")
            dataset_id = 'WESAD'; log.error(f"{log_prefix} Assuming dataset_id '{dataset_id}' as fallback.")

    original_label_fs_lookup = get_sampling_rate(config, 'label', 'label', dataset_id=dataset_id)
    original_label_fs = original_label_fs_lookup if original_label_fs_lookup else 700 # Default fallback
    if not original_label_fs_lookup:
        log.warning(f"{log_prefix} Label Fs not found for dataset '{dataset_id}'. Assuming default: {original_label_fs} Hz.")

    # --- Get Features to Process ---
    chest_features = safe_get(config, ['features_to_use', 'chest'], [])
    wrist_features = safe_get(config, ['features_to_use', 'wrist'], [])
    all_features_to_align = list(dict.fromkeys(chest_features + wrist_features))
    log.debug(f"{log_prefix} Signals to attempt resampling & align: {all_features_to_align}")

    # --- Resample Labels (Crucial Step) ---
    original_labels = safe_get(raw_subj_data, ['label'])
    final_labels = None
    reference_len = 0 # Target length for all signals

    if original_labels is not None and isinstance(original_labels, np.ndarray) and original_labels.size > 0:
        original_labels_flat = original_labels.flatten()
        target_len_float = len(original_labels_flat) * (target_fs / original_label_fs)
        reference_len = int(np.round(target_len_float))

        if reference_len > 0:
            target_indices_float = np.linspace(0, len(original_labels_flat) - 1, reference_len)
            nearest_indices = np.round(target_indices_float).astype(int)
            nearest_indices = np.clip(nearest_indices, 0, len(original_labels_flat) - 1)
            final_labels = original_labels_flat[nearest_indices]
            log.debug(f"{log_prefix} Labels resampled using nearest neighbor. Reference length set to: {reference_len}")
        else:
            log.error(f"{log_prefix} Label resampling resulted in target length <= 0. Cannot process subject.")
            return None
    else:
        log.error(f"{log_prefix} Original labels missing or empty. Cannot process subject.")
        return None

    # --- Resample Signals ---
    subj_processed_signals = {} # Store resampled signals: {(device, signal_key): array}
    subj_sampling_rates = {} # Store final sampling rates: {signal_key_final: rate}
    missing_signals_for_subj = []
    signals_processed_count = 0

    possible_sources = []
    if isinstance(safe_get(raw_subj_data, ['signal', 'chest']), dict): possible_sources.append('chest')
    if isinstance(safe_get(raw_subj_data, ['signal', 'wrist']), dict): possible_sources.append('wrist')

    for signal_key in all_features_to_align:
        signal_found = False
        for device in possible_sources:
            signal_values = safe_get(raw_subj_data, ['signal', device, signal_key])
            if signal_values is not None:
                original_fs = safe_get(raw_subj_data, ['sampling_rates', signal_key])
                if original_fs is None:
                    original_fs = get_sampling_rate(config, signal_key, device, dataset_id=dataset_id)

                if original_fs is None or original_fs <= 0:
                    log.warning(f"{log_prefix} Original Fs missing/invalid for {dataset_id}/{device}/{signal_key}. Skipping.")
                    missing_signals_for_subj.append(f"{device}/{signal_key}")
                    signal_found = True; break

                try:
                    resampled_signal = resample_signal(signal_values, original_fs=original_fs, target_fs=target_fs)
                    if resampled_signal is None or resampled_signal.size == 0:
                        log.warning(f"{log_prefix} Resampling returned None/empty for {device}/{signal_key}. Skipping.")
                        missing_signals_for_subj.append(f"{device}/{signal_key}")
                    else:
                        subj_processed_signals[(device, signal_key)] = resampled_signal
                        subj_sampling_rates[f"{signal_key}_final"] = target_fs
                        signals_processed_count += 1
                        log.debug(f"{log_prefix}   Resampled {device}/{signal_key} from {original_fs}Hz. New shape: {resampled_signal.shape}")
                    signal_found = True; break
                except Exception as resample_e:
                    log.error(f"{log_prefix} Error resampling {device}/{signal_key}: {resample_e}. Skipping.", exc_info=False)
                    missing_signals_for_subj.append(f"{device}/{signal_key}")
                    signal_found = True; break
        if not signal_found:
            log.warning(f"{log_prefix} Signal '{signal_key}' not found in any source ({possible_sources}).")
            missing_signals_for_subj.append(f"{signal_key} (Not Found)")

    if missing_signals_for_subj:
        log.warning(f"{log_prefix} Some signals were missing or failed resampling: {missing_signals_for_subj}")
    if signals_processed_count == 0 and all_features_to_align:
        log.warning(f"{log_prefix} No signals specified in 'features_to_use' were processed successfully.")

    # --- Align Signals to Reference Length ---
    log.info(f"{log_prefix} Aligning {signals_processed_count} processed signal(s) to reference length: {reference_len}")
    final_aligned_subj_data = {
        'signal': {'chest': {}, 'wrist': {}},
        'label': final_labels,
        'sampling_rates': subj_sampling_rates,
        'dataset_id': dataset_id
    }
    alignment_success = True

    for (device, signal_key), resampled_signal in subj_processed_signals.items():
        current_len = resampled_signal.shape[0]
        aligned_signal = None
        if current_len == reference_len:
            aligned_signal = resampled_signal
        elif current_len > reference_len:
            aligned_signal = resampled_signal[:reference_len]
            log.debug(f"{log_prefix}   Truncated {device}/{signal_key} from {current_len} to {reference_len}.")
        else: # current_len < reference_len
            pad_width = reference_len - current_len
            pad_dims = [(0, pad_width)] + [(0, 0)] * (resampled_signal.ndim - 1)
            try:
                aligned_signal = np.pad(resampled_signal, pad_dims, mode='edge')
                log.debug(f"{log_prefix}   Padded {device}/{signal_key} from {current_len} to {reference_len}.")
            except Exception as pad_e:
                log.error(f"{log_prefix} Error padding {device}/{signal_key}: {pad_e}. Skipping subject alignment.")
                alignment_success = False; break

        if not alignment_success: break

        if device not in final_aligned_subj_data['signal']:
            final_aligned_subj_data['signal'][device] = {}
        final_aligned_subj_data['signal'][device][signal_key] = aligned_signal

    # --- Final Return ---
    if alignment_success:
        final_aligned_subj_data['sampling_rates']['label_final'] = target_fs
        log.info(f"{log_prefix} Finished signal resampling and alignment.")
        return final_aligned_subj_data
    else:
        log.error(f"{log_prefix} Signal alignment failed. Subject data discarded.")
        return None
