# feature_extraction.py (Calculates static and interaction features)

import numpy as np
import pandas as pd
from scipy import stats
import warnings
import logging
import time
from joblib import Parallel, delayed
import multiprocessing
from typing import Dict, Any, Optional, Tuple, List, Union

# Assuming utils.py is in the same directory or PYTHONPATH
try:
    from utils import safe_get
except ImportError:
    # Fallback implementation if utils not found
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    logging.warning("Could not import 'safe_get' from 'utils'. Using basic fallback in feature_extraction.py.")

# Attempt to import NeuroKit2
try:
    import neurokit2 as nk
except ImportError:
    nk = None
    logging.error("NeuroKit2 library not found. HRV and advanced EDA features cannot be calculated.")

log = logging.getLogger(__name__)

# ==============================================================================
# == Feature Calculation Helper Functions ==
# ==============================================================================
# Note: These functions are copied from the original preprocessing.py V7
# They calculate features based on the *processed* (resampled/aligned) signals.

# --- Basic Stats ---
def calculate_basic_stats(signal: Optional[np.ndarray], prefix: str) -> Dict[str, Optional[float]]:
    """ Calculates basic statistical features (Mean, Std, Max, Min, Range). Handles potential multi-channel ACC. """
    features = {f'{prefix}_Mean': None, f'{prefix}_Std': None, f'{prefix}_Max': None, f'{prefix}_Min': None, f'{prefix}_Range': None}
    if signal is None or not isinstance(signal, np.ndarray) or signal.size < 2: return features
    try:
        if signal.ndim > 1 and signal.shape[1] == 3:
             valid_rows = ~np.isnan(signal).any(axis=1)
             if np.sum(valid_rows) > 0: signal = np.linalg.norm(signal[valid_rows,:], axis=1)
             else: signal = np.array([])
        elif signal.ndim > 1:
             log.warning(f"Basic stats on multi-channel signal '{prefix}' ({signal.shape[1]} channels). Using mean across channels.")
             signal = np.nanmean(signal, axis=1)
        signal = signal.flatten()
        if signal.size < 2: return features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features[f'{prefix}_Mean'] = np.nanmean(signal)
            features[f'{prefix}_Std'] = np.nanstd(signal)
            valid_signal = signal[~np.isnan(signal)]
            features[f'{prefix}_Max'] = np.nanmax(valid_signal) if valid_signal.size > 0 else None
            features[f'{prefix}_Min'] = np.nanmin(valid_signal) if valid_signal.size > 0 else None
        if features[f'{prefix}_Max'] is not None and features[f'{prefix}_Min'] is not None:
            features[f'{prefix}_Range'] = features[f'{prefix}_Max'] - features[f'{prefix}_Min']
        else: features[f'{prefix}_Range'] = None
    except Exception as e:
        log.warning(f"Error calculating basic stats for '{prefix}': {e}", exc_info=False)
        return {f'{prefix}_Mean': None, f'{prefix}_Std': None, f'{prefix}_Max': None, f'{prefix}_Min': None, f'{prefix}_Range': None}
    return {k: (v if (v is not None and np.isfinite(v)) else None) for k, v in features.items()}

# --- Temp Slope ---
def calculate_temp_slope(signal: Optional[np.ndarray], fs: Optional[float], prefix: str) -> Dict[str, Optional[float]]:
    """ Calculates the slope of the temperature signal using linear regression. """
    feature_name = f'{prefix}_Slope'
    if signal is None or not isinstance(signal, np.ndarray) or signal.size < 5: return {feature_name: None}
    if fs is None or not isinstance(fs, (int, float)) or fs <= 0: return {feature_name: None}
    signal = signal.flatten()
    try:
        time_vector = np.arange(len(signal)) / fs
        valid_mask = ~np.isnan(signal)
        if np.sum(valid_mask) < 2: return {feature_name: None}
        slope, _, _, _, _ = stats.linregress(time_vector[valid_mask], signal[valid_mask])
        return {feature_name: slope if np.isfinite(slope) else None}
    except Exception as e:
        log.warning(f"Could not calculate slope for {prefix}: {e}")
        return {feature_name: None}

# --- EDA Features ---
def calculate_eda_features(signal: Optional[np.ndarray], fs: Optional[float], prefix: str) -> Dict[str, Optional[float]]:
    """ Calculates EDA features (basic stats, SCR peaks, SCL) using NeuroKit2. """
    features = {f'{prefix}_Mean': None, f'{prefix}_Std': None, f'{prefix}_Max': None, f'{prefix}_Min': None, f'{prefix}_Range': None,
                f'SCR_Peaks_N': None, f'SCR_Peaks_Amplitude_Mean': None, f'SCR_Peaks_Amplitude_Std': None,
                f'SCR_Peaks_Amplitude_Sum': None, f'{prefix}_SCL_Mean': None, f'{prefix}_SCL_Std': None}
    if nk is None:
        log.warning(f"calculate_eda_features ({prefix}): NeuroKit2 not available. Calculating basic stats only.")
        basic_stats = calculate_basic_stats(signal, prefix); features.update(basic_stats)
        return {k: (v if (v is not None and np.isfinite(v)) else None) for k, v in features.items()}
    if signal is None or not isinstance(signal, np.ndarray) or signal.size < 10: return features
    if fs is None or not isinstance(fs, (int, float)) or fs <= 0: return features
    signal = signal.flatten()
    try:
        basic_stats = calculate_basic_stats(signal, prefix); features.update(basic_stats)
        eda_cleaned = nk.eda_clean(signal, sampling_rate=fs)
        try: eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=fs, method='highpass')
        except Exception as decomp_e: log.warning(f"NK eda_phasic failed for {prefix}: {decomp_e}. Skipping SCR/SCL features."); eda_decomposed = None
        if eda_decomposed is None: return {k: (v if (v is not None and np.isfinite(v)) else None) for k, v in features.items()}
        scr_signal = eda_decomposed["EDA_Phasic"].values; scl_signal = eda_decomposed["EDA_Tonic"].values
        try:
            peaks_info = nk.eda_peaks(scr_signal, sampling_rate=fs, method='neurokit', amplitude_min=0.01)
            peak_indices = peaks_info[1].get('SCR_Peaks') if peaks_info[1] is not None else None
            num_peaks = len(peak_indices) if peak_indices is not None else 0
            features[f'SCR_Peaks_N'] = float(num_peaks)
            if num_peaks > 0:
                peak_amplitudes = peaks_info[1].get('SCR_Amplitude')
                if peak_amplitudes is not None:
                    valid_amplitudes = peak_amplitudes[~np.isnan(peak_amplitudes)]
                    if len(valid_amplitudes) > 0:
                        features[f'SCR_Peaks_Amplitude_Mean'] = np.mean(valid_amplitudes)
                        features[f'SCR_Peaks_Amplitude_Std'] = np.std(valid_amplitudes)
                        features[f'SCR_Peaks_Amplitude_Sum'] = np.sum(valid_amplitudes)
                else: log.warning(f"NK eda_peaks did not return 'SCR_Amplitude' for {prefix}.")
        except Exception as peak_e: log.warning(f"NK eda_peaks failed for {prefix}: {peak_e}. Skipping SCR peak features.")
        if scl_signal is not None and scl_signal.size > 0:
             features[f'{prefix}_SCL_Mean'] = np.nanmean(scl_signal)
             features[f'{prefix}_SCL_Std'] = np.nanstd(scl_signal)
    except Exception as e: log.warning(f"Could not calculate advanced EDA features for {prefix}: {e}", exc_info=False)
    return {k: (v if (v is not None and np.isfinite(v)) else None) for k, v in features.items()}

# --- HRV from ECG ---
def process_hrv_from_ecg(ecg_signal: Optional[np.ndarray], sampling_rate: float, nk_instance: Any) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """ Processes ECG for R-peaks and HRV features using NeuroKit2. Returns (hrv_features_df, r_peak_indices). """
    if nk_instance is None: log.warning("NK instance missing for HRV."); return None, None
    if ecg_signal is None or not isinstance(ecg_signal, np.ndarray) or ecg_signal.size < 10: return None, None
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0: return None, None
    hrv_features_df = None; rpeaks = None; MIN_RPEAKS_FOR_HRV = 10
    try:
        ecg_signal = ecg_signal.flatten()
        try: signals, info = nk_instance.ecg_process(ecg_signal, sampling_rate=sampling_rate, method='neurokit'); rpeaks = info.get('ECG_R_Peaks') if info else None
        except Exception as process_e: log.warning(f"NK ecg_process failed: {process_e}. Cannot calculate HRV."); return None, None
        if rpeaks is not None and len(rpeaks) > MIN_RPEAKS_FOR_HRV:
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    hrv_features_df = nk_instance.hrv(rpeaks, sampling_rate=sampling_rate, show=False)
                if hrv_features_df is None or hrv_features_df.empty or hrv_features_df.isnull().all().all():
                     log.warning(f"HRV analysis failed or returned empty/NaNs despite {len(rpeaks)} peaks."); hrv_features_df = None
                # else: log.debug(f"HRV calculation successful ({len(hrv_features_df.columns)} features).")
            except Exception as hrv_calc_e: log.warning(f"NK hrv calculation failed: {hrv_calc_e}"); hrv_features_df = None
        elif rpeaks is not None: log.warning(f"Too few R-peaks detected ({len(rpeaks)} <= {MIN_RPEAKS_FOR_HRV}) for reliable HRV analysis.")
        else: log.warning("R-peak detection failed (ecg_process returned no peaks).")
        return hrv_features_df, rpeaks
    except Exception as e: log.error(f"Unexpected error during HRV processing: {e}", exc_info=False); return None, None

# --- Frequency Feature Helper ---
def calculate_frequency_features(signal: Optional[np.ndarray], fs: Optional[float], prefix: str) -> Dict[str, Optional[float]]:
    """ Calculates basic frequency domain features (e.g., peak frequency). """
    features = {f'{prefix}_PeakHz': None}
    if signal is None or not isinstance(signal, np.ndarray) or signal.size < 10: return features
    if fs is None or not isinstance(fs, (int, float)) or fs <= 0: return features
    try:
        if signal.ndim > 1 and signal.shape[1] == 3:
             valid_rows = ~np.isnan(signal).any(axis=1)
             if np.sum(valid_rows) > 0: signal = np.linalg.norm(signal[valid_rows,:], axis=1)
             else: signal = np.array([])
        elif signal.ndim > 1: signal = np.nanmean(signal, axis=1)
        signal = signal.flatten(); valid_signal = signal[~np.isnan(signal)]
        if valid_signal.size < 10: return features
        n = len(valid_signal); fft_vals = np.fft.fft(valid_signal); fft_freq = np.fft.fftfreq(n, d=1/fs)
        half_n = n // 2; fft_mag = np.abs(fft_vals[:half_n]); fft_freq_pos = fft_freq[:half_n]
        if len(fft_mag) > 1:
            peak_idx = np.argmax(fft_mag[1:]) + 1
            peak_frequency = fft_freq_pos[peak_idx]
            features[f'{prefix}_PeakHz'] = peak_frequency if np.isfinite(peak_frequency) else None
        else: features[f'{prefix}_PeakHz'] = None
    except Exception as e: log.warning(f"Could not calculate frequency features for {prefix}: {e}"); features[f'{prefix}_PeakHz'] = None
    return features

# ==============================================================================
# == Static Feature Calculation Pipeline for One Subject ==
# ==============================================================================

def calculate_subject_static_features(
    subject_id: Union[int, str],
    subject_proc_data: Dict, # Expects output from signal_processing module
    config: Dict[str, Any]
) -> Tuple[Union[int, str], Optional[Tuple[pd.DataFrame, Optional[np.ndarray]]]]:
    """
    Worker function (intended for parallel execution) to calculate ALL configured
    static and interaction features for ONE subject.

    Args:
        subject_id (Union[int, str]): The identifier for the subject.
        subject_proc_data (Dict): The processed (resampled/aligned) data dictionary.
        config (Dict[str, Any]): The main configuration dictionary.

    Returns:
        Tuple[Union[int, str], Optional[Tuple[pd.DataFrame, Optional[np.ndarray]]]]:
            A tuple containing:
                - subject_id: The identifier of the processed subject.
                - results_tuple (Optional): (features_df, r_peak_indices) or None on failure.
    """
    worker_log_prefix = f"[Parallel S{subject_id} FeatCalc]"
    all_static_features = {} # Dictionary to hold all calculated features
    final_r_peaks = None # To store R-peaks if calculated from ECG

    if not subject_proc_data or not isinstance(subject_proc_data, dict):
        log.warning(f"{worker_log_prefix} Invalid or empty processed data. Skipping static feature calculation.")
        return subject_id, None

    # --- Get Feature Calculation Flags ---
    calc_hrv = safe_get(config, ['processing', 'calculate_hrv'], True)
    calc_eda = safe_get(config, ['processing', 'calculate_eda_features'], True)
    calc_acc = safe_get(config, ['processing', 'calculate_acc_features'], True)
    calc_temp = safe_get(config, ['processing', 'calculate_temp_features'], True)
    calc_freq = safe_get(config, ['processing', 'calculate_freq_features'], True)
    calc_resp = safe_get(config, ['processing', 'calculate_resp_features'], False)
    calc_interact = safe_get(config, ['processing', 'calculate_interaction_features'], True)
    log.debug(f"{worker_log_prefix} Flags - HRV:{calc_hrv}, EDA:{calc_eda}, ACC:{calc_acc}, TEMP:{calc_temp}, FREQ:{calc_freq}, RESP:{calc_resp}, INTERACT:{calc_interact}")

    # --- Calculate Base Features ---
    # (Logic copied and verified from original preprocessing.py V7)
    # 1. HRV / HR
    if calc_hrv:
        ecg_signal = safe_get(subject_proc_data, ['signal', 'chest', 'ECG']); ecg_fs = safe_get(subject_proc_data, ['sampling_rates', 'ECG_final'])
        hr_signal = safe_get(subject_proc_data, ['signal', 'wrist', 'BVP']); hr_fs = safe_get(subject_proc_data, ['sampling_rates', 'BVP_final'])
        if nk is None: log.warning(f"{worker_log_prefix} NeuroKit2 not available, cannot calculate HRV.")
        elif ecg_signal is not None and ecg_fs is not None:
            log.debug(f"{worker_log_prefix} Calculating HRV from ECG (Fs={ecg_fs}Hz)...")
            hrv_features_df, r_peaks_array = process_hrv_from_ecg(ecg_signal, sampling_rate=ecg_fs, nk_instance=nk)
            if hrv_features_df is not None:
                try: hrv_dict = hrv_features_df.iloc[0].to_dict(); hrv_dict_clean = {k: (v if pd.notna(v) else None) for k, v in hrv_dict.items()}; all_static_features.update(hrv_dict_clean); log.debug(f"{worker_log_prefix} HRV (ECG) added ({len(hrv_dict_clean)} features)."); final_r_peaks = r_peaks_array; log.debug(f"{worker_log_prefix} R-Peaks stored ({len(final_r_peaks) if final_r_peaks is not None else 0} peaks).")
                except Exception as dict_e: log.warning(f"{worker_log_prefix} Could not convert HRV DF to dict: {dict_e}")
            else: log.warning(f"{worker_log_prefix} HRV (ECG) calculation returned None or empty DF.")
        elif hr_signal is not None and hr_fs is not None:
            log.debug(f"{worker_log_prefix} ECG missing, calculating features from HR/BVP (Fs={hr_fs}Hz)...")
            hr_signal_flat = hr_signal.flatten(); all_static_features[f'HR_Mean'] = np.nanmean(hr_signal_flat) if hr_signal_flat.size > 0 else None; all_static_features[f'HR_Std'] = np.nanstd(hr_signal_flat) if hr_signal_flat.size > 0 else None; log.debug(f"{worker_log_prefix} Basic HR features added.")
        else: log.warning(f"{worker_log_prefix} Skipping HRV/HR features: Neither ECG nor HR/BVP available with valid Fs.")

    # 2. EDA
    if calc_eda:
        for device in ['chest', 'wrist']:
            eda_signal = safe_get(subject_proc_data, ['signal', device, 'EDA']); eda_fs = safe_get(subject_proc_data, ['sampling_rates', 'EDA_final']);
            if eda_fs is None: eda_fs = safe_get(subject_proc_data, ['sampling_rates', 'eda_final'])
            if eda_signal is not None and eda_fs is not None:
                prefix = f"{device}_EDA"; log.debug(f"{worker_log_prefix} Calculating EDA features for {device} (Fs={eda_fs}Hz)..."); eda_feats = calculate_eda_features(eda_signal, eda_fs, prefix); all_static_features.update(eda_feats); log.debug(f"{worker_log_prefix} EDA features ({device}) added ({len(eda_feats)} features).")

    # 3. ACC
    if calc_acc:
        for device in ['chest', 'wrist']:
            acc_signal = safe_get(subject_proc_data, ['signal', device, 'ACC']); acc_fs = safe_get(subject_proc_data, ['sampling_rates', 'ACC_final'])
            if acc_fs is None: acc_fs = safe_get(subject_proc_data, ['sampling_rates', 'acc_final'])
            if acc_signal is not None and acc_fs is not None:
                prefix = f"{device}_ACC"; log.debug(f"{worker_log_prefix} Calculating ACC features for {device} (Fs={acc_fs}Hz)...")
                acc_stat_feats = calculate_basic_stats(acc_signal, prefix); all_static_features.update(acc_stat_feats); log.debug(f"{worker_log_prefix} ACC basic stats ({device}) added ({len(acc_stat_feats)} features).")
                if calc_freq:
                    acc_freq_feats = calculate_frequency_features(acc_signal, acc_fs, prefix); all_static_features.update(acc_freq_feats); log.debug(f"{worker_log_prefix} ACC frequency features ({device}) added ({len(acc_freq_feats)} features).")

    # 4. TEMP
    if calc_temp:
        for device in ['chest', 'wrist']:
            temp_signal = safe_get(subject_proc_data, ['signal', device, 'TEMP']); temp_fs = safe_get(subject_proc_data, ['sampling_rates', 'TEMP_final']);
            if temp_fs is None: temp_fs = safe_get(subject_proc_data, ['sampling_rates', 'temp_final'])
            if temp_signal is not None and temp_fs is not None:
                prefix = f"{device}_TEMP"; log.debug(f"{worker_log_prefix} Calculating TEMP features for {device} (Fs={temp_fs}Hz)..."); temp_stats = calculate_basic_stats(temp_signal, prefix); temp_slope = calculate_temp_slope(temp_signal, temp_fs, prefix); all_static_features.update(temp_stats); all_static_features.update(temp_slope); log.debug(f"{worker_log_prefix} TEMP features ({device}) added ({len(temp_stats)+len(temp_slope)} features).")

    # 5. RESP (Placeholder)
    if calc_resp:
        resp_signal = safe_get(subject_proc_data, ['signal', 'chest', 'RESP']); resp_fs = safe_get(subject_proc_data, ['sampling_rates', 'RESP_final'])
        if resp_signal is not None and resp_fs is not None:
             prefix = "chest_RESP"; log.debug(f"{worker_log_prefix} Calculating RESP features (Fs={resp_fs}Hz)...")
             resp_stats = calculate_basic_stats(resp_signal, prefix); all_static_features.update(resp_stats); log.debug(f"{worker_log_prefix} RESP basic stats added ({len(resp_stats)} features).")

    # --- Calculate Interaction Features ---
    if calc_interact:
        log.debug(f"{worker_log_prefix} Calculating interaction features...")
        num_interact_added = 0
        try:
            # Example 1: HRV LF/HF Ratio * Wrist EDA Mean
            hrv_lfhf = all_static_features.get('HRV_LFHF'); wrist_eda_mean = all_static_features.get('wrist_EDA_Mean')
            if hrv_lfhf is not None and wrist_eda_mean is not None: all_static_features['Interact_LFHF_x_WristEDA'] = hrv_lfhf * wrist_eda_mean; num_interact_added += 1
            # Example 2: Mean HR / Wrist Temp Slope
            hr_mean = all_static_features.get('HR_Mean'); wrist_temp_slope = all_static_features.get('wrist_TEMP_Slope')
            if hr_mean is not None and wrist_temp_slope is not None:
                if not np.isclose(wrist_temp_slope, 0, atol=1e-6): all_static_features['Interact_HRmean_div_WristTempSlope'] = hr_mean / wrist_temp_slope; num_interact_added += 1
                else: all_static_features['Interact_HRmean_div_WristTempSlope'] = None
            # Example 3: SCR Peak Count * Chest ACC Std Dev
            scr_peaks_n = all_static_features.get('SCR_Peaks_N'); chest_acc_std = all_static_features.get('chest_ACC_Std')
            if scr_peaks_n is not None and chest_acc_std is not None: all_static_features['Interact_SCRpeaks_x_ChestACCstd'] = scr_peaks_n * chest_acc_std; num_interact_added += 1
            # Example 4: Wrist Temp Mean * Wrist ACC Mean
            wrist_temp_mean = all_static_features.get('wrist_TEMP_Mean'); wrist_acc_mean = all_static_features.get('wrist_ACC_Mean')
            if wrist_temp_mean is not None and wrist_acc_mean is not None: all_static_features['Interact_WristTempMean_x_WristACCmean'] = wrist_temp_mean * wrist_acc_mean; num_interact_added += 1
            log.debug(f"{worker_log_prefix} Interaction features added: {num_interact_added}")
        except Exception as interact_e: log.warning(f"{worker_log_prefix} Error calculating interaction features: {interact_e}")

    # --- Prepare and Return Results ---
    if not all_static_features:
        log.warning(f"{worker_log_prefix} No static features were successfully calculated.")
        return subject_id, None
    else:
        final_features_dict = {k: (v if (v is not None and np.isfinite(v)) else None) for k, v in all_static_features.items()}
        if not any(v is not None for v in final_features_dict.values()):
             log.warning(f"{worker_log_prefix} All calculated static features resulted in None.")
             return subject_id, None
        try:
             final_features_df = pd.DataFrame([final_features_dict])
             log.info(f"{worker_log_prefix} Finished static feature calculation. Total valid features: {len(final_features_dict)}.")
             return subject_id, (final_features_df, final_r_peaks)
        except Exception as df_e:
             log.error(f"{worker_log_prefix} Failed to create DataFrame from features: {df_e}")
             return subject_id, None

# ==============================================================================
# == Parallel Execution Runner ==
# ==============================================================================

def run_static_feature_extraction_parallel(
        processed_data: Dict[Union[int, str], Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Tuple[Dict[Union[int, str], Optional[pd.DataFrame]], Dict[Union[int, str], Optional[np.ndarray]]]:
    """
    Runs the static feature calculation in parallel for all subjects using joblib.

    Args:
        processed_data (Dict): Dictionary containing the processed (resampled/aligned)
                               data for each subject.
        config (Dict): The main configuration dictionary.

    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - static_features_results: Dictionary mapping subject ID to their static feature DataFrame (or None).
            - r_peak_results: Dictionary mapping subject ID to their R-peak indices array (or None).
    """
    log.info("--- Applying Static Feature Processing (Parallel CPU) ---")
    start_feat_time = time.time()
    static_features_results = {}
    r_peak_results = {}

    # --- Pre-checks ---
    any_static_features_requested = any([
        safe_get(config, ['processing', f], True) for f in [
            'calculate_hrv', 'calculate_eda_features', 'calculate_acc_features',
            'calculate_temp_features', 'calculate_freq_features',
            'calculate_resp_features', 'calculate_interaction_features'
        ]
    ])
    if not processed_data: log.warning("Skipping static feature processing: No processed data provided."); return static_features_results, r_peak_results
    if not any_static_features_requested: log.info("Skipping static feature processing: No features requested in config flags."); return static_features_results, r_peak_results

    # --- Determine Number of Parallel Jobs ---
    n_cores = multiprocessing.cpu_count()
    n_jobs_config = safe_get(config, ['processing', 'parallel_n_jobs'], -1)
    if n_jobs_config == -1: n_jobs = max(1, n_cores - 1)
    elif isinstance(n_jobs_config, int) and n_jobs_config > 0: n_jobs = min(n_cores, n_jobs_config)
    else: n_jobs = 1
    log.info(f"Using {n_jobs} parallel jobs for static feature calculation.")

    # --- Prepare Data for Parallel Processing ---
    subject_items_for_features = [(subj_id, subj_data) for subj_id, subj_data in processed_data.items() if subj_data is not None]
    if not subject_items_for_features: log.warning("No valid subjects found in processed_data for feature extraction."); return static_features_results, r_peak_results
    log.info(f"Sending {len(subject_items_for_features)} subjects for parallel static feature processing...")

    # --- Run Parallel Calculation ---
    try:
        parallel_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(calculate_subject_static_features)(subj_id, subj_data, config)
            for subj_id, subj_data in subject_items_for_features
        )
    except Exception as parallel_e:
        log.error(f"Joblib Parallel execution failed: {parallel_e}", exc_info=True)
        return static_features_results, r_peak_results

    # --- Collect Results ---
    subjects_feat_processed_ids = []; subjects_feat_failed_ids = []
    for result_item in parallel_results:
        if result_item is None: continue
        subj_id, result_tuple = result_item
        if result_tuple is not None and isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            features_df, r_peaks_array = result_tuple
            static_features_results[subj_id] = features_df
            r_peak_results[subj_id] = r_peaks_array
            if features_df is not None or r_peaks_array is not None: subjects_feat_processed_ids.append(subj_id)
            else: subjects_feat_failed_ids.append(subj_id)
        else:
            static_features_results[subj_id] = None; r_peak_results[subj_id] = None
            subjects_feat_failed_ids.append(subj_id)

    # --- Log Summary ---
    end_feat_time = time.time()
    log.info("--- Static Feature Parallel Processing Finished ---")
    log.info(f"Feature processing time: {end_feat_time - start_feat_time:.2f} seconds")
    log.info(f"Successfully calculated static features/R-peaks bundle for {len(subjects_feat_processed_ids)} subjects.")
    if subjects_feat_failed_ids: log.warning(f"Failed static feature calculation within worker for {len(subjects_feat_failed_ids)} subjects: {sorted(subjects_feat_failed_ids)}")

    return static_features_results, r_peak_results
