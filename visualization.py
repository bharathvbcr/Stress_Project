# visualization.py (Handles plotting of signals, results, etc.)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For creating custom legend handles
import seaborn as sns
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import torch # Only needed if directly plotting tensors, otherwise remove
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Assuming utils.py is available
try:
    from utils import get_sampling_rate, safe_get
except ImportError:
    # Fallback implementations if utils not found
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    def get_sampling_rate(*args): return None
    logging.warning("Could not import from 'utils'. Using basic fallbacks in visualization.py.")

log = logging.getLogger(__name__)

# Set a default plotting style using seaborn
sns.set_theme(style="whitegrid", palette="muted")

# ==============================================================================
# == Helper Functions ==
# ==============================================================================

def _get_signal_info(
    subject_data: Dict[str, Any],
    config: Dict[str, Any],
    signal_key: str
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    """
    Internal helper to retrieve a signal array, its sampling rate, and source device ('chest' or 'wrist').
    Checks both raw and processed data structures.

    Args:
        subject_data (Dict[str, Any]): Data dictionary for a single subject.
        config (Dict[str, Any]): Configuration dictionary.
        signal_key (str): The key of the signal to retrieve (e.g., 'ECG', 'EDA').

    Returns:
        Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
            - Signal array (numpy)
            - Sampling rate (float)
            - Source device ('chest' or 'wrist')
            Returns (None, None, None) if signal/rate cannot be found.
    """
    source_device = None
    signal_array = None
    sampling_rate = None
    # Use subject ID from data if available for logging, otherwise use '?'
    subj_id_for_log = subject_data.get('subject_id', subject_data.get('subject', '?'))
    log_prefix = f"[GetSignal S{subj_id_for_log}-{signal_key}]"

    # Basic validation
    if not subject_data or not isinstance(subject_data, dict):
        log.warning(f"{log_prefix} Invalid subject_data provided.")
        return None, None, None
    if not config:
        log.warning(f"{log_prefix} Config dictionary not provided.")
        return None, None, None

    # Identify potential sources ('chest', 'wrist') present in the data
    possible_sources = []
    if isinstance(safe_get(subject_data, ['signal', 'chest']), dict): possible_sources.append('chest')
    if isinstance(safe_get(subject_data, ['signal', 'wrist']), dict): possible_sources.append('wrist')
    if not possible_sources:
        log.warning(f"{log_prefix} No 'chest' or 'wrist' keys found under subject_data['signal'].")
        return None, None, None

    log.debug(f"{log_prefix} Searching for signal in sources: {possible_sources}")
    # Iterate through possible sources to find the signal
    for src in possible_sources:
        candidate_signal = safe_get(subject_data, ['signal', src, signal_key])
        if candidate_signal is not None:
            log.debug(f"{log_prefix} Found candidate signal in source '{src}'.")

            # Determine sampling rate: Check processed data first, then config
            processed_fs_key = f"{signal_key}_final" # Key used in processed data
            candidate_fs = safe_get(subject_data, ['sampling_rates', processed_fs_key])
            # Try lowercase key as fallback
            if candidate_fs is None: candidate_fs = safe_get(subject_data, ['sampling_rates', processed_fs_key.lower()])
            # If not in processed data, look up original rate from config
            if candidate_fs is None:
                 dataset_id = subject_data.get('dataset_id', 'WESAD') # Get dataset ID if available
                 try:
                     # Use the utility function to get the rate from config
                     candidate_fs = get_sampling_rate(config, signal_key, src, dataset_id=dataset_id)
                     log.debug(f"{log_prefix} Looked up original Fs from config for {dataset_id}/{src}/{signal_key}: {candidate_fs}")
                 except Exception as e:
                     log.warning(f"{log_prefix} Error looking up Fs from config for {dataset_id}/{src}/{signal_key}: {e}")
                     candidate_fs = None

            # Validate the found sampling rate
            if candidate_fs is not None and isinstance(candidate_fs, (int, float)) and candidate_fs > 0:
                signal_array = candidate_signal
                sampling_rate = float(candidate_fs)
                source_device = src
                log.debug(f"{log_prefix} Confirmed signal from '{source_device}' with Fs={sampling_rate} Hz.")
                break # Stop searching once found with valid Fs
            else:
                log.warning(f"{log_prefix} Found signal in '{src}', but Fs is unknown/invalid ({candidate_fs}).")
                # Keep the first found signal even if Fs is unknown, but prioritize signals with known Fs
                if signal_array is None:
                     signal_array = candidate_signal
                     source_device = src

    # Log final status
    if signal_array is not None and sampling_rate is None:
        log.warning(f"{log_prefix} Signal found in '{source_device}' but sampling rate remains unknown.")
    elif signal_array is None:
        log.error(f"{log_prefix} Signal ultimately not found in any source.")
        return None, None, None

    # Final Checks on the selected signal array
    if signal_array is not None:
        # Ensure it's a numpy array
        if not isinstance(signal_array, np.ndarray):
            try:
                signal_array = np.array(signal_array)
                log.debug(f"{log_prefix} Converted signal to numpy array.")
            except Exception as e:
                log.error(f"{log_prefix} Could not convert signal to numpy array: {e}")
                return None, None, None
        # Check if empty
        if signal_array.size == 0:
            log.warning(f"{log_prefix} Signal array from '{source_device}' exists but is empty.")
            # Return the empty array along with rate/source if found
            return signal_array, sampling_rate, source_device

        log.debug(f"{log_prefix} Final signal shape: {signal_array.shape}, Fs: {sampling_rate}, Source: {source_device}")

    return signal_array, sampling_rate, source_device


def get_available_signals(subject_data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    """
    Gets a sorted list of signal keys available for a subject based on data presence.

    Args:
        subject_data (Dict[str, Any]): Data dictionary for a single subject.
        config (Dict[str, Any]): Configuration dictionary (currently unused here but kept for consistency).

    Returns:
        List[str]: Sorted list of available signal keys (e.g., ['ACC', 'BVP', 'ECG', 'EDA']).
    """
    available_signals = set()
    # Basic validation
    if not subject_data or not isinstance(subject_data, dict):
        log.warning("Cannot get signals: Invalid subject_data provided.")
        return []

    # Access the 'signal' part of the subject data
    signal_data_dict = safe_get(subject_data, ['signal'], {})
    subj_id_for_log = subject_data.get('subject_id', subject_data.get('subject', '?')) # Get subject ID for logging
    if not signal_data_dict:
        log.warning(f"No 'signal' key found in subject data for S{subj_id_for_log}.")
        return []

    # Iterate through sources ('chest', 'wrist') and their signals
    for source, signals_in_source in signal_data_dict.items():
        if isinstance(signals_in_source, dict):
            for sig_key, signal_array in signals_in_source.items():
                # Check if signal_array exists and is not empty
                if signal_array is not None and hasattr(signal_array, 'size') and signal_array.size > 0:
                     available_signals.add(sig_key)
                # else: log.debug(f"Signal '{sig_key}' from '{source}' is None or empty.") # Reduce log noise

    if not available_signals:
        log.warning(f"No available (non-empty) signals found in data for S{subj_id_for_log}.")

    return sorted(list(available_signals)) # Return sorted list

# ==============================================================================
# == Plotting Functions ==
# ==============================================================================

def plot_raw_signal(
    subject_data: Dict[str, Any],
    config: Dict[str, Any],
    subject_id: Union[int, str],
    signal_key: str,
    time_range_sec: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None # Allow passing an existing Axes object
) -> None:
    """
    Plots a segment of a signal (raw or processed) for a selected subject.
    Handles multi-channel signals and cases where sampling rate is unknown.

    Args:
        subject_data (Dict[str, Any]): Data dictionary for the subject.
        config (Dict[str, Any]): Configuration dictionary.
        subject_id (Union[int, str]): Subject identifier.
        signal_key (str): Key of the signal to plot (e.g., 'ECG').
        time_range_sec (Optional[Tuple[float, float]], optional): Start and end time in seconds.
                                                                  If None, uses default duration from config.
        ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on. If None, creates a new figure/axes.
    """
    log_prefix = f"[PlotSignal S{subject_id}-{signal_key}]"
    log.debug(f"{log_prefix} Requested time range: {time_range_sec}")

    # Create figure/axes if not provided
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))
        fig_created = True

    # --- Get Signal Data ---
    # Check for subject data early
    if not subject_data:
        log.warning(f"{log_prefix} No data provided for subject.")
        ax.set_title(f"Signal: {signal_key} (S{subject_id}) - Subject Data Missing")
        ax.text(0.5, 0.5, "Subject data not available", ha='center', va='center', transform=ax.transAxes)
        if fig_created: plt.tight_layout(); plt.show() # Show empty plot if figure was created here
        return

    # Retrieve signal array, sampling rate, and source using the helper
    signal_array, fs, source = _get_signal_info(subject_data, config, signal_key)

    # --- Prepare Plot Title and Labels ---
    plot_title = f'Signal: {signal_key} (S{subject_id}, Source: {source or "Unknown"})'
    x_label = "Time (seconds)" # Default x-axis label

    # --- Handle Different Scenarios ---
    if signal_array is None:
        # Signal data could not be found
        log.error(f"{log_prefix} Signal data not found.")
        plot_title += " - Signal Not Found"
        ax.text(0.5, 0.5, "Signal data not available", ha='center', va='center', transform=ax.transAxes)

    elif signal_array.size == 0:
        # Signal data exists but is empty
        log.warning(f"{log_prefix} Signal data is empty.")
        plot_title += " - Signal Empty"
        ax.text(0.5, 0.5, "Signal data is empty", ha='center', va='center', transform=ax.transAxes)

    elif fs is None or fs <= 0:
        # Sampling rate is unknown, plot against sample index
        log.warning(f"{log_prefix} Sampling rate unknown/invalid ({fs}). Plotting against sample index.")
        data_to_plot = signal_array
        plot_title += ' - Fs Unknown'
        x_label = "Sample Index"
        time_vector = np.arange(data_to_plot.shape[0]) # Use sample indices as x-axis

        # Plot data (handle multi-channel)
        if data_to_plot.ndim > 1 and data_to_plot.shape[1] > 1:
            # Plot each channel separately
            num_channels = data_to_plot.shape[1]
            for i in range(num_channels):
                ax.plot(time_vector, data_to_plot[:, i], label=f'Ch {i+1}', linewidth=1)
            ax.legend()
        else:
            # Plot single channel or flattened data
            ax.plot(time_vector, data_to_plot.flatten(), linewidth=1)
        ax.set_ylabel(f'{signal_key} Amplitude')
        ax.set_xlim(0, len(time_vector) - 1 if len(time_vector) > 1 else 1) # Set x-limits based on samples

    else: # Plot against time (Fs is known)
        # --- Determine Time Segment ---
        if time_range_sec is None:
            # Use default duration if no range is specified
            default_duration = safe_get(config, ['visualization', 'default_plot_duration_s'], 60.0)
            start_sec, end_sec = 0.0, float(default_duration)
        else:
            start_sec, end_sec = float(time_range_sec[0]), float(time_range_sec[1])

        # Convert time range to sample indices
        start_sample = max(0, int(start_sec * fs))
        end_sample = min(signal_array.shape[0], int(end_sec * fs))

        # --- Robust Range Checks ---
        # Ensure start index is valid
        if start_sample >= signal_array.shape[0]:
            start_sample = 0 # Reset to beginning if start is beyond end
            start_sec = 0.0
        # Ensure end index is after start index
        if end_sample <= start_sample:
            end_sample = start_sample + int(fs * 1) # Plot at least 1 second if possible
        # Ensure end index doesn't exceed array bounds
        end_sample = min(signal_array.shape[0], end_sample)
        # Ensure start < end after adjustments, handle edge case where end_sample becomes 0
        if end_sample > 0:
            start_sample = min(start_sample, end_sample - 1)
        else: # If signal is very short and end_sample is 0
            start_sample = 0

        # Recalculate actual start/end time based on adjusted samples
        actual_start_sec = start_sample / fs
        actual_end_sec = end_sample / fs

        log.debug(f"{log_prefix} Calculated samples: [{start_sample}:{end_sample}] for time [{actual_start_sec:.2f}s, {actual_end_sec:.2f}s]")

        # Extract the segment data
        segment_data = signal_array[start_sample:end_sample]
        num_samples_in_segment = segment_data.shape[0]

        # Check if segment is valid for plotting
        if num_samples_in_segment <= 1: # Need at least 2 points to plot a line
            log.warning(f"{log_prefix} Segment empty or only 1 point [{start_sample}:{end_sample}].")
            plot_title += f' | {actual_start_sec:.2f}s - {actual_end_sec:.2f}s | No Data in Range'
            ax.text(0.5, 0.5, "No data or too few points in time range", ha='center', va='center', transform=ax.transAxes)
        else:
            # Create time vector for the x-axis
            time_vector = np.linspace(actual_start_sec, actual_end_sec, num=num_samples_in_segment, endpoint=False)
            plot_title += f' | {actual_start_sec:.2f}s - {actual_end_sec:.2f}s | Fs~{fs:.1f}Hz'

            # Plot data (handle multi-channel) using seaborn for better aesthetics
            if segment_data.ndim > 1 and segment_data.shape[1] > 1:
                num_channels = segment_data.shape[1]
                labels = [f'Ch {i+1}' for i in range(num_channels)]
                for i in range(num_channels):
                    sns.lineplot(x=time_vector, y=segment_data[:, i], label=labels[i], linewidth=1, ax=ax)
                ax.legend()
            else:
                sns.lineplot(x=time_vector, y=segment_data.flatten(), linewidth=1, ax=ax)

            ax.set_ylabel(f'{signal_key} Amplitude')
            ax.set_xlim(actual_start_sec, actual_end_sec) # Set x-limits to the actual plotted range

    # --- Final Touches ---
    ax.set_xlabel(x_label)
    ax.set_title(plot_title, fontsize=10)
    ax.grid(True, linestyle=':')
    if fig_created:
        plt.tight_layout() # Adjust layout only if figure was created here
        plt.show() # Display the plot if figure was created here

def plot_ecg_with_peaks(
    ecg_signal: Optional[np.ndarray],
    r_peak_indices: Optional[np.ndarray],
    true_labels: Optional[np.ndarray],
    sampling_rate: Optional[float],
    subject_id: Union[int, str],
    time_range_sec: Tuple[float, float],
    ax: Optional[plt.Axes] = None
) -> None:
    """
    Plots a segment of processed ECG signal with detected R-peaks and true stress labels.

    Args:
        ecg_signal (Optional[np.ndarray]): Processed ECG signal array (1D).
        r_peak_indices (Optional[np.ndarray]): Array of sample indices where R-peaks occur.
        true_labels (Optional[np.ndarray]): Array of true labels (0 or 1) aligned with ECG signal.
        sampling_rate (Optional[float]): Sampling rate of the ECG signal and labels.
        subject_id (Union[int, str]): Subject identifier.
        time_range_sec (Tuple[float, float]): Start and end time in seconds for the plot.
        ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on. If None, creates a new figure/axes.
    """
    log_prefix = f"[PlotECG S{subject_id}]"
    log.debug(f"{log_prefix} Requested time range: {time_range_sec}")

    # Create figure/axes if not provided
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
        fig_created = True

    # --- Input Validation ---
    if ecg_signal is None or not isinstance(ecg_signal, np.ndarray) or ecg_signal.size == 0:
        log.error(f"{log_prefix} ECG signal empty/None.")
        ax.set_title(f"ECG + R-Peaks + Labels (S{subject_id}) - ECG Data Missing")
        if fig_created: plt.tight_layout(); plt.show()
        return
    plot_labels = True # Flag to control label plotting
    if true_labels is None or not isinstance(true_labels, np.ndarray) or true_labels.size == 0:
        log.warning(f"{log_prefix} True labels empty/None. Will not plot labels.")
        plot_labels = False
    if sampling_rate is None or not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        log.error(f"{log_prefix} Invalid ECG sampling rate ({sampling_rate}).")
        ax.set_title(f"ECG + R-Peaks + Labels (S{subject_id}) - Invalid Sampling Rate")
        if fig_created: plt.tight_layout(); plt.show()
        return

    # --- Calculate Segment ---
    start_sec, end_sec = time_range_sec
    # Determine end sample based on shortest available signal (ECG or Labels)
    max_len = len(ecg_signal)
    if plot_labels: max_len = min(max_len, len(true_labels))

    start_sample = max(0, int(start_sec * sampling_rate))
    end_sample = min(max_len, int(end_sec * sampling_rate))

    # --- Robust Range Checks (similar to plot_raw_signal) ---
    if start_sample >= max_len: start_sample = 0; start_sec = 0.0
    if end_sample <= start_sample: end_sample = start_sample + int(sampling_rate * 1) # Ensure at least 1s if possible
    end_sample = min(max_len, end_sample)
    if end_sample > 0: start_sample = min(start_sample, end_sample - 1)
    else: start_sample = 0

    # Recalculate actual times based on adjusted samples
    actual_start_sec = start_sample / sampling_rate
    actual_end_sec = end_sample / sampling_rate

    log.debug(f"{log_prefix} Calculated samples: [{start_sample}:{end_sample}] for time [{actual_start_sec:.2f}s, {actual_end_sec:.2f}s]")

    # Extract data segments
    segment_ecg = ecg_signal.flatten()[start_sample:end_sample]
    segment_labels_true_signal = true_labels.flatten()[start_sample:end_sample] if plot_labels else None
    num_samples_in_segment = len(segment_ecg)

    # Check if segment is valid
    if num_samples_in_segment <= 1:
        log.warning(f"{log_prefix} Segment empty or only 1 point [{start_sample}:{end_sample}].")
        ax.set_title(f'ECG + R-Peaks + Labels (S{subject_id}) | {actual_start_sec:.2f}s - {actual_end_sec:.2f}s | No Data in Range')
        if fig_created: plt.tight_layout(); plt.show()
        return

    # --- Calculate Time Vector & Find Peaks within Segment ---
    time_vector = np.linspace(actual_start_sec, actual_end_sec, num=num_samples_in_segment, endpoint=False)
    peaks_in_segment_relative = np.array([], dtype=int)
    num_peaks_in_segment = 0

    if r_peak_indices is not None and isinstance(r_peak_indices, np.ndarray) and r_peak_indices.size > 0:
        # Find absolute peak indices within the segment boundaries
        peaks_mask = (r_peak_indices >= start_sample) & (r_peak_indices < end_sample)
        peaks_in_segment_indices = r_peak_indices[peaks_mask]
        # Convert absolute indices to indices relative to the segment start
        peaks_in_segment_relative = peaks_in_segment_indices - start_sample
        # Ensure relative indices are within the bounds of the segment array
        valid_peak_mask = (peaks_in_segment_relative >= 0) & (peaks_in_segment_relative < num_samples_in_segment)
        peaks_in_segment_relative = peaks_in_segment_relative[valid_peak_mask].astype(int)
        num_peaks_in_segment = len(peaks_in_segment_relative)
    elif r_peak_indices is None:
        log.warning(f"{log_prefix} R-peak data is None. Cannot plot peaks.")

    # --- Plotting ---
    plot_title = f'Processed ECG + R-Peaks + Labels (S{subject_id}) | {actual_start_sec:.2f}s - {actual_end_sec:.2f}s | Fs~{sampling_rate:.1f}Hz'

    # Plot ECG signal
    line_ecg, = ax.plot(time_vector, segment_ecg, label='ECG Signal', linewidth=1, color='royalblue', zorder=3)
    ax.set_ylabel('ECG Amplitude')
    ax.set_xlabel('Time (seconds)')
    y_min, y_max = ax.get_ylim() # Get current y-limits for label shading

    # Plot True Stress Labels as shaded background
    handle_true_stress = None
    if plot_labels and segment_labels_true_signal is not None:
        where_stress_true = segment_labels_true_signal == 1
        if np.any(where_stress_true):
            # Use fill_between to shade regions where true stress is 1
            ax.fill_between(time_vector, y_min, y_max, where=where_stress_true, color='orange', alpha=0.3, step='mid', label='True Stress', zorder=1)
            # Create a patch handle for the legend
            handle_true_stress = mpatches.Patch(color='orange', alpha=0.3, label='True Stress')

    # Plot R-Peaks
    peak_label = f'R-Peaks ({num_peaks_in_segment} found)'
    handle_peaks = None
    if num_peaks_in_segment > 0:
        try:
            # Plot peaks as red scatter points on top of the ECG signal
            scatter_peaks = ax.scatter(time_vector[peaks_in_segment_relative], segment_ecg[peaks_in_segment_relative], color='red', s=40, zorder=5, label=peak_label)
            handle_peaks = scatter_peaks # Use the scatter plot object as the legend handle
        except IndexError:
            log.error(f"{log_prefix} Index error plotting R-peaks. Relative indices might be out of bounds.")
            peak_label += " (Plot Error)"
            handle_peaks = mpatches.Circle((0, 0), radius=3, color='red', label=peak_label) # Placeholder handle
        except Exception as e:
            log.error(f"{log_prefix} Unknown error plotting R-peaks: {e}")
            peak_label += " (Plot Error)"
            handle_peaks = mpatches.Circle((0, 0), radius=3, color='red', label=peak_label) # Placeholder handle
    else:
        # Create a placeholder handle if no peaks are found/plotted
        handle_peaks = mpatches.Circle((0, 0), radius=3, color='red', alpha=0.5, label=peak_label)

    # --- Legend and Final Touches ---
    handles = [line_ecg] # Start legend with ECG line
    if handle_true_stress: handles.append(handle_true_stress)
    if handle_peaks: handles.append(handle_peaks)

    ax.legend(handles=[h for h in handles if h is not None], loc='upper right')
    ax.set_title(plot_title, fontsize=10)
    ax.grid(True, linestyle=':')
    ax.set_xlim(actual_start_sec, actual_end_sec) # Set x-limits
    ax.set_ylim(y_min, y_max) # Restore y-limits

    if fig_created:
        plt.tight_layout()
        plt.show()


def plot_raw_vs_resampled(
    subject_id: Union[int, str],
    signal_key: str,
    time_range_sec: Tuple[float, float],
    raw_subject_data: Dict[str, Any],
    processed_subject_data: Dict[str, Any],
    config: Dict[str, Any]
):
    """
    Plots raw and resampled signals side-by-side for comparison.

    Args:
        subject_id (Union[int, str]): Subject identifier.
        signal_key (str): Key of the signal to plot.
        time_range_sec (Tuple[float, float]): Start and end time in seconds.
        raw_subject_data (Dict[str, Any]): Dictionary containing the raw subject data.
        processed_subject_data (Dict[str, Any]): Dictionary containing the processed (resampled) data.
        config (Dict[str, Any]): Configuration dictionary.
    """
    log_prefix = f"[PlotCompare S{subject_id}-{signal_key}]"
    log.debug(f"{log_prefix} Requested time range: {time_range_sec}")
    start_sec, end_sec = time_range_sec

    # Create a figure with two subplots, sharing the x-axis
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    main_title = f"Raw vs. Resampled: S{subject_id} - {signal_key} [{start_sec:.2f}s - {end_sec:.2f}s]"
    fig.suptitle(main_title)

    # --- Plot Raw Signal (Top Subplot) ---
    ax_raw = axes[0]
    try:
        # Use the plot_raw_signal function with the raw data
        plot_raw_signal(
            subject_data=raw_subject_data,
            config=config,
            subject_id=subject_id,
            signal_key=signal_key,
            time_range_sec=time_range_sec,
            ax=ax_raw # Pass the top Axes object
        )
        # Modify the title generated by plot_raw_signal
        raw_title = ax_raw.get_title()
        # Extract Fs info if present
        fs_info_raw = raw_title.split('|')[-1].strip() if '|' in raw_title else 'Fs Unknown'
        ax_raw.set_title(f"Raw Signal ({fs_info_raw})", fontsize=9)
        ax_raw.set_xlabel('') # Remove x-label from top plot
    except Exception as e:
        log.error(f"{log_prefix} Error plotting raw component: {e}", exc_info=True)
        ax_raw.set_title("Error Plotting Raw Signal")
        ax_raw.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax_raw.transAxes)

    # --- Plot Resampled Signal (Bottom Subplot) ---
    ax_res = axes[1]
    try:
        # Use the plot_raw_signal function with the processed data
        plot_raw_signal(
            subject_data=processed_subject_data,
            config=config,
            subject_id=subject_id,
            signal_key=signal_key,
            time_range_sec=time_range_sec,
            ax=ax_res # Pass the bottom Axes object
        )
        # Modify the title
        res_title = ax_res.get_title()
        fs_info_res = res_title.split('|')[-1].strip() if '|' in res_title else 'Fs Unknown'
        ax_res.set_title(f"Resampled Signal ({fs_info_res})", fontsize=9)
        # Style the resampled line differently (e.g., dashed orange)
        for line in ax_res.get_lines():
            line.set_linestyle('--')
            line.set_color('darkorange')
    except Exception as e:
        log.error(f"{log_prefix} Error plotting resampled component: {e}", exc_info=True)
        ax_res.set_title("Error Plotting Resampled Signal")
        ax_res.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax_res.transAxes)

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap
    plt.show()


def plot_predictions_on_signal(
    processed_data: Dict[Union[int, str], Dict[str, Any]],
    config: Dict[str, Any],
    subject_id: Union[int, str],
    signal_key: str,
    time_range_sec: Tuple[float, float],
    window_start_samples: Optional[List[int]] = None, # Absolute start indices of prediction windows
    window_predictions: Optional[List[int]] = None, # Predictions (0 or 1) for each window
    ax: Optional[plt.Axes] = None
) -> None:
    """
    Plots a signal segment with true labels and overlays pre-computed model predictions.

    Args:
        processed_data (Dict): Dictionary containing processed data for subjects.
        config (Dict): Configuration dictionary.
        subject_id (Union[int, str]): Subject identifier.
        signal_key (str): Key of the signal to plot as background (e.g., 'ECG', 'EDA').
        time_range_sec (Tuple[float, float]): Start and end time in seconds.
        window_start_samples (Optional[List[int]]): List of absolute start sample indices
                                                    corresponding to the predictions.
        window_predictions (Optional[List[int]]): List of binary predictions (0 or 1)
                                                  corresponding to each window start.
        ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on. If None, creates a new figure/axes.
    """
    log_prefix = f"[PlotPred S{subject_id}-{signal_key}]"
    log.debug(f"{log_prefix} Requested time range: {time_range_sec}")

    # Create figure/axes if not provided
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 5)) # Wider figure for clarity
        fig_created = True

    # --- Get Config Info ---
    target_fs = safe_get(config, ['processing', 'target_sampling_rate'])
    window_size_sec = safe_get(config, ['windowing', 'window_size_sec'], 60)
    window_samples = 0
    if target_fs and target_fs > 0 and window_size_sec > 0:
        window_samples = int(target_fs * window_size_sec)
    else:
        log.error(f"{log_prefix} Invalid config for prediction mapping (target_fs or window_size_sec).")
        ax.set_title(f"Plot Error (S{subject_id}) - Invalid Config for Prediction Mapping")
        if fig_created: plt.tight_layout(); plt.show()
        return

    # --- Get Data for Subject ---
    if subject_id not in processed_data:
        log.error(f"{log_prefix} Subject {subject_id} not found in processed_data.")
        ax.set_title(f"Plot Error (S{subject_id}) - Processed Data Missing")
        if fig_created: plt.tight_layout(); plt.show()
        return
    subj_proc_data = processed_data[subject_id]

    # Get signal, labels, and sampling rate
    signal_array, signal_fs, signal_source = _get_signal_info(subj_proc_data, config, signal_key)
    true_labels_full = safe_get(subj_proc_data, ['label'])

    # --- Validate Data ---
    if signal_array is None or signal_array.size == 0:
        log.error(f"{log_prefix} Signal '{signal_key}' missing or empty for S{subject_id}.")
        ax.set_title(f"Plot Error (S{subject_id}) - Missing/Empty Signal '{signal_key}'")
        if fig_created: plt.tight_layout(); plt.show()
        return
    if true_labels_full is None or true_labels_full.size == 0:
        log.error(f"{log_prefix} True labels missing or empty for S{subject_id}.")
        ax.set_title(f"Plot Error (S{subject_id}) - Missing/Empty Labels")
        if fig_created: plt.tight_layout(); plt.show()
        return
    if signal_fs is None:
        log.error(f"{log_prefix} Sampling rate for signal '{signal_key}' missing for S{subject_id}.")
        ax.set_title(f"Plot Error (S{subject_id}) - Missing Signal Fs")
        if fig_created: plt.tight_layout(); plt.show()
        return
    # Check if signal Fs matches the target Fs used for windowing/predictions
    if not np.isclose(signal_fs, target_fs):
        log.warning(f"{log_prefix} Signal Fs ({signal_fs}) != target Fs ({target_fs})! Prediction mapping might be inaccurate.")

    # --- Get Segment to Plot ---
    signal_array = signal_array.flatten()
    true_labels_full = true_labels_full.flatten()
    max_len = min(len(signal_array), len(true_labels_full)) # Max length based on available data

    start_sec, end_sec = time_range_sec
    start_sample_sig = max(0, int(start_sec * signal_fs))
    end_sample_sig = min(max_len, int(end_sec * signal_fs))

    # Robust range checks
    if start_sample_sig >= max_len: start_sample_sig = 0; start_sec = 0.0
    if end_sample_sig <= start_sample_sig: end_sample_sig = start_sample_sig + int(signal_fs * 1)
    end_sample_sig = min(max_len, end_sample_sig)
    if end_sample_sig > 0: start_sample_sig = min(start_sample_sig, end_sample_sig - 1)
    else: start_sample_sig = 0

    actual_start_sec = start_sample_sig / signal_fs
    actual_end_sec = end_sample_sig / signal_fs

    log.debug(f"{log_prefix} Calculated samples: [{start_sample_sig}:{end_sample_sig}] for time [{actual_start_sec:.2f}s, {actual_end_sec:.2f}s]")

    # Extract segments
    segment_signal = signal_array[start_sample_sig:end_sample_sig]
    segment_labels_true_signal = true_labels_full[start_sample_sig:end_sample_sig]
    num_samples_in_segment = len(segment_signal)

    if num_samples_in_segment <= 1:
        log.warning(f"{log_prefix} Segment empty or only 1 point [{start_sample_sig}:{end_sample_sig}].")
        ax.set_title(f'Predictions vs True Labels (S{subject_id}) | {actual_start_sec:.1f}s - {actual_end_sec:.1f}s | No Signal Data in Range')
        if fig_created: plt.tight_layout(); plt.show()
        return

    # Create time vector for plotting
    time_vector = np.linspace(actual_start_sec, actual_end_sec, num=num_samples_in_segment, endpoint=False)

    # --- Map Predictions onto the Time Segment ---
    predictions_mapped = np.full(num_samples_in_segment, np.nan) # Initialize with NaN
    plot_predictions = False # Flag to indicate if any predictions fall within the segment

    if window_start_samples is not None and window_predictions is not None:
        if len(window_start_samples) == len(window_predictions):
            # Iterate through each prediction window
            for win_start_sample, win_pred in zip(window_start_samples, window_predictions):
                win_end_sample = win_start_sample + window_samples
                # Find the overlap between the prediction window and the current plot segment
                overlap_start_abs = max(win_start_sample, start_sample_sig)
                overlap_end_abs = min(win_end_sample, end_sample_sig)

                # If there is an overlap
                if overlap_start_abs < overlap_end_abs:
                    # Convert overlap boundaries to indices relative to the segment
                    overlap_start_rel = overlap_start_abs - start_sample_sig
                    overlap_end_rel = overlap_end_abs - start_sample_sig
                    # Clip relative indices to be within the segment bounds
                    start_idx_rel = np.clip(overlap_start_rel, 0, num_samples_in_segment - 1)
                    end_idx_rel = np.clip(overlap_end_rel, 0, num_samples_in_segment) # End index is exclusive

                    # Assign the prediction value to the corresponding part of the mapped array
                    if start_idx_rel < end_idx_rel: # Ensure start < end
                        predictions_mapped[start_idx_rel:end_idx_rel] = win_pred
                        plot_predictions = True # Mark that we have predictions to plot
        else:
            log.warning(f"{log_prefix} Length mismatch between window_start_samples ({len(window_start_samples)}) and window_predictions ({len(window_predictions)}). Cannot map predictions.")
    else:
        log.info(f"{log_prefix} No prediction data (window starts or predictions) provided.")

    # --- Plotting ---
    plot_title = f'Predictions vs True Labels (S{subject_id}) | Signal: {signal_key} ({signal_source or "Unknown"}) | {actual_start_sec:.1f}s - {actual_end_sec:.1f}s'
    ax.set_title(plot_title, fontsize=10)

    # Plot background signal (e.g., ECG or EDA)
    line_signal, = ax.plot(time_vector, segment_signal, color='grey', alpha=0.8, label=f'{signal_key} Signal', linewidth=1.0, zorder=1)
    ax.set_ylabel(f'{signal_key} Amplitude')
    ax.set_xlabel('Time (seconds)')
    y_min_sig, y_max_sig = ax.get_ylim() # Get y-limits for shading

    # Plot True Stress Labels (shaded background)
    handle_true_stress = None
    if segment_labels_true_signal is not None:
        where_stress_true = segment_labels_true_signal == 1
        if np.any(where_stress_true):
            ax.fill_between(time_vector, y_min_sig, y_max_sig, where=where_stress_true, color='orange', alpha=0.3, step='mid', label='True Stress', zorder=2)
            handle_true_stress = mpatches.Patch(color='orange', alpha=0.3, label='True Stress')

    # Plot Predictions (on secondary y-axis)
    handle_pred_stress = None
    ax_pred = None # Initialize secondary axis
    if plot_predictions:
        ax_pred = ax.twinx() # Create a twin Axes sharing the xaxis
        valid_preds_mask = ~np.isnan(predictions_mapped) # Mask for non-NaN predictions
        if np.any(valid_preds_mask):
            # Use step plot for predictions
            line_pred, = ax_pred.step(time_vector[valid_preds_mask], predictions_mapped[valid_preds_mask], where='mid', color='red', linestyle='--', linewidth=1.5, label='Predicted Stress', zorder=3)
            handle_pred_stress = line_pred # Use line object for legend
        else:
            log.warning(f"{log_prefix} No valid predictions mapped in the selected time range.")
            handle_pred_stress = None # Ensure handle is None if no valid preds plotted

        # Configure secondary y-axis for predictions
        ax_pred.set_ylabel('Prediction (0=Non-Stress, 1=Stress)', color='red')
        ax_pred.tick_params(axis='y', labelcolor='red')
        ax_pred.set_yticks([0, 1])
        ax_pred.set_yticklabels(['Non-Stress', 'Stress'])
        ax_pred.set_ylim(-0.1, 1.1) # Set limits slightly outside 0-1

    # --- Legend and Final Touches ---
    # Collect handles for the legend
    handles = [line_signal]
    if handle_true_stress: handles.append(handle_true_stress)
    if handle_pred_stress: handles.append(handle_pred_stress) # Add only if valid

    # Create legend using collected handles
    labels = [h.get_label() for h in handles if h is not None]
    ax.legend(handles, labels, loc='upper right')

    # Remove duplicate legend potentially created by twinx
    if ax_pred and ax_pred.get_legend():
        ax_pred.get_legend().remove()

    ax.grid(True, linestyle=':', axis='x') # Grid only on x-axis
    ax.set_xlim(actual_start_sec, actual_end_sec) # Set x-limits
    ax.set_ylim(y_min_sig, y_max_sig) # Restore original signal y-limits

    if fig_created:
        plt.tight_layout()
        plt.show()


def plot_training_history(history: Dict, results_dir: str):
     """
     Plots training and validation loss and metrics (F1 or Accuracy) from training history.

     Args:
         history (Dict): Dictionary containing lists of metrics per epoch
                         (e.g., history['train_loss'], history['val_loss'], history['val_f1']).
         results_dir (str): Directory to save the plot.
     """
     log.info("Plotting training history...")
     try:
         # Determine the number of epochs from the length of the training loss list
         epochs = range(1, len(history.get('train_loss', [])) + 1)
         if not epochs:
             log.warning("No training history data found in 'history' dictionary. Cannot plot.")
             return

         # Create figure with two subplots
         fig, axs = plt.subplots(1, 2, figsize=(15, 5))

         # --- Plot Loss ---
         axs0_lines = [] # To store lines for legend
         train_loss_data = history.get('train_loss', [])
         val_loss_data = history.get('val_loss', [])

         # Plot training loss if data exists and is valid
         if train_loss_data and any(v is not None and np.isfinite(v) for v in train_loss_data):
             line_train, = axs[0].plot(epochs, train_loss_data, 'bo-', label='Training Loss')
             axs0_lines.append(line_train)
         # Plot validation loss if data exists and is valid
         if val_loss_data and any(v is not None and np.isfinite(v) for v in val_loss_data):
             line_val, = axs[0].plot(epochs, val_loss_data, 'ro--', label='Validation Loss')
             axs0_lines.append(line_val)

         axs[0].set_title('Training and Validation Loss')
         axs[0].set_xlabel('Epochs')
         axs[0].set_ylabel('Loss')
         if axs0_lines: axs[0].legend(handles=axs0_lines) # Add legend only if lines were plotted
         axs[0].grid(True)

         # --- Plot Metrics (Prefer F1, fallback to Accuracy) ---
         metric_key_f1 = 'val_f1'
         metric_key_acc = 'val_accuracy'
         metric_label = 'Validation Metric'
         metric_data = None

         # Check for F1 score first
         val_f1_data = history.get(metric_key_f1, [])
         if val_f1_data and any(v is not None and np.isfinite(v) for v in val_f1_data):
             metric_data = val_f1_data
             metric_label = 'Validation F1-Score'
         else:
             # Fallback to Accuracy if F1 is not available/valid
             val_acc_data = history.get(metric_key_acc, [])
             if val_acc_data and any(v is not None and np.isfinite(v) for v in val_acc_data):
                 metric_data = val_acc_data
                 metric_label = 'Validation Accuracy'

         # Plot the selected metric if data is available
         if metric_data:
             axs[1].plot(epochs, metric_data, 'go--', label=metric_label)
             axs[1].set_title(metric_label)
             axs[1].set_xlabel('Epochs')
             axs[1].set_ylabel(metric_label.split()[-1]) # Use 'F1-Score' or 'Accuracy' as label
             axs[1].legend()
             axs[1].grid(True)
         else:
             # Display message if no valid validation metrics were found
             axs[1].set_title("No Validation Metrics Logged")
             axs[1].text(0.5, 0.5, "Validation metrics missing or invalid", ha='center', va='center', transform=axs[1].transAxes)

         # --- Save Plot ---
         plt.tight_layout()
         # Ensure results directory exists
         os.makedirs(results_dir, exist_ok=True)
         plot_path = os.path.join(results_dir, "training_history.png")
         plt.savefig(plot_path)
         log.info(f"Training history plot saved to {plot_path}")
         plt.close(fig) # Close the figure after saving to free memory

     except Exception as e:
         log.error(f"Failed to plot training history: {e}", exc_info=True)
         # Ensure figure is closed even on error if it was created
         if 'fig' in locals() and plt.fignum_exists(fig.number):
              plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: str,
    set_name: str = "Evaluation"
) -> None:
    """
    Calculates and plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (np.ndarray): Numpy array of true binary labels (0 or 1).
        y_probs (np.ndarray): Numpy array of predicted probabilities for the positive class (1).
        output_dir (str): Directory to save the plot.
        set_name (str, optional): Name of the dataset (e.g., "Validation", "Test") for title/filename.
    """
    log.info(f"Plotting ROC curve for {set_name} set...")
    # Input validation
    if y_true is None or y_probs is None or len(y_true) != len(y_probs) or len(y_true) == 0:
        log.error(f"Invalid input for ROC curve plotting ({set_name}). Labels: {len(y_true) if y_true is not None else 'None'}, Probs: {len(y_probs) if y_probs is not None else 'None'}")
        return
    # ROC AUC is not defined for single-class data
    if len(np.unique(y_true)) < 2:
        log.warning(f"Only one class present in true labels for {set_name}. ROC AUC is not defined. Skipping ROC plot.")
        return

    try:
        # Calculate ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
        # Calculate Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)

        # --- Plotting ---
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})') # Show AUC in legend
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill (AUC = 0.5)') # Baseline
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05]) # Slightly above 1.0 for visibility
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity/Recall)')
        plt.title(f'{set_name} Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':')
        plt.tight_layout()

        # --- Save Plot ---
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        plot_filename = f"roc_curve_{set_name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        log.info(f"{set_name} ROC curve plot saved to {plot_path} (AUC: {roc_auc:.4f})")
        plt.close() # Close the figure context

    except Exception as e:
        log.error(f"Failed to plot ROC curve for {set_name}: {e}", exc_info=True)
        # Ensure plot is closed on error
        if plt.get_fignums(): plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: str,
    set_name: str = "Evaluation"
) -> None:
    """
    Calculates and plots the Precision-Recall (PR) curve.

    Args:
        y_true (np.ndarray): Numpy array of true binary labels (0 or 1).
        y_probs (np.ndarray): Numpy array of predicted probabilities for the positive class (1).
        output_dir (str): Directory to save the plot.
        set_name (str, optional): Name of the dataset (e.g., "Validation", "Test") for title/filename.
    """
    log.info(f"Plotting Precision-Recall curve for {set_name} set...")
    # Input validation
    if y_true is None or y_probs is None or len(y_true) != len(y_probs) or len(y_true) == 0:
        log.error(f"Invalid input for PR curve plotting ({set_name}). Labels: {len(y_true) if y_true is not None else 'None'}, Probs: {len(y_probs) if y_probs is not None else 'None'}")
        return

    try:
        # Calculate Precision-Recall curve points
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs, pos_label=1)
        # Calculate Average Precision (AP), which summarizes the PR curve
        avg_precision = average_precision_score(y_true, y_probs, pos_label=1)
        # Calculate no-skill line (baseline precision = proportion of positive samples)
        no_skill = len(y_true[y_true==1]) / len(y_true) if len(y_true) > 0 else 0

        # --- Plotting ---
        plt.figure(figsize=(7, 7))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})') # Show AP in legend
        plt.plot([0, 1], [no_skill, no_skill], color='grey', lw=2, linestyle='--', label=f'No Skill (AP = {no_skill:.3f})') # Baseline
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05]) # Slightly above 1.0 for visibility
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title(f'{set_name} Precision-Recall Curve')
        plt.legend(loc="lower left") # Often better location for PR curves
        plt.grid(True, linestyle=':')
        plt.tight_layout()

        # --- Save Plot ---
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        plot_filename = f"precision_recall_curve_{set_name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        log.info(f"{set_name} Precision-Recall curve plot saved to {plot_path} (Avg Precision: {avg_precision:.4f})")
        plt.close() # Close the figure context

    except Exception as e:
        log.error(f"Failed to plot Precision-Recall curve for {set_name}: {e}", exc_info=True)
        # Ensure plot is closed on error
        if plt.get_fignums(): plt.close()
