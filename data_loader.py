# data_loader.py (Loads raw data from different sources)
import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

# Assuming utils.py is in the same directory or PYTHONPATH
try:
    from utils import safe_get, get_sampling_rate
except ImportError:
    # Basic fallback implementations if utils cannot be imported
    def safe_get(data_dict, keys, default=None): temp = data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    def get_sampling_rate(config, signal_key, source, dataset_id="WESAD"):
         rate_key = f"{dataset_id}_{source}_{signal_key}".upper()
         possible_keys = [ f"{dataset_id.upper()}_{source.lower()}_{signal_key.lower()}", f"{dataset_id.upper()}_{source.lower()}_{signal_key.upper()}", f"{dataset_id.upper()}_{source.upper()}_{signal_key.lower()}", f"{dataset_id.upper()}_{source.upper()}_{signal_key.upper()}", ]
         fs = None; sampling_rates_dict = safe_get(config, ['sampling_rates'], {})
         fs = next((sampling_rates_dict.get(key) for key in possible_keys if sampling_rates_dict.get(key) is not None), None)
         return fs if fs is not None and isinstance(fs, (int, float)) and fs > 0 else None
    logging.warning("Could not import from 'utils'. Using basic fallbacks for safe_get and get_sampling_rate.")

log = logging.getLogger(__name__)

# ==============================================================================
# == WESAD Data Loading ==
# ==============================================================================

def load_wesad_subject_data(subject_id: int, wesad_config: Dict[str, Any], main_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Loads WESAD .pkl data for a single subject.

    Args:
        subject_id (int): The ID of the subject (e.g., 2, 3, ...).
        wesad_config (Dict[str, Any]): The specific config section for WESAD
                                       (from main_config['datasets']['WESAD']).
        main_config (Dict[str, Any]): The main configuration dictionary (used indirectly
                                      via helpers like get_sampling_rate if needed later).

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the subject's raw data,
                                  including signals, labels, and metadata, or None on failure.
                                  Adds 'dataset_id' and ensures 'subject' key exists.
    """
    log_prefix = f"[WESAD S{subject_id}]" # Prefix for log messages

    # Extract necessary paths and file naming conventions from the WESAD config
    data_path = wesad_config.get('path')
    prefix = wesad_config.get('file_prefix', 'S') # Default prefix 'S'
    suffix = wesad_config.get('file_suffix', '.pkl') # Default suffix '.pkl'

    # Validate the data path
    if not data_path:
        log.error(f"{log_prefix} `path` not specified in WESAD config section.")
        return None

    # Construct the full path to the subject's pickle file
    subject_dir = f"{prefix}{subject_id}"
    file_name = f"{subject_dir}{suffix}"
    file_path = os.path.join(data_path, subject_dir, file_name)
    abs_file_path = os.path.abspath(file_path) # Get absolute path for clarity

    log.info(f"{log_prefix} Attempting to load file: {abs_file_path}")
    try:
        # Open the pickle file for reading in binary mode ('rb')
        with open(abs_file_path, 'rb') as f:
            # Load the data using pickle, specifying encoding for Python 3 compatibility
            # 'latin1' is often needed for older pickle files (like WESAD)
            data = pickle.load(f, encoding='latin1')
        log.info(f"{log_prefix} Successfully loaded raw data structure.")

        # --- Post-loading processing ---
        # Add a dataset identifier for downstream processing
        data['dataset_id'] = 'WESAD'
        # Add subject ID if not present (WESAD structure usually includes it)
        if 'subject' not in data:
            data['subject'] = str(subject_id) # Store as string for consistency

        # --- Add original sampling rates to the raw data structure ---
        # This makes them easily accessible later without needing the config again
        data['sampling_rates'] = {}
        # Iterate through signals expected in WESAD data
        for source in ['chest', 'wrist']:
             if source in data.get('signal', {}):
                 for signal_key in data['signal'][source].keys():
                     # Use the utility function to get the rate from config
                     fs = get_sampling_rate(main_config, signal_key, source, dataset_id='WESAD')
                     if fs:
                         data['sampling_rates'][signal_key] = fs
                         log.debug(f"{log_prefix} Added original Fs for {source}/{signal_key}: {fs} Hz")
                     else:
                         log.warning(f"{log_prefix} Could not find original Fs for {source}/{signal_key} in config.")
        # Add label sampling rate
        label_fs = get_sampling_rate(main_config, 'label', 'label', dataset_id='WESAD')
        if label_fs: data['sampling_rates']['label'] = label_fs
        else: log.warning(f"{log_prefix} Could not find original Fs for label in config.")

        return data

    except FileNotFoundError:
        log.error(f"{log_prefix} Data file not found at {abs_file_path}")
        return None
    except pickle.UnpicklingError as e:
        log.error(f"{log_prefix} Error unpickling data: {e}")
        return None
    except Exception as e:
        log.error(f"{log_prefix} Unexpected error loading raw data: {e}", exc_info=True)
        return None

# ==============================================================================
# == Nurse Dataset Loading ==
# ==============================================================================

def _parse_custom_time(time_str: str) -> Optional[float]:
    """
    Parses custom time strings like 'MM:SS.f' or potentially 'HH:MM:SS.f'
    into total seconds as a float. Returns None if parsing fails.
    Handles potential errors like the '30:00.0' case by returning None.

    Args:
        time_str (str): The time string to parse.

    Returns:
        Optional[float]: Total seconds as a float, or None if parsing failed.
    """
    if not isinstance(time_str, str):
        return None # Input must be a string
    try:
        parts = time_str.split(':')
        if len(parts) == 2: # Assume MM:SS.f format
            minutes_str, seconds_full = parts
            # Split seconds and fractional part (tenths)
            sec_parts = seconds_full.split('.')
            seconds_str = sec_parts[0]
            tenths_str = sec_parts[1] if len(sec_parts) > 1 else '0'

            # Convert parts to integers
            minutes = int(minutes_str)
            seconds = int(seconds_str)
            tenths = int(tenths_str)

            # Basic validation (e.g., seconds < 60, tenths < 10)
            # This handles cases like '30:00.0' where seconds might be 60 if not validated
            if not (0 <= seconds < 60 and 0 <= tenths < 10):
                 # Log the specific invalid value found (optional, can be noisy)
                 # log.warning(f"Invalid time component found: {time_str}")
                 return None # Invalid component

            # Calculate total seconds
            total_seconds = minutes * 60 + seconds + tenths / 10.0
            return total_seconds

        elif len(parts) == 3: # Assume HH:MM:SS.f format
            hours_str, minutes_str, seconds_full = parts
            sec_parts = seconds_full.split('.')
            seconds_str = sec_parts[0]
            tenths_str = sec_parts[1] if len(sec_parts) > 1 else '0'

            hours = int(hours_str)
            minutes = int(minutes_str)
            seconds = int(seconds_str)
            tenths = int(tenths_str)

            # Basic validation
            if not (0 <= minutes < 60 and 0 <= seconds < 60 and 0 <= tenths < 10):
                 # log.warning(f"Invalid time component found: {time_str}")
                 return None # Invalid component

            total_seconds = hours * 3600 + minutes * 60 + seconds + tenths / 10.0
            return total_seconds

        else:
            # Log unexpected format (optional)
            # log.warning(f"Unexpected time format encountered: {time_str}")
            return None # Unexpected format
    except (ValueError, IndexError, TypeError):
        # Log parsing failure for the specific string (optional)
        # log.warning(f"Failed to parse time string: {time_str}")
        return None # Error during parsing/conversion

def load_nurse_dataset(nurse_config: Dict[str, Any], main_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Loads and preprocesses data from the nurse stress CSV dataset.
    Groups data by subject ID ('id' column). Maps columns and labels according to config.
    Handles custom 'MM:SS.f' datetime format by converting to Timedelta.

    Args:
        nurse_config (Dict[str, Any]): The specific config section for NURSE
                                      (from main_config['datasets']['NURSE']).
        main_config (Dict[str, Any]): The main configuration dictionary.

    Returns:
        Optional[Dict[str, Any]]: A dictionary where keys are subject IDs (e.g., "NURSE_1")
                                  and values are dictionaries containing the subject's
                                  processed data ('signal', 'label', 'sampling_rates', etc.),
                                  or None on failure.
    """
    log_prefix = "[Nurse Dataset]"
    file_path = nurse_config.get('path')
    if not file_path:
        log.error(f"{log_prefix} `path` not specified in NURSE config section.")
        return None

    abs_file_path = os.path.abspath(file_path)
    if not os.path.exists(abs_file_path):
        log.error(f"{log_prefix} Data file not found at {abs_file_path}")
        return None

    log.info(f"{log_prefix} Attempting to load nurse data from: {abs_file_path}")
    try:
        # Load CSV, handle potential DtypeWarning if columns have mixed types
        # low_memory=False can help prevent DtypeWarning but uses more memory
        df = pd.read_csv(abs_file_path, low_memory=False)
        log.info(f"{log_prefix} Loaded nurse CSV with shape {df.shape}. Columns: {df.columns.tolist()}")

        # --- Check for required columns ---
        required_cols = ['id', 'datetime', 'EDA', 'HR', 'TEMP', 'X', 'Y', 'Z', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log.error(f"{log_prefix} CSV missing required columns: {missing_cols}.")
            return None

        # --- Parse custom time format to Timedelta ---
        original_rows = len(df)
        log.info(f"{log_prefix} Parsing 'datetime' column (format like MM:SS.f)...")
        # Apply the custom parser function to get total seconds
        df['total_seconds'] = df['datetime'].apply(_parse_custom_time)

        # Drop rows where custom parsing failed (returned None)
        rows_failed_parsing = df['total_seconds'].isna().sum()
        if rows_failed_parsing > 0:
            log.warning(f"{log_prefix} Found {rows_failed_parsing} rows with unparseable datetime format (e.g., '30:00.0' or other issues). Dropping them.")
            df = df.dropna(subset=['total_seconds'])
            log.info(f"{log_prefix} Shape after dropping unparseable times: {df.shape}")
        if df.empty:
            log.error(f"{log_prefix} DataFrame is empty after dropping rows with unparseable datetime. Cannot proceed.")
            return None

        # Convert total seconds to Timedelta for sorting and potential time-based analysis
        df['datetime_td'] = pd.to_timedelta(df['total_seconds'], unit='s')
        log.info(f"{log_prefix} Converted 'total_seconds' to Timedelta ('datetime_td').")

        # Sort by subject ID and the new Timedelta column to ensure chronological order per subject
        df = df.sort_values(by=['id', 'datetime_td'])

        # --- Get Assumed Original Sampling Rates from Config ---
        # Uses get_sampling_rate helper which understands the config structure
        assumed_fs = {
            'EDA': get_sampling_rate(main_config, 'eda', 'wrist', dataset_id='NURSE') or 4.0, # Default 4Hz if not found
            'HR': get_sampling_rate(main_config, 'hr', 'wrist', dataset_id='NURSE') or 1.0, # Default 1Hz if not found
            'TEMP': get_sampling_rate(main_config, 'temp', 'wrist', dataset_id='NURSE') or 4.0, # Default 4Hz
            'XYZ': get_sampling_rate(main_config, 'xyz', 'wrist', dataset_id='NURSE') or 32.0 # Default 32Hz
        }
        # Check if any rate is missing/invalid
        if None in assumed_fs.values() or any(fs <= 0 for fs in assumed_fs.values()):
             log.error(f"{log_prefix} Invalid or missing original sampling rates in config: {assumed_fs}. Check config['sampling_rates'] section for NURSE.")
             # Decide whether to abort or proceed with defaults (already set above)
             # return None # Abort if rates are critical for subsequent processing

        log.warning(f"{log_prefix} Using original sampling rates from config: {assumed_fs}. VERIFY THESE RATES ARE CORRECT for the Nurse dataset.")

        # --- Get Label Mapping from main_config ---
        label_map_config = safe_get(main_config, ['label_mapping'], {})
        # Create map from WESAD label name (e.g., 'stress') -> WESAD numeric ID (e.g., 2)
        name_to_id_map = {v: int(k) for k, v in label_map_config.items() if isinstance(v, str) and k.isdigit()}
        # Create map from NURSE CSV label value (e.g., 'Work') -> WESAD numeric ID (e.g., 2)
        nurse_csv_label_to_numeric_id = {}
        # Find the target WESAD name for each special nurse config key
        nurse_baseline_target = label_map_config.get("nurse_baseline_label_value")
        nurse_stress_target = label_map_config.get("nurse_stress_label_value")
        nurse_other_target = label_map_config.get("nurse_other_label_value")

        # Map the *values* from the CSV 'label' column to the target WESAD numeric IDs
        # This requires knowing the actual string values present in the 'label' column of the CSV
        # Example: If CSV contains 'Rest', 'Work', 'Other'
        # We need to map 'Rest' -> ID for baseline, 'Work' -> ID for stress, 'Other' -> ID for transient
        # Assuming the CSV contains labels like 'baseline', 'stress', 'transient' directly:
        if nurse_baseline_target in name_to_id_map:
             nurse_csv_label_to_numeric_id['baseline'] = name_to_id_map[nurse_baseline_target]
        if nurse_stress_target in name_to_id_map:
             nurse_csv_label_to_numeric_id['stress'] = name_to_id_map[nurse_stress_target]
        if nurse_other_target in name_to_id_map:
             nurse_csv_label_to_numeric_id['transient'] = name_to_id_map[nurse_other_target]
             # Map any other potential values in the CSV label column to transient as well
             # You might need to inspect the unique values in df['label'] to be comprehensive
             # Example: nurse_csv_label_to_numeric_id['Break'] = name_to_id_map[nurse_other_target]

        # Find the ID for 'transient' or default to 0
        transient_id = name_to_id_map.get('transient', 0)
        log.warning(f"{log_prefix} Using label mapping CSV->Numeric: {nurse_csv_label_to_numeric_id}. Unknown labels map to {transient_id}.")
        if not nurse_csv_label_to_numeric_id:
             log.error(f"{log_prefix} Label mapping failed. Check 'label_mapping' in config and CSV 'label' column values.")
             return None

        # --- Group data by subject ---
        grouped_data_by_subject = {}
        processed_subject_count = 0
        subjects_to_load_nurse = nurse_config.get('subjects') # Get specific list if provided

        if df.empty: # Double check df is not empty before grouping
             log.error(f"{log_prefix} DataFrame became empty, cannot group by subject.")
             return None

        for subject_id, group_df in df.groupby('id'):
            # If specific subjects are listed in the config, only process those
            if subjects_to_load_nurse is not None and subject_id not in subjects_to_load_nurse:
                continue

            log.debug(f"{log_prefix} Processing subject ID: {subject_id}")
            group_df = group_df.copy() # Avoid SettingWithCopyWarning

            # Prepare signal dictionary for this subject
            subject_signals = {'wrist': {}} # Nurse data is assumed to be from wrist
            subject_rates = {} # Store original rates used for this subject

            # Extract signals and map to standard keys (e.g., HR -> BVP, XYZ -> ACC)
            if 'EDA' in group_df.columns:
                # Ensure data is float, handle potential non-numeric entries
                eda_vals = pd.to_numeric(group_df['EDA'], errors='coerce').values
                if np.isnan(eda_vals).any(): log.warning(f"{log_prefix} S{subject_id}: NaNs found in EDA after coercion.")
                subject_signals['wrist']['EDA'] = eda_vals
                subject_rates['EDA'] = assumed_fs['EDA']
            if 'HR' in group_df.columns:
                hr_vals = pd.to_numeric(group_df['HR'], errors='coerce').values
                if np.isnan(hr_vals).any(): log.warning(f"{log_prefix} S{subject_id}: NaNs found in HR after coercion.")
                subject_signals['wrist']['BVP'] = hr_vals # Map HR to BVP
                subject_rates['BVP'] = assumed_fs['HR']
            if 'TEMP' in group_df.columns:
                temp_vals = pd.to_numeric(group_df['TEMP'], errors='coerce').values
                if np.isnan(temp_vals).any(): log.warning(f"{log_prefix} S{subject_id}: NaNs found in TEMP after coercion.")
                subject_signals['wrist']['TEMP'] = temp_vals
                subject_rates['TEMP'] = assumed_fs['TEMP']
            if 'X' in group_df.columns and 'Y' in group_df.columns and 'Z' in group_df.columns:
                # Ensure ACC columns are numeric
                x_vals = pd.to_numeric(group_df['X'], errors='coerce')
                y_vals = pd.to_numeric(group_df['Y'], errors='coerce')
                z_vals = pd.to_numeric(group_df['Z'], errors='coerce')
                acc_vals = np.stack([x_vals, y_vals, z_vals], axis=-1)
                if np.isnan(acc_vals).any(): log.warning(f"{log_prefix} S{subject_id}: NaNs found in ACC (X/Y/Z) after coercion.")
                subject_signals['wrist']['ACC'] = acc_vals # Combine X,Y,Z into one array
                subject_rates['ACC'] = assumed_fs['XYZ']

            # Map labels using the derived map
            try:
                 # Map the string labels from the CSV to the numeric IDs
                 # Fill any labels not in the map with the transient ID
                 mapped_labels = group_df['label'].map(nurse_csv_label_to_numeric_id).fillna(transient_id)
                 labels_numeric = mapped_labels.values.astype(int)
            except Exception as label_e:
                log.error(f"{log_prefix} Failed to map labels for subject {subject_id}: {label_e}. Check CSV 'label' values and config. Skipping subject.")
                continue

            # Create a unique subject ID string to avoid collision with WESAD IDs
            nurse_subj_id_str = f"NURSE_{subject_id}"
            # Store the processed data for this subject
            grouped_data_by_subject[nurse_subj_id_str] = {
                'signal': subject_signals,
                'label': labels_numeric,
                'sampling_rates': subject_rates, # Store original rates used
                'dataset_id': 'NURSE', # Add dataset identifier
                'subject': nurse_subj_id_str # Store prefixed subject ID
            }
            processed_subject_count += 1

        log.info(f"{log_prefix} Processed {processed_subject_count} subjects from nurse dataset.")
        return grouped_data_by_subject

    except pd.errors.EmptyDataError:
        log.error(f"{log_prefix} CSV file is empty: {abs_file_path}")
        return None
    except Exception as e:
        log.error(f"{log_prefix} Unexpected error loading/processing nurse data: {e}", exc_info=True)
        return None


# ==============================================================================
# == Combined Data Loading ==
# ==============================================================================

def load_all_datasets(config: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """
    Loads data for all datasets specified and enabled in config['datasets'].
    Calls the appropriate loading function based on dataset ID and type.

    Args:
        config (Dict[str, Any]): The main configuration dictionary.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]: A tuple containing:
            - all_datasets_data: Dictionary mapping subject ID (str) to their loaded raw data dict.
                                 Subject IDs are strings (e.g., "2", "NURSE_1").
            - subjects_loaded_ids: List of subject IDs successfully loaded.
            - subjects_failed_ids: List of subject/dataset IDs that failed to load.
    """
    all_datasets_data = {} # Dictionary to hold data for all subjects from all datasets
    subjects_loaded_ids = [] # List to track successfully loaded subject IDs
    subjects_failed_ids = [] # List to track failed loads

    # Validate config
    if not config:
        log.error("Configuration dictionary is missing.")
        return {}, [], []

    # Get the 'datasets' section from the config
    datasets_config = safe_get(config, ['datasets'], {})
    if not datasets_config:
        log.error("No 'datasets' section found in config.")
        return {}, [], []

    log.info(f"--- Starting Raw Data Loading ---")

    # Iterate through each dataset defined in the config
    for dataset_id, dataset_config in datasets_config.items():
        # Check if dataset_config is a valid dictionary and the 'load' flag is true
        if not isinstance(dataset_config, dict) or not dataset_config.get('load', False):
            log.info(f"Skipping dataset '{dataset_id}' (load flag is false or config invalid).")
            continue

        log.info(f"--- Loading Dataset: {dataset_id} ---")
        # Infer file type or get from config (default to pkl for WESAD if not specified)
        dataset_type = dataset_config.get('file_type', 'pkl' if dataset_id == 'WESAD' else None)

        # --- WESAD Loading Logic ---
        if dataset_id == 'WESAD' and dataset_type == 'pkl':
            subjects_to_load = dataset_config.get('subjects', [])
            if not subjects_to_load:
                log.warning(f"No subjects specified for WESAD in config.")
                continue
            log.info(f"Attempting to load {len(subjects_to_load)} WESAD subjects: {subjects_to_load}")
            for subject_id in subjects_to_load:
                try:
                    # WESAD IDs are typically integers
                    subj_id_int = int(subject_id)
                except (ValueError, TypeError):
                    log.warning(f"Invalid WESAD subject ID '{subject_id}'. Skipping.")
                    subjects_failed_ids.append(f"WESAD_{subject_id}")
                    continue

                # Load data for a single WESAD subject using the dedicated function
                subject_data = load_wesad_subject_data(subj_id_int, dataset_config, config)
                if subject_data:
                    # Use original WESAD ID as key, but convert to string for consistency
                    subj_key = str(subj_id_int)
                    all_datasets_data[subj_key] = subject_data
                    subjects_loaded_ids.append(subj_key)
                else:
                    # Log failure for this specific subject
                    subjects_failed_ids.append(f"WESAD_{subj_id_int}")

        # --- NURSE Loading Logic ---
        elif dataset_id == 'NURSE' and dataset_type == 'csv':
             # Load the entire nurse dataset (grouped by subject internally)
             nurse_data_dict = load_nurse_dataset(dataset_config, config)
             if nurse_data_dict:
                 # Update the main dictionary with data for each nurse subject
                 all_datasets_data.update(nurse_data_dict)
                 # Keys in nurse_data_dict are already prefixed (e.g., "NURSE_1")
                 loaded_nurse_ids = list(nurse_data_dict.keys())
                 subjects_loaded_ids.extend(loaded_nurse_ids)
                 log.info(f"Successfully loaded data for {len(loaded_nurse_ids)} nurse subjects.")
             else:
                 log.error("Failed to load Nurse dataset.")
                 subjects_failed_ids.append(f"{dataset_id}_LOAD_FAILED") # Mark dataset failure

        # --- Add elif blocks here for other dataset types ---
        # elif dataset_id == 'OTHER_DATASET' and dataset_type == 'some_format':
        #    other_data = load_other_dataset(...) # Implement load_other_dataset
        #    if other_data:
        #        all_datasets_data.update(other_data)
        #        subjects_loaded_ids.extend(list(other_data.keys()))
        #    else:
        #        subjects_failed_ids.append(f"{dataset_id}_LOAD_FAILED")

        else:
            log.warning(f"Unsupported dataset ID '{dataset_id}' or file_type '{dataset_type}'. Skipping.")

        log.info(f"--- Finished Loading {dataset_id} ---")

    # --- Final Summary ---
    log.info("--- All Raw Data Loading Finished ---")
    log.info(f"Successfully loaded data for {len(subjects_loaded_ids)} total subjects: {sorted(subjects_loaded_ids)}")
    if subjects_failed_ids:
        log.warning(f"Failed to load data for {len(subjects_failed_ids)} subjects/datasets: {sorted(subjects_failed_ids)}")
    else:
        log.info("All specified subjects/datasets loaded successfully.")
    if not all_datasets_data:
        log.critical("No subject data was loaded successfully from any dataset.")

    # Return the combined data, list of loaded IDs, and list of failed IDs
    return all_datasets_data, subjects_loaded_ids, subjects_failed_ids
