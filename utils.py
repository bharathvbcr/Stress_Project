# utils.py (Updated with clear_files option in load_preprocessed_data)

import json
import os
import pprint # For pretty printing dictionaries (optional)
import logging
from typing import Dict, Any, Optional, List, Union, Tuple

import torch

# Import joblib safely for loading/saving processed data
try:
    import joblib
except ImportError:
    joblib = None
    logging.warning("Joblib library not found. Loading/saving .joblib files will fail.")

# Setup basic logging if not already done by the main script
log = logging.getLogger(__name__)
# Example: Configure logging if running this file directly or if needed elsewhere
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

# --- Configuration Loading ---
def load_config(config_path: str = 'config.json') -> Optional[Dict[str, Any]]:
    """
    Loads configuration from a JSON file.

    Args:
        config_path (str): The path to the configuration JSON file.

    Returns:
        Optional[Dict[str, Any]]: The loaded configuration dictionary, or None on failure.
    """
    log.info(f"Attempting to load configuration from: {config_path}")
    abs_config_path = os.path.abspath(config_path) # Get absolute path

    # Check if the file exists
    if not os.path.exists(abs_config_path):
        log.error(f"Configuration file not found at {abs_config_path}")
        return None
    try:
        # Open and load the JSON file
        with open(abs_config_path, 'r') as f:
            config = json.load(f)
        log.info("Configuration loaded successfully.")

        # --- Validation and Directory Creation (Optional but Recommended) ---
        # Get the save_paths section, default to empty dict if not found
        save_paths = config.get('save_paths', {})
        if save_paths:
            log.info("Checking and creating save directories specified in config...")
            for key, path_str in save_paths.items():
                 # Ensure the path is a non-empty string
                 if isinstance(path_str, str) and path_str.strip():
                     try:
                         abs_path = os.path.abspath(path_str)
                         # Create directories recursively if they don't exist
                         os.makedirs(abs_path, exist_ok=True)
                         log.debug(f"Ensured directory exists: {abs_path} (for key '{key}')")
                     except OSError as e:
                         log.error(f"Error creating directory '{abs_path}' for key '{key}': {e}")
                     except Exception as e:
                          log.error(f"Unexpected error handling path '{path_str}' for key '{key}': {e}")
                 # else: log.warning(f"Invalid or empty path specified for save_paths key '{key}': {path_str}") # Optional warning
        else:
            log.warning("No 'save_paths' section found in config. Output saving might fail.")

        # Example: Validate features_to_use (ensure lists exist)
        if 'features_to_use' not in config:
            log.warning("'features_to_use' section missing in config.")
            config['features_to_use'] = {'chest': [], 'wrist': []} # Provide default empty lists
        elif not isinstance(config['features_to_use'], dict):
             log.error("'features_to_use' must be a dictionary. Using defaults.")
             config['features_to_use'] = {'chest': [], 'wrist': []}
        else:
             # Ensure 'chest' and 'wrist' keys exist within features_to_use
             if 'chest' not in config['features_to_use']: config['features_to_use']['chest'] = []
             if 'wrist' not in config['features_to_use']: config['features_to_use']['wrist'] = []

        # Add more validation as needed (e.g., check types, ranges for numerical values)

        return config
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON from {abs_config_path}: {e}")
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred loading config: {e}", exc_info=True)
        return None

# --- Sampling Rate Retrieval ---
def get_sampling_rate(config: Dict[str, Any], signal_key: str, source: str, dataset_id: str) -> Optional[float]:
    """
    Gets the original sampling rate for a specific signal from a specific dataset
    using the structured config. Handles different casing conventions.

    Args:
        config (Dict[str, Any]): The loaded configuration dictionary.
        signal_key (str): The key for the signal (e.g., 'ECG', 'eda').
        source (str): The source of the signal (e.g., 'chest', 'wrist').
        dataset_id (str): The identifier for the dataset (e.g., 'WESAD', 'NURSE').

    Returns:
        Optional[float]: The sampling rate as a float, or None if not found/invalid.
    """
    # Check if config and the sampling_rates section exist
    if not config or 'sampling_rates' not in config:
        log.warning(f"Cannot get sampling rate: 'sampling_rates' missing in config.")
        return None

    sampling_rates_dict = config['sampling_rates']
    # Prepare variations of keys for matching (case-insensitive)
    dataset_id_upper = dataset_id.upper()
    source_lower = source.lower()
    signal_key_lower = signal_key.lower()
    signal_key_upper = signal_key.upper() # Handle cases like ECG vs ecg

    # Construct potential keys based on common conventions
    # e.g., WESAD_wrist_bvp, WESAD_WRIST_BVP, WESAD_wrist_BVP, NURSE_wrist_hr
    possible_keys = [
        f"{dataset_id_upper}_{source_lower}_{signal_key_lower}",
        f"{dataset_id_upper}_{source_lower}_{signal_key_upper}",
        f"{dataset_id_upper}_{source.upper()}_{signal_key_lower}", # WESAD_WRIST_bvp
        f"{dataset_id_upper}_{source.upper()}_{signal_key_upper}", # WESAD_WRIST_BVP
        # Add other potential variations if needed (e.g., if source is sometimes capitalized differently)
    ]

    fs = None
    found_key = None
    # Iterate through possible keys to find a match
    for key in possible_keys:
        fs = sampling_rates_dict.get(key)
        if fs is not None:
            found_key = key
            break # Found a match

    # Validate the found sampling rate
    if fs is not None and isinstance(fs, (int, float)) and fs > 0:
        # log.debug(f"Found sampling rate for {dataset_id}/{source}/{signal_key}: {fs} Hz (using key '{found_key}')")
        return float(fs)
    else:
        # Log a warning if the rate wasn't found or was invalid
        # log.warning(f"Sampling rate not found or invalid for {dataset_id}/{source}/{signal_key}. Checked keys: {possible_keys}")
        return None


def setup_logging(level=logging.INFO):
    """Configures basic logging for the project."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Safe Dictionary Access ---
def safe_get(data_dict: Optional[Dict], keys: List[str], default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary. Returns default if any key is missing
    or if the intermediate value is not a dictionary.

    Args:
        data_dict (Optional[Dict]): The dictionary to access.
        keys (List[str]): A list of keys representing the path to the desired value.
        default (Any, optional): The value to return if the path is not found. Defaults to None.

    Returns:
        Any: The value found at the nested path, or the default value.
    """
    # Check if the initial input is a dictionary
    if not isinstance(data_dict, dict):
        return default

    temp_dict = data_dict
    # Iterate through the keys to traverse the dictionary
    for key in keys:
        try:
            # Check if the current level is still a dictionary
            if isinstance(temp_dict, dict):
                temp_dict = temp_dict[key] # Move to the next level
            else:
                # Intermediate value is not a dictionary, cannot go deeper
                return default
        except (KeyError, TypeError, IndexError): # Handle missing keys or invalid indexing
            # Key not found or trying to index a non-dict/non-list item
            return default
    # Return the final value if the path was fully traversed
    return temp_dict


def infer_model_feature_dims(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    Best-effort inference of sequence/static feature dimensions from config.

    This is used by inference and benchmark code paths that need to rebuild a
    model outside the training DataModule.
    """
    explicit_channels = safe_get(config, ['model_config', 'num_channels'])
    if not isinstance(explicit_channels, int) or explicit_channels <= 0:
        explicit_channels = safe_get(config, ['model', 'num_channels'])

    if isinstance(explicit_channels, int) and explicit_channels > 0:
        input_dim_sequence = explicit_channels
    else:
        features_to_use = safe_get(config, ['features_to_use'], None)
        if not isinstance(features_to_use, dict):
            features_to_use = safe_get(config, ['processing', 'features_to_use'], {})
        input_dim_sequence = sum(
            len(feature_list)
            for feature_list in features_to_use.values()
            if isinstance(feature_list, list)
        )
        if input_dim_sequence <= 0:
            input_dim_sequence = 8

    static_features = safe_get(config, ['static_features_to_use'], None)
    if not isinstance(static_features, list):
        static_features = safe_get(config, ['processing', 'static_features_to_use'], [])
    input_dim_static = len(static_features) if isinstance(static_features, list) else 0
    return input_dim_sequence, input_dim_static


def resolve_hf_path(config: Dict[str, Any]) -> str:
    """Return the configured Arrow/HF dataset cache path."""
    return safe_get(config, ['hf_path'], "./outputs/processed_data_hf")


def extract_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the wrapped model parameters from a Lightning checkpoint.
    Falls back to returning the provided mapping unchanged when it already
    resembles a raw model state_dict.
    """
    raw_state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in raw_state):
        return {
            key[len("model."):]: value
            for key, value in raw_state.items()
            if key.startswith("model.")
        }
    return raw_state


def get_runtime_capabilities() -> Dict[str, Any]:
    """Collect a compact runtime capability report for logging and health checks."""
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_available else False
    precision = "bf16-mixed" if bf16_supported else ("16-mixed" if cuda_available else "32")

    try:
        import tensorrt  # type: ignore  # noqa: F401
        tensorrt_available = True
    except ImportError:
        tensorrt_available = False

    try:
        import torch_tensorrt  # type: ignore  # noqa: F401
        torch_tensorrt_available = True
    except ImportError:
        torch_tensorrt_available = False

    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": str(device),
        "device_name": device_name,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "bf16_supported": bf16_supported,
        "precision": precision,
        "tensorrt_available": tensorrt_available,
        "torch_tensorrt_available": torch_tensorrt_available,
    }


def configure_torch_runtime(runtime_caps: Optional[Dict[str, Any]] = None, enable_cudnn_benchmark: bool = True) -> Dict[str, Any]:
    """Enable safe runtime optimizations when CUDA is available."""
    runtime_caps = runtime_caps or get_runtime_capabilities()
    if runtime_caps["cuda_available"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
    return runtime_caps


def log_runtime_capabilities(logger: logging.Logger, runtime_caps: Optional[Dict[str, Any]] = None, prefix: str = "Runtime") -> Dict[str, Any]:
    """Log a one-line runtime capability summary."""
    runtime_caps = runtime_caps or get_runtime_capabilities()
    logger.info(
        "%s: torch=%s cuda=%s device=%s device_name=%s precision=%s tensorrt=%s torch_tensorrt=%s",
        prefix,
        runtime_caps["torch_version"],
        runtime_caps["cuda_available"],
        runtime_caps["device"],
        runtime_caps["device_name"],
        runtime_caps["precision"],
        runtime_caps["tensorrt_available"],
        runtime_caps["torch_tensorrt_available"],
    )
    return runtime_caps

# --- Loading Preprocessed Data ---
def load_preprocessed_data(config: Dict[str, Any], clear_files: bool = False) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Loads the preprocessed data, static features, and R-peak results from .joblib files,
    OR clears (deletes) these files if clear_files is set to True.

    Args:
        config (Dict[str, Any]): The loaded configuration dictionary containing save_paths.
        clear_files (bool, optional): If True, delete the files instead of loading.
                                      Defaults to False.

    Returns:
        If clear_files is False:
            A tuple containing:
                - processed_data (Dict | None): Dictionary of processed signals and labels.
                - static_features_results (Dict | None): Dictionary of calculated static features.
                - r_peak_results (Dict | None): Dictionary of calculated R-peak indices.
        If clear_files is True:
            Returns (None, None, None) after attempting deletion.
    """
    if clear_files:
        log.info("--- Attempting to CLEAR preprocessed data .joblib files ---")
    else:
        log.info("--- Attempting to LOAD preprocessed data from .joblib files ---")

    processed_data = None
    static_features_results = None
    r_peak_results = None

    # Check if joblib is available (only needed for loading)
    if not clear_files and joblib is None:
        log.error("Joblib library not found. Cannot load .joblib files.")
        return None, None, None

    # Get paths from config using safe_get
    processed_dir = safe_get(config, ['save_paths', 'processed_data'])
    static_feat_dir = safe_get(config, ['save_paths', 'static_features_results'])
    # R-peaks are assumed to be saved in the same directory as static features
    r_peak_dir = static_feat_dir

    # Define file paths using os.path.join for cross-platform compatibility
    # Ensure paths are constructed only if the directory paths are valid strings
    proc_file_path = os.path.join(os.path.abspath(processed_dir), "processed_aligned_data.joblib") if processed_dir and isinstance(processed_dir, str) else None
    feat_file_path = os.path.join(os.path.abspath(static_feat_dir), "static_features_results.joblib") if static_feat_dir and isinstance(static_feat_dir, str) else None
    rpeak_file_path = os.path.join(os.path.abspath(r_peak_dir), "r_peak_indices.joblib") if r_peak_dir and isinstance(r_peak_dir, str) else None

    # Dictionary mapping descriptive names to file paths
    files_to_process = {
        "Processed Data": proc_file_path,
        "Static Features": feat_file_path,
        "R-Peak Indices": rpeak_file_path
    }

    # --- Conditional Logic: Clear or Load ---
    if clear_files:
        # --- Deletion Logic ---
        log.warning("Clear Files requested. Deleting saved processed data files...")
        for name, file_path in files_to_process.items():
            if file_path: # Check if path was resolved correctly
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path) # Attempt to delete the file
                        log.info(f"Successfully deleted {name} file: {file_path}")
                    except OSError as e:
                        log.error(f"Error deleting {name} file {file_path}: {e}", exc_info=True)
                    except Exception as e:
                         log.error(f"Unexpected error deleting {name} file {file_path}: {e}", exc_info=True)
                else:
                    # File doesn't exist, no need to delete
                    log.warning(f"{name} file not found at {file_path}. Cannot delete.")
            else:
                 # Path was not configured or invalid
                 log.warning(f"Path for {name} not configured or invalid. Skipping deletion.")
        return None, None, None # Return None after clearing attempt

    else:
        # --- Loading Logic ---
        # Load processed_data
        if proc_file_path:
            if os.path.exists(proc_file_path):
                try:
                    processed_data = joblib.load(proc_file_path)
                    log.info(f"Successfully loaded Processed Data from: {proc_file_path}")
                except Exception as e:
                    log.error(f"Failed to load Processed Data from {proc_file_path}: {e}", exc_info=True)
            else:
                # File doesn't exist
                log.warning(f"Processed Data file not found at: {proc_file_path}")
        else:
            # Path not configured
            log.warning("Path 'save_paths.processed_data' not found or invalid in config. Cannot load processed data.")

        # Load static_features_results
        if feat_file_path:
            if os.path.exists(feat_file_path):
                try:
                    static_features_results = joblib.load(feat_file_path)
                    log.info(f"Successfully loaded Static Features from: {feat_file_path}")
                except Exception as e:
                    log.error(f"Failed to load Static Features from {feat_file_path}: {e}", exc_info=True)
            else:
                log.warning(f"Static Features file not found at: {feat_file_path}")
        else:
            log.warning("Path 'save_paths.static_features_results' not found or invalid in config. Cannot load static features.")

        # Load r_peak_results
        if rpeak_file_path:
            if os.path.exists(rpeak_file_path):
                try:
                    r_peak_results = joblib.load(rpeak_file_path)
                    log.info(f"Successfully loaded R-Peak Indices from: {rpeak_file_path}")
                except Exception as e:
                    log.error(f"Failed to load R-Peak Indices from {rpeak_file_path}: {e}", exc_info=True)
            else:
                log.warning(f"R-Peak Indices file not found at: {rpeak_file_path}")
        else:
            log.warning("Path for R-peak indices ('save_paths.static_features_results') not found or invalid in config. Cannot load R-peaks.")

        # Return loaded data (can be None if loading failed or files didn't exist)
        return processed_data, static_features_results, r_peak_results
