# utils.py (Updated with clear_files option in load_preprocessed_data)

import glob
import json
import os
import pprint # For pretty printing dictionaries (optional)
import logging
import time
from functools import wraps
from typing import Dict, Any, Optional, List, Union, Tuple

import torch

from mlx_backend import (
    configure_mlx_runtime,
    get_requested_backend,
    is_mlx_available,
    mlx_version,
)

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

def timeit(func):
    """
    Decorator to measure and log the execution time of a function.
    Useful for profiling pre-processing and setup stages.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        log.info(f"[Profiler] {func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper

def check_gpu_features():
    """
    Checks and logs advanced CUDA features useful for stress detection training.
    """
    if not torch.cuda.is_available():
        return {"cuda": False}

    props = torch.cuda.get_device_properties(0)
    features = {
        "cuda": True,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "bf16": torch.cuda.is_bf16_supported(),
        "tf32": torch.cuda.get_tf32_threshold() > 0 or props.major >= 8,
        "fp16_native": props.major >= 7,
        "tensor_cores": props.major >= 7,
        "total_mem_gb": props.total_memory / 1e9
    }

    log.info(f"=== Hardware Capability Check ===")
    log.info(f"GPU: {features['name']} (CC {features['capability']})")
    log.info(f"Support: BF16={features['bf16']}, TF32={features['tf32']}, FP16={features['fp16_native']}")
    log.info(f"Architecture: {'Ampere+' if props.major >= 8 else ('Turing' if props.major == 7 else 'Legacy')}")
    log.info(f"=================================")
    return features

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
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend.is_built()) if mps_backend and hasattr(mps_backend, "is_built") else False
    mps_available = bool(mps_backend.is_available()) if mps_backend and hasattr(mps_backend, "is_available") else False
    requested_device = os.environ.get("STRESS_TORCH_DEVICE", "auto").strip().lower()

    if requested_device in {"cuda", "gpu"} and cuda_available:
        device_type = "cuda"
    elif requested_device == "mps" and mps_available:
        device_type = "mps"
    elif requested_device == "cpu":
        device_type = "cpu"
    elif cuda_available:
        device_type = "cuda"
    elif mps_available:
        device_type = "mps"
    else:
        device_type = "cpu"

    device = torch.device(device_type)
    if device_type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    elif device_type == "mps":
        device_name = "Apple MPS"
    else:
        device_name = "cpu"

    bf16_supported = torch.cuda.is_bf16_supported() if cuda_available else False
    precision = (
        "bf16-mixed"
        if device_type == "cuda" and bf16_supported
        else ("16-mixed" if device_type == "cuda" else "32")
    )
    device_count = torch.cuda.device_count() if cuda_available else (1 if device_type == "mps" else 0)

    try:
        mlx_available = is_mlx_available()
    except Exception:
        mlx_available = False

    backend_request = get_requested_backend()
    mlx_runtime = configure_mlx_runtime() if mlx_available else {}

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

    cuda_python_available = False
    cuda_python_version = None        # CUDA *driver* version as seen through cuda-python (e.g. "13.3")
    cuda_python_package_version = None  # The installed cuda-python *package* version (e.g. "1.0.0")
    if cuda_available:
        try:
            from cuda import bindings  # type: ignore  # noqa: F401
            from cuda.bindings import driver
            err, = driver.cuInit(0)
            if err == driver.CUresult.CUDA_SUCCESS:
                cuda_python_available = True
                err, ver = driver.cuDriverGetVersion()
                if err == driver.CUresult.CUDA_SUCCESS:
                    major = ver // 1000
                    minor = (ver % 1000) // 10
                    cuda_python_version = f"{major}.{minor}"
                # Attempt to read the installed package version (cuda-python or cuda-bindings)
                try:
                    from importlib.metadata import version as _pkg_version, PackageNotFoundError
                    try:
                        cuda_python_package_version = _pkg_version("cuda-python")
                    except PackageNotFoundError:
                        cuda_python_package_version = _pkg_version("cuda-bindings")
                except Exception:
                    pass
        except Exception:
            pass

    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "mlx_available": mlx_available,
        "mlx_version": mlx_version() if mlx_available else None,
        "backend_request": backend_request,
        "mlx_device": mlx_runtime.get("mlx_device"),
        "mlx_compile": mlx_runtime.get("mlx_compile", False),
        "device": str(device),
        "device_type": device_type,
        "device_name": device_name,
        "device_count": device_count,
        "accelerator": "gpu" if device_type == "cuda" else device_type,
        "devices": "auto" if device_type == "cuda" and device_count > 1 else 1,
        "strategy": "ddp_find_unused_parameters_false" if device_type == "cuda" and device_count > 1 else "auto",
        "bf16_supported": bf16_supported,
        "precision": precision,
        "requested_device": requested_device,
        "device_request_satisfied": requested_device in {"", "auto"} or requested_device == device_type or (requested_device == "gpu" and device_type == "cuda"),
        "tensorrt_available": tensorrt_available,
        "torch_tensorrt_available": torch_tensorrt_available,
        "cuda_python_available": cuda_python_available,
        "cuda_python_version": cuda_python_version,          # CUDA driver version (e.g. "13.3")
        "cuda_python_package_version": cuda_python_package_version,  # Package version (e.g. "1.0.0")
    }


def select_torch_device(runtime_caps: Optional[Dict[str, Any]] = None) -> torch.device:
    """Return the best PyTorch device selected by get_runtime_capabilities()."""
    runtime_caps = runtime_caps or get_runtime_capabilities()
    return torch.device(runtime_caps.get("device_type") or runtime_caps.get("device", "cpu"))


def configure_torch_runtime(runtime_caps: Optional[Dict[str, Any]] = None, enable_cudnn_benchmark: bool = True) -> Dict[str, Any]:
    """Enable safe runtime optimizations for the selected PyTorch backend."""
    runtime_caps = runtime_caps or get_runtime_capabilities()
    if runtime_caps.get("device_type") == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
    elif runtime_caps.get("device_type") == "mps":
        torch.set_float32_matmul_precision("high")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        watermark = os.environ.get("STRESS_MPS_HIGH_WATERMARK_RATIO", "").strip()
        if watermark:
            try:
                torch.mps.set_per_process_memory_fraction(float(watermark))
            except Exception as exc:
                log.warning("Could not apply STRESS_MPS_HIGH_WATERMARK_RATIO=%s: %s", watermark, exc)
    if runtime_caps.get("mlx_available"):
        configure_mlx_runtime()
    return runtime_caps


def log_runtime_capabilities(logger: logging.Logger, runtime_caps: Optional[Dict[str, Any]] = None, prefix: str = "Runtime") -> Dict[str, Any]:
    """Log a one-line runtime capability summary."""
    runtime_caps = runtime_caps or get_runtime_capabilities()
    logger.info(
        "%s: torch=%s cuda=%s cuda_python=%s(driver=%s,pkg=%s) mps=%s mlx=%s mlx_ver=%s backend=%s device=%s device_name=%s accelerator=%s devices=%s precision=%s tensorrt=%s torch_tensorrt=%s",
        prefix,
        runtime_caps["torch_version"],
        runtime_caps["cuda_available"],
        runtime_caps.get("cuda_python_available", False),
        runtime_caps.get("cuda_python_version", "unknown"),
        runtime_caps.get("cuda_python_package_version", "unknown"),
        runtime_caps["mps_available"],
        runtime_caps["mlx_available"],
        runtime_caps.get("mlx_version", "n/a"),
        runtime_caps.get("backend_request", "auto"),
        runtime_caps["device"],
        runtime_caps["device_name"],
        runtime_caps["accelerator"],
        runtime_caps["devices"],
        runtime_caps["precision"],
        runtime_caps["tensorrt_available"],
        runtime_caps["torch_tensorrt_available"],
    )
    if runtime_caps.get("requested_device") not in {"", "auto"} and not runtime_caps.get("device_request_satisfied", True):
        logger.warning(
            "Requested STRESS_TORCH_DEVICE=%s but selected %s because the requested backend is unavailable.",
            runtime_caps["requested_device"],
            runtime_caps["device_type"],
        )
    if runtime_caps["mlx_available"]:
        logger.info(
            "%s: MLX is available. Set STRESS_BACKEND=mlx (or keep auto on Apple silicon) "
            "for LSTM, CNN-LSTM, Transformer, and PatchTST; TimesFM uses a hybrid "
            "PyTorch backbone + MLX head on Apple silicon.",
            prefix,
        )
    return runtime_caps

# --- Loading Preprocessed Data ---

PROCESSED_SUBJECT_SUFFIX = "_processed.joblib"
PROCESSED_FILE_MAP_NAME = "processed_file_map.joblib"


def _basename_any_os(path: str) -> str:
    """
    Return the final component of a path written by either OS.

    The file map records absolute paths from whichever machine ran preprocessing,
    so a map produced on Windows carries backslash-separated paths that
    posixpath.basename() would hand back unsplit.
    """
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def resolve_processed_file_map(config: Dict[str, Any]) -> Dict[Union[int, str], str]:
    """
    Resolve the {subject_id: file_path} map for the per-subject processed .joblib
    files written by preprocess_all_subjects().

    Downstream stages consume paths and load each subject lazily
    (data_pipeline._calculate_input_dims, windowing.create_all_subject_windows),
    which is what keeps the whole cohort out of memory.

    Resolution order:
        1. processed_file_map.joblib, with every entry re-anchored by filename to the
           configured processed_data directory. The map stores absolute paths from the
           machine that ran preprocessing, so they are wrong after the project moves
           between machines or operating systems.
        2. A directory scan for '*_processed.joblib' when the map is absent,
           unreadable, malformed, or resolves to no file that exists.

    Args:
        config (Dict[str, Any]): Loaded configuration containing save_paths.processed_data.

    Returns:
        Dict[Union[int, str], str]: Subject ID -> existing absolute .joblib path.
            Empty if the directory is unconfigured, missing, or holds no subject files.
    """
    processed_dir = safe_get(config, ['save_paths', 'processed_data'])
    if not processed_dir or not isinstance(processed_dir, str):
        log.error("Path 'save_paths.processed_data' not found or invalid in config. Cannot resolve processed files.")
        return {}

    abs_processed_dir = os.path.abspath(processed_dir)
    if not os.path.isdir(abs_processed_dir):
        log.error(f"Processed data directory does not exist: {abs_processed_dir}")
        return {}

    file_map: Dict[Union[int, str], str] = {}
    map_path = os.path.join(abs_processed_dir, PROCESSED_FILE_MAP_NAME)

    # --- Route 1: the file map written alongside the per-subject files ---
    if joblib is not None and os.path.exists(map_path):
        stored_map = None
        try:
            stored_map = joblib.load(map_path)
        except Exception as e:
            log.warning(f"Failed to read {PROCESSED_FILE_MAP_NAME} ({e}). Falling back to a directory scan.")

        if isinstance(stored_map, dict):
            unresolved = []
            for subj_id, stored_path in stored_map.items():
                if not isinstance(stored_path, str) or not stored_path:
                    unresolved.append(subj_id)
                    continue
                if os.path.exists(stored_path):
                    file_map[subj_id] = os.path.abspath(stored_path)
                    continue
                # Recorded path is stale (different machine or OS) - re-anchor by filename.
                candidate = os.path.join(abs_processed_dir, _basename_any_os(stored_path))
                if os.path.exists(candidate):
                    file_map[subj_id] = candidate
                else:
                    unresolved.append(subj_id)
            if unresolved:
                log.warning(
                    f"{len(unresolved)} of {len(stored_map)} entries in {PROCESSED_FILE_MAP_NAME} "
                    f"point at files that do not exist and were dropped: {sorted(map(str, unresolved))}"
                )
            if file_map:
                log.info(f"Resolved {len(file_map)} subject file(s) from {map_path}")
        elif stored_map is not None:
            log.warning(
                f"{PROCESSED_FILE_MAP_NAME} did not contain a dictionary "
                f"(got {type(stored_map).__name__}). Falling back to a directory scan."
            )

    # --- Route 2: scan for the files the pipeline always writes next to the map ---
    if not file_map:
        # glob.escape: a directory named e.g. 'run[1]' would otherwise be read as a pattern.
        scan_pattern = os.path.join(glob.escape(abs_processed_dir), f"*{PROCESSED_SUBJECT_SUFFIX}")
        for path in sorted(glob.glob(scan_pattern)):
            subj_id = _basename_any_os(path)[: -len(PROCESSED_SUBJECT_SUFFIX)]
            if subj_id:
                file_map[subj_id] = path
        if file_map:
            log.info(
                f"Recovered {len(file_map)} subject file(s) by scanning {abs_processed_dir} "
                f"(no usable {PROCESSED_FILE_MAP_NAME})."
            )

    if not file_map:
        log.error(
            f"No per-subject '*{PROCESSED_SUBJECT_SUFFIX}' files found in {abs_processed_dir}. "
            "Run the preprocessing pipeline first."
        )
    return file_map


def load_preprocessed_data(config: Dict[str, Any], clear_files: bool = False) -> Tuple[Optional[Dict[Union[int, str], str]], Optional[Dict], Optional[Dict]]:
    """
    Loads the artifacts written by preprocess_all_subjects(), OR clears (deletes)
    them if clear_files is set to True.

    Returns the same triple shape that preprocess_all_subjects() returns, so a run
    resumed from disk is interchangeable with a fresh one. The first element is a
    {subject_id: file_path} map, NOT the signal data itself: get_data_splits() and
    create_all_subject_windows() load each subject lazily from its path so the
    cohort is never resident in memory at once.

    Args:
        config (Dict[str, Any]): The loaded configuration dictionary containing save_paths.
        clear_files (bool, optional): If True, delete the files instead of loading.
                                      Defaults to False.

    Returns:
        If clear_files is False:
            A tuple containing:
                - processed_file_paths (Dict | None): Subject ID -> processed .joblib path.
                  None if no subject file could be resolved.
                - static_features_results (Dict | None): Dictionary of calculated static features.
                - r_peak_results (Dict | None): Dictionary of calculated R-peak indices.
        If clear_files is True:
            Returns (None, None, None) after attempting deletion.
    """
    if clear_files:
        log.info("--- Attempting to CLEAR preprocessed data .joblib files ---")
    else:
        log.info("--- Attempting to LOAD preprocessed data from .joblib files ---")

    processed_file_paths = None
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
    feat_file_path = os.path.join(os.path.abspath(static_feat_dir), "static_features_results.joblib") if static_feat_dir and isinstance(static_feat_dir, str) else None
    rpeak_file_path = os.path.join(os.path.abspath(r_peak_dir), "r_peak_indices.joblib") if r_peak_dir and isinstance(r_peak_dir, str) else None

    # --- Conditional Logic: Clear or Load ---
    if clear_files:
        # --- Deletion Logic ---
        log.warning("Clear Files requested. Deleting saved processed data files...")

        # Per-subject files and their map, as written by preprocess_all_subjects().
        # The legacy 'processed_aligned_data.joblib' monolith is deliberately left in
        # place: nothing in the pipeline writes it any more, so deleting it here would
        # destroy an artifact this function can no longer regenerate.
        subject_file_map = resolve_processed_file_map(config)
        files_to_process = {f"Processed Subject '{subj_id}'": path
                            for subj_id, path in subject_file_map.items()}
        if processed_dir and isinstance(processed_dir, str):
            files_to_process["Processed File Map"] = os.path.join(os.path.abspath(processed_dir), PROCESSED_FILE_MAP_NAME)
        files_to_process["Static Features"] = feat_file_path
        files_to_process["R-Peak Indices"] = rpeak_file_path
        log.info(f"Clearing {len(subject_file_map)} per-subject file(s) plus map, static features and R-peaks.")

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
        # Resolve the per-subject processed files (paths, loaded lazily downstream)
        resolved_map = resolve_processed_file_map(config)
        if resolved_map:
            processed_file_paths = resolved_map
            log.info(f"Successfully resolved Processed Data for {len(processed_file_paths)} subject(s).")

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
        return processed_file_paths, static_features_results, r_peak_results
