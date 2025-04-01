# pytorch_datasets.py (Defines Dataset class and creates DataLoaders)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import sys
import multiprocessing
import collections
from typing import Dict, Tuple, List, Optional, Any, Union

# Assuming utils.py is available
try:
    from utils import safe_get
except ImportError:
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    logging.warning("Could not import 'safe_get' from 'utils'. Using basic fallback in pytorch_datasets.py.")

log = logging.getLogger(__name__)

# Default value for padding missing static features
DEFAULT_STATIC_VALUE = 0.0

# ==============================================================================
# == PyTorch Dataset Class ==
# ==============================================================================
class SequenceWindowDataset(Dataset):
    """
    Custom PyTorch Dataset for windowed sequence data.

    Handles sequence features, static features (including missing ones),
    labels, and window metadata (subject ID, start index).

    Converts input lists of numpy arrays into stacked PyTorch tensors.
    """
    def __init__(self,
                 sequence_features: List[np.ndarray],
                 static_features: List[np.ndarray],
                 labels: List[int],
                 subject_ids: List[Union[int, str]], # Allow string IDs (e.g., "NURSE_1")
                 window_start_indices: List[int]):
        """
        Initializes the Dataset.

        Args:
            sequence_features: List of numpy arrays, each (window_length, num_sequence_features).
            static_features: List of numpy arrays, each (num_static_features,) or potentially empty array ().
            labels: List of integer labels (0 or 1).
            subject_ids: List of subject IDs corresponding to each window.
            window_start_indices: List of absolute start sample indices for each window.

        Raises:
            ValueError: If input lists are empty or have inconsistent lengths, or if
                        feature dimensions are inconsistent within sequence or static lists.
        """
        # --- Input Validation ---
        # Check if essential lists are non-empty
        if not (sequence_features and labels and subject_ids and window_start_indices):
            # Note: static_features can be an empty list if no static features are used
            raise ValueError("Input lists (sequence, labels, subject_ids, window_start_indices) must be non-empty.")
        # Check if all lists have the same length
        list_lengths = [len(lst) for lst in [sequence_features, static_features, labels, subject_ids, window_start_indices]]
        if len(set(list_lengths)) != 1:
            raise ValueError(f"Input lists must have the same length. Got lengths: {list_lengths}")

        # --- Process Sequence Features ---
        try:
            # Determine dimensions from the first window
            first_seq_shape = sequence_features[0].shape
            if len(first_seq_shape) != 2:
                 raise ValueError(f"Sequence features must be 2D (window_length, num_features). Got shape {first_seq_shape} for first element.")
            self.window_length = first_seq_shape[0]
            self.num_sequence_features = first_seq_shape[1]

            processed_seq_features = []
            # Validate shape consistency across all sequence windows
            for i, seq_win in enumerate(sequence_features):
                 if not isinstance(seq_win, np.ndarray) or seq_win.ndim != 2:
                     raise ValueError(f"Sequence window at index {i} is not a 2D numpy array (shape: {seq_win.shape if hasattr(seq_win, 'shape') else type(seq_win)}).")
                 if seq_win.shape[0] != self.window_length:
                     raise ValueError(f"Inconsistent sequence window length at index {i}. Expected {self.window_length}, got {seq_win.shape[0]}.")
                 if seq_win.shape[1] != self.num_sequence_features:
                     raise ValueError(f"Inconsistent sequence feature dimension at index {i}. Expected {self.num_sequence_features}, got {seq_win.shape[1]}.")
                 processed_seq_features.append(seq_win)

            # Stack validated sequence windows into a single tensor
            self.sequence_features = torch.tensor(np.stack(processed_seq_features, axis=0), dtype=torch.float32)
            log.info(f"Stacked {len(sequence_features)} sequence feature windows.")
        except Exception as e:
            log.error(f"Error processing or stacking sequence features: {e}", exc_info=True)
            raise

        # --- Process Static Features (Handle potential empty arrays/list) ---
        self.num_static_features = 0 # Default to 0
        processed_static_features = []

        if static_features: # Check if the list itself is not empty
            # Find the first valid (non-empty) static feature vector to determine dimension
            first_valid_static_idx = -1
            first_static_shape = None
            for i, static_vec in enumerate(static_features):
                if isinstance(static_vec, np.ndarray) and static_vec.size > 0:
                    first_valid_static_idx = i
                    first_static_shape = static_vec.shape
                    break

            # If a valid static vector was found, determine the dimension
            if first_valid_static_idx != -1 and first_static_shape is not None and len(first_static_shape) > 0:
                # Assume 1D vector (num_features,) or take last dim if > 1D
                self.num_static_features = first_static_shape[0] if len(first_static_shape) == 1 else first_static_shape[-1]
                log.info(f"Determined static feature dimension: {self.num_static_features}")

                # Create a default vector for padding missing/invalid ones
                default_static_vector = np.full(self.num_static_features, DEFAULT_STATIC_VALUE, dtype=np.float32)

                # Validate and process all static feature vectors
                for i, static_vec in enumerate(static_features):
                    if isinstance(static_vec, np.ndarray) and static_vec.size > 0:
                        static_vec_flat = static_vec.flatten() # Ensure 1D
                        # Check if dimension matches
                        if static_vec_flat.shape[0] != self.num_static_features:
                            log.error(f"Inconsistent static feature shape at index {i}. Expected ({self.num_static_features},), got {static_vec.shape}. Using default padding.")
                            processed_static_features.append(default_static_vector)
                        else:
                            processed_static_features.append(static_vec_flat)
                    else:
                        # Pad empty or invalid vectors
                        # log.debug(f"Padding empty/invalid static vector at index {i}.")
                        processed_static_features.append(default_static_vector)
            else:
                # All provided static vectors were empty or invalid
                log.warning("All provided static feature vectors were empty or invalid. Treating as 0 static features.")
                self.num_static_features = 0
                # Create list of empty arrays matching the number of samples
                processed_static_features = [np.array([], dtype=np.float32)] * len(static_features)
        else:
            # static_features list itself was empty
            log.info("No static features provided to dataset. Setting num_static_features to 0.")
            self.num_static_features = 0
            processed_static_features = [np.array([], dtype=np.float32)] * len(sequence_features) # Match length

        # --- Stack Static Features ---
        try:
            if self.num_static_features > 0 and processed_static_features:
                 # Stack the processed (potentially padded) static vectors
                 self.static_features = torch.tensor(np.stack(processed_static_features, axis=0), dtype=torch.float32)
            else:
                 # Handle case where num_static_features is 0 or list is empty/contains only empty arrays
                 # Create an empty tensor with shape (N, 0)
                 self.static_features = torch.empty((len(sequence_features), 0), dtype=torch.float32)
            log.info(f"Stacked {len(processed_static_features)} static feature vectors (Dim: {self.num_static_features}).")
        except Exception as e:
            log.error(f"Error stacking static features: {e}", exc_info=True)
            raise

        # --- Process Labels ---
        try:
            self.labels = torch.tensor(labels, dtype=torch.float32) # Use float for BCEWithLogitsLoss
        except Exception as e:
            log.error(f"Error converting labels to tensor: {e}", exc_info=True)
            raise

        # --- Store Metadata ---
        self.subject_ids_list = subject_ids # Keep original list (can contain strings)
        # Try converting subject IDs to tensor if they are all integers, otherwise use placeholder
        try:
             if all(isinstance(x, int) for x in subject_ids):
                 self.subject_ids_tensor = torch.tensor(subject_ids, dtype=torch.int64)
             else:
                  log.warning("Subject IDs contain non-integers. Storing as list. Tensor version will be placeholder (zeros).")
                  # Create a placeholder tensor if IDs are not all integers
                  self.subject_ids_tensor = torch.zeros(len(subject_ids), dtype=torch.int64)
        except Exception as e:
             log.error(f"Error converting subject IDs to tensor: {e}. Using placeholder.")
             self.subject_ids_tensor = torch.zeros(len(subject_ids), dtype=torch.int64)

        try:
            self.window_start_indices = torch.tensor(window_start_indices, dtype=torch.int64)
        except Exception as e:
            log.error(f"Error converting window start indices to tensor: {e}", exc_info=True)
            raise

        # --- Log Final Shapes ---
        log.info(f"Created SequenceWindowDataset with {len(self.labels)} windows.")
        log.info(f"  Sequence Feature tensor shape: {self.sequence_features.shape}")
        log.info(f"  Static Feature tensor shape: {self.static_features.shape}")
        log.info(f"  Label tensor shape: {self.labels.shape}")
        # Log tensor shape, noting it might be a placeholder if IDs were strings
        log.info(f"  Subject ID tensor shape: {self.subject_ids_tensor.shape} {'(Placeholder if IDs are strings)' if not all(isinstance(x, int) for x in subject_ids) else ''}")
        log.info(f"  Window Start tensor shape: {self.window_start_indices.shape}")

    def __len__(self) -> int:
        """Returns the total number of windows in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single window's data.

        Args:
            idx (int): The index of the window to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Sequence features tensor (window_length, num_sequence_features)
                - Static features tensor (num_static_features,) or (0,) if none
                - Label tensor (scalar)
                - Subject ID tensor (scalar, potentially placeholder)
                - Window start index tensor (scalar)
        """
        return (
            self.sequence_features[idx],
            self.static_features[idx], # Will have shape (0,) if no static features
            self.labels[idx],
            self.subject_ids_tensor[idx], # Return the tensor version
            self.window_start_indices[idx]
        )

# ==============================================================================
# == DataLoader Creation ==
# ==============================================================================

def create_pytorch_dataloaders(
    train_data: Tuple[List, ...],
    val_data: Tuple[List, ...],
    test_data: Tuple[List, ...],
    batch_size: int,
    config: Dict[str, Any] # Pass config for hardware options
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates PyTorch Datasets and DataLoaders for train, validation, and test sets.
    Includes hardware optimization options (num_workers, pin_memory).

    Args:
        train_data (Tuple[List, ...]): Training data lists (output from sampling.py).
        val_data (Tuple[List, ...]): Validation data lists (output from data_splitting.py).
        test_data (Tuple[List, ...]): Test data lists (output from data_splitting.py).
        batch_size (int): The desired batch size.
        config (Dict[str, Any]): Configuration dictionary for hardware settings.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
            DataLoaders for train, validation, and test sets. Returns None for a
            loader if the corresponding dataset is empty or creation fails.
    """
    log.info("Creating Datasets and DataLoaders (with hardware opts)...")
    train_loader, val_loader, test_loader = None, None, None

    # --- Hardware Enhancement Settings ---
    # Determine num_workers based on OS and config (default to 0 for safety)
    # Config key: config['processing']['dataloader_num_workers']
    num_workers_config = safe_get(config, ['processing', 'dataloader_num_workers'], 0)
    max_workers = multiprocessing.cpu_count() # Get available CPU cores
    num_workers = 0 # Default

    if isinstance(num_workers_config, int) and num_workers_config > 0:
        # Use specified number, capped by available cores
        num_workers = min(num_workers_config, max_workers)
        if sys.platform == 'win32': # Check if running on Windows
            log.warning(f"num_workers > 0 ({num_workers}) on Windows can sometimes cause issues. Set to 0 if problems arise.")
    elif num_workers_config == -1: # Use max available (minus 1 for safety)
         num_workers = max(0, max_workers - 1)
         if sys.platform == 'win32' and num_workers > 0:
            log.warning(f"num_workers > 0 ({num_workers}) on Windows can sometimes cause issues. Set to 0 if problems arise.")
    # else: num_workers remains 0

    # Pin memory if using CUDA (can speed up CPU->GPU data transfer)
    pin_memory = torch.cuda.is_available()

    # Use persistent workers if num_workers > 0 (can speed up epoch starts)
    # Requires PyTorch 1.7+
    persist_workers = num_workers > 0

    log.info(f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persist_workers}")
    # --- End Hardware Settings ---

    # --- Helper to Log Class Distribution ---
    def log_imbalance(labels_list: List, name: str):
        """Logs the class distribution for a dataset split."""
        if not labels_list:
            log.info(f"{name} Label Dist: Empty.")
            return
        try:
            labels_int = np.array(labels_list).astype(int)
            counts = collections.Counter(labels_int)
            total = len(labels_int)
            # Format distribution as counts and percentages
            dist_percent = {k: f"{v} ({v/total*100:.1f}%)" for k, v in sorted(counts.items())}
            log.info(f"{name} Label Dist ({total} windows): {dist_percent}")
        except Exception as e:
            log.warning(f"Could not log label distribution for {name}: {e}")

    # Log distributions before creating DataLoaders
    log.info("--- Final Set Class Distribution ---")
    log_imbalance(train_data[2], "Training Set (Sampled)")
    log_imbalance(val_data[2], "Validation Set")
    log_imbalance(test_data[2], "Test Set")

    # Check for single-class sets (can cause issues with some metrics/losses)
    if len(np.unique(np.array(train_data[2]))) < 2 and len(train_data[2]) > 0 : log.warning("Sampled Training set appears to contain only one class.")
    if len(np.unique(np.array(val_data[2]))) < 2 and len(val_data[2]) > 0: log.warning("Validation set appears to contain only one class.")
    if len(np.unique(np.array(test_data[2]))) < 2 and len(test_data[2]) > 0: log.warning("Test set appears to contain only one class.")

    # --- Create Datasets and DataLoaders ---
    try:
        # --- Training Loader ---
        if train_data[0]: # Check if sequence list is non-empty
            train_dataset = SequenceWindowDataset(*train_data)
            # Ensure batch size is not larger than dataset size
            actual_batch_size_train = min(batch_size, len(train_dataset))
            if actual_batch_size_train > 0:
                 # Use hardware settings in DataLoader constructor
                 train_loader = DataLoader(
                     train_dataset,
                     batch_size=actual_batch_size_train,
                     shuffle=True, # Shuffle training data
                     num_workers=num_workers,
                     drop_last=True, # Drop last incomplete batch during training
                     pin_memory=pin_memory,
                     persistent_workers=persist_workers # Only True if num_workers > 0
                 )
                 # Handle edge case where drop_last=True makes loader empty
                 if len(train_loader) == 0 and len(train_dataset) > 0:
                      log.warning("Drop_last=True resulted in 0 batches for training. Retrying with drop_last=False.")
                      train_loader = DataLoader(
                          train_dataset, batch_size=actual_batch_size_train, shuffle=True,
                          num_workers=num_workers, drop_last=False, pin_memory=pin_memory,
                          persistent_workers=persist_workers
                      )
                 log.info(f"Train loader created (sampled data). Batches: {len(train_loader)}")
            else:
                 log.error(f"Sampled train dataset too small or empty for batching (batch size {batch_size}, dataset size {len(train_dataset)}). Train loader not created.")
        else:
            log.warning("Training set empty. Train loader not created.")

        # --- Validation Loader ---
        if val_data[0]: # Check if sequence list is non-empty
            val_dataset = SequenceWindowDataset(*val_data)
            actual_batch_size_val = min(batch_size, len(val_dataset))
            if actual_batch_size_val > 0:
                 # Use hardware settings (shuffle=False for validation)
                 val_loader = DataLoader(
                     val_dataset,
                     batch_size=actual_batch_size_val,
                     shuffle=False,
                     num_workers=num_workers,
                     pin_memory=pin_memory,
                     persistent_workers=persist_workers
                 )
                 log.info(f"Validation loader created. Batches: {len(val_loader)}")
            else:
                 log.warning("Validation dataset too small or empty for batching. Validation loader not created.")
        else:
            log.info("Validation set empty. Validation loader not created.") # Changed from warning to info

        # --- Test Loader ---
        if test_data[0]: # Check if sequence list is non-empty
            test_dataset = SequenceWindowDataset(*test_data)
            actual_batch_size_test = min(batch_size, len(test_dataset))
            if actual_batch_size_test > 0:
                 # Use hardware settings (shuffle=False for test)
                 test_loader = DataLoader(
                     test_dataset,
                     batch_size=actual_batch_size_test,
                     shuffle=False,
                     num_workers=num_workers,
                     pin_memory=pin_memory,
                     persistent_workers=persist_workers
                 )
                 log.info(f"Test loader created. Batches: {len(test_loader)}")
            else:
                 log.warning("Test dataset too small or empty for batching. Test loader not created.")
        else:
            log.warning("Test set empty. Test loader not created.")

    except Exception as e:
        log.error(f"Failed to create Datasets or DataLoaders: {e}", exc_info=True)
        # Return None for all loaders on critical error
        return None, None, None

    return train_loader, val_loader, test_loader
