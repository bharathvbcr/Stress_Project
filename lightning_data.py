import lightning as L
import torch
import os
import logging
import sys
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from typing import Dict, Any, Optional, Union

from data_pipeline import get_data_splits
from training import _calculate_pos_weight
from utils import safe_get, resolve_hf_path

log = logging.getLogger(__name__)

class StressDataModule(L.LightningDataModule):
    """
    Modernized DataModule using Hugging Face Datasets (Arrow) for high-speed I/O.
    Supports memory-mapping for 'instant-load' of large multi-modal datasets.
    """
    def __init__(self, 
                 processed_data_paths: Dict[Union[int, str], str],
                 static_features_results: Dict[Union[int, str], Optional[Any]],
                 config: Dict[str, Any]):
        super().__init__()
        self.processed_data_paths = processed_data_paths
        self.static_features_results = static_features_results
        self.config = config
        self.batch_size = safe_get(config, ['training_config', 'batch_size'], 64)
        
        # Path for persisted HF/Arrow data
        self.hf_path = resolve_hf_path(config)
        workers_default = 2 if sys.platform == "win32" else min(8, os.cpu_count() or 4)
        self.num_workers = safe_get(config, ['processing', 'dataloader_num_workers'], workers_default)
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            self.num_workers = workers_default
        self.num_workers = min(self.num_workers, os.cpu_count() or self.num_workers or 1)
        self.pin_memory = torch.cuda.is_available()
        
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.pos_weight = None

    def setup(self, stage: Optional[str] = None):
        """
        Loads Arrow datasets if they exist, otherwise runs the pipeline 
        and persists them for future instant-loads.
        """
        if os.path.exists(self.hf_path) and os.listdir(self.hf_path):
            log.info(f"Loading memory-mapped Arrow datasets from {self.hf_path}...")
            self.train_ds = load_from_disk(os.path.join(self.hf_path, "train"))
            self.val_ds = load_from_disk(os.path.join(self.hf_path, "val"))
            self.test_ds = load_from_disk(os.path.join(self.hf_path, "test"))
            
            # Infer dimensions
            sample = self.train_ds[0]
            self.input_dim_sequence = len(sample['sequence'][0]) # (L, F)
            self.input_dim_static = len(sample['static'])
            log.info("Dataset dimensions inferred from Arrow storage.")
        else:
            log.info("Arrow cache not found. Running legacy data engine and converting...")
            splits = get_data_splits(self.processed_data_paths, self.static_features_results, self.config)
            train_tup, val_tup, test_tup, self.input_dim_sequence, self.input_dim_static = splits
            
            # Convert and Persist
            from convert_to_hf import convert_split_to_hf
            self.train_ds = convert_split_to_hf(train_tup)
            self.val_ds = convert_split_to_hf(val_tup)
            self.test_ds = convert_split_to_hf(test_tup)
            
            log.info(f"Persisting Arrow datasets to {self.hf_path} for future instant-load...")
            os.makedirs(self.hf_path, exist_ok=True)
            self.train_ds.save_to_disk(os.path.join(self.hf_path, "train"))
            self.val_ds.save_to_disk(os.path.join(self.hf_path, "val"))
            self.test_ds.save_to_disk(os.path.join(self.hf_path, "test"))

        # Format as PyTorch tensors
        for ds in [self.train_ds, self.val_ds, self.test_ds]:
            ds.set_format(type='torch', columns=['sequence', 'static', 'label'])

        # Calculate pos_weight for imbalance
        self.pos_weight = self._calculate_pos_weight()

    def train_dataloader(self):
        num_workers = self.num_workers
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "pin_memory": self.pin_memory,
            "num_workers": num_workers,
            "persistent_workers": (num_workers > 0),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(self.train_ds, **loader_kwargs)

    def val_dataloader(self):
        num_workers = min(self.num_workers, 2)
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "pin_memory": self.pin_memory,
            "num_workers": num_workers,
            "persistent_workers": (num_workers > 0),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(self.val_ds, **loader_kwargs)

    def test_dataloader(self):
        num_workers = min(self.num_workers, 2)
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "pin_memory": self.pin_memory,
            "num_workers": num_workers,
            "persistent_workers": (num_workers > 0),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(self.test_ds, **loader_kwargs)

    def _calculate_pos_weight(self):
        # Optimized pos_weight calculation from Arrow dataset
        labels_column = self.train_ds.data.column("label")
        labels = labels_column.to_pylist()
        num_pos = float(sum(labels))
        num_neg = len(labels) - num_pos
            
        if num_pos == 0: return torch.tensor([1.0])
        return torch.tensor([num_neg / num_pos])
