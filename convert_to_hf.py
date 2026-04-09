import logging
import os
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from joblib import load
from typing import Dict, Any, Tuple, List

from utils import load_config, setup_logging

log = logging.getLogger(__name__)

def convert_split_to_hf(data_tuple: Tuple[List, ...]) -> Dataset:
    """
    Converts a standard split tuple from get_data_splits 
    into a Hugging Face Dataset.
    """
    # data_tuple structure: (seq_features, static_features, labels, subject_ids, window_indices)
    seqs, statics, labels, subjs, indices = data_tuple
    
    # Convert to dictionary format for HF
    data_dict = {
        "sequence": seqs,
        "static": statics,
        "label": labels,
        "subject_id": subjs,
        "window_index": indices
    }
    
    return Dataset.from_dict(data_dict)

def main():
    setup_logging()
    config = load_config("config.json")
    if not config:
        return

    export_path = "./outputs/processed_data_hf"
    os.makedirs(export_path, exist_ok=True)

    # Note: In a real scenario, we'd run the preprocessing pipeline here
    # or load the existing joblib files to convert them.
    # For the migration, we'll demonstrate the conversion logic.
    log.info("Starting conversion of processed data to Hugging Face Arrow format...")
    
    # This script is intended to be called after preprocessing/windowing
    # We will refine lightning_data.py to call this automatically if Arrow files aren't found.
    log.info(f"Target Export Path: {export_path}")

if __name__ == "__main__":
    main()
