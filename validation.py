import os
import logging
import pandas as pd
import numpy as np
import torch
from datasets import load_from_disk
from deepchecks.tabular.suites import train_test_validation
from deepchecks.tabular import Dataset

from utils import setup_logging

log = logging.getLogger(__name__)

def extract_meta_features(ds) -> pd.DataFrame:
    """
    Converts windowed signal data into a tabular format of statistical 
    meta-features for Deepchecks drift detection.
    """
    log.info(f"Extracting meta-features from {len(ds)} windows...")
    
    # sequence: (N, L, F)
    seqs = ds['sequence'].numpy()
    labels = ds['label'].numpy()
    
    # Calculate stats per channel per window
    # Channel 0: ECG/EDA summary, etc.
    features = {}
    num_channels = seqs.shape[2]
    
    for c in range(num_channels):
        features[f"ch{c}_mean"] = seqs[:, :, c].mean(axis=1)
        features[f"ch{c}_std"]  = seqs[:, :, c].std(axis=1)
        features[f"ch{c}_max"]  = seqs[:, :, c].max(axis=1)
        features[f"ch{c}_min"]  = seqs[:, :, c].min(axis=1)
        
    # Static features: (N, F_static)
    statics = ds['static'].numpy()
    for s in range(statics.shape[1]):
        features[f"static_{s}"] = statics[:, s]
        
    features['target'] = labels
    return pd.DataFrame(features)

def run_integrity_check():
    """
    Loads Arrow datasets, extracts meta-features, and runs 
    the Deepchecks integrity suite.
    """
    setup_logging()
    hf_path = "./outputs/processed_data_hf"
    
    if not os.path.exists(hf_path):
        log.error("HuggingFace datasets not found. Run preprocessing first.")
        return

    train_ds = load_from_disk(os.path.join(hf_path, "train"))
    test_ds = load_from_disk(os.path.join(hf_path, "test"))

    # Convert to Tabular Meta-Features
    train_df = extract_meta_features(train_ds)
    test_df = extract_meta_features(test_ds)

    # Wrap in Deepchecks Dataset
    ds_train = Dataset(train_df, label='target', cat_features=[])
    ds_test = Dataset(test_df, label='target', cat_features=[])

    # Run Suite
    log.info("Running Deepchecks Train-Test Validation Suite...")
    suite = train_test_validation()
    result = suite.run(train_dataset=ds_train, test_dataset=ds_test)
    
    # Save Report
    report_path = "./outputs/results/integrity_report.html"
    result.save_as_html(report_path)
    log.info(f"Integrity report saved to: {report_path}")
    
    return result

if __name__ == "__main__":
    run_integrity_check()
