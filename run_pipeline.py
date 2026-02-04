import logging
import torch
import os
import sys
from utils import load_config, setup_logging
from data_loader import load_all_datasets
from preprocessing import preprocess_all_subjects
from data_pipeline import prepare_dataloaders
from models import get_model
from training import train_model
from evaluation import evaluate_model

# Initialize logging immediately
setup_logging()
log = logging.getLogger(__name__)

def main():
    log.info("=== Starting Stress Detection Pipeline (Full Run) ===")
    
    # 1. Config
    log.info("Stage 1: Loading Configuration")
    config = load_config('config.json')
    if not config:
        log.error("Failed to load configuration. Exiting.")
        return

    # 2. Data Loading
    log.info("Stage 2: Loading Raw Data")
    raw_data, loaded_ids, failed_ids = load_all_datasets(config)
    if not raw_data:
        log.error("No data loaded. Exiting.")
        return
    log.info(f"Loaded raw data for {len(loaded_ids)} subjects.")
    
    # 3. Preprocessing
    log.info("Stage 3: Preprocessing (Resampling, Alignment, Feature Extraction)")
    processed_data, static_features, r_peaks = preprocess_all_subjects(raw_data, loaded_ids, config)
    if not processed_data:
        log.error("Preprocessing failed. Exiting.")
        return
    
    # 4. Pipeline (Windowing, Splitting, Sampling, Loader Creation)
    log.info("Stage 4: Preparing Pipeline (Windows, Splits, Sampling, Loaders)")
    train_loader, val_loader, test_loader, seq_dim, static_dim = prepare_dataloaders(
        processed_data, static_features, config
    )
    if not train_loader:
        log.error("Failed to create DataLoaders. Exiting.")
        return
    
    # 5. Model
    log.info("Stage 5: Building Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if device.type == 'cuda':
        log.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        model = get_model(config, seq_dim, static_dim)
        model.to(device)
    except Exception as e:
        log.error(f"Failed to build model: {e}", exc_info=True)
        return
    
    # 6. Training
    log.info("Stage 6: Training Model")
    try:
        best_state, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            config, 
            device, 
            config['save_paths'].get('models')
        )
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        return
    
    # 7. Evaluation
    log.info("Stage 7: Evaluating on Test Set")
    if best_state:
        log.info("Loading best model state for evaluation...")
        model.load_state_dict(best_state)
    else:
        log.warning("Using final model state (no best state saved).")

    try:
        results = evaluate_model(
            model, 
            test_loader, 
            None, # Criterion optional for evaluation
            device, 
            config, 
            config['save_paths'].get('results'), 
            set_name="Test Set"
        )
        if results:
            log.info("Pipeline completed successfully.")
            print("\nFinal Test Metrics:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1 Score: {results['f1_score']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
    except Exception as e:
        log.error(f"Evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        log.critical(f"Unhandled exception in main pipeline: {e}", exc_info=True)
        sys.exit(1)
