# Stress Detection Project

This project implements a comprehensive deep learning pipeline for detecting stress using physiological signals collected from wearable devices (WESAD and NURSE datasets). It features a modular architecture handling data loading, advanced signal preprocessing, feature extraction, and classification using state-of-the-art neural networks.

## 🚀 Key Features

*   **Multi-Modal Data Fusion:** Integrates data from chest (ECG, EDA, EMG, ACC, Temp) and wrist (BVP, EDA, ACC, Temp) devices.
*   **Advanced Models:**
    *   **StressCNNLSTM:** Hybrid architecture combining 1D-CNNs for local feature extraction, Bi-LSTMs for temporal dynamics, and Attention mechanisms.
    *   **StressTransformer:** Transformer-based encoder for capturing long-range dependencies.
    *   **StressLSTM:** Baseline architecture with late fusion.
*   **Robust Preprocessing:**
    *   Automatic signal resampling and alignment.
    *   **Bio-signal Analysis:** Computes HRV (Heart Rate Variability), EDA (Phasic/Tonic), and statistical features using `neurokit2` and `scikit-learn`.
    *   **Parallel Processing:** Optimized feature extraction using `joblib`.
*   **Imbalance Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) and **Focal Loss** / Class Weighting to address severe class imbalance.
*   **Optimized Pipeline:**
    *   **Hyperparameter Tuning:** Integrated Optuna tuning loop.
    *   **Modular Design:** Separated concerns for loading, splitting, sampling, and training.
    *   **Interactive Visualization:** Jupyter widgets for exploring raw signals, predictions, and model performance.

## 📂 Project Structure

```text
C:\Users\bhara\Downloads\Code\StressProject\
├── run_pipeline.py                 # 🚀 MAIN ENTRY POINT: Runs the full end-to-end pipeline
├── config.json                     # Central configuration for paths, models, and training
├── requirements.txt                # Project dependencies
│
├── data_loader.py                  # Loads raw WESAD/NURSE data
├── preprocessing.py                # Signal resampling, alignment, and feature extraction orchestration
├── signal_processing.py            # Low-level signal resampling logic
├── feature_extraction.py           # Computation of static features (HRV, EDA peaks, etc.)
│
├── data_pipeline.py                # Pipeline orchestration: windowing, splitting, sampling, dataloaders
├── windowing.py                    # Splits signals into overlapping windows
├── data_splitting.py               # Group-stratified train/val/test splitting
├── sampling.py                     # Handles class imbalance (SMOTE, Random Oversampling)
├── pytorch_datasets.py             # Custom PyTorch Dataset and DataLoader creation
│
├── models.py                       # PyTorch model definitions (LSTM, CNN-LSTM, Transformer)
├── losses.py                       # Custom loss functions (FocalLoss)
├── training.py                     # Training loops, validation, and early stopping
├── evaluation.py                   # Metrics (F1, AUC), threshold optimization, and reporting
├── tuning.py                       # Optuna hyperparameter optimization script
│
├── visualization.py                # Plotting utilities (Signal, ROC, Confusion Matrix)
├── widget_setup.py                 # Interactive Jupyter widgets
├── utils.py                        # Helpers for config, logging, and I/O
│
└── outputs/                        # Generated artifacts
    ├── models/                     # Saved model weights (.pth)
    ├── processed_data/             # Cached preprocessed data (.joblib)
    └── results/                    # Evaluation metrics (.json) and plots (.png)
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bharathvbcr/Stress_Project.git
    cd StressProject
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries: `torch`, `numpy`, `pandas`, `neurokit2`, `scikit-learn`, `optuna`, `shap`, `joblib`.*

## ⚙️ Configuration

The `config.json` file controls the entire pipeline. Key sections:

*   **`datasets`**: Paths to WESAD/NURSE data. **Update the `path` values to match your local system.**
*   **`features_to_use`**: Define which sensor channels to use (e.g., `["ECG", "EDA"]`).
*   **`static_features_to_use`**: List of computed features to include (HRV, statistical moments).
*   **`windowing`**: Set `window_size_sec` (default 60s) and `window_overlap` (default 0.5).
*   **`model_config`**: Select model `type` (`CNN-LSTM`, `TRANSFORMER`, `LSTM`) and architecture parameters.
*   **`training_config`**: Hyperparameters (LR, Batch Size, Epochs) and `sampling_strategy` (`smote` or `random`).

## 🚀 Usage

### 1. Run the Full Pipeline
The easiest way to run the project is via the main script. This handles data loading, processing, training, and evaluation in one go.

```bash
python run_pipeline.py
```
*Check the console output for detailed logs regarding data loading status, split sizes, and training progress.*

### 2. Hyperparameter Tuning
To optimize model performance using Optuna:

```bash
python tuning.py
```
*This will run multiple trials to find the best hyperparameters (learning rate, layers, etc.) and save them to `outputs/results/best_hyperparameters.json`.*

### 3. Interactive Notebook
For exploration and visualization, use the Jupyter Notebook:
`Baseline_Calibration_for_Stress_Response.ipynb`

*   **Interactive Plots:** Visualize raw signals vs. resampled signals.
*   **Prediction Analysis:** Overlay model predictions on true signal labels.
*   **HRV Analysis:** Inspect ECG signals with detected R-peaks.

## 📊 Pipeline Stages

1.  **Data Loading:** Reads raw pickle/CSV files.
2.  **Preprocessing:**
    *   **Resampling:** Downsamples signals to a common target rate (e.g., 64Hz).
    *   **Feature Extraction:** Calculates HRV metrics using original high-freq ECG data.
3.  **Windowing:** Slices continuous signals into fixed-length windows (e.g., 60s).
4.  **Splitting:** Performs **Subject-Group Stratified Split** to ensure no subject leakage between Train/Val/Test.
5.  **Sampling:** Applies **SMOTE** or Random Oversampling to the Training set to balance stress/non-stress classes.
6.  **Training:** Trains the PyTorch model with **Early Stopping** and **ReduceLROnPlateau**.
7.  **Evaluation:** Computes Accuracy, F1-Score, Precision, Recall, and ROC-AUC on the Test set. Optimizes the decision threshold for maximum F1.

## 📈 Outputs

Artifacts are saved in the `outputs/` directory:
*   `processed_aligned_data.joblib`: Cached processed signals.
*   `static_features_results.joblib`: Cached feature dataframes.
*   `best_model.pth`: State dictionary of the best trained model.
*   `results/`: Contains:
    *   `test_evaluation_results.json`: Full metrics report.
    *   `confusion_matrix_test.png`: Visual confusion matrix.
    *   `roc_curve_test.png`: ROC Curve.
    *   `training_history.png`: Loss and F1 score curves.