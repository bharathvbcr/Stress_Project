# Stress Detection Project

This project focuses on the detection of stress using physiological signals collected from wearable devices (chest and wrist). It implements a deep learning pipeline that processes multi-modal data (ECG, EDA, EMG, ACC, TEMP, BVP), extracts features, and classifies stress states using advanced neural network architectures.

## Project Overview

The system is designed to work with standard stress detection datasets like **WESAD** (Wearable Stress and Affect Detection) and **NURSE**. It includes a robust data processing pipeline and configurable deep learning models to classify data into states such as "baseline", "stress", and "transient".

Key features include:
-   **Multi-modal Data Support:** Handles signals from both chest (ECG, EDA, EMG, Temp, ACC) and wrist (BVP, EDA, Temp, ACC) devices.
-   **Advanced Models:**
    -   **StressLSTM:** A baseline Long Short-Term Memory network with late fusion of static features.
    -   **StressCNNLSTM:** A hybrid CNN-LSTM architecture with 1D Convolutional layers for feature extraction, LSTM for temporal sequencing, and an optional Attention mechanism. It uses early fusion for static features.
    -   **StressTransformer:** A state-of-the-art Transformer Encoder model (like BERT) for capturing long-range dependencies in physiological signals.
-   **Comprehensive Pipeline:** Automated data loading, signal resampling, alignment, windowing, and feature extraction (including HRV and statistical features).
-   **Optimized Tuning:** High-performance hyperparameter tuning pipeline (using Optuna) that reuses data splits for 10x-50x faster execution.
-   **Configurable Experiments:** All aspects of the pipeline (datasets, features, model hyperparameters, training settings) are controlled via a central `config.json` file.

## Project Structure

```text
C:\Users\bhara\Downloads\Code\StressProject\
├── Baseline_Calibration_for_Stress_Response.ipynb  # Main entry point for training and evaluation
├── config.json                                     # Configuration file for the entire pipeline
├── data_pipeline.py                                # Orchestrates data loading, windowing, and splitting
├── models.py                                       # PyTorch model definitions (StressLSTM, StressCNNLSTM)
├── preprocessing.py                                # Signal processing and feature extraction
├── training.py                                     # Training loops and validation logic
├── utils.py                                        # Utility functions
├── outputs/                                        # Generated artifacts
│   ├── models/                                     # Saved model weights (.pth)
│   ├── processed_data/                             # Preprocessed datasets (.joblib)
│   └── results/                                    # Evaluation metrics and plots
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd StressProject
    ```

2.  **Install Dependencies:**
    Ensure you have Python installed. The key libraries required are:
    -   `numpy`
    -   `pandas`
    -   `torch` (PyTorch)
    -   `scikit-learn`
    -   `neurokit2` (for physiological feature extraction)
    -   `joblib`
    -   `matplotlib` / `seaborn` (for visualization)

    You can install them via pip:
    ```bash
    pip install numpy pandas torch scikit-learn neurokit2 joblib matplotlib seaborn
    ```

## Configuration

The `config.json` file allows you to customize the experiment without changing code. Key sections include:

-   **`datasets`**: Paths and settings for WESAD and NURSE datasets.
-   **`features_to_use`**: Select which signals (e.g., "chest_ECG", "wrist_BVP") to include in the sequence input.
-   **`static_features_to_use`**: List of statistical features (e.g., HRV metrics, means, std devs) to calculate and use.
-   **`model_config`**: Choose the model type (`CNN-LSTM` or `LSTM`) and tune hyperparameters like layers, dropout, filters, and attention heads.
-   **`training_config`**: Set batch size, learning rate, epochs, and loss function.

## Usage

1.  **Configure Data Paths:**
    Open `config.json` and ensure the paths under `datasets` point to the actual locations of your WESAD or NURSE datasets on your machine.

2.  **Run the Pipeline:**
    The project is primarily driven by the Jupyter Notebook:
    `Baseline_Calibration_for_Stress_Response.ipynb`

    Open this notebook in Jupyter Lab or VS Code and execute the cells. It will:
    -   Load configuration.
    -   Process raw data (if not already cached).
    -   Create training, validation, and test splits.
    -   Initialize the selected model (e.g., StressCNNLSTM).
    -   Train the model and save the best weights.
    -   Evaluate performance and generate plots.

## Models

### StressCNNLSTM
A hybrid model designed to capture both local patterns and long-term dependencies.
-   **CNN Encoder:** 1D Conv layers extract features from physiological signal windows.
-   **Early Fusion:** Static features (demographics, computed stats) are concatenated with CNN outputs.
-   **Bi-LSTM:** Processes the sequence of features to understand temporal dynamics.
-   **Attention (Optional):** Weighs the importance of different time steps before classification.

### StressTransformer
A modern architecture leveraging the Transformer Encoder mechanism.
-   **Transformer Encoder:** Uses self-attention mechanisms to weigh the influence of different parts of the signal sequence on each other, capturing complex long-range dependencies often missed by RNNs.
-   **Positional Encoding:** Injects temporal order information into the sequence.
-   **Global Pooling:** Aggregates the encoded sequence for classification.
-   **Flexible Fusion:** Supports both early and late fusion of static features.

### StressLSTM
A simpler baseline model.
-   **LSTM:** Processes the time-series data.
-   **Late Fusion:** Static features are concatenated with the LSTM output just before the final classification layer.

## Outputs

After running the notebook, check the `outputs/` directory:
-   **`outputs/models/`**: Contains the saved `best_model.pth`.
-   **`outputs/results/`**: Contains evaluation metrics (`test_evaluation_results.json`) and visualization plots (confusion matrices, training history).
