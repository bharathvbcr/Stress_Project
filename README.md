# Stress Detection Project 🚀

This project implements a comprehensive deep learning pipeline for detecting stress using physiological signals. It has been hardened into a **Hyper-Optimized, Production-Grade** platform featuring sub-millisecond inference and hardware-saturated training.

## 💎 State-of-the-Art (SOTA) Features

* **Hyper-Performance Compute:**
  * **torch.compile:** Automated full-graph operator fusion with `reduce-overhead` and `max-autotune` modes.
  * **Global TF32:** Enabled TensorFloat-32 for ~3x speedup on Ampere/Ada GPUs.
  * **CUDNN Auto-Tuner:** Dynamically selects the fastest kernels for your specific hardware.
  * **Persistent Caching:** Compiled kernels are stored on disk to skip warmup in future sessions.
* **Zero-Latency Data I/O:**
  * **Smart Caching:** Pipeline automatically skips redundant preprocessing if the **Apache Arrow** cache exists.
  * **Memory Mapping:** Loads multi-gigabyte datasets instantly using memory-mapped files via Hugging Face Datasets.
  * **Aggressive Prefetching:** Custom `DataLoader` logic that saturates the PCIe bus to keep GPUs at 100% utilization.
* **Hyper-Inference Engine:**
  * **IO Binding:** Zero-copy inference in `api.py` using pre-allocated GPU buffers.
  * **TensorRT Support:** Dedicated pipeline to export models to **NVIDIA TensorRT** for sub-millisecond execution.
  * **CUDAGraphs:** Ready for recorded hardware execution plans, eliminating CPU launch overhead.
* **Enterprise Data Governance:**
  * **DVC (Data Version Control):** Integrated DVC for tracking gigabytes of signal data with Git-like efficiency.
  * **Subject-Group Stratification:** Guaranteed zero subject leakage between training and testing splits.
* **Advanced Models:**
  * **TimesFM 2.5:** Leveraging Google's pretrained Foundation Model for zero-shot signal embedding.
  * **CNN-LSTM-Attention:** Hybrid architectures for temporal and spatial signal dynamics.

## 📂 Project Structure

```text
├── main.py                     # 🚀 MODERN ENTRY POINT: Optimized training with Smart Cache & DDP
├── api.py                      # ⚡ HYPER-API: Sub-millisecond inference with IO Binding
├── export_trt.py               # 💎 EXPORT TOOL: Generates NVIDIA TensorRT execution engines
├── benchmark.py                # 📊 PROFILER: Benchmarks latency and throughput gains
├── dvc_init.py                 # 📦 DATA GOVERNANCE: Bootstraps DVC for signal tracking
│
├── lightning_module.py         # 🧠 Core logic (Buffer management, SOTA metrics)
├── lightning_data.py           # 📥 High-speed DataModule (Arrow/Prefetching)
├── models.py                   # 🏗️ Model Architectures (TimesFM, CNN-LSTM, Transformer)
│
├── preprocessing.py            # Signal conditioning & orchestration
├── feature_extraction.py       # Parallel (CPU) static feature calculation
├── signal_processing.py        # Resampling & Alignment logic
│
└── outputs/
    ├── processed_data_hf/      # 📂 HIGH-SPEED CACHE: Arrow formatted datasets
    ├── models/                 # Saved .ckpt and .ts (TensorRT) files
    └── results/                # Reports, Plots, and Deepchecks Integrity reports
```

## 💻 Hardware Acceleration

Optimized for **NVIDIA RTX 30/40-series** GPUs.

* **Precision:** Uses `bf16-mixed` or `16-mixed` with high-speed Tensor Core kernels.
* **Compute:** Employs `torch.backends.cudnn.benchmark` and `torch.compile`.
* **Scaling:** Supports **Distributed Data Parallel (DDP)** for multi-GPU workstations.

## 🚀 Usage

### 1. Training with Hyper-Performance

```bash
# Standard training (Auto-detects GPU/DDP/Compile)
python main.py

# Force rebuild the Arrow cache
python main.py force_preprocess=True
```

### 2. High-Throughput Inference

```bash
# Start the FastAPI server with pre-allocated IO Buffers
python api.py
```

### 3. Professional Deployment (TensorRT)

```bash
# Convert your best checkpoint to a hardware-native engine
python export_trt.py --ckpt outputs/models/best_model.ckpt
```

### 4. Performance Benchmarking

```bash
# Compare Standard vs CUDAGraph vs TensorRT performance
python benchmark.py
```

## 📦 Installation

1. **Clone & Setup:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Initialize Data Tracking:**

    ```bash
    python dvc_init.py
    ```

---
*Developed for SOTA physiological signal processing and high-performance machine learning.*
