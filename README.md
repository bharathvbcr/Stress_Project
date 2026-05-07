# StressProject

StressProject is a physiological time-series machine learning project for
stress detection. It supports end-to-end preprocessing, subject-safe dataset
splitting, PyTorch Lightning training, TimesFM 2.5 foundation-model experiments,
FastAPI inference, benchmarking, TensorRT export, validation, and repository
governance tooling.

The root repository owns the stress-prediction pipeline. The `timesfm/`
directory is a packaged TimesFM workspace with its own guidance and should be
treated as a separate subproject when making changes under that tree.

## Capabilities

- **Signal processing pipeline**: load physiological signals, resample and align
  channels, extract static features, build time-series windows, and generate
  train/validation/test splits without subject leakage.
- **Model training**: train LSTM, CNN-LSTM, Transformer, and TimesFM-backed
  stress classifiers with PyTorch Lightning.
- **Runtime optimization**: detect CUDA, precision, TF32, TensorRT, and
  Torch-TensorRT availability through shared runtime utilities.
- **Inference service**: serve trained checkpoints through a FastAPI API with
  health reporting, input validation, checkpoint loading, sequence
  pad/truncate handling, and preallocated inference buffers.
- **Foundation-model path**: run dedicated TimesFM 2.5 training through
  `run_pipeline_timesfm.py` and `timesfm_wrapper.py`.
- **Validation and benchmarking**: run Deepchecks integrity checks, latency
  benchmarks, TensorRT export, and runtime smoke tests.
- **Repository intelligence**: use `docs/repo-map.md`, `docs/repo-map.json`,
  AGENTS guidance, and GitNexus to navigate ownership and impact safely.

## Architecture

```mermaid
flowchart TD
  User["User / CLI"] --> Config["Configuration<br/>conf/*.yaml or config.json"]
  Config --> Entrypoint{"Entrypoint"}

  Entrypoint --> Train["main.py<br/>standard Lightning training"]
  Entrypoint --> TimesFMTrain["run_pipeline_timesfm.py<br/>TimesFM training"]
  Entrypoint --> Service["api.py<br/>FastAPI inference"]
  Entrypoint --> Bench["benchmark.py<br/>latency benchmark"]
  Entrypoint --> Export["export_trt.py<br/>TensorRT export"]
  Entrypoint --> Validate["validation.py<br/>Deepchecks validation"]

  Train --> Cache{"Arrow dataset cache?"}
  Cache -- "missing or forced" --> Preprocess["preprocessing.py<br/>load, resample, align, label"]
  Preprocess --> Features["feature_extraction.py<br/>static features and HRV"]
  Features --> Windows["windowing.py<br/>window construction"]
  Windows --> Splits["data_pipeline.py<br/>subject-safe splits"]
  Cache -- "available" --> DataModule["lightning_data.py<br/>Arrow/HF DataModule"]
  Splits --> DataModule

  DataModule --> ModelFactory["models.py<br/>get_model"]
  TimesFMTrain --> ModelFactory
  ModelFactory --> Classical["LSTM / CNN-LSTM / Transformer"]
  ModelFactory --> TimesFM["StressTimesFM"]
  TimesFM --> TimesFMWrapper["timesfm_wrapper.py<br/>TimesFM 2.5 embeddings"]

  Classical --> Lightning["lightning_module.py<br/>loss, metrics, steps"]
  TimesFM --> Lightning
  Lightning --> Trainer["Lightning Trainer<br/>precision, DDP, checkpoints"]
  Trainer --> Checkpoints["outputs/models<br/>*.ckpt"]

  Checkpoints --> Service
  Service --> Runtime["utils.py<br/>runtime and checkpoint helpers"]
  Runtime --> Health["GET /health"]
  Runtime --> Predict["POST /predict<br/>validate, pad/truncate, infer"]

  Checkpoints --> Export
  Checkpoints --> Bench
  DataModule --> Validate

  RepoMap["docs/repo-map.md/json<br/>ownership map"] -. guides .-> Entrypoint
  Tests["tests/test_runtime_paths.py<br/>runtime smoke tests"] -. verifies .-> DataModule
  Tests -. verifies .-> Service
  Tests -. verifies .-> Bench
```

## Project Structure

```text
.
|-- conf/                       # Hydra configuration groups
|   |-- config.yaml             # Main config composition
|   |-- dataset/wesad.yaml      # Dataset metadata and sampling rates
|   |-- model/*.yaml            # Model-specific settings
|   |-- processing/default.yaml # Preprocessing/windowing settings
|   `-- training/standard.yaml  # Training settings
|-- docs/
|   |-- repo-map.md             # Canonical repo ownership and dependency map
|   |-- repo-map.json           # Machine-readable repo map
|   `-- repo-map-refresh.ps1    # Map refresh helper
|-- tests/
|   `-- test_runtime_paths.py   # DataModule/API/benchmark smoke tests
|-- timesfm/                    # Packaged TimesFM project and local source
|-- main.py                     # Primary Lightning training entrypoint
|-- run_pipeline.py             # Legacy/alternate training pipeline
|-- run_pipeline_timesfm.py     # TimesFM 2.5 training pipeline
|-- run_lightning.py            # Lightning run helper
|-- api.py                      # FastAPI inference server
|-- benchmark.py                # Runtime latency benchmark
|-- export_trt.py               # TensorRT export helper
|-- validation.py               # Deepchecks integrity validation
|-- data_loader.py              # Raw data loading
|-- preprocessing.py            # Preprocessing orchestration
|-- signal_processing.py        # Resampling and alignment
|-- feature_extraction.py       # Static feature and HRV extraction
|-- windowing.py                # Time-series window construction
|-- data_pipeline.py            # Splits, sampling, DataLoader construction
|-- lightning_data.py           # LightningDataModule
|-- lightning_module.py         # LightningModule wrapper
|-- models.py                   # Model architectures and factory
|-- timesfm_wrapper.py          # TimesFM backbone loader and extractor
|-- utils.py                    # Config, runtime, checkpoint utilities
|-- config.json                 # Legacy/runtime JSON config
`-- requirements.txt            # Python dependencies
```

Runtime and experiment artifacts are not source-of-truth by default:
`outputs/`, `scratch/`, `.venv/`, and `__pycache__/`.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build that matches the local CUDA
runtime before installing project dependencies. TensorRT export additionally
requires `tensorrt` and `torch-tensorrt`.

## Common Workflows

### Train the Standard Pipeline

```powershell
python main.py
```

Regenerate the Arrow/Hugging Face dataset cache:

```powershell
python main.py force_preprocess=True
```

### Train the TimesFM Pipeline

```powershell
python run_pipeline_timesfm.py
```

### Start the Inference API

```powershell
python api.py
```

Check service health:

```powershell
curl http://localhost:8000/health
```

Submit a prediction request:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"sequence\": [[0,0,0,0,0,0,0,0]]}"
```

The API loads the newest checkpoint from `outputs/models`, rebuilds the model
from configuration, validates the request, pads or truncates sequences to the
configured context length, and returns stress probability, label, and
confidence.

### Benchmark, Validate, and Export

```powershell
python benchmark.py
python validation.py
python export_trt.py --ckpt outputs/models/best_model.ckpt
```

### Run Tests

```powershell
python -m pytest tests/test_runtime_paths.py
```

The runtime smoke tests cover Lightning batch handling, Arrow-backed
DataModule batches, API initialization and prediction on CPU, and benchmark
execution.

## Useful Docs

| Document | Purpose |
| --- | --- |
| `docs/repo-map.md` | Canonical ownership matrix, source index, and dependency anchors. Start here before broad edits. |
| `docs/repo-map.json` | Machine-readable version of the repo map for automation and tooling. |
| `docs/repo-map-refresh.ps1` | Regenerates the markdown and JSON repo maps after structural file changes. |
| `AGENTS.md` | Top-level agent policy for navigation, ownership, GitNexus use, and repo-map upkeep. |
| `CLAUDE.md` | Parallel top-level assistant guidance for this repository. |
| `timesfm/AGENTS.md` | Additional guidance for the packaged TimesFM subproject. |
| `conf/config.yaml` | Main Hydra config composition. |
| `config.json` | Legacy/runtime JSON config used by API and utility paths. |
| `tests/test_runtime_paths.py` | Minimal runtime verification surface for the main Python paths. |

Refresh the repo map after structural moves or ownership changes:

```powershell
pwsh .\docs\repo-map-refresh.ps1 -RepoRoot (Get-Location).Path
```

GitNexus is configured for this repository. Use it to inspect unfamiliar
execution flows, run impact analysis before symbol edits, and detect affected
flows before committing.
