import glob
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CUDA_PYTHON_AVAILABLE = False
try:
    from cuda.bindings import driver, runtime
    err, = driver.cuInit(0)
    if err == driver.CUresult.CUDA_SUCCESS:
        CUDA_PYTHON_AVAILABLE = True
except Exception:
    pass

from models import get_model
from mlx_backend import configure_mlx_runtime, resolve_inference_backend, resolve_timesfm_hybrid_enabled
from mlx_convert import ensure_mlx_checkpoint
from mlx_hybrid import build_timesfm_hybrid, timesfm_hybrid_predict
from mlx_models import get_mlx_model, load_mlx_checkpoint
from utils import (
    configure_torch_runtime,
    extract_model_state_dict,
    infer_model_feature_dims,
    load_config,
    log_runtime_capabilities,
    safe_get,
    setup_logging,
)

# Setup
setup_logging()
log = logging.getLogger(__name__)

from starlette.middleware.base import BaseHTTPMiddleware

class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        log.info(f"Request: {request.url.path} latency={process_time:.4f}s")
        return response

def _activate_cuda_context():
    if model_state.get("cuda_context") is not None:
        try:
            from cuda import core
            d = core.Device(0)
            prev_ctx = d.set_current(model_state["cuda_context"])
            return prev_ctx
        except Exception:
            pass
    return None


def _deactivate_cuda_context(prev_ctx):
    if prev_ctx is not None:
        try:
            from cuda import core
            d = core.Device(0)
            d.set_current(prev_ctx)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_best_model()
    yield
    # Cleanup low-level CUDA resources
    if model_state.get("cuda_stream") is not None:
        try:
            model_state["cuda_stream"].close()
            log.info("Low-level CUDA Stream closed.")
        except Exception:
            pass
    if model_state.get("cuda_context") is not None:
        try:
            model_state["cuda_context"].close()
            log.info("CUDA Green Context closed.")
        except Exception:
            pass


app = FastAPI(title="PhysioPulse Expert API", version="4.0.0", lifespan=lifespan)
app.add_middleware(LatencyMiddleware)


class InferenceRequest(BaseModel):
    sequence: List[List[float]]
    static: Optional[List[float]] = None


class InferenceResponse(BaseModel):
    stress_probability: float
    is_stress: bool
    confidence: float
    latency_ms: float # Performance tracking


model_state = {
    "module": None,
    "config": None,
    "input_buffer": None,
    "static_buffer": None,
    "target_len": 512,
    "num_channels": 0,
    "input_dim_static": 0,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "backend": "torch",
    "mlx_head": None,
    "runtime": {},
    "checkpoint_path": None,
    "threshold": 0.5,
    "threshold_source": "default",
    "cuda_context": None,
    "cuda_stream": None,
}


def _select_best_checkpoint(model_dir: str, prefer_mlx: bool = False) -> Optional[str]:
    if prefer_mlx:
        mlx_ckpts = glob.glob(os.path.join(model_dir, "*_mlx.npz"))
        if mlx_ckpts:
            return max(mlx_ckpts, key=os.path.getmtime)

    ckpts = glob.glob(os.path.join(model_dir, "*.ckpt"))
    if not ckpts:
        return None

    loss_re = re.compile(r"val_loss=([0-9]+(?:\.[0-9]+)?)")

    def val_loss(path: str) -> float:
        match = loss_re.search(os.path.basename(path))
        return float(match.group(1)) if match else float("inf")

    val_loss_ckpts = [path for path in ckpts if val_loss(path) != float("inf")]
    phase2_ckpts = [
        path for path in val_loss_ckpts
        if "best_model_phase2" in os.path.basename(path)
    ]
    full_phase2_ckpts = [
        path for path in phase2_ckpts
        if os.path.getsize(path) > 50_000_000
    ]
    candidates = full_phase2_ckpts or phase2_ckpts or val_loss_ckpts
    if candidates:
        return min(candidates, key=lambda path: (val_loss(path), -os.path.getmtime(path)))
    return max(ckpts, key=os.path.getmtime)


def _load_calibrated_threshold(results_dir: str, checkpoint_path: str) -> tuple[float, str]:
    checkpoint_name = os.path.basename(checkpoint_path)
    default = (0.5, "default")
    if not os.path.isdir(results_dir):
        return default

    eval_files = sorted(
        glob.glob(os.path.join(results_dir, "eval_pass*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    latest_valid = None
    for path in eval_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        recorded_checkpoint = os.path.basename(str(payload.get("checkpoint", "")))
        calibrated = payload.get("test_calibrated_threshold", {})
        threshold = calibrated.get("threshold")
        if threshold is None:
            threshold = payload.get("validation", {}).get("best_threshold")
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            continue
        if 0.0 <= threshold_value <= 1.0:
            if recorded_checkpoint == checkpoint_name:
                return threshold_value, path
            if latest_valid is None:
                latest_valid = (threshold_value, f"{path} (latest calibration; checkpoint mismatch)")
    if latest_valid is not None:
        return latest_valid
    return default


def _resolve_prediction_threshold(config: dict, checkpoint_path: str) -> tuple[float, str]:
    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    configured = inference_cfg.get("threshold") if isinstance(inference_cfg, dict) else None
    if configured is not None:
        try:
            threshold_value = float(configured)
            if 0.0 <= threshold_value <= 1.0:
                return threshold_value, "config.inference.threshold"
        except (TypeError, ValueError):
            log.warning("Ignoring invalid inference threshold from config: %r", configured)

    results_dir = safe_get(config, ['save_paths', 'results'], "./outputs/results")
    return _load_calibrated_threshold(results_dir, checkpoint_path)


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = extract_model_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Checkpoint load missing keys: %s", missing)
    if unexpected:
        log.warning("Checkpoint load unexpected keys: %s", unexpected)
    return model


def initialize_model_state(config_path: str = "config.json", model_dir: str = "./outputs/models") -> None:
    runtime_caps = configure_torch_runtime()
    log_runtime_capabilities(log, runtime_caps, prefix="API Runtime")

    config = load_config(config_path)
    model_state["runtime"] = runtime_caps
    model_state["config"] = config
    model_state["device"] = torch.device(runtime_caps["device"])
    model_state["backend"] = "torch"
    model_state["module"] = None
    model_state["input_buffer"] = None
    model_state["static_buffer"] = None
    model_state["checkpoint_path"] = None
    model_state["threshold"] = 0.5
    model_state["threshold_source"] = "default"
    
    # Close any existing stream or green context before re-initializing
    if model_state.get("cuda_stream") is not None:
        try:
            model_state["cuda_stream"].close()
        except Exception:
            pass
        model_state["cuda_stream"] = None
    if model_state.get("cuda_context") is not None:
        try:
            model_state["cuda_context"].close()
        except Exception:
            pass
        model_state["cuda_context"] = None

    if not config:
        log.error("Configuration failed to load. API in limited mode.")
        return

    input_dim_sequence, input_dim_static = infer_model_feature_dims(config)
    target_len = safe_get(config, ['model_config', 'timesfm_context_len'], 512)

    model_state["target_len"] = int(target_len)
    model_state["num_channels"] = int(input_dim_sequence)
    model_state["input_dim_static"] = int(input_dim_static)

    model_type = safe_get(config, ["model_config", "type"], "LSTM").upper()
    timesfm_hybrid = (
        model_type == "TIMESFM"
        and resolve_timesfm_hybrid_enabled(runtime_caps.get("backend_request"))
    )
    use_mlx = timesfm_hybrid or (
        resolve_inference_backend(
            model_type,
            None,
            mlx_available=bool(runtime_caps.get("mlx_available")),
            requested=runtime_caps.get("backend_request"),
        )
        == "mlx"
    )
    checkpoint_path = _select_best_checkpoint(model_dir, prefer_mlx=use_mlx)
    if checkpoint_path is None and use_mlx:
        checkpoint_path = _select_best_checkpoint(model_dir, prefer_mlx=False)
        use_mlx = False
    if checkpoint_path and (checkpoint_path.endswith(".mlx.npz") or checkpoint_path.endswith("_mlx.npz")):
        use_mlx = True
        timesfm_hybrid = False
    if checkpoint_path is None:
        log.error("No model checkpoints found. API in limited mode.")
        return

    log.info(
        "API loading checkpoint: %s (backend=%s)",
        checkpoint_path,
        "mlx_hybrid" if timesfm_hybrid else ("mlx" if use_mlx else "torch"),
    )

    try:
        if timesfm_hybrid:
            configure_mlx_runtime()
            torch_model, mlx_head, _ = build_timesfm_hybrid(
                config,
                checkpoint_path,
                input_dim_sequence,
                input_dim_static,
                device=model_state["device"],
            )
            model_state["module"] = torch_model
            model_state["mlx_head"] = mlx_head
            model_state["backend"] = "mlx_hybrid"
            model_state["input_buffer"] = None
            model_state["static_buffer"] = None
        elif use_mlx:
            configure_mlx_runtime()
            mlx_ckpt = checkpoint_path
            if not checkpoint_path.endswith(".npz"):
                auto_convert = os.environ.get("STRESS_MLX_AUTO_CONVERT", "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if auto_convert:
                    mlx_ckpt = ensure_mlx_checkpoint(
                        config,
                        checkpoint_path,
                        input_dim_sequence,
                        input_dim_static,
                    )
            model = get_mlx_model(config, input_dim_sequence=input_dim_sequence, input_dim_static=input_dim_static)
            load_mlx_checkpoint(model, mlx_ckpt)
            # Inference parity with the torch path: eval mode disables Dropout
            # and makes BatchNorm use running stats instead of normalizing the
            # single-sample request batch by its own mean/var.
            model.eval()
            model_state["module"] = model
            # Compile the frozen-weight forward once (large win for LSTM/CNN-LSTM
            # recurrences; no-op for attention models). Weights never change
            # after load, so capturing them in the compiled graph is safe.
            mlx_compile = os.environ.get("STRESS_MLX_COMPILE", "1").strip().lower() in {"1", "true", "yes", "on"}
            if mlx_compile:
                import mlx.core as mx

                model_state["mlx_forward"] = mx.compile(lambda s, st: model(s, st))
            else:
                model_state["mlx_forward"] = model
            model_state["mlx_head"] = None
            model_state["backend"] = "mlx"
            model_state["device"] = torch.device("cpu")
            model_state["input_buffer"] = None
            model_state["static_buffer"] = None
        else:
            model = get_model(config, input_dim_sequence=input_dim_sequence, input_dim_static=input_dim_static)
            model = _load_model_weights(model, checkpoint_path)
            model = model.to(model_state["device"]).eval()
            model_state["module"] = model
            model_state["backend"] = "torch"
        model_state["checkpoint_path"] = checkpoint_path
        threshold, threshold_source = _resolve_prediction_threshold(config, checkpoint_path)
        model_state["threshold"] = threshold
        model_state["threshold_source"] = threshold_source
        # Train/serve parity: apply the same per-channel normalization used in training.
        model_state["seq_mean"], model_state["seq_std"] = _load_norm_stats(
            config, model_state["num_channels"]
        )

        if not use_mlx and not timesfm_hybrid:
            model_state["input_buffer"] = torch.zeros(
                (1, model_state["target_len"], model_state["num_channels"]),
                device=model_state["device"],
                dtype=torch.float32,
            )
            if model_state["input_dim_static"] > 0:
                model_state["static_buffer"] = torch.zeros(
                    (1, model_state["input_dim_static"]),
                    device=model_state["device"],
                    dtype=torch.float32,
                )
        # Initialize Green Context if requested via environment variables
        sm_count = int(os.environ.get("STRESS_GREEN_CONTEXT_SM_COUNT", "0"))
        if not use_mlx and not timesfm_hybrid and sm_count > 0 and CUDA_PYTHON_AVAILABLE and model_state["device"].type == "cuda":
            try:
                from cuda import core
                d = core.Device(0)
                # Primary context bound to thread first
                d.set_current()
                groups, rem = d.resources.sm.split(core.SMResourceOptions(count=sm_count))
                worker_idx = int(os.environ.get("STRESS_WORKER_IDX", "0"))
                chosen_group = groups[worker_idx % len(groups)]
                opt = core.ContextOptions(resources=[chosen_group])
                model_state["cuda_context"] = d.create_context(opt)
                d.set_current(model_state["cuda_context"])
                log.info("Initialized virtualized CUDA Green Context with SM count: %d", chosen_group.sm_count)
            except Exception as e:
                log.warning("Could not initialize CUDA Green Context (falling back to primary context): %s", e)

        # Pre-allocate Stream in the current context
        if not use_mlx and not timesfm_hybrid and CUDA_PYTHON_AVAILABLE and model_state["device"].type == "cuda":
            try:
                from cuda import core
                d = core.Device(0)
                if model_state.get("cuda_context") is not None:
                    d.set_current(model_state["cuda_context"])
                else:
                    d.set_current()
                model_state["cuda_stream"] = d.create_stream()
                log.info("Initialized low-level CUDA Stream for async H2D copies.")
            except Exception as stream_exc:
                log.warning("Could not pre-allocate low-level CUDA Stream: %s", stream_exc)

        log.info(
            "API model ready. target_len=%s channels=%s static_dim=%s threshold=%.4f source=%s",
            model_state["target_len"],
            model_state["num_channels"],
            model_state["input_dim_static"],
            model_state["threshold"],
            model_state["threshold_source"],
        )
    except Exception as exc:
        log.error("Failed to initialize inference model: %s", exc, exc_info=True)


def _load_norm_stats(cfg, num_channels):
    """Load train-time per-channel normalization stats so inference matches training."""
    out_dir = safe_get(cfg, ['save_paths', 'processed_data'], './outputs/processed_data')
    path = os.path.join(os.path.abspath(out_dir), 'sequence_norm_stats.json')
    if not os.path.exists(path):
        log.info("No sequence_norm_stats.json found; inference will use raw (unnormalized) inputs.")
        return None, None
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            stats = json.load(fh)
        mean = np.asarray(stats['mean'], dtype=np.float32)
        std = np.asarray(stats['std'], dtype=np.float32)
        if mean.shape[0] != num_channels or std.shape[0] != num_channels:
            log.warning(
                "Norm stats channel count (%s) != model channels (%s); skipping normalization.",
                mean.shape[0], num_channels,
            )
            return None, None
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        log.info("Loaded sequence normalization stats from %s", path)
        return mean, std
    except Exception as exc:
        log.warning("Failed to load normalization stats (%s); inference uses raw inputs.", exc)
        return None, None


def _autocast_context():
    runtime_caps = model_state["runtime"]
    device = model_state["device"]
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if runtime_caps.get("precision") == "bf16-mixed" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_best_model():
    initialize_model_state()


@app.get("/health")
def health():
    runtime_caps = model_state["runtime"] or {}
    return {
        "status": "healthy" if model_state["module"] is not None else "degraded",
        "model_loaded": model_state["module"] is not None,
        "torch_version": runtime_caps.get("torch_version"),
        "cuda_available": runtime_caps.get("cuda_available", False),
        "device": runtime_caps.get("device", "cpu"),
        "device_name": runtime_caps.get("device_name", "cpu"),
        "backend": model_state.get("backend", "torch"),
        "mlx_available": runtime_caps.get("mlx_available", False),
        "precision": runtime_caps.get("precision", "32"),
        "tensorrt_available": runtime_caps.get("tensorrt_available", False),
        "checkpoint_path": model_state["checkpoint_path"],
        "threshold": model_state["threshold"],
        "threshold_source": model_state["threshold_source"],
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if model_state["module"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    start_p = time.perf_counter()
    prev_ctx = _activate_cuda_context()
    try:
        seq_np = np.asarray(request.sequence, dtype=np.float32)
        if seq_np.ndim != 2:
            raise HTTPException(status_code=400, detail="sequence must be 2D: [time, channels].")
        if seq_np.shape[0] == 0:
            raise HTTPException(status_code=400, detail="sequence must contain at least one timestep.")
        if seq_np.shape[1] != model_state["num_channels"]:
            raise HTTPException(
                status_code=400,
                detail=f"expected {model_state['num_channels']} channels, got {seq_np.shape[1]}",
            )

        target_len = model_state["target_len"]
        if seq_np.shape[0] > target_len:
            seq_np = seq_np[-target_len:]
        elif seq_np.shape[0] < target_len:
            pad_width = target_len - seq_np.shape[0]
            seq_np = np.pad(seq_np, ((pad_width, 0), (0, 0)), mode="edge")

        # Apply training-time per-channel normalization (train/serve parity).
        seq_mean = model_state.get("seq_mean")
        seq_std = model_state.get("seq_std")
        if seq_mean is not None and seq_std is not None:
            seq_np = (seq_np - seq_mean) / seq_std

        # Ensure array is contiguous in memory before accessing raw pointer
        seq_np = np.ascontiguousarray(np.asarray(seq_np, dtype=np.float32))

        if model_state.get("backend") == "mlx_hybrid":
            import mlx.core as mx

            prep_time = time.perf_counter() - start_p
            inf_start = time.perf_counter()
            seq_tensor = torch.from_numpy(seq_np).to(device=model_state["device"], dtype=torch.float32).unsqueeze(0)
            static_tensor = None
            if model_state["input_dim_static"] > 0:
                static_values = request.static or ([0.0] * model_state["input_dim_static"])
                static_tensor = torch.tensor([static_values], device=model_state["device"], dtype=torch.float32)
            prob = timesfm_hybrid_predict(
                model_state["module"],
                model_state["mlx_head"],
                seq_tensor,
                static_tensor,
            )
            inf_time = time.perf_counter() - inf_start
            total_latency = (time.perf_counter() - start_p) * 1000
            threshold = model_state["threshold"]
            return InferenceResponse(
                stress_probability=prob,
                is_stress=prob > threshold,
                confidence=min(1.0, abs(prob - threshold) / max(threshold, 1.0 - threshold, 1e-6)),
                latency_ms=total_latency,
            )

        if model_state.get("backend") == "mlx":
            import mlx.core as mx

            prep_time = time.perf_counter() - start_p
            inf_start = time.perf_counter()
            seq_mx = mx.array(seq_np[None, ...], dtype=mx.float32)
            static_mx = None
            if model_state["input_dim_static"] > 0:
                static_values = request.static or ([0.0] * model_state["input_dim_static"])
                if len(static_values) != model_state["input_dim_static"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"expected {model_state['input_dim_static']} static features, got {len(static_values)}",
                    )
                static_mx = mx.array([static_values], dtype=mx.float32)
            forward = model_state.get("mlx_forward") or model_state["module"]
            logits = forward(seq_mx, static_mx)
            prob = float(mx.sigmoid(logits.reshape(-1))[0].item())
            inf_time = time.perf_counter() - inf_start
            total_latency = (time.perf_counter() - start_p) * 1000
            threshold = model_state["threshold"]
            return InferenceResponse(
                stress_probability=prob,
                is_stress=prob > threshold,
                confidence=min(1.0, abs(prob - threshold) / max(threshold, 1.0 - threshold, 1e-6)),
                latency_ms=total_latency,
            )

        buffer_bytes = model_state["input_buffer"].nelement() * model_state["input_buffer"].element_size()
        
        stream = model_state.get("cuda_stream")
        stream_copy_triggered = False
        
        if CUDA_PYTHON_AVAILABLE and model_state["device"].type == "cuda" and seq_np.nbytes == buffer_bytes:
            if stream is not None:
                err, = runtime.cudaMemcpyAsync(
                    model_state["input_buffer"].data_ptr(),
                    seq_np.ctypes.data,
                    seq_np.nbytes,
                    runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    int(stream.handle)
                )
                if err == runtime.cudaError_t.cudaSuccess:
                    stream_copy_triggered = True
                else:
                    runtime.cudaMemcpy(
                        model_state["input_buffer"].data_ptr(),
                        seq_np.ctypes.data,
                        seq_np.nbytes,
                        runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
                    )
            else:
                runtime.cudaMemcpy(
                    model_state["input_buffer"].data_ptr(),
                    seq_np.ctypes.data,
                    seq_np.nbytes,
                    runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
                )
        else:
            seq_tensor = torch.from_numpy(seq_np).to(device=model_state["device"], dtype=torch.float32)
            model_state["input_buffer"][0].copy_(seq_tensor, non_blocking=(model_state["device"].type == "cuda"))

        static_tensor = None
        if model_state["input_dim_static"] > 0:
            static_values = request.static or ([0.0] * model_state["input_dim_static"])
            if len(static_values) != model_state["input_dim_static"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"expected {model_state['input_dim_static']} static features, got {len(static_values)}",
                )
            static_np = np.ascontiguousarray(np.asarray(static_values, dtype=np.float32))
            static_buffer_bytes = model_state["static_buffer"].nelement() * model_state["static_buffer"].element_size()
            
            if stream_copy_triggered and static_np.nbytes == static_buffer_bytes:
                err, = runtime.cudaMemcpyAsync(
                    model_state["static_buffer"].data_ptr(),
                    static_np.ctypes.data,
                    static_np.nbytes,
                    runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    int(stream.handle)
                )
                if err != runtime.cudaError_t.cudaSuccess:
                    model_state["static_buffer"][0].copy_(
                        torch.from_numpy(static_np).to(device=model_state["device"], dtype=torch.float32),
                        non_blocking=True,
                    )
            elif CUDA_PYTHON_AVAILABLE and model_state["device"].type == "cuda" and static_np.nbytes == static_buffer_bytes:
                err, = runtime.cudaMemcpy(
                    model_state["static_buffer"].data_ptr(),
                    static_np.ctypes.data,
                    static_np.nbytes,
                    runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
                )
                if err != runtime.cudaError_t.cudaSuccess:
                    model_state["static_buffer"][0].copy_(
                        torch.from_numpy(static_np).to(device=model_state["device"], dtype=torch.float32),
                        non_blocking=True,
                    )
            else:
                model_state["static_buffer"][0].copy_(
                    torch.from_numpy(static_np).to(device=model_state["device"], dtype=torch.float32),
                    non_blocking=(model_state["device"].type == "cuda"),
                )
            static_tensor = model_state["static_buffer"]

        if stream_copy_triggered and stream is not None:
            stream.sync()

        prep_time = time.perf_counter() - start_p

        inf_start = time.perf_counter()
        with torch.no_grad():
            with _autocast_context():
                logits = model_state["module"](model_state["input_buffer"], static_tensor)
                prob = torch.sigmoid(logits.squeeze()).item()

        inf_time = time.perf_counter() - inf_start
        total_latency = (time.perf_counter() - start_p) * 1000 # ms

        log.debug(f"API Inference: prep={prep_time*1000:.2f}ms, compute={inf_time*1000:.2f}ms")

        threshold = model_state["threshold"]
        return InferenceResponse(
            stress_probability=prob,
            is_stress=prob > threshold,
            confidence=min(1.0, abs(prob - threshold) / max(threshold, 1.0 - threshold, 1e-6)),
            latency_ms=total_latency
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _deactivate_cuda_context(prev_ctx)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
