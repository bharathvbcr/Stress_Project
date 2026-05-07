import glob
import logging
import os
from contextlib import asynccontextmanager
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import get_model
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_best_model()
    yield


app = FastAPI(title="PhysioPulse Expert API", version="4.0.0", lifespan=lifespan)


class InferenceRequest(BaseModel):
    sequence: List[List[float]]
    static: Optional[List[float]] = None


class InferenceResponse(BaseModel):
    stress_probability: float
    is_stress: bool
    confidence: float


model_state = {
    "module": None,
    "config": None,
    "input_buffer": None,
    "static_buffer": None,
    "target_len": 512,
    "num_channels": 0,
    "input_dim_static": 0,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "runtime": {},
    "checkpoint_path": None,
}


def _select_best_checkpoint(model_dir: str) -> Optional[str]:
    ckpts = glob.glob(os.path.join(model_dir, "*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getctime)


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
    model_state["module"] = None
    model_state["input_buffer"] = None
    model_state["static_buffer"] = None
    model_state["checkpoint_path"] = None

    if not config:
        log.error("Configuration failed to load. API in limited mode.")
        return

    input_dim_sequence, input_dim_static = infer_model_feature_dims(config)
    target_len = safe_get(config, ['model_config', 'timesfm_context_len'], 512)

    model_state["target_len"] = int(target_len)
    model_state["num_channels"] = int(input_dim_sequence)
    model_state["input_dim_static"] = int(input_dim_static)

    checkpoint_path = _select_best_checkpoint(model_dir)
    if checkpoint_path is None:
        log.error("No model checkpoints found. API in limited mode.")
        return

    log.info("API loading checkpoint: %s", checkpoint_path)

    try:
        model = get_model(config, input_dim_sequence=input_dim_sequence, input_dim_static=input_dim_static)
        model = _load_model_weights(model, checkpoint_path)
        model = model.to(model_state["device"]).eval()
        model_state["module"] = model
        model_state["checkpoint_path"] = checkpoint_path

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
        log.info(
            "API model ready. target_len=%s channels=%s static_dim=%s",
            model_state["target_len"],
            model_state["num_channels"],
            model_state["input_dim_static"],
        )
    except Exception as exc:
        log.error("Failed to initialize inference model: %s", exc, exc_info=True)


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
        "precision": runtime_caps.get("precision", "32"),
        "tensorrt_available": runtime_caps.get("tensorrt_available", False),
        "checkpoint_path": model_state["checkpoint_path"],
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if model_state["module"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

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
            static_np = np.asarray(static_values, dtype=np.float32)
            model_state["static_buffer"][0].copy_(
                torch.from_numpy(static_np).to(device=model_state["device"], dtype=torch.float32),
                non_blocking=(model_state["device"].type == "cuda"),
            )
            static_tensor = model_state["static_buffer"]

        with torch.no_grad():
            with _autocast_context():
                logits = model_state["module"](model_state["input_buffer"], static_tensor)
                prob = torch.sigmoid(logits.squeeze()).item()

        return InferenceResponse(
            stress_probability=prob,
            is_stress=prob > 0.5,
            confidence=abs(prob - 0.5) * 2,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
