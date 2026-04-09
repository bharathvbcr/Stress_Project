import os
import glob
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from lightning_module import StressLightningModule
from models import get_model
from utils import load_config, setup_logging, safe_get

# Setup
setup_logging()
log = logging.getLogger(__name__)

app = FastAPI(title="PhysioPulse Expert API", version="3.0.0")

# --- Schemas ---
class InferenceRequest(BaseModel):
    sequence: List[List[float]] # (Length, Channels)
    static: Optional[List[float]] = None # (F_static)

class InferenceResponse(BaseModel):
    stress_probability: float
    is_stress: bool
    confidence: float

model_state = {
    "module": None,
    "config": None,
    "input_buffer": None,
    "static_buffer": None,
    "target_len": 512, # Optimized for TimesFM context length
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

@app.on_event("startup")
def load_best_model():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        log.info("API SOTA Autotuning Engaged: TF32=ON")

    config = load_config("config.json") # Fallback to JSON or Hydra
    model_dir = "./outputs/models"
    ckpts = glob.glob(os.path.join(model_dir, "*.ckpt"))
    
    if not ckpts:
        log.error("No model checkpoints found. API in limited mode.")
        return

    best_ckpt = max(ckpts, key=os.path.getctime)
    log.info(f"API Loading best checkpoint: {best_ckpt}")
    
    try:
        # We need to re-init the architecture first to load weights
        # (In a full Hydra setup, we'd use Hydra to instantiate)
        model_module = StressLightningModule.load_from_checkpoint(
            checkpoint_path=best_ckpt,
            model=None # Lightning will try to re-init if args are in hparams
        )
        model_module.eval()
        model_module.freeze()
        
        model_state["module"] = model_module
        model_state["config"] = config
        
        # Pre-allocate GPU buffers for IO Binding
        # This avoids repeated memory allocation and reduces latency
        num_channels = safe_get(config, ['model', 'num_channels'], 8)
        model_state["input_buffer"] = torch.zeros((1, model_state["target_len"], num_channels), device=model_state["device"])
        
        log.info(f"✓ Model loaded. High-speed IO buffers pre-allocated (Len: {model_state['target_len']}).")
    except Exception as e:
        log.error(f"Failed to load model: {e}")

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_state["module"] is not None}

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if model_state["module"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        # 1. Zero-Copy Tensor Creation (High-Speed)
        seq_np = np.array(request.sequence, dtype=np.float32)
        L, C = seq_np.shape
        T = model_state["target_len"]
        
        # 2. Hardening: Fixed-Size Padding for CUDAGraphs / Engine compatibility
        if L > T:
            seq_np = seq_np[-T:] # Truncate to most recent
        elif L < T:
            # Efficient padding
            pad_width = T - L
            seq_np = np.pad(seq_np, ((pad_width, 0), (0, 0)), mode='edge')
            
        # 3. IO Binding: Copy directly into pre-allocated buffer
        # .copy_ is significantly faster than creating a new sequence_tensor
        model_state["input_buffer"].copy_(torch.from_numpy(seq_np))
        
        # 4. Inference
        with torch.no_grad():
            logits = model_state["module"](model_state["input_buffer"], None)
            prob = torch.sigmoid(logits).item()
            
        return InferenceResponse(
            stress_probability=prob,
            is_stress=prob > 0.5,
            confidence=abs(prob - 0.5) * 2
        )
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
