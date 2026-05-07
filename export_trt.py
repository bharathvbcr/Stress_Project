import torch
import torch.nn as nn
import os
import logging
import argparse

# Check for TensorRT and TorchTensorRT
try:
    import tensorrt as trt
    import torch_tensorrt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TRT-Export")

from models import get_model
from utils import (
    configure_torch_runtime,
    extract_model_state_dict,
    infer_model_feature_dims,
    load_config,
    log_runtime_capabilities,
    safe_get,
)


def export_to_trt(ckpt_path, output_path, config_path="config.json", input_shape=None):
    """
    Exports a trained StressProject model to a TensorRT engine.
    """
    if not TRT_AVAILABLE:
        log.error("TensorRT or Torch-TensorRT not found. Please install them to use this script.")
        log.info("Installation: pip install tensorrt torch-tensorrt")
        return
    runtime_caps = configure_torch_runtime()
    log_runtime_capabilities(log, runtime_caps, prefix="TensorRT Export Runtime")
    if not runtime_caps["cuda_available"]:
        log.error("TensorRT export requires a CUDA-enabled PyTorch runtime.")
        return

    config = load_config(config_path)
    if not config:
        log.error("Failed to load configuration from %s", config_path)
        return

    log.info(f"Loading checkpoint: {ckpt_path}")
    try:
        input_dim_sequence, input_dim_static = infer_model_feature_dims(config)
        target_len = safe_get(config, ['model_config', 'timesfm_context_len'], 512)
        input_shape = input_shape or (1, target_len, input_dim_sequence)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = get_model(config, input_dim_sequence=input_dim_sequence, input_dim_static=input_dim_static)
        model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
        model = model.eval().cuda()
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return

    log.info(f"Exporting to TensorRT with input shape {input_shape}...")
    
    # 1. Tracing the model
    # TRT needs a fixed-size trace for maximum optimization
    example_input = torch.randn(input_shape).cuda()
    
    # 2. Compile with Torch-TensorRT
    # Using FP16 for massive speedup on 30-series/40-series GPUs
    try:
        trt_model = torch_tensorrt.compile(model, 
            inputs=[torch_tensorrt.Input(input_shape)],
            enabled_precisions={torch.float16}, # FP16 half-precision
            workspace_size=1 << 30 # 1GB workspace
        )
        
        # 3. Save the compiled engine (as a TorchScript module)
        torch.jit.save(trt_model, output_path)
        log.info(f"✓ TensorRT engine saved to: {output_path}")
    except Exception as e:
        log.error(f"TensorRT compilation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--out", type=str, default="./outputs/models/model_trt.ts", help="Output path for TRT engine")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON config file")
    args = parser.parse_args()
    
    export_to_trt(args.ckpt, args.out, config_path=args.config)
