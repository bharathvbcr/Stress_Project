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

def export_to_trt(ckpt_path, output_path, input_shape=(1, 512, 8)):
    """
    Exports a trained StressProject model to a TensorRT engine.
    """
    if not TRT_AVAILABLE:
        log.error("TensorRT or Torch-TensorRT not found. Please install them to use this script.")
        log.info("Installation: pip install tensorrt torch-tensorrt")
        return

    log.info(f"Loading checkpoint: {ckpt_path}")
    # Lazy import to avoid dependency issues
    from lightning_module import StressLightningModule
    
    try:
        model_module = StressLightningModule.load_from_checkpoint(ckpt_path, model=None)
        model = model_module.model.eval().cuda()
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
    args = parser.parse_args()
    
    export_to_trt(args.ckpt, args.out)
