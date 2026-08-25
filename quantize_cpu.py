# quantize_cpu.py — Post-Training Static Quantization for CPU Deployment
import argparse
import logging
import os
import sys

import torch

from models import get_model
from utils import extract_model_state_dict, infer_model_feature_dims, load_config, safe_get

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Quantization")

MLX_NATIVE_TYPES = frozenset({"LSTM", "CNN-LSTM", "TRANSFORMER", "PATCHTST", "PATCH-TST"})


def _resolve_backend(request: str, model_type: str) -> str:
    request = (request or "auto").strip().lower()
    if request in {"mlx", "torch"}:
        return request
    try:
        from mlx_backend import is_mlx_available, resolve_inference_backend

        if is_mlx_available() and resolve_inference_backend(model_type, None) == "mlx":
            return "mlx"
    except ImportError:
        pass
    return "torch"


def quantize_model(ckpt_path, output_path, config_path="config.json", backend: str = "auto"):
    """
    Applies post-training static quantization (INT8) to PyTorch models for CPU speedup.

    MLX-native checkpoints (``.mlx.npz``) and TimesFM hybrid models are not supported
  here — use ``mlx_benchmark.py`` or hybrid head export instead.
    """
    config = load_config(config_path)
    model_type = safe_get(config, ["model_config", "type"], "LSTM").upper()
    resolved = _resolve_backend(backend, model_type)

    if ckpt_path.endswith(".mlx.npz") or ckpt_path.endswith("_mlx.npz"):
        log.error(
            "Checkpoint %s is an MLX weights file. PyTorch PTQ does not apply. "
            "Use mlx_benchmark.py for MLX latency measurement.",
            ckpt_path,
        )
        sys.exit(1)

    if model_type == "TIMESFM":
        log.error(
            "TimesFM uses a PyTorch backbone with an optional MLX classifier head. "
            "PyTorch PTQ is not supported. Export the MLX head with export_timesfm_mlx_artifacts "
            "or use STRESS_BACKEND=mlx_hybrid for inference."
        )
        sys.exit(1)

    if resolved == "mlx" and model_type in MLX_NATIVE_TYPES:
        log.error(
            "Model type %s is trained/deployed via MLX on Apple Silicon. "
            "PyTorch static quantization does not apply to MLX .npz weights. "
            "Pass --backend torch to quantize a PyTorch checkpoint explicitly.",
            model_type,
        )
        sys.exit(1)

    from benchmark import measure_latency_and_memory

    input_dim_sequence, input_dim_static = infer_model_feature_dims(config)
    target_len = safe_get(config, ["model_config", "timesfm_context_len"], 512)
    input_shape = (1, target_len, input_dim_sequence)

    log.info("Loading model for CPU quantization: %s", ckpt_path)
    model = get_model(config, input_dim_sequence=input_dim_sequence, input_dim_static=input_dim_static)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if hasattr(model, "lora_adapters"):
        from timesfm_lora import restore_timesfm_lora_adapters

        restore_timesfm_lora_adapters(model, state_dict, config=config)
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    model_prepared = torch.quantization.prepare(model)

    log.info("Calibrating quantization observers...")
    for _ in range(50):
        dummy_seq = torch.randn(input_shape)
        dummy_static = (
            torch.randn(1, input_dim_static) if input_dim_static > 0 else None
        )
        _ = model_prepared(dummy_seq, dummy_static)

    log.info("Converting to INT8 quantized model...")
    model_int8 = torch.quantization.convert(model_prepared)

    torch.save(model_int8.state_dict(), output_path)
    log.info("Quantized model saved to: %s", output_path)

    log.info("=== Benchmarking CPU performance ===")
    example_input = torch.randn(input_shape)

    log.info("Base CPU (FP32):")
    base_avg = measure_latency_and_memory(model, example_input, 100, "cpu", "Base FP32")

    log.info("Quantized CPU (INT8):")
    q_avg = measure_latency_and_memory(model_int8, example_input, 100, "cpu", "Quantized INT8")

    log.info("CPU QUANTIZATION SPEEDUP: %.2fx", base_avg["avg"] / q_avg["avg"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CPU static quantization (not for MLX/TimesFM).")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/models/model_int8.pth")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "mlx"],
        default="auto",
        help="Refuse MLX-native paths unless torch is forced.",
    )
    args = parser.parse_args()

    quantize_model(args.ckpt, args.out, config_path=args.config, backend=args.backend)
