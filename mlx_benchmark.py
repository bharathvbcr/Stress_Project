# mlx_benchmark.py — MLX inference benchmarking for Apple Silicon
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np

from mlx_backend import (
    cast_mlx_model_parameters,
    configure_mlx_runtime,
    is_mlx_available,
    maybe_compile_mlx_callable,
    resolve_mlx_dtype,
    resolve_mlx_dtype_name,
)
from mlx_models import get_mlx_model, load_mlx_checkpoint
from utils import safe_get

log = logging.getLogger("MLXBenchmark")


def measure_mlx_latency(
    model,
    seq,
    static,
    num_iters: int = 200,
    warmup_iters: int = 20,
    compile_model: bool = False,
) -> Dict[str, float]:
    import mlx.core as mx

    if hasattr(model, "eval"):
        model.eval()  # measure inference behaviour (dropout/BN off)

    forward = maybe_compile_mlx_callable(
        lambda s, st: model(s, st),
        enabled=compile_model,
    )

    # Force the forward *output* to materialize. Evaluating model.parameters()
    # only realizes the (already-materialized) weights and never executes the
    # forward graph, so the loop below would otherwise time nothing.
    for _ in range(warmup_iters):
        mx.eval(forward(seq, static))

    latencies = []
    for _ in range(num_iters):
        start = time.perf_counter()
        out = forward(seq, static)
        mx.eval(out)
        latencies.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(latencies, dtype=np.float64)
    avg = float(np.mean(arr))
    return {
        "avg": avg,
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "throughput": 1000.0 / max(avg, 1e-9),
    }


def run_mlx_benchmark(
    config: Dict[str, Any],
    input_dim_sequence: int,
    input_dim_static: int,
    input_shape: Tuple[int, int, int] = (1, 512, 8),
    checkpoint_path: str | None = None,
    num_iters: int = 500,
) -> Dict[str, Dict[str, float]]:
    if not is_mlx_available():
        raise ImportError("MLX is not installed. Use: pip install -r requirements-apple.txt")

    import mlx.core as mx

    mlx_info = configure_mlx_runtime()
    dtype = resolve_mlx_dtype()
    dtype_name = resolve_mlx_dtype_name()
    log.info(
        "MLX benchmark runtime: version=%s device=%s compile=%s dtype=%s",
        mlx_info.get("mlx_version"),
        mlx_info.get("mlx_device"),
        mlx_info.get("mlx_compile"),
        dtype_name,
    )

    model = get_mlx_model(config, input_dim_sequence, input_dim_static)
    if checkpoint_path:
        load_mlx_checkpoint(model, checkpoint_path)
    cast_mlx_model_parameters(model, dtype)

    batch, seq_len, channels = input_shape
    seq = mx.random.normal((batch, seq_len, channels)).astype(dtype)
    static = (
        mx.random.normal((batch, input_dim_static)).astype(dtype)
        if input_dim_static > 0
        else None
    )

    base_key = f"MLX {dtype_name}"
    results: Dict[str, Dict[str, float]] = {}
    log.info("=== %s ===", base_key)
    results[base_key] = measure_mlx_latency(model, seq, static, num_iters=num_iters)

    if mlx_info.get("mlx_compile"):
        compile_key = f"{base_key} compile"
        log.info("=== %s (mx.compile) ===", compile_key)
        results[compile_key] = measure_mlx_latency(
            model,
            seq,
            static,
            num_iters=num_iters,
            compile_model=True,
        )
        base = results[base_key]["avg"]
        compiled = results[compile_key]["avg"]
        log.info("  Compile speedup: %.2fx", base / max(compiled, 1e-9))

    return results


if __name__ == "__main__":
    import argparse
    import json

    from utils import infer_model_feature_dims, load_config

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Benchmark MLX inference latency.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default=None, help="Optional *_mlx.npz checkpoint.")
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config) or {}
    seq_dim, static_dim = infer_model_feature_dims(cfg)
    target_len = int(safe_get(cfg, ["model_config", "timesfm_context_len"], 512))
    results = run_mlx_benchmark(
        cfg,
        seq_dim,
        static_dim,
        input_shape=(1, target_len, seq_dim),
        checkpoint_path=args.checkpoint,
        num_iters=args.iters,
    )
    print(json.dumps(results, indent=2))
