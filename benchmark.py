# benchmark.py — Model Inference Benchmarking Suite
import torch
import time
import logging
import numpy as np
import gc
from typing import Any, Dict, List, Optional, Tuple, Union

from utils import configure_torch_runtime, log_runtime_capabilities, select_torch_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Benchmark")

def measure_latency_and_memory(model, x, num_iters: int, device_type: str, desc: str, warmup_iters: int = 10) -> Dict[str, float]:
    """
    Measures P50, P90, P99 latency, throughput, and memory footprint.
    Returns a dictionary of statistics.
    """
    latencies = []

    # Reset memory tracking
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

    log.info(f"Running {desc} ({num_iters} iterations)...")

    # Warmup
    for _ in range(warmup_iters):
        _ = model(x)
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

    # Benchmark loop
    for _ in range(num_iters):
        if device_type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event)) # in ms
        elif device_type == "mps":
            if hasattr(torch, "mps"):
                torch.mps.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if hasattr(torch, "mps"):
                torch.mps.synchronize()
            latencies.append((time.perf_counter() - start) * 1000) # in ms
        else:
            start = time.perf_counter()
            _ = model(x)
            latencies.append((time.perf_counter() - start) * 1000) # in ms

    latencies = np.array(latencies)
    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    throughput = 1000.0 / avg # 1000ms / ms_per_sample = samples_per_sec

    log.info(f"  [{desc}] Latency: Avg={avg:.3f}ms | P50={p50:.3f}ms | P90={p90:.3f}ms | P99={p99:.3f}ms")
    log.info(f"  [{desc}] Throughput: {throughput:.2f} samples/sec")

    peak_mem = 0.0
    if device_type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        log.info(f"  [{desc}] Peak VRAM: {peak_mem:.2f} MB")

    return {
        "avg": avg,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "throughput": throughput,
        "peak_vram_mb": peak_mem
    }

def run_benchmark(model, input_shape: Tuple[int, int, int] = (1, 512, 8), num_iters: int = 500) -> Dict[str, Dict[str, float]]:
    """
    Runs a comprehensive benchmarking suite on the provided model.
    Compares Standard, AMP, torch.compile, and CUDAGraphs.
    """
    runtime_caps = configure_torch_runtime()
    log_runtime_capabilities(log, runtime_caps, prefix="Benchmark Runtime")
    device = select_torch_device(runtime_caps)
    model = model.to(device).eval()
    
    x = torch.randn(input_shape, device=device)
    results = {}
    
    log.info("=== Standard PyTorch (FP32) ===")
    results["Standard FP32"] = measure_latency_and_memory(model, x, num_iters, device.type, "Standard FP32")
    base_avg = results["Standard FP32"]["avg"]

    if device.type == "cuda":
        log.info("=== Automatic Mixed Precision (AMP - FP16) ===")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Wrapper to ensure autocast is applied during timing loop
            class AMPWrapper:
                def __init__(self, m): self.m = m
                def __call__(self, inp): return self.m(inp)

            results["AMP FP16"] = measure_latency_and_memory(AMPWrapper(model), x, num_iters, device.type, "AMP FP16")
            log.info(f"  ⚡ AMP Speedup: {base_avg / results['AMP FP16']['avg']:.2f}x")

        log.info("=== torch.compile (Graph Fusion) ===")
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            results["torch.compile"] = measure_latency_and_memory(compiled_model, x, num_iters, device.type, "torch.compile")
            log.info(f"  ⚡ Compile Speedup: {base_avg / results['torch.compile']['avg']:.2f}x")
        except Exception as e:
            log.warning(f"torch.compile failed: {e}")

        if hasattr(torch.cuda, "make_graphed_callables"):
            log.info("=== CUDAGraphs ===")
            try:
                # CUDAGraphs require fixed inputs, which we have.
                graphed_model = torch.cuda.make_graphed_callables(model, (x,))
                results["CUDAGraphs"] = measure_latency_and_memory(graphed_model, x, num_iters, device.type, "CUDAGraphs")
                log.info(f"  ⚡ CUDAGraph Speedup: {base_avg / results['CUDAGraphs']['avg']:.2f}x")
            except Exception as e:
                log.warning(f"CUDAGraph capture failed: {e}")
    else:
        log.info("Skipping hardware-accelerated benchmarks because CUDA is unavailable.")

    return results

if __name__ == "__main__":
    from models import get_model

    config = {
        "model_config": {
            "type": "LSTM",
            "lstm_layers": [64],
            "dropout": 0.1,
            "bidirectional": True,
        }
    }
    model = get_model(config, input_dim_sequence=8, input_dim_static=0)
    run_benchmark(model)
