import torch
import time
import logging

from utils import configure_torch_runtime, log_runtime_capabilities

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Benchmark")

def run_benchmark(model, input_shape=(1, 512, 8), num_iters=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    runtime_caps = configure_torch_runtime()
    log_runtime_capabilities(log, runtime_caps, prefix="Benchmark Runtime")
    
    # Pre-allocate input
    x = torch.randn(input_shape, device=device)
    
    # Warmup
    log.info("Warming up kernels...")
    for _ in range(50):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Standard Benchmark
    log.info(f"Running Standard Benchmark ({num_iters} iterations)...")
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = max(end - start, 1e-9)
    std_latency = elapsed / num_iters * 1000
    std_throughput = num_iters / elapsed
    
    log.info(f"Standard Latency: {std_latency:.4f} ms")
    log.info(f"Standard Throughput: {std_throughput:.2f} samples/sec")

    # CUDAGraphs Benchmark (If possible)
    if device.type == "cuda" and hasattr(torch.cuda, "make_graphed_callables"):
        log.info("Capturing CUDAGraph...")
        try:
            graphed_model = torch.cuda.make_graphed_callables(model, (x,))
            
            log.info(f"Running CUDAGraph Benchmark ({num_iters} iterations)...")
            start = time.perf_counter()
            for _ in range(num_iters):
                _ = graphed_model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            elapsed = max(end - start, 1e-9)
            graph_latency = elapsed / num_iters * 1000
            graph_throughput = num_iters / elapsed
            
            log.info(f"CUDAGraph Latency: {graph_latency:.4f} ms")
            log.info(f"CUDAGraph Throughput: {graph_throughput:.2f} samples/sec")
            log.info(f"🚀 CUDAGraph Speedup: {std_latency / graph_latency:.2f}x")
        except Exception as e:
            log.warning(f"CUDAGraph capture failed (common for non-static graphs): {e}")
    else:
        log.info("Skipping CUDAGraph benchmark because CUDA is unavailable.")

if __name__ == "__main__":
    from models import get_model

    # Create a small config for smoke testing the benchmark entrypoint.
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
