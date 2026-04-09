import torch
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Benchmark")

def run_benchmark(model, input_shape=(1, 512, 8), num_iters=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Pre-allocate input
    x = torch.randn(input_shape, device=device)
    
    # Warmup
    log.info("Warming up kernels...")
    for _ in range(50):
        _ = model(x)
    torch.cuda.synchronize()
    
    # Standard Benchmark
    log.info(f"Running Standard Benchmark ({num_iters} iterations)...")
    start = time.time()
    for _ in range(num_iters):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    
    std_latency = (end - start) / num_iters * 1000
    std_throughput = num_iters / (end - start)
    
    log.info(f"Standard Latency: {std_latency:.4f} ms")
    log.info(f"Standard Throughput: {std_throughput:.2f} samples/sec")

    # CUDAGraphs Benchmark (If possible)
    if hasattr(torch.cuda, "make_graphed_callables"):
        log.info("Capturing CUDAGraph...")
        try:
            graphed_model = torch.cuda.make_graphed_callables(model, (x,))
            
            log.info(f"Running CUDAGraph Benchmark ({num_iters} iterations)...")
            start = time.time()
            for _ in range(num_iters):
                _ = graphed_model(x)
            torch.cuda.synchronize()
            end = time.time()
            
            graph_latency = (end - start) / num_iters * 1000
            graph_throughput = num_iters / (end - start)
            
            log.info(f"CUDAGraph Latency: {graph_latency:.4f} ms")
            log.info(f"CUDAGraph Throughput: {graph_throughput:.2f} samples/sec")
            log.info(f"🚀 CUDAGraph Speedup: {std_latency / graph_latency:.2f}x")
        except Exception as e:
            log.warning(f"CUDAGraph capture failed (common for non-static graphs): {e}")

if __name__ == "__main__":
    from models import get_model
    from omegaconf import OmegaConf
    
    # Create a dummy config for testing
    cfg = OmegaConf.create({
        "model": {"type": "lstm", "num_channels": 8, "hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "bidirectional": True}
    })
    
    model = get_model(cfg, 8)
    run_benchmark(model)
