# performance_guard.py — Latency SLA Enforcement Script
import torch
import json
import os
import sys
import logging
import argparse
from benchmark import run_benchmark, measure_latency_and_memory
from models import get_model
from utils import load_config, infer_model_feature_dims, safe_get, select_torch_device, configure_torch_runtime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerformanceGuard")

def run_guard(config_path="config.json", baseline_path="perf_baseline.json", threshold=1.10, num_iters=200):
    """
    Enforces performance SLAs by comparing current inference latency against a baseline.
    If P99 latency degrades by more than 'threshold' (default 10%), the guard fails.
    """
    config = load_config(config_path)
    if not config:
        log.error("Failed to load config.")
        sys.exit(1)

    # 1. Initialize Model
    input_dim_sequence, input_dim_static = infer_model_feature_dims(config)
    target_len = safe_get(config, ['model_config', 'timesfm_context_len'], 512)
    input_shape = (1, target_len, input_dim_sequence)
    
    runtime_caps = configure_torch_runtime()
    device = select_torch_device(runtime_caps)
    model = get_model(config, input_dim_sequence=input_dim_sequence, input_dim_static=input_dim_static)
    model = model.to(device).eval()
    
    x = torch.randn(input_shape, device=device)
    
    # 2. Run Benchmark (FP32 baseline)
    log.info("Starting performance regression check...")
    res = measure_latency_and_memory(model, x, num_iters, device.type, "Regression Check", warmup_iters=20)
    current_p99 = res["p99"]
    
    # 3. Load or Create Baseline
    if not os.path.exists(baseline_path):
        log.warning(f"No baseline found at {baseline_path}. Creating new baseline.")
        baseline = {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "p99_latency_ms": current_p99,
            "threshold": threshold
        }
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=4)
        log.info(f"✓ Baseline established: {current_p99:.4f}ms (P99)")
        return

    with open(baseline_path, "r") as f:
        baseline = json.load(f)
    
    base_p99 = baseline.get("p99_latency_ms") or baseline.get("avg_latency_ms") # Migration support
    diff = current_p99 / base_p99
    
    log.info(f"Baseline P99: {base_p99:.4f}ms")
    log.info(f"Current P99:  {current_p99:.4f}ms")
    log.info(f"Performance Ratio: {diff:.2f}x")

    # 4. Enforce SLA
    if diff > threshold:
        log.error(f"❌ PERFORMANCE REGRESSION DETECTED (Tail Latency)!")
        log.error(f"P99 is {((diff-1)*100):.1f}% slower than baseline (Max allowed: {((threshold-1)*100):.1f}%)")
        sys.exit(1)
    else:
        log.info(f"✅ Performance Guard Passed. Tail latency within limits.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--baseline", type=str, default="perf_baseline.json")
    parser.add_argument("--update", action="store_true", help="Update the baseline with current results")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.10,
        help="Max allowed P99 ratio vs baseline (e.g. 1.10 = 10%% regression budget). "
        "Use a looser value on shared CI runners (they are noisy).",
    )
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations.")
    args = parser.parse_args()

    if args.update and os.path.exists(args.baseline):
        os.remove(args.baseline)

    run_guard(
        config_path=args.config,
        baseline_path=args.baseline,
        threshold=args.threshold,
        num_iters=args.iters,
    )
