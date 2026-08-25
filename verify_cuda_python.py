import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("cuda-python-check")


def _get_cuda_python_package_version() -> str:
    """Return the installed cuda-python (or cuda-bindings) package version string."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("cuda-python")
        except PackageNotFoundError:
            return version("cuda-bindings")
    except Exception:
        return "unknown"

def check_cuda_python():
    log.info("Testing CUDA Python 1.0 (cuda-python) initialization...")
    try:
        from cuda.bindings import driver, runtime
        log.info("Successfully imported 'cuda.bindings' (driver, runtime) modules.")
    except ImportError as e:
        log.error(f"Failed to import 'cuda.bindings': {e}")
        return False

    # Initialize CUDA Driver API
    try:
        err, = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            log.error(f"cuInit failed with result code: {err}")
            return False
        log.info("CUDA Driver API successfully initialized via cuInit().")
    except Exception as e:
        log.error(f"Error during cuInit(): {e}")
        return False

    # Get device count
    try:
        err, device_count = driver.cuDeviceGetCount()
        if err != driver.CUresult.CUDA_SUCCESS:
            log.error(f"cuDeviceGetCount failed: {err}")
            return False
        log.info(f"Number of CUDA devices detected: {device_count}")
        if device_count == 0:
            log.warning("No CUDA devices found.")
            return False
    except Exception as e:
        log.error(f"Error getting device count: {e}")
        return False

    # Inspect device 0
    try:
        err, device = driver.cuDeviceGet(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            log.error(f"cuDeviceGet failed for device 0: {err}")
            return False

        err, name = driver.cuDeviceGetName(256, device)
        if err != driver.CUresult.CUDA_SUCCESS:
            log.error(f"cuDeviceGetName failed: {err}")
            name_str = "Unknown"
        else:
            name_str = name.decode('utf-8').strip('\x00')
        log.info(f"Device 0 Name: {name_str}")

        err_major, cc_major = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
        err_minor, cc_minor = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
        if err_major != driver.CUresult.CUDA_SUCCESS or err_minor != driver.CUresult.CUDA_SUCCESS:
            log.error("Failed to get compute capability attributes.")
            return False
        log.info(f"Device Compute Capability: {cc_major}.{cc_minor}")
        
        err, total_mem = driver.cuDeviceTotalMem(device)
        if err == driver.CUresult.CUDA_SUCCESS:
            log.info(f"Device Total Memory: {total_mem / (1024**3):.2f} GB")
    except Exception as e:
        log.error(f"Error inspecting device 0: {e}")
        return False

    # Get driver version
    try:
        err, driver_version = driver.cuDriverGetVersion()
        if err == driver.CUresult.CUDA_SUCCESS:
            major = driver_version // 1000
            minor = (driver_version % 1000) // 10
            log.info(f"CUDA Driver Version: {major}.{minor} (Raw: {driver_version})")
    except Exception as e:
        log.error(f"Error getting driver version: {e}")

    # Report the installed package version for audit trails
    pkg_ver = _get_cuda_python_package_version()
    log.info("cuda-python package version: %s", pkg_ver)
    log.info("CUDA Python 1.0 status verification completed successfully!")
    return True

def benchmark_memory_transfers():
    log.info("\n=== Running Memory Bandwidth Benchmark (CUDA Python 1.0 vs PyTorch) ===")
    try:
        from cuda.bindings import runtime
        import numpy as np
        import torch
    except ImportError as e:
        log.error(f"Required libraries missing for benchmarking: {e}")
        return False

    size_bytes = 100 * 1024 * 1024  # 100 MB data size
    log.info(f"Payload Size: {size_bytes / (1024**2):.1f} MB")

    # --- NumPy & CUDA Python 1.0 (Low-level cudaMemcpy) ---
    # Create 100MB float64 array
    h_src = np.random.randn(size_bytes // 8).astype(np.float64)
    h_dst = np.zeros_like(h_src)

    # Measure allocation latency
    t_alloc_start = time.perf_counter()
    err, d_ptr = runtime.cudaMalloc(size_bytes)
    t_alloc = (time.perf_counter() - t_alloc_start) * 1000
    if err != runtime.cudaError_t.cudaSuccess:
        log.error(f"cudaMalloc failed to allocate 100MB: {err}")
        return False

    # Warm up transfers
    for _ in range(5):
        err, = runtime.cudaMemcpy(d_ptr, h_src.ctypes.data, size_bytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice)
        if err != runtime.cudaError_t.cudaSuccess:
            log.error(f"Warm-up H2D cudaMemcpy failed: {err}")
            runtime.cudaFree(d_ptr)
            return False
        err, = runtime.cudaMemcpy(h_dst.ctypes.data, d_ptr, size_bytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        if err != runtime.cudaError_t.cudaSuccess:
            log.error(f"Warm-up D2H cudaMemcpy failed: {err}")
            runtime.cudaFree(d_ptr)
            return False

    # Benchmark Host to Device (H2D)
    # cudaDeviceSynchronize() is called before timing to ensure the GPU is idle.
    # Error is checked after the loop to avoid disrupting timing measurements.
    iters = 100
    runtime.cudaDeviceSynchronize()
    last_h2d_err = runtime.cudaError_t.cudaSuccess
    t_start = time.perf_counter()
    for _ in range(iters):
        err, = runtime.cudaMemcpy(d_ptr, h_src.ctypes.data, size_bytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice)
        last_h2d_err = err
    t_h2d = (time.perf_counter() - t_start) / iters
    if last_h2d_err != runtime.cudaError_t.cudaSuccess:
        log.error(f"H2D benchmark cudaMemcpy failed: {last_h2d_err}")
        runtime.cudaFree(d_ptr)
        return False
    h2d_gbps = (size_bytes / 1e9) / t_h2d

    # Benchmark Device to Host (D2H)
    runtime.cudaDeviceSynchronize()
    last_d2h_err = runtime.cudaError_t.cudaSuccess
    t_start = time.perf_counter()
    for _ in range(iters):
        err, = runtime.cudaMemcpy(h_dst.ctypes.data, d_ptr, size_bytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        last_d2h_err = err
    t_d2h = (time.perf_counter() - t_start) / iters
    if last_d2h_err != runtime.cudaError_t.cudaSuccess:
        log.error(f"D2H benchmark cudaMemcpy failed: {last_d2h_err}")
        runtime.cudaFree(d_ptr)
        return False
    d2h_gbps = (size_bytes / 1e9) / t_d2h

    # Cleanup CUDA Python device pointer
    err, = runtime.cudaFree(d_ptr)
    if err != runtime.cudaError_t.cudaSuccess:
        log.warning(f"cudaFree failed (non-fatal): {err}")

    log.info(f"  [CUDA Python] Alloc time: {t_alloc:.3f} ms")
    log.info(f"  [CUDA Python] Host-to-Device (H2D): {h2d_gbps:.2f} GB/s (Avg: {t_h2d*1000:.3f} ms)")
    log.info(f"  [CUDA Python] Device-to-Host (D2H): {d2h_gbps:.2f} GB/s (Avg: {t_d2h*1000:.3f} ms)")

    # --- PyTorch (High-level tensor.cuda()) ---
    # Create 100MB float32 tensor
    t_tensor = torch.randn(size_bytes // 4, dtype=torch.float32)

    # Warm up PyTorch transfers
    for _ in range(5):
        _ = t_tensor.cuda()
        torch.cuda.synchronize()

    # Benchmark PyTorch Host to Device (H2D)
    t_start = time.perf_counter()
    for _ in range(iters):
        d_tensor = t_tensor.cuda()
    torch.cuda.synchronize()
    t_pt_h2d = (time.perf_counter() - t_start) / iters
    pt_h2d_gbps = (size_bytes / 1e9) / t_pt_h2d

    # Benchmark PyTorch Device to Host (D2H)
    t_start = time.perf_counter()
    for _ in range(iters):
        _ = d_tensor.cpu()
    torch.cuda.synchronize()
    t_pt_d2h = (time.perf_counter() - t_start) / iters
    pt_d2h_gbps = (size_bytes / 1e9) / t_pt_d2h

    log.info(f"  [PyTorch] Host-to-Device (H2D): {pt_h2d_gbps:.2f} GB/s (Avg: {t_pt_h2d*1000:.3f} ms)")
    log.info(f"  [PyTorch] Device-to-Host (D2H): {pt_d2h_gbps:.2f} GB/s (Avg: {t_pt_d2h*1000:.3f} ms)")

    # Efficiency Comparison
    log.info(f"  ⚡ H2D Speed Ratio: CUDA Python / PyTorch = {h2d_gbps / pt_h2d_gbps:.2f}x")
    log.info(f"  ⚡ D2H Speed Ratio: CUDA Python / PyTorch = {d2h_gbps / pt_d2h_gbps:.2f}x")
    
    log.info("Memory Bandwidth Benchmark completed successfully!")
    return True

if __name__ == "__main__":
    status_ok = check_cuda_python()
    if status_ok:
        benchmark_ok = benchmark_memory_transfers()
        sys.exit(0 if benchmark_ok else 1)
    sys.exit(1)
