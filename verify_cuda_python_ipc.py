import sys
import time
import logging
import multiprocessing
from multiprocessing.reduction import ForkingPickler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("cuda-ipc-check")

# Reduce serialization boilerplates for raw CUDA structures in multiprocessing
def serialize_ipc_handle(handle):
    return bytes(handle)

def check_ipc_requirements():
    try:
        from cuda.bindings import driver, runtime
        err, = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            log.error(f"cuInit failed: {err}")
            return False
        
        # IPC is only supported on Linux/Windows on devices with compute capability >= 2.0
        # and not in SLI/Multi-GPU configurations without peer access or on Tegra/WSL in some configurations.
        err, dev_count = driver.cuDeviceGetCount()
        if dev_count == 0:
            log.warning("No CUDA devices found for IPC.")
            return False

        # On Windows, IPC memory handles require TCC driver mode.
        # WDDM driver mode does not support cross-process memory handle maps.
        if sys.platform == "win32":
            err, tcc = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrTccDriver, 0)
            if err == runtime.cudaError_t.cudaSuccess and tcc == 0:
                log.warning("Windows CUDA driver is in WDDM mode. IPC memory sharing is not supported (requires TCC mode).")
                return False
            
        return True
    except Exception as e:
        log.error(f"CUDA initialization error: {e}")
        return False

def child_process_reader(handle_bytes, size_bytes, barrier):
    try:
        from cuda.bindings import runtime
        import numpy as np
        
        # Child process initializes CUDA
        err, = runtime.cudaSetDevice(0)
        if err != runtime.cudaError_t.cudaSuccess:
            log.error(f"[Child] cudaSetDevice failed: {err}")
            return
            
        # Reconstruct the IPC handle from bytes
        # cudaIpcMemHandle_t is a byte array structure
        handle = runtime.cudaIpcMemHandle_t()
        import ctypes
        ctypes.memmove(handle.getPtr(), handle_bytes, len(handle_bytes))
        
        log.info("[Child] Opening shared memory handle...")
        err, d_ptr = runtime.cudaIpcOpenMemHandle(handle, 0)
        if err != runtime.cudaError_t.cudaSuccess:
            log.error(f"[Child] cudaIpcOpenMemHandle failed: {err}")
            return
            
        log.info("[Child] Shared memory handle successfully opened.")
        
        # Wait for parent to write data
        barrier.wait()
        
        # Read data back to host
        h_dst = np.zeros(size_bytes // 4, dtype=np.float32)
        log.info("[Child] Reading shared memory contents from GPU...")
        err, = runtime.cudaMemcpy(
            h_dst.ctypes.data,
            d_ptr,
            size_bytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        if err != runtime.cudaError_t.cudaSuccess:
            log.error(f"[Child] cudaMemcpy failed: {err}")
            return
            
        # Verify read contents
        log.info(f"[Child] Data read verify (first 5 elements): {h_dst[:5]}")
        expected = np.arange(5, dtype=np.float32)
        if np.allclose(h_dst[:5], expected):
            log.info("[Child] Success! Data verified successfully.")
        else:
            log.error("[Child] Failure: Data verification mismatch.")
            
        # Clean up
        log.info("[Child] Closing shared memory handle...")
        err, = runtime.cudaIpcCloseMemHandle(d_ptr)
        if err != runtime.cudaError_t.cudaSuccess:
            log.warning(f"[Child] cudaIpcCloseMemHandle warning: {err}")
            
    except Exception as e:
        log.error(f"[Child] Error in reader process: {e}")

def run_ipc_demo():
    log.info("=== Running CUDA Python 1.0 IPC memory sharing demo ===")
    from cuda.bindings import runtime
    import numpy as np
    
    # 1MB buffer size
    size_bytes = 1024 * 1024
    
    log.info("[Parent] Allocating device memory...")
    err, d_ptr = runtime.cudaMalloc(size_bytes)
    if err != runtime.cudaError_t.cudaSuccess:
        log.error(f"[Parent] cudaMalloc failed: {err}")
        return False
        
    # Get IPC handle for allocated memory
    log.info("[Parent] Creating IPC memory handle...")
    err, handle = runtime.cudaIpcGetMemHandle(d_ptr)
    if err != runtime.cudaError_t.cudaSuccess:
        log.error(f"[Parent] cudaIpcGetMemHandle failed: {err}")
        runtime.cudaFree(d_ptr)
        return False
        
    # Serialize handle to bytes to safely pass across processes
    handle_bytes = handle.reserved
    
    # Setup process synchronization barrier
    barrier = multiprocessing.Barrier(2)
    
    # Spawn child process
    p = multiprocessing.Process(
        target=child_process_reader,
        args=(handle_bytes, size_bytes, barrier)
    )
    p.start()
    
    # Write data to device memory from parent
    h_src = np.arange(size_bytes // 4, dtype=np.float32)
    log.info("[Parent] Writing verification data to GPU...")
    err, = runtime.cudaMemcpy(
        d_ptr,
        h_src.ctypes.data,
        size_bytes,
        runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    if err != runtime.cudaError_t.cudaSuccess:
        log.error(f"[Parent] cudaMemcpy failed: {err}")
        p.terminate()
        runtime.cudaFree(d_ptr)
        return False
        
    # Trigger barrier to let child know data is ready
    log.info("[Parent] Triggering data-ready signal.")
    barrier.wait()
    
    p.join()
    
    # Free device memory
    log.info("[Parent] Freeing device memory...")
    err, = runtime.cudaFree(d_ptr)
    if err != runtime.cudaError_t.cudaSuccess:
        log.warning(f"[Parent] cudaFree warning: {err}")
        
    log.info("[Parent] IPC demo completed successfully.")
    return True

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    if check_ipc_requirements():
        success = run_ipc_demo()
        sys.exit(0 if success else 1)
    else:
        log.warning("System does not support CUDA IPC. Skipping demo.")
        sys.exit(0)
