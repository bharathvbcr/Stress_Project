# callbacks.py (Telemetry & Diagnostic Callbacks for PyTorch Lightning)
import logging
import psutil
import time
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

# ==============================================================================
# == Internal Helpers ==
# ==============================================================================

def _as_positive_int(value: Any, default: int) -> int:
    try:
        val = int(value)
        return val if val > 0 else default
    except (ValueError, TypeError):
        return default

def _cuda_device(device: Any) -> Optional[torch.device]:
    if isinstance(device, torch.device) and device.type == "cuda":
        return device
    if isinstance(device, str) and "cuda" in device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return None

def _maybe_cuda_synchronize(device: Any, enabled: bool = True) -> bool:
    if not enabled:
        return False
    d = _cuda_device(device)
    if d is not None:
        torch.cuda.synchronize(device=d)
        return True
    return False

def _cuda_memory_snapshot(device: Any) -> Optional[Dict[str, float]]:
    d = _cuda_device(device)
    if d is None:
        return None
    try:
        return {
            "allocated_gb": torch.cuda.memory_allocated(device=d) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device=d) / 1e9,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device=d) / 1e9,
        }
    except Exception:
        return None

def _format_cuda_snapshot(snapshot: Optional[Dict[str, float]]) -> str:
    if not snapshot:
        return ""
    return " CUDA: alloc=%.2fGB res=%.2fGB peak=%.2fGB" % (
        snapshot["allocated_gb"],
        snapshot["reserved_gb"],
        snapshot["peak_allocated_gb"],
    )

def _batch_size(batch: Any) -> int:
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        return _batch_size(batch[0])
    if isinstance(batch, dict):
        # Look for typical keys
        for k in ["sequence", "x", "input", "features"]:
            if k in batch:
                return _batch_size(batch[k])
        # Fallback to first value
        return _batch_size(next(iter(batch.values())))
    if hasattr(batch, "shape"):
        return batch.shape[0]
    if hasattr(batch, "__len__"):
        return len(batch)
    return 1

def _batch_shapes(batch: Any) -> Any:
    if isinstance(batch, (list, tuple)):
        return [_batch_shapes(i) for i in batch]
    if isinstance(batch, dict):
        return {k: _batch_shapes(v) for k, v in batch.items()}
    if hasattr(batch, "shape"):
        return list(batch.shape)
    return type(batch).__name__

# ==============================================================================
# == Callbacks ==
# ==============================================================================

class HistoryCallback(L.Callback):
    """
    Captures training and validation metrics for final reporting.
    """
    def __init__(self):
        super().__init__()
        self.history = []

    def _scalar(self, v):
        if hasattr(v, "item"): return v.item()
        return float(v)

    def on_train_epoch_end(self, trainer, pl_module):
        # Keep real epoch metrics (train_loss, val_f1, ...); drop namespaced
        # telemetry (time/, perf/, gpu/, grad/, weights/, sys/, diag/) so history
        # isn't polluted with last-step instrumentation values.
        metrics = {
            k: self._scalar(v)
            for k, v in trainer.logged_metrics.items()
            if "/" not in k
        }
        metrics["epoch"] = trainer.current_epoch + 1
        metrics["step"] = trainer.global_step
        self.history.append(metrics)


class BatchTimeLoggingCallback(L.Callback):
    """
    Advanced telemetry for identifying I/O vs. Compute bottlenecks.
    Calculates throughput, per-batch latency, and CUDA memory health.
    """
    def __init__(
        self,
        log_interval: int = 50,
        cuda_sync_interval: int = 1,
        memory_log_interval: int = 100,
        epoch_memory_summary: bool = True,
    ):
        super().__init__()
        self.log_interval = _as_positive_int(log_interval, 50)
        self.cuda_sync_interval = _as_positive_int(cuda_sync_interval, 0)
        self.memory_log_interval = _as_positive_int(memory_log_interval, 100)
        self.epoch_memory_summary = epoch_memory_summary

        # Stats
        self.batch_start_time = None
        self.processing_start_time = None
        self.last_load_time = 0.0
        
        # Breakdown timers
        self.before_backward_time = None
        self.after_backward_time = None
        self.before_optimizer_step_time = None

        # Cumulative
        self.total_samples = 0
        self.num_batches = 0
        self.total_load_time = 0.0
        self.total_compute_time = 0.0
        self.total_forward_loss_time = 0.0
        self.total_backward_time = 0.0
        self.total_pre_optimizer_time = 0.0
        self.total_optimizer_time = 0.0
        self.optimizer_step_count = 0
        self.cuda_sync_failures = 0

        # Memory tracking
        self.first_gpu_alloc_gb = None
        self.max_gpu_alloc_gb = 0.0
        self.last_gpu_alloc_gb = 0.0

        # Reset per epoch
        self.epoch_start_time = None
        self.epoch_samples = 0
        self.epoch_batches = 0
        self.epoch_load_time = 0.0
        self.epoch_compute_time = 0.0
        self.epoch_forward_loss_time = 0.0
        self.epoch_backward_time = 0.0
        self.epoch_pre_optimizer_time = 0.0
        self.epoch_optimizer_time = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()
        self.epoch_samples = 0
        self.epoch_batches = 0
        self.epoch_load_time = 0.0
        self.epoch_compute_time = 0.0
        self.epoch_forward_loss_time = 0.0
        self.epoch_backward_time = 0.0
        self.epoch_pre_optimizer_time = 0.0
        self.epoch_optimizer_time = 0.0
        self.last_load_time = 0.0
        self.batch_start_time = None
        self.processing_start_time = None
        
        log.info("[Training Stage] START train_epoch epoch=%s", trainer.current_epoch + 1)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.batch_start_time = None
        self.last_load_time = 0.0
        self.val_start_time = time.perf_counter()
        log.info("[Training Stage] START validation epoch=%s", trainer.current_epoch + 1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(self, 'val_start_time'):
            duration = time.perf_counter() - self.val_start_time
            pl_module.log("time/validation_epoch_sec", duration)
            log.info(
                "[Training Stage] END validation epoch=%s elapsed=%.2fs%s",
                trainer.current_epoch + 1,
                duration,
                _format_cuda_snapshot(_cuda_memory_snapshot(pl_module.device)),
            )

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.perf_counter() - self.epoch_start_time if self.epoch_start_time else 0.0
        epoch_throughput = self.epoch_samples / max(elapsed, 1e-6)
        
        # Calculate load ratio
        load_ratio = self.epoch_load_time / max(elapsed, 1e-6)
        
        log.info(
            "[Training Stage] END train_epoch epoch=%s elapsed=%.2fs samples=%s batches=%s throughput=%.2f samples/sec load_ratio=%.2f%%%s",
            trainer.current_epoch + 1,
            elapsed,
            self.epoch_samples,
            self.epoch_batches,
            epoch_throughput,
            load_ratio * 100,
            _format_cuda_snapshot(_cuda_memory_snapshot(pl_module.device)),
        )
        
        if load_ratio > 0.3:
            log.warning("High Data Loading Overhead detected (%.1f%%). GPU is starving for data.", load_ratio * 100)

        cuda_device = _cuda_device(pl_module.device)
        if cuda_device is not None:
            try:
                stats = torch.cuda.memory_stats(device=cuda_device)
                inactive_split_bytes = stats.get("inactive_split.all.current", 0)
                pl_module.log("gpu/fragmentation_mb", inactive_split_bytes / 1e6, on_epoch=True)
            except Exception as exc:
                log.warning("CUDA memory stats failed: %s", exc)

            if self.epoch_memory_summary:
                try:
                    mem_summary = torch.cuda.memory_summary(device=cuda_device, abbreviated=True)
                    log.info("Epoch %s CUDA memory summary:\n%s", trainer.current_epoch + 1, mem_summary)
                except Exception as exc:
                    log.warning("CUDA memory summary failed: %s", exc)

    def _should_log_step(self, batch_idx: int) -> bool:
        return batch_idx == 0 or (batch_idx + 1) % self.log_interval == 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        self.current_batch_idx = batch_idx
        # Gate the per-step pl_module.log calls; cumulative timers still accumulate
        # every batch so epoch summaries remain accurate.
        self.log_this_batch = self._should_log_step(batch_idx)
        self.before_backward_time = None
        self.after_backward_time = None
        self.before_optimizer_step_time = None

        if self.batch_start_time is not None:
            load_time = now - self.batch_start_time
            self.last_load_time = load_time
            self.total_load_time += load_time
            self.epoch_load_time += load_time
            if self.log_this_batch:
                pl_module.log("time/data_load", load_time, on_step=True, on_epoch=False)

        if batch_idx == 0:
            log.info("Epoch %s batch 0 shapes: %s", trainer.current_epoch + 1, _batch_shapes(batch))

        self.processing_start_time = now

    def _should_sync_batch(self, batch_idx: int) -> bool:
        return self.cuda_sync_interval > 0 and (
            batch_idx == 0 or (batch_idx + 1) % self.cuda_sync_interval == 0
        )

    def _sync_for_batch(self, pl_module, batch_idx: int) -> None:
        if not self._should_sync_batch(batch_idx):
            return
        try:
            _maybe_cuda_synchronize(pl_module.device, enabled=True)
        except Exception as exc:
            self.cuda_sync_failures += 1
            log.warning("CUDA synchronize failed at batch %s: %s", batch_idx, exc)

    def on_before_backward(self, trainer, pl_module, loss):
        self._sync_for_batch(pl_module, self.current_batch_idx)
        now = time.perf_counter()
        self.before_backward_time = now
        if self.processing_start_time is not None:
            forward_loss_time = now - self.processing_start_time
            self.total_forward_loss_time += forward_loss_time
            self.epoch_forward_loss_time += forward_loss_time
            if getattr(self, "log_this_batch", True):
                pl_module.log("time/forward_loss_sec", forward_loss_time, on_step=True, on_epoch=False)

    def on_after_backward(self, trainer, pl_module):
        self._sync_for_batch(pl_module, self.current_batch_idx)
        now = time.perf_counter()
        self.after_backward_time = now
        if self.before_backward_time is not None:
            backward_time = now - self.before_backward_time
            self.total_backward_time += backward_time
            self.epoch_backward_time += backward_time
            if getattr(self, "log_this_batch", True):
                pl_module.log("time/backward_sec", backward_time, on_step=True, on_epoch=False)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self._sync_for_batch(pl_module, self.current_batch_idx)
        now = time.perf_counter()
        self.before_optimizer_step_time = now
        if self.after_backward_time is not None:
            pre_optimizer_time = now - self.after_backward_time
            self.total_pre_optimizer_time += pre_optimizer_time
            self.epoch_pre_optimizer_time += pre_optimizer_time
            if getattr(self, "log_this_batch", True):
                pl_module.log("time/pre_optimizer_sec", pre_optimizer_time, on_step=True, on_epoch=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync_for_batch(pl_module, batch_idx)
        now = time.perf_counter()
        compute_time = now - self.processing_start_time if self.processing_start_time else 0.0
        log_this_batch = getattr(self, "log_this_batch", True)
        optimizer_time = 0.0
        if self.before_optimizer_step_time is not None:
            optimizer_time = now - self.before_optimizer_step_time
            self.total_optimizer_time += optimizer_time
            self.epoch_optimizer_time += optimizer_time
            self.optimizer_step_count += 1
            if log_this_batch:
                pl_module.log("time/optimizer_step_sec", optimizer_time, on_step=True, on_epoch=False)

        step_time = self.last_load_time + compute_time
        self.total_compute_time += compute_time
        self.epoch_compute_time += compute_time

        batch_size = _batch_size(batch)
        self.total_samples += batch_size
        self.num_batches += 1
        self.epoch_samples += batch_size
        self.epoch_batches += 1

        compute_throughput = batch_size / max(compute_time, 1e-6)
        step_throughput = batch_size / max(step_time, 1e-6)
        if log_this_batch:
            pl_module.log("time/batch_processing", compute_time, on_step=True, on_epoch=False)
            pl_module.log("time/step_total_sec", step_time, on_step=True, on_epoch=False)
            pl_module.log("perf/throughput_samples_per_sec", step_throughput, on_step=True, on_epoch=False)
            pl_module.log("perf/compute_samples_per_sec", compute_throughput, on_step=True, on_epoch=False)

        snapshot = None
        if batch_idx == 0 or (batch_idx + 1) % self.memory_log_interval == 0:
            snapshot = _cuda_memory_snapshot(pl_module.device)
        if snapshot:
            self.last_gpu_alloc_gb = snapshot["allocated_gb"]
            self.max_gpu_alloc_gb = max(self.max_gpu_alloc_gb, snapshot["allocated_gb"])
            if self.first_gpu_alloc_gb is None:
                self.first_gpu_alloc_gb = snapshot["allocated_gb"]
            pl_module.log("gpu/memory_allocated_gb", snapshot["allocated_gb"], on_step=True, on_epoch=False)
            pl_module.log("gpu/memory_reserved_gb", snapshot["reserved_gb"], on_step=True, on_epoch=False)
            pl_module.log("gpu/memory_peak_allocated_gb", snapshot["peak_allocated_gb"], on_step=True, on_epoch=False)

        total_batches = getattr(trainer, "num_training_batches", "unknown")
        if batch_idx == 0 or (batch_idx + 1) % self.log_interval == 0:
            log.info(
                "Train step epoch=%s batch=%s/%s global_step=%s load=%.4fs compute=%.4fs optimizer=%.4fs step=%.4fs batch_size=%s throughput=%.2f samples/sec%s",
                trainer.current_epoch + 1,
                batch_idx + 1,
                total_batches,
                trainer.global_step,
                self.last_load_time,
                compute_time,
                optimizer_time,
                step_time,
                batch_size,
                step_throughput,
                _format_cuda_snapshot(snapshot),
            )

        self.batch_start_time = time.perf_counter()
        self.before_backward_time = None
        self.after_backward_time = None
        self.before_optimizer_step_time = None


class CheckpointTimerCallback(L.Callback):
    """
    Measures and logs the time spent saving checkpoints to disk.
    """
    def __init__(self):
        super().__init__()
        self.save_start = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.save_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.save_start:
            duration = time.time() - self.save_start
            log.info(f"Checkpoint saved to disk in {duration:.2f} seconds.")
            pl_module.log("time/checkpoint_io", duration)
            self.save_start = None


class GradientNormLoggingCallback(L.Callback):
    """
    Logs the total gradient norm and weight statistics to identify training issues.
    """
    def __init__(self, log_interval: int = 1, weight_stats: bool = True):
        super().__init__()
        self.log_interval = _as_positive_int(log_interval, 1)
        self.weight_stats = weight_stats
        # Count backward passes locally: trainer.global_step only advances per
        # optimizer step, so under gradient accumulation it would trigger this
        # expensive norm computation on every micro-batch of the logged step.
        self._backward_calls = 0

    def on_after_backward(self, trainer, pl_module):
        self._backward_calls += 1
        if self._backward_calls % self.log_interval != 0:
            return
        norms = [p.grad.detach().norm(2) for p in pl_module.parameters() if p.grad is not None]
        if not norms:
            return
        total_norm = torch.norm(torch.stack([n.to(norms[0].device) for n in norms]), 2)
        pl_module.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.weight_stats:
            return
        max_abs_values = []
        mean_values = []
        std_values = []
        for name, param in pl_module.model.named_parameters():
            if param.requires_grad and "weight" in name:
                weights = param.detach()
                max_abs_values.append(weights.abs().max())
                mean_values.append(weights.mean())
                std_values.append(weights.std())

        if max_abs_values:
            device = max_abs_values[0].device
            global_abs_max = torch.stack([v.to(device) for v in max_abs_values]).max()
            global_mean = torch.stack([v.to(device) for v in mean_values]).mean()
            global_std = torch.stack([v.to(device) for v in std_values]).mean()
            pl_module.log("weights/global_abs_max", global_abs_max, on_step=False, on_epoch=True)
            pl_module.log("weights/global_mean", global_mean, on_step=False, on_epoch=True)
            pl_module.log("weights/global_std", global_std, on_step=False, on_epoch=True)
            log.info(
                "Epoch %s weight summary: abs_max=%.4f mean=%.4f avg_std=%.4f",
                trainer.current_epoch + 1,
                global_abs_max.detach().cpu().item(),
                global_mean.detach().cpu().item(),
                global_std.detach().cpu().item(),
            )


class NumericalStabilityCallback(L.Callback):
    """
    Monitors training for numerical instability (NaNs/Infs).
    Checks loss, gradients, and model weights.
    """
    def __init__(self, check_every_n_steps: int = 50):
        super().__init__()
        self.check_every_n_steps = _as_positive_int(check_every_n_steps, 50)
        # Local counters: trainer.global_step is constant across the micro-batches
        # of one accumulated step, so it would run these scans repeatedly.
        self._batch_end_calls = 0
        self._backward_calls = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_end_calls += 1
        if self._batch_end_calls % self.check_every_n_steps != 0:
            return

        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None:
            l_val = loss.item() if hasattr(loss, "item") else float(loss)
            if np.isnan(l_val) or np.isinf(l_val):
                log.critical(f"NaN/Inf detected in LOSS at step {trainer.global_step}! Aborting.")
                trainer.should_stop = True

    def on_after_backward(self, trainer, pl_module):
        self._backward_calls += 1
        if self._backward_calls % self.check_every_n_steps != 0:
            return

        for name, param in pl_module.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    log.error(f"NaN/Inf detected in GRADIENTS of {name} at step {trainer.global_step}!")

    def on_train_epoch_end(self, trainer, pl_module):
        for name, param in pl_module.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                log.critical(f"NaN/Inf detected in WEIGHTS of {name} at epoch {trainer.current_epoch}!")
                trainer.should_stop = True


class SystemDiagnosticsCallback(L.Callback):
    """
    Logs system-level metrics (CPU, RAM, Disk) to detect environment pressure.
    """
    def on_train_epoch_start(self, trainer, pl_module):
        cpu_pct = psutil.cpu_percent()
        ram_pct = psutil.virtual_memory().percent
        disk_pct = psutil.disk_usage("/").percent
        
        pl_module.log("sys/cpu_pct", cpu_pct, on_epoch=True)
        pl_module.log("sys/ram_pct", ram_pct, on_epoch=True)
        pl_module.log("sys/disk_pct", disk_pct, on_epoch=True)
        
        log.info(f"System Health [Epoch {trainer.current_epoch+1}]: CPU={cpu_pct}% | RAM={ram_pct}% | Disk={disk_pct}%")


class ModelComplexityCallback(L.Callback):
    """
    Logs parameter count and estimated model size in bytes.
    Useful for comparing architectures or verifying freeze logic.
    """
    def on_train_start(self, trainer, pl_module):
        total_params = sum(p.numel() for p in pl_module.model.parameters())
        trainable_params = sum(p.numel() for p in pl_module.model.parameters() if p.requires_grad)
        
        # Estimate size in MB (assuming FP32)
        est_size_mb = total_params * 4 / 1e6
        
        pl_module.log("model/total_params", float(total_params))
        pl_module.log("model/trainable_params", float(trainable_params))
        pl_module.log("model/size_mb", est_size_mb)
        
        log.info(f"Model Complexity: {total_params:,} total params | {trainable_params:,} trainable | ~{est_size_mb:.2f} MB")

def get_gpu_diagnostics(device: torch.device):
    """
    Returns a dictionary of GPU diagnostic information.
    """
    if device.type != "cuda":
        return {}
    
    try:
        props = torch.cuda.get_device_properties(device)
        return {
            "GPU Device": torch.cuda.get_device_name(device),
            "Total GPU Memory (GB)": round(props.total_memory / 1e9, 2),
            "Allocated Memory (GB)": round(torch.cuda.memory_allocated(device) / 1e9, 2),
            "Reserved Memory (GB)": round(torch.cuda.memory_reserved(device) / 1e9, 2),
            "Compute Capability": f"{props.major}.{props.minor}",
            "Multi-Processor Count": props.multi_processor_count
        }
    except Exception as e:
        log.warning(f"GPU diagnostics failed: {e}")
        return {}

def log_gpu_diagnostics(log_obj, device: torch.device):
    """
    Logs GPU diagnostic information using the provided logger.
    """
    diag = get_gpu_diagnostics(device)
    if diag:
        log_obj.info("=== GPU Diagnostics ===")
        for k, v in diag.items():
            log_obj.info(f"{k}: {v}")
        log_obj.info("=======================")
