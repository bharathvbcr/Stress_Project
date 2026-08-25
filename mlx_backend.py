# mlx_backend.py — Apple MLX runtime selection and helpers
from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict, Optional, Set

import numpy as np

log = logging.getLogger(__name__)

MLX_SUPPORTED_MODELS: Set[str] = frozenset({
    "LSTM",
    "PATCHTST",
    "PATCH-TST",
    "TRANSFORMER",
    "CNN-LSTM",
})
_DEFAULT_BACKEND = "auto"

# STRESS_MLX_DTYPE aliases -> canonical mlx dtype attribute names.
_MLX_DTYPE_ALIASES = {
    "float32": "float32",
    "fp32": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
}
_DEFAULT_MLX_DTYPE = "float32"


def _normalize_backend(value: Optional[str]) -> str:
    if not value:
        return _DEFAULT_BACKEND
    normalized = value.strip().lower()
    if normalized in {"gpu", "cuda"}:
        return "cuda"
    if normalized in {"mps", "metal"}:
        return "mps"
    if normalized in {"mlx", "apple"}:
        return "mlx"
    if normalized in {"cpu", "auto"}:
        return normalized
    log.warning("Unknown STRESS_BACKEND=%r; using auto.", value)
    return _DEFAULT_BACKEND


def is_mlx_available() -> bool:
    try:
        import mlx.core as _mlx_core  # noqa: F401
        return True
    except ImportError:
        return False


def mlx_version() -> Optional[str]:
    if not is_mlx_available():
        return None
    try:
        import mlx.core as mx
        return str(mx.__version__)
    except Exception:
        return None


def mlx_supports_model(model_type: str) -> bool:
    return model_type.strip().upper() in MLX_SUPPORTED_MODELS


def mlx_supports_inference(model_type: str) -> bool:
    """Return True when MLX can serve inference (full model or TimesFM hybrid head)."""
    model = model_type.strip().upper()
    return mlx_supports_model(model) or model == "TIMESFM"


def get_requested_backend() -> str:
    return _normalize_backend(os.environ.get("STRESS_BACKEND"))


def resolve_training_backend(
    model_type: str,
    *,
    mlx_available: bool,
    cuda_available: bool,
    mps_available: bool,
    requested: Optional[str] = None,
) -> str:
    """Return ``mlx`` or ``torch`` for the training entrypoint."""
    request = _normalize_backend(requested or get_requested_backend())
    model = model_type.strip().upper()

    if request == "mlx":
        if mlx_available and mlx_supports_model(model):
            return "mlx"
        log.warning(
            "STRESS_BACKEND=mlx requested but MLX is unavailable or unsupported for %s; using torch.",
            model,
        )
        return "torch"

    if request in {"cuda", "mps", "cpu"}:
        return "torch"

    # auto: prefer MLX on Apple silicon for supported models
    if (
        platform.system() == "Darwin"
        and platform.machine() in {"arm64", "aarch64"}
        and mlx_available
        and mlx_supports_model(model)
        and not cuda_available
    ):
        return "mlx"

    return "torch"


def _cuda_available_flag() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_inference_backend(
    model_type: str,
    checkpoint_path: Optional[str],
    *,
    mlx_available: bool,
    requested: Optional[str] = None,
) -> str:
    """Return ``mlx`` or ``torch`` for API inference."""
    if checkpoint_path and checkpoint_path.endswith(".mlx.npz"):
        return "mlx" if mlx_available else "torch"

    request = _normalize_backend(requested or get_requested_backend())
    model = model_type.strip().upper()

    if request in {"cuda", "mps", "cpu"}:
        return "torch"

    if (
        request == "auto"
        and platform.system() == "Darwin"
        and platform.machine() in {"arm64", "aarch64"}
        and mlx_available
        and mlx_supports_inference(model)
        and not _cuda_available_flag()
    ):
        return "mlx"

    if request == "mlx" and mlx_available and mlx_supports_inference(model):
        return "mlx"
    if request == "mlx":
        log.warning("STRESS_BACKEND=mlx requested for inference but falling back to torch.")
    return "torch"


def resolve_timesfm_hybrid_enabled(requested: Optional[str] = None) -> bool:
    request = _normalize_backend(requested or get_requested_backend())
    if request in {"mps", "cuda", "cpu"}:
        return False
    return is_mlx_available() and request in {"mlx", "auto"}


def configure_mlx_runtime() -> Dict[str, Any]:
    """Apply MLX defaults for Apple-silicon execution."""
    info: Dict[str, Any] = {
        "mlx_available": is_mlx_available(),
        "mlx_version": mlx_version(),
        "mlx_compile": False,
    }
    if not info["mlx_available"]:
        return info

    import mlx.core as mx

    compile_requested = os.environ.get("STRESS_MLX_COMPILE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    info["mlx_compile"] = compile_requested
    info["mlx_dtype"] = resolve_mlx_dtype_name()

    default_device = os.environ.get("STRESS_MLX_DEVICE", "gpu").strip().lower()
    if default_device in {"cpu"}:
        mx.set_default_device(mx.cpu)
        info["mlx_device"] = "cpu"
    else:
        mx.set_default_device(mx.gpu)
        info["mlx_device"] = "gpu"

    return info


def resolve_mlx_dtype_name() -> str:
    """Return the canonical MLX compute dtype name from ``STRESS_MLX_DTYPE``.

    float32 is the default; bfloat16/float16 are opt-in for reduced-precision
    training and inference on Apple silicon (roughly halves activation/weight
    memory and improves throughput on M-series GPUs).
    """
    raw = os.environ.get("STRESS_MLX_DTYPE", _DEFAULT_MLX_DTYPE).strip().lower()
    name = _MLX_DTYPE_ALIASES.get(raw)
    if name is None:
        log.warning("Unknown STRESS_MLX_DTYPE=%r; using %s.", raw, _DEFAULT_MLX_DTYPE)
        return _DEFAULT_MLX_DTYPE
    return name


def resolve_mlx_dtype():
    """Return the mx.Dtype selected via ``STRESS_MLX_DTYPE`` (see resolve_mlx_dtype_name)."""
    import mlx.core as mx

    return getattr(mx, resolve_mlx_dtype_name())


def cast_mlx_model_parameters(model, dtype=None):
    """Cast a model's floating-point parameters to ``dtype`` (mixed precision).

    No-op when the target dtype is float32. Integer/bool parameters are left
    untouched. Loss computation should still be done in float32 by the caller
    (cast logits up) for numerical stability.
    """
    import mlx.core as mx

    dtype = dtype or resolve_mlx_dtype()
    if dtype == mx.float32:
        return model
    model.apply(lambda p: p.astype(dtype) if mx.issubdtype(p.dtype, mx.floating) else p)
    return model


def numpy_to_mlx(array: np.ndarray):
    import mlx.core as mx
    return mx.array(array)


def mlx_to_numpy(array) -> np.ndarray:
    return np.asarray(array)


def maybe_compile_mlx_callable(fn, enabled: bool = True):
    if not enabled or not is_mlx_available():
        return fn
    import mlx.core as mx
    return mx.compile(fn)
