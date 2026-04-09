"""
timesfm_wrapper.py
==================
Utilities for loading the TimesFM 2.5 backbone and extracting per-channel
embeddings to use as features in the StressProject classification pipeline.

TimesFM is a univariate time-series model, so we pass each physiological
channel as an independent series and pool the resulting embeddings.

References
----------
- Paper   : https://arxiv.org/abs/2310.10688  (ICML 2024)
- HF Hub  : google/timesfm-2.5-200m-pytorch
- GitHub  : https://github.com/google-research/timesfm
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.amp

# Set TF32 for Ampere+ GPUs (RTX 3070 Ti)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import: fail gracefully if timesfm is not installed
# ---------------------------------------------------------------------------
try:
    import timesfm  # type: ignore
    from huggingface_hub import hf_hub_download

    _TIMESFM_AVAILABLE = True
    log.info("timesfm package detected (version OK).")
except ImportError:
    _TIMESFM_AVAILABLE = False
    timesfm = None  # type: ignore
    hf_hub_download = None  # type: ignore
    log.warning(
        "timesfm is NOT installed. If you want to use the TIMESFM model, "
        "run:  pip install 'timesfm[torch]>=2.5.0'"
    )

# ---------------------------------------------------------------------------
# Embedding dimensionality for TimesFM 2.5 (200M)
# The model's d_model is 512 – this is what the backbone produces per timestep
# before the decoder projection head.
# ---------------------------------------------------------------------------
# hidden dim for the 200M model is 1280
TIMESFM_EMBED_DIM = 1280


# ---------------------------------------------------------------------------
# Internal: load the pretrained backbone (singleton-style, cached)
# ---------------------------------------------------------------------------
_CACHED_BACKBONE: Optional[nn.Module] = None
_CACHED_CHECKPOINT: Optional[str] = None


def _load_backbone(
    checkpoint: str,
    context_len: int,
    horizon: int,
    normalize_inputs: bool,
    device: torch.device,
) -> nn.Module:
    """
    Download (on first call) and return the TimesFM 2.5 backbone.
    Subsequent calls with the same checkpoint reuse the cached instance.
    """
    global _CACHED_BACKBONE, _CACHED_CHECKPOINT

    if _CACHED_BACKBONE is not None and _CACHED_CHECKPOINT == checkpoint:
        log.debug("Reusing cached TimesFM backbone.")
        return _CACHED_BACKBONE

    if not _TIMESFM_AVAILABLE:
        raise RuntimeError(
            "Cannot load TimesFM backbone: the 'timesfm' package is not installed. "
            "Install it with:  pip install \"timesfm[torch] @ git+https://github.com/google-research/timesfm.git\""
        )

    log.info(
        f"Initializing TimesFM 2.5 backbone (checkpoint: {checkpoint})..."
    )

    try:
        # 1. Instantiate the 200M parameter model class
        # Note: version 2.5 takes torch_compile in __init__
        backbone = timesfm.TimesFM_2p5_200M_torch(torch_compile=torch.cuda.is_available())

        # 2. Download and load weights manually (Hugging Face Hub)
        # We bypass .from_pretrained() because of a signature mismatch in the 2.5 torch mixin.
        log.info(f"Downloading weights from HF hub: {checkpoint}")
        weights_path = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
        
        log.info(f"Loading checkpoint weights into internal module from: {weights_path}")
        backbone.model.load_checkpoint(weights_path, torch_compile=torch.cuda.is_available())

        # 3. Configure for inference
        backbone.compile(
            timesfm.ForecastConfig(
                max_context=context_len,
                max_horizon=horizon,
                normalize_inputs=normalize_inputs,
                use_continuous_quantile_head=False,
            )
        )

        backbone.model = backbone.model.to(device)
        backbone.model.eval()
        _CACHED_BACKBONE = backbone
        _CACHED_CHECKPOINT = checkpoint
        log.info("TimesFM backbone loaded and aligned successfully.")
        return backbone
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load/initialize TimesFM 2.5: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Embedding extractor: hooks into the backbone, collects hidden states
# ---------------------------------------------------------------------------
class TimesFMEmbeddingExtractor(nn.Module):
    """
    Wraps the TimesFM backbone and exposes a forward() method that
    returns a flat embedding vector per input window.

    Strategy
    --------
    We register a forward hook on the last transformer-attention / MLP block
    to capture the model's internal representations *before* the forecasting
    decoder head.  These representations are then:
      1. Averaged over the time dimension  →  (batch, embed_dim)
      2. Returned for downstream classification

    If hook capture fails (API change), we fall back to using the raw
    `point_forecast` output as a weak embedding proxy.
    """

    def __init__(
        self,
        checkpoint: str,
        context_len: int,
        horizon: int,
        normalize_inputs: bool,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.checkpoint = checkpoint
        self.context_len = context_len
        self.horizon = horizon
        self.normalize_inputs = normalize_inputs
        self.device = device

        # Will be populated in _attach()
        self._backbone: Optional[nn.Module] = None
        self._hook_handle = None
        self._captured_hidden: Optional[torch.Tensor] = None
        self.embed_dim: int = TIMESFM_EMBED_DIM
        self._hook_ok: bool = False

    # ------------------------------------------------------------------
    def _attach(self) -> None:
        """Lazy-load backbone and attach the embedding hook."""
        if self._backbone is not None:
            return

        self._backbone = _load_backbone(
            self.checkpoint,
            self.context_len,
            self.horizon,
            self.normalize_inputs,
            self.device,
        )
        self._hook_ok = self._register_hook()

    def _register_hook(self) -> bool:
        """
        Try to find the last suitable sub-module (LayerNorm / Linear) in the
        TimesFM backbone and attach a forward hook to capture its output.

        Returns True if a hook was successfully attached, False otherwise.
        """
        target_module: Optional[nn.Module] = None

        # Walk all modules to find suitable hook targets (RMSNorm or LayerNorm)
        # In version 2.5, we looking for names like 'stacked_xf.19.post_ff_ln'
        available_norms = []
        for name, mod in self._backbone.model.named_modules():
            type_name = type(mod).__name__.lower()
            if "norm" in type_name or name.endswith("_ln"):
                available_norms.append((name, mod))

        if available_norms:
            # We target the very last normalization layer in the transformer stack
            last_ln_name, target_module = available_norms[-1]
            log.info(
                f"TimesFM hook attached to: '{last_ln_name}' ({type(target_module).__name__}) "
                f"for embedding extraction."
            )
        else:
            log.warning(
                "Could not find a suitable Norm layer in TimesFM 2.5 backbone. "
                "Falling back to forecast-output proxy embedding."
            )
            return False

        def _hook_fn(module, input, output):  # noqa: ARG001
            # output shape from LayerNorm: (batch, seq, d_model) or (batch, d_model)
            self._captured_hidden = output.detach()

        self._hook_handle = target_module.register_forward_hook(_hook_fn)
        return True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for a batch of univariate time-series windows.

        Parameters
        ----------
        x : torch.Tensor, shape (N, T)
            Batch of N univariate windows, each of length T timesteps.

        Returns
        -------
        embedding : torch.Tensor, shape (N, embed_dim)
        """
        self._attach()
        N, T = x.shape

        # Truncate / pad context to configured length
        if T > self.context_len:
            x = x[:, -self.context_len:]
        elif T < self.context_len:
            pad = torch.zeros(N, self.context_len - T, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)

        self._captured_hidden = None

        with torch.no_grad():
            # Enable Mixed Precision for the backbone pass
            device_type = x.device.type
            # Prefer Bfloat16 on Ampere+ GPUs (like RTX 3070 Ti)
            dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                try:
                    # 1. Prepare inputs: reshape to (N, -1, patch_size)
                    patch_size = self._backbone.model.p
                    patched_x = x.reshape(N, -1, patch_size)
                    
                    # 2. Create masks (everything valid = False)
                    # Use empty + fill_ for faster allocation than zeros_like
                    masks = torch.empty_like(patched_x, dtype=torch.bool, device=x.device).fill_(False)
                    
                    # 3. Call internal forward pass directly to preserve batching
                    # In 2.5, calling the backbone.model(...) triggers the transformer stack
                    # and our hook catches the last activation.
                    self._backbone.model(patched_x, masks)
                    
                except Exception as exc:
                    log.warning(
                        f"TimesFM forward pass raised: {exc!r}. "
                        "Returning zero embeddings for this batch."
                    )
                    return torch.zeros(N, self.embed_dim, device=x.device)

        if self._hook_ok and self._captured_hidden is not None:
            h = self._captured_hidden  # (N, T', d_model) or (N, d_model)
            if h.ndim == 3:
                h = h.mean(dim=1)       # temporal mean pooling → (N, d_model)
            # If the device/dtype differs (hook captures on CPU when backbone is CPU)
            return h.to(device=x.device, dtype=torch.float32)
        else:
            # Fallback: point_forecast not captured — return zeros with correct dim
            log.debug("Hook did not capture hidden state; returning zero embedding.")
            return torch.zeros(N, self.embed_dim, device=x.device)

    def remove_hook(self) -> None:
        """Detach the forward hook (call during teardown)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def freeze_backbone(self) -> None:
        """Freeze all TimesFM backbone parameters (training phase 1)."""
        if self._backbone is not None:
            for p in self._backbone.parameters():
                p.requires_grad_(False)
            log.info("TimesFM backbone frozen (classification head only will train).")

    def unfreeze_backbone(self, last_n_blocks: int = 4) -> None:
        """
        Unfreeze the last `last_n_blocks` transformer blocks for fine-tuning
        (training phase 2).  Set last_n_blocks=-1 to unfreeze everything.
        """
        if self._backbone is None:
            return

        # First freeze everything
        for p in self._backbone.parameters():
            p.requires_grad_(False)

        if last_n_blocks == -1:
            for p in self._backbone.parameters():
                p.requires_grad_(True)
            log.info("All TimesFM backbone parameters unfrozen for fine-tuning.")
            return

        # In version 2.5, blocks are in model.stacked_xf
        block_modules = []
        if hasattr(self._backbone.model, "stacked_xf"):
            block_modules = list(self._backbone.model.stacked_xf)
        
        if not block_modules:
            # Fallback: try to find anything that looks like a block
            for mod in self._backbone.modules():
                if "Transformer" in type(mod).__name__:
                    block_modules.append(mod)

        if not block_modules:
            # Fallback: unfreeze all if architecture unclear
            for p in self._backbone.parameters():
                p.requires_grad_(True)
            log.warning(
                "Could not locate Transformer blocks; unfreezing all backbone params."
            )
            return

        # Unfreeze last N blocks
        blocks_to_unfreeze = block_modules[-last_n_blocks:]
        for block in blocks_to_unfreeze:
            for p in block.parameters():
                p.requires_grad_(True)

        n_trainable = sum(
            p.numel() for p in self._backbone.parameters() if p.requires_grad
        )
        log.info(
            f"Unfroze last {last_n_blocks} TimesFM block(s) for fine-tuning. "
            f"Trainable backbone params: {n_trainable:,}"
        )


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """Return True if the timesfm package is importable."""
    return _TIMESFM_AVAILABLE
