# timesfm_lora.py — Low-rank adapters for TimesFM Phase-2 fine-tuning on Apple Silicon
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

log = logging.getLogger(__name__)


class LoRAAdapter(nn.Module):
    """Additive low-rank adapter: delta(x) = (x @ A^T @ B^T) * scaling."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_a.T @ self.lora_b.T) * self.scaling


def _lora_forward_hook(adapter: LoRAAdapter):
    def hook(_module, inputs, output):
        return output + adapter(inputs[0])

    return hook


def apply_lora_to_linear(
    linear: nn.Linear,
    rank: int,
    alpha: float,
    adapter: Optional[LoRAAdapter] = None,
) -> Tuple[LoRAAdapter, torch.utils.hooks.RemovableHandle]:
    """Freeze a Linear layer and attach a trainable LoRA adapter via forward hook."""
    linear.weight.requires_grad_(False)
    if linear.bias is not None:
        linear.bias.requires_grad_(False)
    if adapter is None:
        adapter = LoRAAdapter(linear.in_features, linear.out_features, rank, alpha)
    device = linear.weight.device
    dtype = linear.weight.dtype
    adapter = adapter.to(device=device, dtype=dtype)
    handle = linear.register_forward_hook(_lora_forward_hook(adapter))
    return adapter, handle


def attach_lora_adapter(
    linear: nn.Linear,
    adapter: LoRAAdapter,
) -> torch.utils.hooks.RemovableHandle:
    """Attach a pre-built LoRA adapter to a frozen Linear layer."""
    linear.weight.requires_grad_(False)
    if linear.bias is not None:
        linear.bias.requires_grad_(False)
    adapter = adapter.to(device=linear.weight.device, dtype=linear.weight.dtype)
    return linear.register_forward_hook(_lora_forward_hook(adapter))


def iter_lora_target_linears(model, last_n_blocks: int):
    """Yield Linear layers in the last N TimesFM stacked_xf blocks (deterministic order)."""
    if hasattr(model, "_ensure_attached"):
        model._ensure_attached()
    backbone = model.extractor._backbone.model
    blocks = backbone.stacked_xf[-last_n_blocks:]
    for block in blocks:
        for mod in block.modules():
            if isinstance(mod, nn.Linear):
                yield mod


def has_lora_weights(state_dict: Dict) -> bool:
    return any(str(key).startswith("lora_adapters.") for key in state_dict)


def count_lora_adapters(state_dict: Dict) -> int:
    indices = set()
    for key in state_dict:
        if not str(key).startswith("lora_adapters."):
            continue
        parts = str(key).split(".")
        if len(parts) > 1 and parts[1].isdigit():
            indices.add(int(parts[1]))
    return len(indices)


def infer_lora_rank(state_dict: Dict, index: int = 0) -> int:
    key = f"lora_adapters.{index}.lora_a"
    if key not in state_dict:
        raise KeyError(f"Missing LoRA weight key: {key}")
    return int(state_dict[key].shape[0])


def clear_lora_hooks(model) -> None:
    handles = getattr(model, "_lora_hook_handles", None)
    if handles:
        for handle in handles:
            handle.remove()
    model._lora_hook_handles = []


def restore_timesfm_lora_adapters(
    model,
    state_dict: Dict,
    *,
    last_n_blocks: Optional[int] = None,
    config: Optional[Dict] = None,
) -> bool:
    """Reattach LoRA forward hooks after loading a checkpoint (inference/eval)."""
    if not has_lora_weights(state_dict):
        return False

    from utils import safe_get

    if hasattr(model, "_ensure_attached"):
        model._ensure_attached()

    n_adapters = count_lora_adapters(state_dict)
    if last_n_blocks is None and config is not None:
        last_n_blocks = int(
            safe_get(config, ["model_config", "timesfm_finetune_last_n_blocks"], 4)
        )
    if last_n_blocks is None:
        last_n_blocks = int(getattr(model, "_lora_last_n_blocks", 4))

    rank = infer_lora_rank(state_dict)
    alpha = resolve_timesfm_lora_alpha()
    linears = list(iter_lora_target_linears(model, last_n_blocks))
    if len(linears) < n_adapters:
        log.warning(
            "LoRA restore: expected %d adapters but only %d Linear targets in last %d block(s).",
            n_adapters,
            len(linears),
            last_n_blocks,
        )

    clear_lora_hooks(model)
    adapters = nn.ModuleList()
    handles = []
    for idx, linear in enumerate(linears[:n_adapters]):
        prefix = f"lora_adapters.{idx}."
        adapter = LoRAAdapter(linear.in_features, linear.out_features, rank, alpha)
        adapter.lora_a.data.copy_(state_dict[f"{prefix}lora_a"])
        adapter.lora_b.data.copy_(state_dict[f"{prefix}lora_b"])
        handle = attach_lora_adapter(linear, adapter)
        adapters.append(adapter)
        handles.append(handle)

    model.lora_adapters = adapters
    model._lora_hook_handles = handles
    model._lora_last_n_blocks = last_n_blocks
    if hasattr(model, "extractor") and model.extractor._backbone is not None:
        model.extractor._backbone.model.eval()
    log.info("Restored %d TimesFM LoRA adapter(s) for inference.", n_adapters)
    return True


def apply_lora_to_blocks(
    blocks: nn.ModuleList,
    last_n_blocks: int,
    rank: int,
    alpha: float,
) -> Tuple[nn.ModuleList, List[LoRAAdapter], List[torch.utils.hooks.RemovableHandle]]:
    """Attach LoRA adapters to Linear layers in the last N transformer blocks."""
    if last_n_blocks <= 0:
        return nn.ModuleList(), []

    selected = list(blocks[-last_n_blocks:])
    adapters: List[LoRAAdapter] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for block in selected:
        for mod in block.modules():
            if isinstance(mod, nn.Linear):
                adapter, handle = apply_lora_to_linear(mod, rank, alpha)
                adapters.append(adapter)
                handles.append(handle)
    log.info(
        "Attached %d LoRA adapter(s) across the last %d TimesFM block(s) (rank=%s, alpha=%s).",
        len(adapters),
        last_n_blocks,
        rank,
        alpha,
    )
    return nn.ModuleList(adapters), adapters, handles


def resolve_timesfm_lora_rank() -> int:
    return int(os.environ.get("TIMESFM_LORA_RANK", "8"))


def resolve_timesfm_lora_alpha() -> float:
    return float(os.environ.get("TIMESFM_LORA_ALPHA", "16"))


def should_use_timesfm_lora_phase2(runtime_caps: Dict) -> bool:
    """Prefer LoRA over full backbone unfreeze when enabled (default on Apple MLX hybrid)."""
    explicit = os.environ.get("TIMESFM_LORA_PHASE2", "auto").strip().lower()
    if explicit in {"0", "false", "no", "off"}:
        return False
    if explicit in {"1", "true", "yes", "on"}:
        return True
    try:
        from mlx_backend import resolve_timesfm_hybrid_enabled

        return bool(
            resolve_timesfm_hybrid_enabled(runtime_caps.get("backend_request"))
            and runtime_caps.get("mlx_available")
        )
    except ImportError:
        return False
