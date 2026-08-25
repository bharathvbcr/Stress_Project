# mlx_hybrid.py — Hybrid PyTorch + MLX inference (TimesFM backbone + MLX head)
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from mlx_convert import convert_timesfm_classifier, load_torch_state_dict
from mlx_models import MlxTimesFMClassifier, save_mlx_checkpoint
from utils import safe_get, select_torch_device

log = logging.getLogger(__name__)


def build_timesfm_hybrid(
    config: Dict[str, Any],
    checkpoint_path: str,
    input_dim_sequence: int,
    input_dim_static: int,
    device: Optional[torch.device] = None,
) -> Tuple[Any, MlxTimesFMClassifier, int]:
    """Load TimesFM PyTorch backbone and MLX classifier head from a checkpoint."""
    from models import get_model

    device = device or select_torch_device()
    torch_model = get_model(config, input_dim_sequence, input_dim_static)
    torch_sd = load_torch_state_dict(checkpoint_path)
    torch_model.load_state_dict(torch_sd, strict=False)

    from timesfm_lora import restore_timesfm_lora_adapters

    restore_timesfm_lora_adapters(torch_model, torch_sd, config=config)
    torch_model = torch_model.to(device).eval()

    model_config = safe_get(config, ["model_config"], {})
    head_hidden = int(safe_get(model_config, ["timesfm_head_hidden_dim"], 128))
    dropout = float(safe_get(model_config, ["dropout"], 0.2))
    embed_dim = int(getattr(torch_model, "embed_dim", safe_get(model_config, ["timesfm_embed_dim"], 1280)))
    fc_in = embed_dim + input_dim_static

    mlx_head = MlxTimesFMClassifier(fc_in, head_hidden, dropout)
    convert_timesfm_classifier(mlx_head, torch_sd)
    return torch_model, mlx_head, embed_dim


def timesfm_hybrid_predict(
    torch_model,
    mlx_head,
    seq_tensor: torch.Tensor,
    static_tensor: Optional[torch.Tensor] = None,
) -> float:
    """Run TimesFM embedding in PyTorch and classify with MLX head."""
    import mlx.core as mx

    with torch.no_grad():
        torch_model._ensure_attached()
        batch = seq_tensor.shape[0]
        n, t, f = seq_tensor.shape
        x_flat = seq_tensor.permute(0, 2, 1).reshape(n * f, t)
        channel_embeds = torch_model.extractor(x_flat)
        channel_embeds = channel_embeds.view(n, f, torch_model.embed_dim)
        pooled = channel_embeds.mean(dim=1)

        if torch_model.input_dim_static > 0:
            if static_tensor is None:
                static_tensor = torch.zeros(batch, torch_model.input_dim_static, device=seq_tensor.device)
            fused = torch.cat([pooled, static_tensor], dim=1)
        else:
            fused = pooled

    fused_mx = mx.array(fused.detach().cpu().numpy(), dtype=mx.float32)
    logits = mlx_head(fused_mx)
    return float(mx.sigmoid(logits.reshape(-1))[0].item())


def collect_timesfm_hybrid_probabilities(
    torch_model,
    mlx_head,
    dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect split probabilities from TimesFM hybrid inference."""
    import mlx.core as mx

    from mlx_training import _iter_batches

    # The PyTorch backbone runs per batch (bounded device memory), but fused
    # embeddings are accumulated on the host and handed to MLX exactly once.
    # The previous implementation paid a numpy->MLX conversion, a head forward
    # graph build, and an mx.eval per batch; with unified memory a single
    # conversion + one head forward over the full split is strictly cheaper.
    fused_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    torch_model.eval()
    mlx_head.eval()

    for batch in _iter_batches(dataset, batch_size):
        seq = batch["sequence"]
        static = batch.get("static")
        labels = batch["label"]
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)
        if static is not None and not isinstance(static, torch.Tensor):
            static = torch.tensor(static, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)

        seq = seq.to(device)
        if static is not None:
            static = static.to(device)

        with torch.no_grad():
            torch_model._ensure_attached()
            n, t, f = seq.shape
            x_flat = seq.permute(0, 2, 1).reshape(n * f, t)
            channel_embeds = torch_model.extractor(x_flat)
            channel_embeds = channel_embeds.view(n, f, torch_model.embed_dim)
            pooled = channel_embeds.mean(dim=1)
            if torch_model.input_dim_static > 0:
                if static is None:
                    static = torch.zeros(n, torch_model.input_dim_static, device=device)
                fused = torch.cat([pooled, static], dim=1)
            else:
                fused = pooled

        fused_chunks.append(fused.detach().cpu().numpy().astype(np.float32, copy=False))
        label_chunks.append(labels.reshape(-1).cpu().numpy())

    if not fused_chunks:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    fused_np = np.concatenate(fused_chunks, axis=0)
    logits = mlx_head(mx.array(fused_np, dtype=mx.float32))
    probs = mx.sigmoid(logits.astype(mx.float32)).reshape(-1)
    mx.eval(probs)

    return (
        np.asarray(probs).astype(np.float64),
        np.concatenate(label_chunks).astype(np.float64),
    )


def export_timesfm_head_mlx(
    config: Dict[str, Any],
    checkpoint_path: str,
    output_path: str,
    input_dim_static: int,
    embed_dim: int = 1280,
) -> str:
    """Export only the TimesFM classifier head to an MLX checkpoint."""
    torch_sd = load_torch_state_dict(checkpoint_path)
    model_config = safe_get(config, ["model_config"], {})
    head_hidden = int(safe_get(model_config, ["timesfm_head_hidden_dim"], 128))
    dropout = float(safe_get(model_config, ["dropout"], 0.2))
    fc_in = embed_dim + input_dim_static
    mlx_head = MlxTimesFMClassifier(fc_in, head_hidden, dropout)
    convert_timesfm_classifier(mlx_head, torch_sd)
    save_mlx_checkpoint(mlx_head, output_path)
    log.info("Exported TimesFM MLX head to %s", output_path)
    return output_path


def export_timesfm_mlx_artifacts(
    config: Dict[str, Any],
    torch_model,
    *,
    checkpoint_path: Optional[str] = None,
    input_dim_static: int,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Export TimesFM MLX head from the live model or a Phase-2 checkpoint."""
    import os

    if os.environ.get("STRESS_MLX_EXPORT_HEAD", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None

    output_dir = output_dir or safe_get(config, ["save_paths", "models"], "outputs/models")
    os.makedirs(output_dir, exist_ok=True)
    head_path = os.path.join(output_dir, "best_timesfm_head_mlx.npz")
    embed_dim = int(getattr(torch_model, "embed_dim", safe_get(config, ["model_config", "timesfm_embed_dim"], 1280)))

    if checkpoint_path and os.path.isfile(checkpoint_path):
        return export_timesfm_head_mlx(
            config,
            checkpoint_path,
            head_path,
            input_dim_static,
            embed_dim=embed_dim,
        )

    classifier_sd = {
        k: v for k, v in torch_model.state_dict().items() if k.startswith("classifier.")
    }
    if not classifier_sd:
        log.warning("No classifier weights found on TimesFM model; skipping MLX head export.")
        return None

    model_config = safe_get(config, ["model_config"], {})
    head_hidden = int(safe_get(model_config, ["timesfm_head_hidden_dim"], 128))
    dropout = float(safe_get(model_config, ["dropout"], 0.2))
    fc_in = embed_dim + input_dim_static
    mlx_head = MlxTimesFMClassifier(fc_in, head_hidden, dropout)
    convert_timesfm_classifier(mlx_head, classifier_sd)
    save_mlx_checkpoint(mlx_head, head_path)
    log.info("Exported TimesFM MLX head from live model to %s", head_path)
    return head_path
