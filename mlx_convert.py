# mlx_convert.py — Convert PyTorch/Lightning checkpoints to MLX weights
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mlx_backend import mlx_supports_model
from mlx_models import get_mlx_model, load_mlx_checkpoint, save_mlx_checkpoint
from utils import extract_model_state_dict, safe_get

log = logging.getLogger(__name__)

_STRIP_PREFIXES = (
    "model.",
    "lightning_module.model.",
    "module.",
)


def torch_to_mx(tensor) -> Any:
    import mlx.core as mx

    return mx.array(np.asarray(tensor.detach().cpu(), dtype=np.float32))


def normalize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in _STRIP_PREFIXES:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


def load_torch_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return normalize_state_dict(extract_model_state_dict(checkpoint))


def _set_lstm_from_torch(
    mlx_lstm,
    weight_ih,
    weight_hh,
    bias_ih,
    bias_hh,
) -> None:
    mlx_lstm.update(
        {
            "Wx": torch_to_mx(weight_ih),
            "Wh": torch_to_mx(weight_hh),
            "bias": torch_to_mx(bias_ih + bias_hh),
        }
    )


def _convert_lstm_stack(
    mlx_model,
    torch_sd: Dict[str, Any],
    prefix_torch: str,
    prefix_mlx: str,
    bidirectional: bool,
) -> None:
    layer_idx = 0
    while f"{prefix_torch}.{layer_idx}.weight_ih_l0" in torch_sd:
        if bidirectional:
            fwd = getattr(mlx_model, f"{prefix_mlx}_{layer_idx}_fwd")
            bwd = getattr(mlx_model, f"{prefix_mlx}_{layer_idx}_bwd")
            _set_lstm_from_torch(
                fwd,
                torch_sd[f"{prefix_torch}.{layer_idx}.weight_ih_l0"],
                torch_sd[f"{prefix_torch}.{layer_idx}.weight_hh_l0"],
                torch_sd[f"{prefix_torch}.{layer_idx}.bias_ih_l0"],
                torch_sd[f"{prefix_torch}.{layer_idx}.bias_hh_l0"],
            )
            _set_lstm_from_torch(
                bwd,
                torch_sd[f"{prefix_torch}.{layer_idx}.weight_ih_l0_reverse"],
                torch_sd[f"{prefix_torch}.{layer_idx}.weight_hh_l0_reverse"],
                torch_sd[f"{prefix_torch}.{layer_idx}.bias_ih_l0_reverse"],
                torch_sd[f"{prefix_torch}.{layer_idx}.bias_hh_l0_reverse"],
            )
        else:
            layer = getattr(mlx_model, f"{prefix_mlx}_{layer_idx}")
            _set_lstm_from_torch(
                layer,
                torch_sd[f"{prefix_torch}.{layer_idx}.weight_ih_l0"],
                torch_sd[f"{prefix_torch}.{layer_idx}.weight_hh_l0"],
                torch_sd[f"{prefix_torch}.{layer_idx}.bias_ih_l0"],
                torch_sd[f"{prefix_torch}.{layer_idx}.bias_hh_l0"],
            )
        layer_idx += 1


def _convert_linear_module(mlx_linear, weight, bias=None) -> None:
    update = {"weight": torch_to_mx(weight)}
    if bias is not None and hasattr(mlx_linear, "bias"):
        update["bias"] = torch_to_mx(bias)
    mlx_linear.update(update)


def _convert_layer_norm(mlx_norm, weight, bias) -> None:
    mlx_norm.update({"weight": torch_to_mx(weight), "bias": torch_to_mx(bias)})


def _convert_torch_encoder_layer_to_mlx(mlx_layer, torch_prefix: str, torch_sd: Dict[str, Any]) -> None:
    import mlx.core as mx

    d_model = torch_sd[f"{torch_prefix}.norm1.weight"].shape[0]
    in_proj = torch_sd[f"{torch_prefix}.self_attn.in_proj_weight"].cpu().numpy()
    flat: Dict[str, Any] = {}
    for idx, name in enumerate(["query_proj", "key_proj", "value_proj"]):
        flat[f"attention.{name}.weight"] = mx.array(in_proj[idx * d_model : (idx + 1) * d_model])
    in_proj_bias_key = f"{torch_prefix}.self_attn.in_proj_bias"
    if in_proj_bias_key in torch_sd:
        bias = torch_sd[in_proj_bias_key].cpu().numpy()
        flat["attention.query_proj.bias"] = mx.array(bias[:d_model])
        flat["attention.key_proj.bias"] = mx.array(bias[d_model : 2 * d_model])
        flat["attention.value_proj.bias"] = mx.array(bias[2 * d_model :])
    flat["attention.out_proj.weight"] = torch_to_mx(torch_sd[f"{torch_prefix}.self_attn.out_proj.weight"])
    out_bias_key = f"{torch_prefix}.self_attn.out_proj.bias"
    if out_bias_key in torch_sd:
        flat["attention.out_proj.bias"] = torch_to_mx(torch_sd[out_bias_key])
    flat["ln1.weight"] = torch_to_mx(torch_sd[f"{torch_prefix}.norm1.weight"])
    flat["ln1.bias"] = torch_to_mx(torch_sd[f"{torch_prefix}.norm1.bias"])
    flat["ln2.weight"] = torch_to_mx(torch_sd[f"{torch_prefix}.norm2.weight"])
    flat["ln2.bias"] = torch_to_mx(torch_sd[f"{torch_prefix}.norm2.bias"])
    flat["linear1.weight"] = torch_to_mx(torch_sd[f"{torch_prefix}.linear1.weight"])
    flat["linear1.bias"] = torch_to_mx(torch_sd[f"{torch_prefix}.linear1.bias"])
    flat["linear2.weight"] = torch_to_mx(torch_sd[f"{torch_prefix}.linear2.weight"])
    flat["linear2.bias"] = torch_to_mx(torch_sd[f"{torch_prefix}.linear2.bias"])

    fd_path = _scratch_npz(flat)
    try:
        mlx_layer.load_weights(fd_path)
    finally:
        os.remove(fd_path)


def _scratch_npz(flat: Dict[str, Any]) -> str:
    import mlx.core as mx
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    mx.savez(path, **flat)
    return path


def _convert_transformer_encoder(torch_sd: Dict[str, Any], prefix_torch: str, mlx_encoder) -> None:
    layer_idx = 0
    while f"{prefix_torch}.layers.{layer_idx}.norm1.weight" in torch_sd:
        mlx_layer = mlx_encoder.layers[layer_idx]
        _convert_torch_encoder_layer_to_mlx(
            mlx_layer,
            f"{prefix_torch}.layers.{layer_idx}",
            torch_sd,
        )
        layer_idx += 1


def _convert_positional_encoding(mlx_pos_encoder, torch_sd: Dict[str, Any], prefix: str = "pos_encoder") -> None:
    pe_key = f"{prefix}.pe"
    if pe_key not in torch_sd:
        return
    import mlx.core as mx

    pe = torch_sd[pe_key].cpu().numpy()
    if pe.ndim == 3:
        pe = np.squeeze(pe, axis=1)
    mlx_pos_encoder.pe = mx.array(pe.astype(np.float32))


def _convert_sequential_linear_head(torch_sd: Dict[str, Any], prefix_torch: str, mlx_seq) -> None:
    index_map = {
        0: (f"{prefix_torch}.0.weight", f"{prefix_torch}.0.bias"),
        3: (f"{prefix_torch}.3.weight", f"{prefix_torch}.3.bias"),
    }
    for layer_idx, (weight_key, bias_key) in index_map.items():
        if weight_key not in torch_sd:
            continue
        linear = mlx_seq.layers[layer_idx]
        _convert_linear_module(linear, torch_sd[weight_key], torch_sd.get(bias_key))


def convert_lstm_model(mlx_model, torch_sd: Dict[str, Any]) -> None:
    bidirectional = any(k.endswith("_reverse") for k in torch_sd if k.startswith("lstm_layers_stacked"))
    _convert_lstm_stack(mlx_model, torch_sd, "lstm_layers_stacked", "lstm", bidirectional)
    _convert_linear_module(mlx_model.fc, torch_sd["fc.weight"], torch_sd.get("fc.bias"))


def convert_patchtst_model(mlx_model, torch_sd: Dict[str, Any]) -> None:
    _convert_linear_module(
        mlx_model.patch_projection,
        torch_sd["patch_projection.weight"],
        torch_sd.get("patch_projection.bias"),
    )
    _convert_positional_encoding(mlx_model.pos_encoder, torch_sd)
    _convert_layer_norm(mlx_model.norm, torch_sd["norm.weight"], torch_sd["norm.bias"])
    _convert_transformer_encoder(torch_sd, "transformer_encoder", mlx_model.transformer_encoder)
    _convert_sequential_linear_head(torch_sd, "fc_head", mlx_model.fc_head)


def convert_transformer_model(mlx_model, torch_sd: Dict[str, Any]) -> None:
    _convert_linear_module(
        mlx_model.seq_embedding,
        torch_sd["seq_embedding.weight"],
        torch_sd.get("seq_embedding.bias"),
    )
    _convert_positional_encoding(mlx_model.pos_encoder, torch_sd)
    _convert_transformer_encoder(torch_sd, "transformer_encoder", mlx_model.transformer_encoder)
    _convert_sequential_linear_head(torch_sd, "fc_head", mlx_model.fc_head)


def convert_cnn_lstm_model(mlx_model, torch_sd: Dict[str, Any]) -> None:
    import mlx.core as mx

    block_num = 0
    idx = 0
    while f"cnn_encoder.{idx}.weight" in torch_sd:
        block = getattr(mlx_model, f"cnn_{block_num}")
        weight = torch_sd[f"cnn_encoder.{idx}.weight"]
        block.conv.update({"weight": mx.array(np.transpose(weight.cpu().numpy(), (0, 2, 1)))})
        if f"cnn_encoder.{idx}.bias" in torch_sd:
            block.conv.update({"bias": torch_to_mx(torch_sd[f"cnn_encoder.{idx}.bias"])})
        bn_idx = idx + 1
        if f"cnn_encoder.{bn_idx}.weight" in torch_sd:
            _convert_layer_norm(
                block.bn,
                torch_sd[f"cnn_encoder.{bn_idx}.weight"],
                torch_sd[f"cnn_encoder.{bn_idx}.bias"],
            )
        block_num += 1
        idx += 4

    bidirectional = any("reverse" in k for k in torch_sd if k.startswith("lstm_layers_stacked"))
    _convert_lstm_stack(mlx_model, torch_sd, "lstm_layers_stacked", "lstm", bidirectional)

    if mlx_model.attention is not None:
        embed_dim = torch_sd["attention.in_proj_weight"].shape[1]
        in_proj = torch_sd["attention.in_proj_weight"].cpu().numpy()
        flat: Dict[str, Any] = {}
        import mlx.core as mx

        for idx, name in enumerate(["query_proj", "key_proj", "value_proj"]):
            flat[f"{name}.weight"] = mx.array(in_proj[idx * embed_dim : (idx + 1) * embed_dim])
        if "attention.in_proj_bias" in torch_sd:
            bias = torch_sd["attention.in_proj_bias"].cpu().numpy()
            flat["query_proj.bias"] = mx.array(bias[:embed_dim])
            flat["key_proj.bias"] = mx.array(bias[embed_dim : 2 * embed_dim])
            flat["value_proj.bias"] = mx.array(bias[2 * embed_dim :])
        flat["out_proj.weight"] = torch_to_mx(torch_sd["attention.out_proj.weight"])
        if "attention.out_proj.bias" in torch_sd:
            flat["out_proj.bias"] = torch_to_mx(torch_sd["attention.out_proj.bias"])
        path = _scratch_npz(flat)
        try:
            mlx_model.attention.load_weights(path)
        finally:
            os.remove(path)

    _convert_linear_module(mlx_model.fc, torch_sd["fc.weight"], torch_sd.get("fc.bias"))


def convert_timesfm_classifier(mlx_head, torch_sd: Dict[str, Any]) -> None:
    """Convert StressTimesFM classifier.* weights into MlxTimesFMClassifier."""
    _convert_layer_norm(mlx_head.norm, torch_sd["classifier.0.weight"], torch_sd["classifier.0.bias"])
    _convert_linear_module(mlx_head.lin1, torch_sd["classifier.1.weight"], torch_sd["classifier.1.bias"])
    _convert_linear_module(mlx_head.lin2, torch_sd["classifier.4.weight"], torch_sd["classifier.4.bias"])
    _convert_linear_module(mlx_head.lin3, torch_sd["classifier.7.weight"], torch_sd["classifier.7.bias"])


def mlx_head_state_to_torch_classifier(mlx_head) -> Dict[str, Any]:
    """Export MlxTimesFMClassifier weights as PyTorch classifier.* keys."""
    import torch

    return {
        "classifier.0.weight": torch.tensor(np.array(mlx_head.norm.weight)),
        "classifier.0.bias": torch.tensor(np.array(mlx_head.norm.bias)),
        "classifier.1.weight": torch.tensor(np.array(mlx_head.lin1.weight)),
        "classifier.1.bias": torch.tensor(np.array(mlx_head.lin1.bias)),
        "classifier.4.weight": torch.tensor(np.array(mlx_head.lin2.weight)),
        "classifier.4.bias": torch.tensor(np.array(mlx_head.lin2.bias)),
        "classifier.7.weight": torch.tensor(np.array(mlx_head.lin3.weight)),
        "classifier.7.bias": torch.tensor(np.array(mlx_head.lin3.bias)),
    }


def apply_mlx_head_to_torch_model(torch_model, mlx_head) -> None:
    """Copy MLX classifier head weights into a StressTimesFM PyTorch model."""
    classifier_sd = mlx_head_state_to_torch_classifier(mlx_head)
    missing, unexpected = torch_model.load_state_dict(classifier_sd, strict=False)
    if unexpected:
        log.warning(
            "Unexpected keys while syncing MLX head to PyTorch (showing 5): %s",
            unexpected[:5],
        )
    if missing:
        log.debug("MLX head sync left %d non-classifier keys unchanged", len(missing))


def convert_checkpoint_to_mlx(
    config: Dict[str, Any],
    checkpoint_path: str,
    output_path: str,
    *,
    input_dim_sequence: int,
    input_dim_static: int,
    atol: float = 0.05,
    validate: bool = True,
) -> Dict[str, Any]:
    """Convert a PyTorch checkpoint into an MLX weight file."""
    model_type = safe_get(config, ["model_config", "type"], "LSTM").upper()
    if model_type == "TIMESFM":
        from mlx_models import MlxTimesFMClassifier

        torch_sd = load_torch_state_dict(checkpoint_path)
        model_config = safe_get(config, ["model_config"], {})
        head_hidden = int(safe_get(model_config, ["timesfm_head_hidden_dim"], 128))
        dropout = float(safe_get(model_config, ["dropout"], 0.2))
        embed_dim = int(safe_get(model_config, ["timesfm_embed_dim"], 1280))
        fc_in = embed_dim + input_dim_static
        mlx_head = MlxTimesFMClassifier(fc_in, head_hidden, dropout)
        convert_timesfm_classifier(mlx_head, torch_sd)
        save_mlx_checkpoint(mlx_head, output_path)
        return {
            "model_type": model_type,
            "output_path": output_path,
            "converted": "classifier_head_only",
            "max_abs_diff": None,
        }

    if not mlx_supports_model(model_type):
        raise ValueError(f"Model type {model_type!r} cannot be converted to MLX.")

    torch_sd = load_torch_state_dict(checkpoint_path)
    mlx_model = get_mlx_model(config, input_dim_sequence, input_dim_static)

    converters = {
        "LSTM": convert_lstm_model,
        "PATCHTST": convert_patchtst_model,
        "PATCH-TST": convert_patchtst_model,
        "TRANSFORMER": convert_transformer_model,
        "CNN-LSTM": convert_cnn_lstm_model,
    }
    converters[model_type](mlx_model, torch_sd)
    save_mlx_checkpoint(mlx_model, output_path)

    report: Dict[str, Any] = {
        "model_type": model_type,
        "output_path": output_path,
        "converted": "full_model",
        "max_abs_diff": None,
    }

    if validate:
        report["max_abs_diff"] = _validate_conversion(
            config,
            checkpoint_path,
            mlx_model,
            input_dim_sequence,
            input_dim_static,
            atol=atol,
        )
    return report


def _validate_conversion(
    config: Dict[str, Any],
    checkpoint_path: str,
    mlx_model,
    input_dim_sequence: int,
    input_dim_static: int,
    atol: float,
) -> float:
    import torch
    from models import get_model

    torch_model = get_model(config, input_dim_sequence, input_dim_static)
    torch_sd = load_torch_state_dict(checkpoint_path)
    torch_model.load_state_dict(torch_sd, strict=False)
    torch_model.eval()

    seq_len = int(safe_get(config, ["model_config", "timesfm_context_len"], 512))
    seq_t = torch.randn(2, seq_len, input_dim_sequence)
    static_t = torch.randn(2, input_dim_static) if input_dim_static > 0 else None

    import mlx.core as mx

    seq_m = mx.array(seq_t.numpy())
    static_m = mx.array(static_t.numpy()) if static_t is not None else None

    with torch.no_grad():
        out_t = torch_model(seq_t, static_t)
    out_m = mlx_model(seq_m, static_m)
    max_diff = float(np.max(np.abs(out_t.numpy() - np.array(out_m))))
    log.info("Conversion validation max_abs_diff=%.6f (atol=%.3f)", max_diff, atol)
    if max_diff > atol:
        log.warning(
            "Converted MLX model diverges from PyTorch reference (%.4f > %.4f). "
            "MLX inference may still be usable but should be re-validated.",
            max_diff,
            atol,
        )
    return max_diff


def ensure_mlx_checkpoint(
    config: Dict[str, Any],
    checkpoint_path: str,
    input_dim_sequence: int,
    input_dim_static: int,
    cache_dir: Optional[str] = None,
) -> str:
    """Return an MLX checkpoint path, converting and caching from PyTorch if needed."""
    if checkpoint_path.endswith(".mlx.npz") or checkpoint_path.endswith("_mlx.npz"):
        return checkpoint_path

    cache_dir = cache_dir or os.path.join(os.path.dirname(checkpoint_path), "mlx_cache")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    cached = os.path.join(cache_dir, f"{base}.mlx.npz")
    if os.path.exists(cached) and os.path.getmtime(cached) >= os.path.getmtime(checkpoint_path):
        return cached

    log.info("Converting PyTorch checkpoint to MLX cache: %s -> %s", checkpoint_path, cached)
    convert_checkpoint_to_mlx(
        config,
        checkpoint_path,
        cached,
        input_dim_sequence=input_dim_sequence,
        input_dim_static=input_dim_static,
        validate=True,
    )
    return cached
