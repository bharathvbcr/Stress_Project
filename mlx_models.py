# mlx_models.py — MLX-native stress-detection models for Apple Silicon
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

from mlx_backend import mlx_supports_model, resolve_mlx_dtype
from utils import safe_get

log = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn

    _MLX_OK = True
except ImportError:
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_OK = False


def _require_mlx():
    if not _MLX_OK:
        raise ImportError("MLX is required for mlx_models. Install with: pip install mlx")
    return mx, nn


def _gelu(x):
    return nn.GELU()(x)


def _extract_patches(x, patch_len: int, patch_stride: int):
    """x: (N, F, L) -> (N, F, P, patch_len) as a zero-copy strided view.

    Replaces a Python slice-and-stack loop (which materializes and copies every
    patch) with a single ``as_strided`` view. ``as_strided`` reads the raw
    buffer, so the input is made contiguous first (the caller passes a
    transposed, non-contiguous view).
    """
    n, f, length = x.shape
    if length < patch_len:
        pad = patch_len - length
        x = mx.pad(x, [(0, 0), (0, 0), (pad, 0)])
        length = patch_len

    num_patches = (length - patch_len) // patch_stride + 1
    x = mx.contiguous(x)
    return mx.as_strided(
        x,
        shape=(n, f, num_patches, patch_len),
        strides=(f * length, length, patch_stride, 1),
    )


def _resolve_conv_padding(kernel: int, stride: int, padding) -> int:
    if isinstance(padding, str) and str(padding).lower() == "same":
        return kernel // 2
    try:
        return max(0, int(padding))
    except (TypeError, ValueError):
        return 0


def _run_bidirectional_lstm(fwd_layer, bwd_layer, x):
    fwd_out, _ = fwd_layer(x)
    rev = x[:, ::-1, :]
    bwd_out, _ = bwd_layer(rev)
    bwd_out = bwd_out[:, ::-1, :]
    return mx.concatenate([fwd_out, bwd_out], axis=-1)


if _MLX_OK:

    class MlxTransformerEncoderLayer(nn.Module):
        """Transformer encoder layer with attention biases (matches PyTorch)."""

        def __init__(
            self,
            dims: int,
            num_heads: int,
            mlp_dims: Optional[int] = None,
            dropout: float = 0.0,
            activation=_gelu,
            norm_first: bool = True,
        ):
            super().__init__()
            mlp_dims = mlp_dims or dims * 4
            self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
            self.ln1 = nn.LayerNorm(dims)
            self.ln2 = nn.LayerNorm(dims)
            self.linear1 = nn.Linear(dims, mlp_dims)
            self.linear2 = nn.Linear(mlp_dims, dims)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = activation
            self.norm_first = norm_first

        def __call__(self, x, mask):
            if self.norm_first:
                y = self.ln1(x)
                y = self.attention(y, y, y, mask)
                y = self.dropout1(y)
                x = x + y

                y = self.ln2(x)
                y = self.linear1(y)
                y = self.activation(y)
                y = self.dropout2(y)
                y = self.linear2(y)
                return x + y

            y = self.attention(x, x, x, mask)
            y = self.dropout1(y)
            x = self.ln1(x + y)

            y = self.linear1(x)
            y = self.activation(y)
            y = self.dropout2(y)
            y = self.linear2(y)
            return self.ln2(x + y)

    class MlxTransformerEncoder(nn.Module):
        """PyTorch-compatible encoder stack (no trailing LayerNorm)."""

        def __init__(
            self,
            num_layers: int,
            dims: int,
            num_heads: int,
            mlp_dims: Optional[int] = None,
            dropout: float = 0.0,
            activation=_gelu,
            norm_first: bool = True,
        ):
            super().__init__()
            self.layers = [
                MlxTransformerEncoderLayer(
                    dims,
                    num_heads,
                    mlp_dims,
                    dropout,
                    activation,
                    norm_first,
                )
                for _ in range(num_layers)
            ]

        def __call__(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
            return x

    class MlxPositionalEncoding(nn.Module):
        def __init__(self, d_model: int, dropout: float, max_len: int):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            position = mx.arange(max_len)[:, None]
            div_term = mx.exp(mx.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = mx.concatenate([mx.sin(position * div_term), mx.cos(position * div_term)], axis=1)
            if pe.shape[1] < d_model:
                pe = mx.pad(pe, [(0, 0), (0, d_model - pe.shape[1])])
            elif pe.shape[1] > d_model:
                pe = pe[:, :d_model]
            self.pe = pe

        def __call__(self, x):
            seq_len = x.shape[1]
            return self.dropout(x + self.pe[:seq_len][None, :, :])

    class MlxConvBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding,
            activation: str,
            dropout: float,
        ):
            super().__init__()
            pad = _resolve_conv_padding(kernel_size, stride, padding)
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
            )
            self.bn = nn.BatchNorm(out_channels)
            self.act = nn.Tanh() if activation == "tanh" else nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def __call__(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
            return self.dropout(x)

    class MlxStressLSTM(nn.Module):
        """MLX LSTM classifier mirroring models.StressLSTM (late fusion)."""

        def __init__(
            self,
            input_dim_sequence: int,
            input_dim_static: int,
            model_config: Dict[str, Any],
            output_dim: int = 1,
        ):
            super().__init__()
            hidden_dims = safe_get(model_config, ["lstm_layers"], [64])
            dropout = float(safe_get(model_config, ["dropout"], 0.0))
            bidirectional = bool(safe_get(model_config, ["bidirectional"], False))

            self.input_dim_sequence = input_dim_sequence
            self.input_dim_static = input_dim_static
            self.bidirectional = bidirectional
            self.hidden_dim_last = hidden_dims[-1]

            current_dim = input_dim_sequence
            for idx, hidden_dim in enumerate(hidden_dims):
                if bidirectional:
                    setattr(self, f"lstm_{idx}_fwd", nn.LSTM(current_dim, hidden_dim))
                    setattr(self, f"lstm_{idx}_bwd", nn.LSTM(current_dim, hidden_dim))
                    current_dim = hidden_dim * 2
                else:
                    setattr(self, f"lstm_{idx}", nn.LSTM(current_dim, hidden_dim))
                    current_dim = hidden_dim
            self._num_lstm_layers = len(hidden_dims)

            self.manual_dropout = (
                nn.Dropout(dropout) if len(hidden_dims) > 1 and dropout > 0 else None
            )
            fc_input = current_dim + input_dim_static
            self.fc = nn.Linear(fc_input, output_dim)

        def __call__(self, x_sequence, x_static=None):
            batch = x_sequence.shape[0]
            if self.input_dim_static > 0 and x_static is None:
                x_static = mx.zeros((batch, self.input_dim_static))

            lstm_out = x_sequence
            for idx in range(self._num_lstm_layers):
                if self.bidirectional:
                    lstm_out = _run_bidirectional_lstm(
                        getattr(self, f"lstm_{idx}_fwd"),
                        getattr(self, f"lstm_{idx}_bwd"),
                        lstm_out,
                    )
                else:
                    lstm_out, _ = getattr(self, f"lstm_{idx}")(lstm_out)
                if self.manual_dropout is not None and idx < self._num_lstm_layers - 1:
                    lstm_out = self.manual_dropout(lstm_out)

            if self.bidirectional:
                forward_last = lstm_out[:, -1, : self.hidden_dim_last]
                backward_first = lstm_out[:, 0, self.hidden_dim_last :]
                final_repr = mx.concatenate([forward_last, backward_first], axis=1)
            else:
                final_repr = lstm_out[:, -1, :]
            if self.input_dim_static > 0:
                combined = mx.concatenate([final_repr, x_static], axis=1)
            else:
                combined = final_repr
            return self.fc(combined)

    class MlxStressPatchTST(nn.Module):
        """MLX PatchTST classifier mirroring models.StressPatchTST."""

        def __init__(
            self,
            input_dim_sequence: int,
            input_dim_static: int,
            model_config: Dict[str, Any],
            output_dim: int = 1,
        ):
            super().__init__()
            patch_len = int(safe_get(model_config, ["patch_len"], 16))
            patch_stride = int(safe_get(model_config, ["patch_stride"], max(1, patch_len // 2)))
            d_model = int(safe_get(model_config, ["patchtst_dim"], 128))
            nhead = int(safe_get(model_config, ["patchtst_heads"], 4))
            num_layers = int(safe_get(model_config, ["patchtst_layers"], 3))
            dim_feedforward = int(safe_get(model_config, ["patchtst_ff_dim"], d_model * 4))
            dropout = float(safe_get(model_config, ["dropout"], 0.1))
            max_tokens = int(safe_get(model_config, ["patchtst_max_tokens"], 8192))

            if d_model % nhead != 0:
                valid_heads = [h for h in range(min(nhead, d_model), 0, -1) if d_model % h == 0]
                nhead = valid_heads[0] if valid_heads else 1
                log.warning("Adjusted patchtst_heads to %s for MLX PatchTST.", nhead)

            self.input_dim_sequence = input_dim_sequence
            self.input_dim_static = input_dim_static
            self.patch_len = patch_len
            self.patch_stride = patch_stride

            self.patch_projection = nn.Linear(patch_len, d_model)
            self.pos_encoder = MlxPositionalEncoding(d_model, dropout, max_tokens)
            self.transformer_encoder = MlxTransformerEncoder(
                num_layers,
                d_model,
                nhead,
                mlp_dims=dim_feedforward,
                dropout=dropout,
                activation=_gelu,
                norm_first=True,
            )
            self.norm = nn.LayerNorm(d_model)
            fc_input = d_model + input_dim_static
            self.fc_head = nn.Sequential(
                nn.Linear(fc_input, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, output_dim),
            )

        def __call__(self, x_sequence, x_static=None):
            batch, _, _ = x_sequence.shape
            x = mx.transpose(x_sequence, (0, 2, 1))
            patches = _extract_patches(x, self.patch_len, self.patch_stride)
            _, channels, patch_count, _ = patches.shape
            tokens = self.patch_projection(patches)
            tokens = mx.reshape(tokens, (batch, channels * patch_count, -1))
            tokens = self.pos_encoder(tokens)
            encoded = self.transformer_encoder(tokens, None)
            pooled = mx.mean(self.norm(encoded), axis=1)

            if self.input_dim_static > 0:
                if x_static is None:
                    x_static = mx.zeros((batch, self.input_dim_static))
                pooled = mx.concatenate([pooled, x_static], axis=1)
            return self.fc_head(pooled)

    class MlxStressTransformer(nn.Module):
        """MLX Transformer classifier mirroring models.StressTransformer."""

        def __init__(
            self,
            input_dim_sequence: int,
            input_dim_static: int,
            model_config: Dict[str, Any],
            output_dim: int = 1,
        ):
            super().__init__()
            d_model = int(safe_get(model_config, ["transformer_dim"], 64))
            nhead = int(safe_get(model_config, ["transformer_heads"], 4))
            num_layers = int(safe_get(model_config, ["transformer_layers"], 2))
            dim_feedforward = int(safe_get(model_config, ["transformer_ff_dim"], 128))
            dropout = float(safe_get(model_config, ["dropout"], 0.1))

            if d_model % nhead != 0:
                valid_heads = [h for h in range(min(nhead, d_model), 0, -1) if d_model % h == 0]
                nhead = valid_heads[0] if valid_heads else 1
                log.warning("Adjusted transformer_heads to %s for MLX Transformer.", nhead)

            self.input_dim_sequence = input_dim_sequence
            self.input_dim_static = input_dim_static

            self.seq_embedding = nn.Linear(input_dim_sequence, d_model)
            self.pos_encoder = MlxPositionalEncoding(d_model, dropout, max_len=5000)
            self.transformer_encoder = MlxTransformerEncoder(
                num_layers,
                d_model,
                nhead,
                mlp_dims=dim_feedforward,
                dropout=dropout,
                activation=_gelu,
                norm_first=False,
            )
            fc_input = d_model + input_dim_static
            self.fc_head = nn.Sequential(
                nn.Linear(fc_input, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, output_dim),
            )

        def __call__(self, x_sequence, x_static=None):
            batch = x_sequence.shape[0]
            x = self.seq_embedding(x_sequence)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x, None)
            x = mx.mean(x, axis=1)

            if self.input_dim_static > 0:
                if x_static is None:
                    x_static = mx.zeros((batch, self.input_dim_static))
                x = mx.concatenate([x, x_static], axis=1)
            return self.fc_head(x)

    class MlxStressCNNLSTM(nn.Module):
        """MLX CNN-LSTM classifier mirroring models.StressCNNLSTM."""

        def __init__(
            self,
            input_dim_sequence: int,
            input_dim_static: int,
            model_config: Dict[str, Any],
            output_dim: int = 1,
        ):
            super().__init__()
            cnn_filters = safe_get(model_config, ["cnn_filters"], [32, 64])
            cnn_kernels = safe_get(model_config, ["cnn_kernels"], [7, 5])
            cnn_stride = int(safe_get(model_config, ["cnn_stride"], 1))
            cnn_padding = safe_get(model_config, ["cnn_padding"], "same")
            cnn_activation = str(safe_get(model_config, ["cnn_activation"], "relu")).lower()
            lstm_hidden_dims = safe_get(model_config, ["lstm_layers"], [128])
            dropout = float(safe_get(model_config, ["dropout"], 0.3))
            bidirectional = bool(safe_get(model_config, ["bidirectional"], True))
            attn_heads = int(safe_get(model_config, ["attn_heads"], 4))

            self.input_dim_static = input_dim_static
            self.bidirectional = bidirectional
            self.hidden_dim_last_lstm = lstm_hidden_dims[-1]
            self._num_lstm_layers = len(lstm_hidden_dims)
            self.use_attention = attn_heads > 0

            for idx, (filters, kernel) in enumerate(zip(cnn_filters, cnn_kernels)):
                in_ch = input_dim_sequence if idx == 0 else cnn_filters[idx - 1]
                setattr(
                    self,
                    f"cnn_{idx}",
                    MlxConvBlock(
                        in_ch,
                        filters,
                        int(kernel),
                        cnn_stride,
                        cnn_padding,
                        cnn_activation,
                        dropout,
                    ),
                )
            self._num_cnn_layers = len(cnn_filters)
            self.cnn_output_channels = cnn_filters[-1]

            lstm_input_dim = self.cnn_output_channels + input_dim_static
            current_lstm_input = lstm_input_dim
            for idx, hidden_dim in enumerate(lstm_hidden_dims):
                if bidirectional:
                    setattr(self, f"lstm_{idx}_fwd", nn.LSTM(current_lstm_input, hidden_dim))
                    setattr(self, f"lstm_{idx}_bwd", nn.LSTM(current_lstm_input, hidden_dim))
                    current_lstm_input = hidden_dim * 2
                else:
                    setattr(self, f"lstm_{idx}", nn.LSTM(current_lstm_input, hidden_dim))
                    current_lstm_input = hidden_dim

            self.manual_lstm_dropout = (
                nn.Dropout(dropout) if len(lstm_hidden_dims) > 1 and dropout > 0 else None
            )
            self.lstm_output_dim = self.hidden_dim_last_lstm * (2 if bidirectional else 1)
            if self.use_attention:
                if self.lstm_output_dim % attn_heads != 0:
                    attn_heads = next(
                        (h for h in range(attn_heads, 0, -1) if self.lstm_output_dim % h == 0),
                        1,
                    )
                    log.warning(
                        "attn_heads adjusted to %s for MLX CNN-LSTM (dim=%s).",
                        attn_heads,
                        self.lstm_output_dim,
                    )
                self.attention = nn.MultiHeadAttention(self.lstm_output_dim, attn_heads, bias=True)
            else:
                self.attention = None

            self.fc = nn.Linear(self.lstm_output_dim, output_dim)

        def _encode_cnn(self, x_sequence):
            x = x_sequence
            for idx in range(self._num_cnn_layers):
                x = getattr(self, f"cnn_{idx}")(x)
            return x

        def _run_lstm_stack(self, lstm_input):
            lstm_out = lstm_input
            for idx in range(self._num_lstm_layers):
                if self.bidirectional:
                    lstm_out = _run_bidirectional_lstm(
                        getattr(self, f"lstm_{idx}_fwd"),
                        getattr(self, f"lstm_{idx}_bwd"),
                        lstm_out,
                    )
                else:
                    lstm_out, _ = getattr(self, f"lstm_{idx}")(lstm_out)
                if self.manual_lstm_dropout is not None and idx < self._num_lstm_layers - 1:
                    lstm_out = self.manual_lstm_dropout(lstm_out)
            return lstm_out

        def __call__(self, x_sequence, x_static=None):
            batch, _, _ = x_sequence.shape
            if self.input_dim_static > 0 and x_static is None:
                x_static = mx.zeros((batch, self.input_dim_static))

            cnn_output = self._encode_cnn(x_sequence)
            seq_len = cnn_output.shape[1]

            if self.input_dim_static > 0:
                static_expanded = mx.broadcast_to(
                    x_static[:, None, :],
                    (batch, seq_len, self.input_dim_static),
                )
                lstm_input = mx.concatenate([cnn_output, static_expanded], axis=-1)
            else:
                lstm_input = cnn_output

            lstm_out = self._run_lstm_stack(lstm_input)

            if self.use_attention and self.attention is not None:
                attn_output = self.attention(lstm_out, lstm_out, lstm_out)
                aggregated = mx.mean(attn_output, axis=1)
            elif self.bidirectional:
                forward_last = lstm_out[:, -1, : self.hidden_dim_last_lstm]
                backward_first = lstm_out[:, 0, self.hidden_dim_last_lstm :]
                aggregated = mx.concatenate([forward_last, backward_first], axis=1)
            else:
                aggregated = lstm_out[:, -1, :]

            return self.fc(aggregated)

    class MlxTimesFMClassifier(nn.Module):
        """MLX classification head for StressTimesFM (backbone stays in PyTorch)."""

        def __init__(self, fc_in: int, head_hidden_dim: int, dropout: float, output_dim: int = 1):
            super().__init__()
            self.norm = nn.LayerNorm(fc_in)
            self.lin1 = nn.Linear(fc_in, head_hidden_dim)
            self.lin2 = nn.Linear(head_hidden_dim, head_hidden_dim // 2)
            self.lin3 = nn.Linear(head_hidden_dim // 2, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.gelu = nn.GELU()

        def __call__(self, fused_features):
            x = self.norm(fused_features)
            x = self.gelu(self.lin1(x))
            x = self.dropout(x)
            x = self.gelu(self.lin2(x))
            x = self.dropout(x)
            return self.lin3(x)


def get_mlx_model(
    config: Dict[str, Any],
    input_dim_sequence: int,
    input_dim_static: int,
):
    _require_mlx()
    model_type = safe_get(config, ["model_config", "type"], "LSTM").upper()
    model_config = safe_get(config, ["model_config"], {})
    if not mlx_supports_model(model_type):
        raise ValueError(f"Model type {model_type!r} is not supported by the MLX backend.")

    if model_type in {"PATCHTST", "PATCH-TST"}:
        return MlxStressPatchTST(input_dim_sequence, input_dim_static, model_config)
    if model_type == "TRANSFORMER":
        return MlxStressTransformer(input_dim_sequence, input_dim_static, model_config)
    if model_type == "CNN-LSTM":
        return MlxStressCNNLSTM(input_dim_sequence, input_dim_static, model_config)
    if model_type == "LSTM":
        lstm_config = {
            "lstm_layers": safe_get(model_config, ["lstm_layers"], [64]),
            "dropout": safe_get(model_config, ["dropout"], 0.0),
            "bidirectional": safe_get(model_config, ["bidirectional"], False),
        }
        return MlxStressLSTM(input_dim_sequence, input_dim_static, lstm_config)
    raise ValueError(f"Unsupported MLX model type: {model_type}")


def save_mlx_checkpoint(model, path: str) -> None:
    _require_mlx()
    model.save_weights(path)


def load_mlx_checkpoint(model, path: str) -> None:
    _require_mlx()
    model.load_weights(path)


def batch_to_mlx(batch: Dict[str, Any]) -> Tuple[Any, Optional[Any], Any]:
    """Convert a HuggingFace/pyarrow batch dict to MLX arrays.

    Features are cast to the compute dtype from ``STRESS_MLX_DTYPE`` (float32
    default, bfloat16/float16 opt-in); labels always stay float32 so the loss
    is computed at full precision.
    """
    _require_mlx()
    dtype = resolve_mlx_dtype()
    seq = mx.array(batch["sequence"], dtype=dtype)
    labels = mx.array(batch["label"], dtype=mx.float32).reshape(-1, 1)
    static = None
    if "static" in batch and batch["static"] is not None:
        static = mx.array(batch["static"], dtype=dtype)
    return seq, static, labels
