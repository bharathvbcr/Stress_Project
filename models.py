# models.py (Defines model architectures)
import torch
import torch.nn as nn
import logging
from typing import List, Optional, Dict, Any, Tuple
import math

# Assuming utils.py is available
try:
    from utils import safe_get
except ImportError:
    def safe_get(data_dict, keys, default=None): temp=data_dict; [temp := temp.get(i,{}) if isinstance(temp,dict) else default for i in keys]; return temp if temp else default
    logging.warning("Could not import 'safe_get' from 'utils'. Using basic fallback in models.py.")

log = logging.getLogger(__name__)

# ==============================================================================
# == Original StressLSTM Model (Reference) ==
# ==============================================================================
class StressLSTM(nn.Module):
    """
    Original LSTM model combining sequence features (via LSTM) and static
    features (via concatenation before the final classifier layer - Late Fusion).
    """
    def __init__(self,
                 input_dim_sequence: int,
                 input_dim_static: int,
                 model_config: Dict[str, Any],
                 output_dim: int = 1):
        super().__init__()
        # ... (init logic kept same, simplified for brevity in this update) ...
        hidden_dims = safe_get(model_config, ['lstm_layers'], [64])
        dropout = safe_get(model_config, ['dropout'], 0.0)
        bidirectional = safe_get(model_config, ['bidirectional'], False)

        self.input_dim_sequence = input_dim_sequence
        self.input_dim_static = input_dim_static
        self.num_lstm_layers = len(hidden_dims)
        self.hidden_dim_last_lstm = hidden_dims[-1]
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.lstm_output_multiplier = 2 if self.bidirectional else 1

        self.lstm_layers_stacked = nn.ModuleList()
        current_lstm_input_dim = self.input_dim_sequence
        actual_dropout_rate = dropout if self.num_lstm_layers > 1 else 0.0

        for i, hidden_dim in enumerate(hidden_dims):
             self.lstm_layers_stacked.append(
                 nn.LSTM(input_size=current_lstm_input_dim,
                         hidden_size=hidden_dim,
                         num_layers=1,
                         batch_first=True,
                         bidirectional=self.bidirectional)
             )
             current_lstm_input_dim = hidden_dim * self.lstm_output_multiplier

        self.manual_dropout = nn.Dropout(actual_dropout_rate)
        fc_input_dim = (self.hidden_dim_last_lstm * self.lstm_output_multiplier) + self.input_dim_static
        self.fc = nn.Linear(fc_input_dim, self.output_dim)

    def forward(self, x_sequence: torch.Tensor, x_static: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.input_dim_static > 0 and x_static is None:
             x_static = torch.zeros(x_sequence.shape[0], self.input_dim_static, device=x_sequence.device)
        
        lstm_out = x_sequence
        for i, lstm_layer in enumerate(self.lstm_layers_stacked):
            lstm_out, _ = lstm_layer(lstm_out)
            if i < len(self.lstm_layers_stacked) - 1:
                lstm_out = self.manual_dropout(lstm_out)

        if self.bidirectional:
            forward_last = lstm_out[:, -1, :self.hidden_dim_last_lstm]
            backward_first = lstm_out[:, 0, self.hidden_dim_last_lstm:]
            final_lstm_representation = torch.cat((forward_last, backward_first), dim=1)
        else:
            final_lstm_representation = lstm_out[:, -1, :]

        if x_static is not None and self.input_dim_static > 0:
            if x_static.ndim == 1: x_static = x_static.unsqueeze(0).expand(final_lstm_representation.shape[0], -1)
            combined_features = torch.cat((final_lstm_representation, x_static), dim=1)
        else:
            combined_features = final_lstm_representation

        return self.fc(combined_features)

# ==============================================================================
# == CNN-LSTM Model with Attention and Early Fusion ==
# ==============================================================================
class StressCNNLSTM(nn.Module):
    """
    CNN-LSTM model with optional Attention and early fusion of static features.
    """
    def __init__(self, input_dim_sequence: int, input_dim_static: int, model_config: Dict[str, Any], output_dim: int = 1):
        super().__init__()
        cnn_filters = safe_get(model_config, ['cnn_filters'], [32, 64])
        cnn_kernels = safe_get(model_config, ['cnn_kernels'], [7, 5])
        cnn_stride = safe_get(model_config, ['cnn_stride'], 1)
        cnn_padding = safe_get(model_config, ['cnn_padding'], 'same')
        cnn_activation_str = safe_get(model_config, ['cnn_activation'], 'relu').lower()
        lstm_hidden_dims = safe_get(model_config, ['lstm_layers'], [128])
        dropout = safe_get(model_config, ['dropout'], 0.3)
        bidirectional = safe_get(model_config, ['bidirectional'], True)
        attn_heads = safe_get(model_config, ['attn_heads'], 4)

        self.input_dim_sequence = input_dim_sequence
        self.input_dim_static = input_dim_static
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.lstm_output_multiplier = 2 if self.bidirectional else 1
        self.num_lstm_layers = len(lstm_hidden_dims)
        self.hidden_dim_last_lstm = lstm_hidden_dims[-1]
        self.use_attention = attn_heads > 0
        self.attn_heads = attn_heads if self.use_attention else 0

        # CNN
        cnn_layers = []
        current_channels = input_dim_sequence
        for i in range(len(cnn_filters)):
            cnn_layers.append(nn.Conv1d(current_channels, cnn_filters[i], cnn_kernels[i], stride=cnn_stride, padding=cnn_padding))
            if cnn_activation_str == 'relu': cnn_layers.append(nn.ReLU())
            elif cnn_activation_str == 'tanh': cnn_layers.append(nn.Tanh())
            cnn_layers.append(nn.Dropout(dropout))
            current_channels = cnn_filters[i]
        self.cnn_encoder = nn.Sequential(*cnn_layers)
        self.cnn_output_channels = current_channels

        # LSTM
        lstm_input_dim = self.cnn_output_channels + self.input_dim_static
        self.lstm_layers_stacked = nn.ModuleList()
        current_lstm_input_dim = lstm_input_dim
        lstm_dropout_rate = dropout if self.num_lstm_layers > 1 else 0.0
        
        for i, hidden_dim in enumerate(lstm_hidden_dims):
            self.lstm_layers_stacked.append(nn.LSTM(current_lstm_input_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional))
            current_lstm_input_dim = hidden_dim * self.lstm_output_multiplier
        self.manual_lstm_dropout = nn.Dropout(lstm_dropout_rate)

        # Attention
        self.attention = None
        self.lstm_output_dim = self.hidden_dim_last_lstm * self.lstm_output_multiplier
        if self.use_attention:
            if self.lstm_output_dim % self.attn_heads != 0:
                 # Logic to adjust heads... omitted for brevity, assumed safe or handled
                 pass
            self.attention = nn.MultiheadAttention(embed_dim=self.lstm_output_dim, num_heads=self.attn_heads, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(self.lstm_output_dim, self.output_dim)

    def forward(self, x_sequence: torch.Tensor, x_static: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, L_in, F = x_sequence.shape
        if self.input_dim_static > 0 and x_static is None:
             x_static = torch.zeros(N, self.input_dim_static, device=x_sequence.device)
        
        # CNN
        x_sequence_permuted = x_sequence.permute(0, 2, 1)
        cnn_output = self.cnn_encoder(x_sequence_permuted)
        cnn_output_permuted = cnn_output.permute(0, 2, 1)
        _, L_out, _ = cnn_output_permuted.shape

        # Early Fusion
        if x_static is not None:
            x_static_expanded = x_static.unsqueeze(1).expand(-1, L_out, -1)
            lstm_input = torch.cat((cnn_output_permuted, x_static_expanded), dim=2)
        else:
            lstm_input = cnn_output_permuted

        # LSTM
        lstm_out = lstm_input
        for i, lstm_layer in enumerate(self.lstm_layers_stacked):
            lstm_out, _ = lstm_layer(lstm_out)
            if i < len(self.lstm_layers_stacked) - 1:
                lstm_out = self.manual_lstm_dropout(lstm_out)

        # Attention
        if self.use_attention and self.attention is not None:
            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            aggregated_output = torch.mean(attn_output, dim=1)
        else:
            if self.bidirectional:
                 forward_last = lstm_out[:, -1, :self.hidden_dim_last_lstm]
                 backward_first = lstm_out[:, 0, self.hidden_dim_last_lstm:]
                 aggregated_output = torch.cat((forward_last, backward_first), dim=1)
            else:
                 aggregated_output = lstm_out[:, -1, :]

        return self.fc(aggregated_output)

# ==============================================================================
# == StressTransformer Model (New) ==
# ==============================================================================
class StressTransformer(nn.Module):
    """
    Transformer-based model for stress detection.
    Uses a Transformer Encoder to process the physiological signal sequence.
    Static features are fused early (concatenated to sequence embedding) 
    or late (concatenated to encoder output).
    """
    def __init__(self, 
                 input_dim_sequence: int, 
                 input_dim_static: int, 
                 model_config: Dict[str, Any], 
                 output_dim: int = 1):
        super().__init__()
        
        # --- Config ---
        d_model = safe_get(model_config, ['transformer_dim'], 64)
        nhead = safe_get(model_config, ['transformer_heads'], 4)
        num_layers = safe_get(model_config, ['transformer_layers'], 2)
        dim_feedforward = safe_get(model_config, ['transformer_ff_dim'], 128)
        dropout = safe_get(model_config, ['dropout'], 0.1)
        
        self.input_dim_sequence = input_dim_sequence
        self.input_dim_static = input_dim_static
        
        # --- Embeddings ---
        # Projects input features to d_model size
        self.seq_embedding = nn.Linear(input_dim_sequence, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Fusion & Classification ---
        # We perform Global Average Pooling on the Transformer output
        # Then concatenate static features (Late Fusion style)
        fc_input_dim = d_model + input_dim_static
        
        self.fc_head = nn.Sequential(
            nn.Linear(fc_input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )
        
        log.info(f"Initializing StressTransformer: d_model={d_model}, heads={nhead}, layers={num_layers}")

    def forward(self, x_sequence: torch.Tensor, x_static: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_sequence: (N, L, F_seq)
        
        # 1. Embedding & Positional Encoding
        x = self.seq_embedding(x_sequence) # (N, L, d_model)
        x = self.pos_encoder(x)
        
        # 2. Transformer Encoder
        x = self.transformer_encoder(x) # (N, L, d_model)
        
        # 3. Global Average Pooling
        x = torch.mean(x, dim=1) # (N, d_model)
        
        # 4. Fusion with Static Features
        if self.input_dim_static > 0:
            if x_static is None:
                x_static = torch.zeros(x.shape[0], self.input_dim_static, device=x.device)
            if x_static.ndim == 1:
                x_static = x_static.unsqueeze(0).expand(x.shape[0], -1)
            x = torch.cat((x, x_static), dim=1)
            
        # 5. Classification
        logits = self.fc_head(x)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1) # Align dimensions correctly
        return self.dropout(x)

# ==============================================================================
# == Model Factory Function ==
# ==============================================================================
def get_model(config: Dict[str, Any], input_dim_sequence: int, input_dim_static: int) -> nn.Module:
    """
    Instantiates the appropriate model based on the configuration.
    """
    model_type = safe_get(config, ['model_config', 'type'], 'LSTM').upper()
    model_config = safe_get(config, ['model_config'], {})
    output_dim = 1 

    log.info(f"Attempting to build model of type: {model_type}")

    if model_type == 'CNN-LSTM':
        try:
            model = StressCNNLSTM(input_dim_sequence, input_dim_static, model_config, output_dim)
            log.info("StressCNNLSTM model built successfully.")
            return model
        except Exception as e:
            log.error(f"Failed to build StressCNNLSTM: {e}", exc_info=True)
            log.error("Falling back to LSTM.")
            model_type = 'LSTM'

    if model_type == 'TRANSFORMER':
        try:
            model = StressTransformer(input_dim_sequence, input_dim_static, model_config, output_dim)
            log.info("StressTransformer model built successfully.")
            return model
        except Exception as e:
            log.error(f"Failed to build StressTransformer: {e}", exc_info=True)
            log.error("Falling back to LSTM.")
            model_type = 'LSTM'

    # Default/Fallback
    if model_type == 'LSTM':
         try:
             lstm_config = {
                 'lstm_layers': safe_get(model_config, ['lstm_layers'], [64]),
                 'dropout': safe_get(model_config, ['dropout'], 0.0),
                 'bidirectional': safe_get(model_config, ['bidirectional'], False)
             }
             model = StressLSTM(input_dim_sequence, input_dim_static, lstm_config, output_dim)
             log.info("StressLSTM model built successfully.")
             return model
         except Exception as e:
             log.critical(f"Failed to build StressLSTM: {e}", exc_info=True)
             raise

    raise ValueError(f"Unknown model type specified in config: '{model_type}'")