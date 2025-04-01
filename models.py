# models.py (Defines model architectures)
import torch
import torch.nn as nn
import logging
from typing import List, Optional, Dict, Any, Tuple

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
        """
        Initializes the StressLSTM model.

        Args:
            input_dim_sequence (int): Number of features per time step in sequence data.
            input_dim_static (int): Number of static features.
            model_config (Dict[str, Any]): Dictionary containing model hyperparameters.
                                           Expected keys: 'lstm_layers', 'dropout', 'bidirectional'.
            output_dim (int, optional): Number of output units (1 for binary classification). Defaults to 1.

        Raises:
            ValueError: If input dimensions or config parameters are invalid.
        """
        super().__init__()
        # --- Get parameters from model_config using safe_get for robustness ---
        hidden_dims = safe_get(model_config, ['lstm_layers'], [64]) # Default: one layer with 64 units
        dropout = safe_get(model_config, ['dropout'], 0.0) # Default: no dropout
        bidirectional = safe_get(model_config, ['bidirectional'], False) # Default: unidirectional

        # --- Input Validation ---
        if not isinstance(input_dim_sequence, int) or input_dim_sequence <= 0:
            raise ValueError("input_dim_sequence must be a positive integer.")
        if not isinstance(input_dim_static, int) or input_dim_static < 0:
            raise ValueError("input_dim_static must be a non-negative integer.")
        if not hidden_dims or not isinstance(hidden_dims, list) or not all(isinstance(h, int) and h > 0 for h in hidden_dims):
            raise ValueError("model_config['lstm_layers'] must be a non-empty list of positive integers.")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if not isinstance(dropout, (float, int)) or not (0.0 <= dropout < 1.0):
            raise ValueError("model_config['dropout'] must be a float between 0.0 and 1.0.")
        if not isinstance(bidirectional, bool):
            raise ValueError("model_config['bidirectional'] must be a boolean.")

        # --- Store dimensions and config ---
        self.input_dim_sequence = input_dim_sequence
        self.input_dim_static = input_dim_static
        self.num_lstm_layers = len(hidden_dims)
        self.hidden_dim_last_lstm = hidden_dims[-1] # Hidden dim of the final LSTM layer
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        # LSTM output dimension multiplier (2 if bidirectional, 1 otherwise)
        self.lstm_output_multiplier = 2 if self.bidirectional else 1

        log.info(f"Initializing StressLSTM model:")
        log.info(f"  Sequence Input Dim: {self.input_dim_sequence}")
        log.info(f"  Static Input Dim: {self.input_dim_static}")
        log.info(f"  LSTM Hidden Dims: {hidden_dims}")
        log.info(f"  Num LSTM Layers: {self.num_lstm_layers}")
        log.info(f"  Bidirectional: {self.bidirectional}")
        log.info(f"  Dropout: {dropout}")
        log.info(f"  Output Dim (Classifier): {self.output_dim}")

        # --- LSTM Layers ---
        # Create a ModuleList to hold the LSTM layers
        self.lstm_layers_stacked = nn.ModuleList()
        current_lstm_input_dim = self.input_dim_sequence # Input dim for the first LSTM layer
        # Apply dropout between LSTM layers only if there's more than one layer
        actual_dropout_rate = dropout if self.num_lstm_layers > 1 else 0.0
        log.info(f"  Manual Dropout Rate (between LSTM layers): {actual_dropout_rate}")

        for i, hidden_dim in enumerate(hidden_dims):
             self.lstm_layers_stacked.append(
                 nn.LSTM(input_size=current_lstm_input_dim,
                         hidden_size=hidden_dim,
                         num_layers=1, # Each element in hidden_dims defines one layer here
                         batch_first=True, # Input tensor shape: (batch, seq_len, features)
                         bidirectional=self.bidirectional)
             )
             # Update input dimension for the next LSTM layer
             current_lstm_input_dim = hidden_dim * self.lstm_output_multiplier

        # Manual dropout layer applied between LSTM layers during forward pass
        self.manual_dropout = nn.Dropout(actual_dropout_rate)

        # --- Final Fully Connected Layer ---
        # Input dimension is the output dim of the last LSTM + static features dim
        fc_input_dim = (self.hidden_dim_last_lstm * self.lstm_output_multiplier) + self.input_dim_static
        self.fc = nn.Linear(fc_input_dim, self.output_dim)
        log.info(f"  FC Layer Input Dim (LSTM Output + Static): {fc_input_dim}")
        log.info(f"StressLSTM model initialized successfully.")

    def forward(self, x_sequence: torch.Tensor, x_static: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining LSTM output with static features (Late Fusion).

        Args:
            x_sequence (torch.Tensor): Input sequence tensor (batch, seq_len, input_dim_sequence).
            x_static (Optional[torch.Tensor]): Input static features tensor (batch, input_dim_static).
                                                Can be None if input_dim_static is 0.

        Returns:
            torch.Tensor: Output logits tensor (batch, output_dim).
        """
        # --- Input Validation ---
        if x_sequence.shape[-1] != self.input_dim_sequence:
            raise ValueError(f"Sequence input feature dimension mismatch. Expected {self.input_dim_sequence}, got {x_sequence.shape[-1]}")
        if x_static is not None and self.input_dim_static > 0 and x_static.shape[-1] != self.input_dim_static:
            raise ValueError(f"Static input dimension mismatch. Expected {self.input_dim_static}, got {x_static.shape[-1]}")
        # Handle cases where static features are expected but not provided, or vice versa
        if self.input_dim_static > 0 and x_static is None:
             log.warning(f"Model expects {self.input_dim_static} static features but None provided. Padding with zeros.")
             # Create zero tensor matching batch size and expected static dim
             zeros_static = torch.zeros(x_sequence.shape[0], self.input_dim_static, device=x_sequence.device, dtype=x_sequence.dtype)
             x_static = zeros_static
        elif self.input_dim_static == 0 and x_static is not None and x_static.numel() > 0:
             log.warning(f"Model expects 0 static features but received non-empty static tensor {x_static.shape}. Ignoring.")
             x_static = None # Ignore provided static features

        # --- LSTM Processing ---
        lstm_out = x_sequence
        hidden_states = None # Not explicitly used here, but returned by LSTM
        num_stacked_layers = len(self.lstm_layers_stacked)
        for i, lstm_layer in enumerate(self.lstm_layers_stacked):
            lstm_out, hidden_states = lstm_layer(lstm_out)
            # Apply manual dropout between layers (except after the last one)
            if i < num_stacked_layers - 1:
                lstm_out = self.manual_dropout(lstm_out)

        # --- Extract Final LSTM Representation ---
        # Select the output from the last time step (or combine first/last for bidirectional)
        if self.bidirectional:
            # Concatenate the last output of the forward pass and the first output of the backward pass
            # lstm_out shape: (batch, seq_len, hidden_dim * 2)
            forward_last = lstm_out[:, -1, :self.hidden_dim_last_lstm]
            backward_first = lstm_out[:, 0, self.hidden_dim_last_lstm:]
            final_lstm_representation = torch.cat((forward_last, backward_first), dim=1)
        else:
            # Use the output of the last time step
            # lstm_out shape: (batch, seq_len, hidden_dim)
            final_lstm_representation = lstm_out[:, -1, :]

        # --- Combine with Static Features (Late Fusion) ---
        if x_static is not None: # This implies self.input_dim_static > 0
            # Ensure static features have the correct batch dimension if needed
            if x_static.ndim == 1: # If static features are 1D, unsqueeze for batch
                 x_static = x_static.unsqueeze(0).expand(final_lstm_representation.shape[0], -1)
            # Concatenate along the feature dimension
            combined_features = torch.cat((final_lstm_representation, x_static), dim=1)
        else:
            # No static features to combine
            combined_features = final_lstm_representation

        # --- Final Classification Layer ---
        logits = self.fc(combined_features)
        return logits


# ==============================================================================
# == CNN-LSTM Model with Attention and Early Fusion ==
# ==============================================================================
class StressCNNLSTM(nn.Module):
    """
    CNN-LSTM model with optional Attention and early fusion of static features.
    Processes sequence data with 1D CNNs, combines with static features (optional),
    feeds into LSTM layers, optionally applies MultiheadAttention, then classifies.

    Args:
        input_dim_sequence (int): Number of features per time step in the sequence.
        input_dim_static (int): Number of static features.
        model_config (Dict[str, Any]): Dictionary containing model parameters. Expected keys:
                                       'cnn_filters', 'cnn_kernels', 'cnn_stride', 'cnn_padding',
                                       'cnn_activation', 'lstm_layers', 'dropout', 'bidirectional',
                                       'attn_heads' (int, optional): Number of attention heads. If <= 0, attention is skipped.
        output_dim (int, optional): Number of output classes (default: 1 for binary sigmoid).
    """
    def __init__(self,
                 input_dim_sequence: int,
                 input_dim_static: int,
                 model_config: Dict[str, Any],
                 output_dim: int = 1):
        super().__init__()

        # --- Get CNN parameters ---
        cnn_filters = safe_get(model_config, ['cnn_filters'], [32, 64])
        cnn_kernels = safe_get(model_config, ['cnn_kernels'], [7, 5])
        cnn_stride = safe_get(model_config, ['cnn_stride'], 1)
        cnn_padding = safe_get(model_config, ['cnn_padding'], 'same') # 'same' or integer
        cnn_activation_str = safe_get(model_config, ['cnn_activation'], 'relu').lower()

        # --- Get LSTM/General parameters ---
        lstm_hidden_dims = safe_get(model_config, ['lstm_layers'], [128])
        dropout = safe_get(model_config, ['dropout'], 0.3)
        bidirectional = safe_get(model_config, ['bidirectional'], True)

        # --- Get Attention parameters ---
        # If attn_heads <= 0, attention layer is skipped
        attn_heads = safe_get(model_config, ['attn_heads'], 4) # e.g., 4 attention heads

        # --- Input Validation ---
        if not isinstance(input_dim_sequence, int) or input_dim_sequence <= 0: raise ValueError("input_dim_sequence must be > 0.")
        if not isinstance(input_dim_static, int) or input_dim_static < 0: raise ValueError("input_dim_static must be >= 0.")
        if not isinstance(output_dim, int) or output_dim <= 0: raise ValueError("output_dim must be > 0.")
        if not isinstance(dropout, (float, int)) or not (0.0 <= dropout < 1.0): raise ValueError("model_config['dropout'] must be between 0.0 and 1.0.")
        if not isinstance(bidirectional, bool): raise ValueError("model_config['bidirectional'] must be a boolean.")
        if not isinstance(lstm_hidden_dims, list) or not lstm_hidden_dims or not all(isinstance(h, int) and h > 0 for h in lstm_hidden_dims): raise ValueError("model_config['lstm_layers'] must be a non-empty list of positive integers.")
        if not isinstance(cnn_filters, list) or not cnn_filters or not all(isinstance(f, int) and f > 0 for f in cnn_filters): raise ValueError("model_config['cnn_filters'] must be a non-empty list of positive integers.")
        if not isinstance(cnn_kernels, list) or len(cnn_kernels) != len(cnn_filters): raise ValueError("model_config['cnn_kernels'] must be a list of same length as cnn_filters.")
        if not all(isinstance(k, int) and k > 0 for k in cnn_kernels): raise ValueError("model_config['cnn_kernels'] elements must be positive integers.")
        if not isinstance(cnn_stride, int) or cnn_stride <= 0: raise ValueError("model_config['cnn_stride'] must be a positive integer.")
        if not isinstance(attn_heads, int): raise ValueError("model_config['attn_heads'] must be an integer.")
        # --- End Validation ---

        # --- Store dimensions and config ---
        self.input_dim_sequence = input_dim_sequence
        self.input_dim_static = input_dim_static
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.lstm_output_multiplier = 2 if self.bidirectional else 1
        self.num_lstm_layers = len(lstm_hidden_dims)
        self.hidden_dim_last_lstm = lstm_hidden_dims[-1]
        self.use_attention = attn_heads > 0 # Flag to enable/disable attention
        self.attn_heads = attn_heads if self.use_attention else 0
        self.config_ref = model_config # Store config for potential later use (e.g., calculating output length)

        log.info(f"Initializing StressCNNLSTM model (Attention={'Enabled' if self.use_attention else 'Disabled'}, Early Fusion):")
        log.info(f"  Sequence Input Dim: {self.input_dim_sequence}")
        log.info(f"  Static Input Dim: {self.input_dim_static}")
        log.info(f"  CNN Filters: {cnn_filters}")
        log.info(f"  CNN Kernels: {cnn_kernels}")
        log.info(f"  CNN Stride: {cnn_stride}")
        log.info(f"  CNN Padding: {cnn_padding}")
        log.info(f"  CNN Activation: {cnn_activation_str}")
        log.info(f"  LSTM Hidden Dims: {lstm_hidden_dims}")
        log.info(f"  Bidirectional LSTM: {self.bidirectional}")
        if self.use_attention: log.info(f"  Attention Heads: {self.attn_heads}")
        log.info(f"  Dropout: {dropout}")
        log.info(f"  Output Dim (Classifier): {self.output_dim}")

        # --- CNN Layers ---
        cnn_layers = []
        current_channels = input_dim_sequence # Input channels for the first CNN layer
        for i in range(len(cnn_filters)):
            # Note: padding='same' requires PyTorch 1.9+ for stride > 1.
            # If using older PyTorch, calculate padding manually or use integer padding.
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=cnn_filters[i],
                    kernel_size=cnn_kernels[i],
                    stride=cnn_stride,
                    padding=cnn_padding
                )
            )
            # Add activation function
            if cnn_activation_str == 'relu': cnn_layers.append(nn.ReLU())
            elif cnn_activation_str == 'tanh': cnn_layers.append(nn.Tanh())
            # Add dropout after activation
            cnn_layers.append(nn.Dropout(dropout))
            # Update channels for the next layer
            current_channels = cnn_filters[i]
        # Create sequential container for CNN layers
        self.cnn_encoder = nn.Sequential(*cnn_layers)
        self.cnn_output_channels = current_channels # Channels output by the last CNN layer

        # --- LSTM Layers ---
        # Input dimension for LSTM is the output channels from CNN + static features (Early Fusion)
        lstm_input_dim = self.cnn_output_channels + self.input_dim_static
        log.info(f"  LSTM Input Dim (CNN Output Channels + Static Dim): {lstm_input_dim}")
        self.lstm_layers_stacked = nn.ModuleList()
        current_lstm_input_dim = lstm_input_dim
        # Apply dropout between LSTM layers only if more than one layer
        lstm_dropout_rate = dropout if self.num_lstm_layers > 1 else 0.0
        log.info(f"  Manual Dropout Rate (between LSTM layers): {lstm_dropout_rate}")

        for i, hidden_dim in enumerate(lstm_hidden_dims):
            self.lstm_layers_stacked.append(
                nn.LSTM(input_size=current_lstm_input_dim,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=self.bidirectional)
            )
            current_lstm_input_dim = hidden_dim * self.lstm_output_multiplier
        # Manual dropout layer applied between LSTM layers
        self.manual_lstm_dropout = nn.Dropout(lstm_dropout_rate)

        # --- Attention Layer (Optional) ---
        self.attention = None
        self.lstm_output_dim = self.hidden_dim_last_lstm * self.lstm_output_multiplier
        if self.use_attention:
            # Ensure embed_dim (LSTM output) is divisible by num_heads
            if self.lstm_output_dim % self.attn_heads != 0:
                original_heads = self.attn_heads
                # Find the largest divisor <= original_heads
                for h in range(original_heads, 0, -1):
                    if self.lstm_output_dim % h == 0:
                        self.attn_heads = h
                        break
                if self.attn_heads != original_heads:
                     log.warning(f"LSTM output dim ({self.lstm_output_dim}) not divisible by attn_heads ({original_heads}). Adjusted heads to {self.attn_heads}.")
                # This should ideally not happen if lstm_output_dim > 0
                if self.attn_heads == 0:
                     log.error(f"Cannot find valid number of attention heads for LSTM output dim {self.lstm_output_dim}. Disabling attention.")
                     self.use_attention = False # Disable attention if no valid head count found

            # Initialize attention layer only if still enabled
            if self.use_attention:
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.lstm_output_dim,
                    num_heads=self.attn_heads,
                    dropout=dropout, # Use same dropout rate? Or add separate attn_dropout config
                    batch_first=True # IMPORTANT: Ensure batch dimension comes first
                )
                log.info(f"  Attention Layer: Embed Dim={self.lstm_output_dim}, Heads={self.attn_heads}")

        # --- Final Fully Connected Layer ---
        # Input dimension depends on whether attention is used
        fc_input_dim = self.lstm_output_dim # Input is LSTM output (potentially after attention)
        self.fc = nn.Linear(fc_input_dim, self.output_dim)
        log.info(f"  FC Layer Input Dim (from {'Attention' if self.use_attention else 'LSTM'} output): {fc_input_dim}")
        log.info(f"StressCNNLSTM model initialized successfully.")

    def _calculate_cnn_output_length(self, L_in: int) -> int:
        """ Helper to calculate the sequence length after CNN layers (if needed). """
        # This needs access to the config used during init.
        # Note: This calculation might be complex for 'same' padding with stride > 1.
        L_out = L_in
        cnn_kernels = safe_get(self.config_ref, ['cnn_kernels'], [])
        cnn_stride = safe_get(self.config_ref, ['cnn_stride'], 1)
        cnn_padding_val = safe_get(self.config_ref, ['cnn_padding'], 'same')

        for i in range(len(cnn_kernels)):
            kernel = cnn_kernels[i]
            stride = cnn_stride # Assuming same stride for all layers for simplicity

            # Calculate padding value based on config ('same' or integer)
            if isinstance(cnn_padding_val, str) and cnn_padding_val.lower() == 'same':
                # Formula for 'same' padding (simplified, may differ slightly from PyTorch for stride>1)
                # For stride=1, padding = (kernel - 1) // 2
                # For stride>1, it's more complex and depends on input size.
                # PyTorch handles this internally, but for manual calculation:
                if stride == 1:
                     padding = (kernel - 1) // 2
                else:
                     # Approximate calculation for stride > 1
                     padding = max(0, ( (L_out - 1) * stride - L_out + kernel ) // 2)
                     log.warning(f"Calculating 'same' padding manually with stride > 1 ({stride}) can be complex. Result may differ from PyTorch. Consider using integer padding.")
            elif isinstance(cnn_padding_val, int):
                padding = cnn_padding_val
            else:
                raise ValueError(f"Invalid cnn_padding value: {cnn_padding_val}")

            # Standard formula for Conv1d output length
            L_out = (L_out + 2 * padding - (kernel - 1) - 1) // stride + 1
        return L_out

    def forward(self,
                x_sequence: torch.Tensor,
                x_static: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        Forward pass for CNN-LSTM with optional Attention and Early Fusion.

        Args:
            x_sequence (torch.Tensor): Input sequence tensor (batch, seq_len, input_dim_sequence).
            x_static (Optional[torch.Tensor]): Input static features tensor (batch, input_dim_static).

        Returns:
            torch.Tensor: Output logits tensor (batch, output_dim).
        """
        # Input shapes: x_sequence (N, L_in, F), x_static (N, S)
        N, L_in, F = x_sequence.shape

        # --- Input Validation (similar to StressLSTM) ---
        if F != self.input_dim_sequence: raise ValueError(f"Sequence input feature dim mismatch. Expected {self.input_dim_sequence}, got {F}")
        if x_static is not None and self.input_dim_static > 0 and x_static.shape[-1] != self.input_dim_static: raise ValueError(f"Static input dim mismatch. Expected {self.input_dim_static}, got {x_static.shape[-1]}")
        if self.input_dim_static > 0 and x_static is None:
             log.warning(f"Model expects {self.input_dim_static} static features but None provided. Padding static features with zeros.")
             x_static = torch.zeros(N, self.input_dim_static, device=x_sequence.device, dtype=x_sequence.dtype)
        elif self.input_dim_static == 0 and x_static is not None and x_static.numel() > 0:
             log.warning(f"Model expects 0 static features but received non-empty static tensor {x_static.shape}. Ignoring.")
             x_static = None
        elif self.input_dim_static == 0:
             x_static = None # Ensure x_static is None if no static features expected

        # --- CNN Processing ---
        # Conv1d expects input shape (N, C_in, L_in)
        x_sequence_permuted = x_sequence.permute(0, 2, 1) # -> (N, F, L_in)
        cnn_output = self.cnn_encoder(x_sequence_permuted) # -> (N, C_out, L_out)
        # Permute back for LSTM: (N, L_out, C_out)
        cnn_output_permuted = cnn_output.permute(0, 2, 1)
        N_out, L_out, C_out = cnn_output_permuted.shape # Get output shape after CNN

        # --- Early Fusion (Combine CNN output with static features) ---
        if x_static is not None: # If static features are used
            # Expand static features to match the sequence length dimension (L_out)
            # x_static shape: (N, S) -> (N, 1, S) -> (N, L_out, S)
            x_static_expanded = x_static.unsqueeze(1).expand(-1, L_out, -1)
            # Concatenate along the feature dimension (dim=2)
            lstm_input = torch.cat((cnn_output_permuted, x_static_expanded), dim=2) # -> (N, L_out, C_out + S)
        else:
            # If no static features, LSTM input is just the CNN output
            lstm_input = cnn_output_permuted # -> (N, L_out, C_out)

        # --- LSTM Processing ---
        lstm_out = lstm_input
        hidden_states = None
        num_stacked_layers = len(self.lstm_layers_stacked)
        for i, lstm_layer in enumerate(self.lstm_layers_stacked):
            lstm_out, hidden_states = lstm_layer(lstm_out)
            if i < num_stacked_layers - 1:
                lstm_out = self.manual_lstm_dropout(lstm_out)
        # lstm_out shape: (N, L_out, lstm_output_dim) where lstm_output_dim = H * num_directions

        # --- Attention Processing (Optional) ---
        if self.use_attention and self.attention is not None:
            # MultiheadAttention expects query, key, value. Use lstm_out for all.
            # Input shape (N, L_out, lstm_output_dim) - Matches batch_first=True
            attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # attn_output shape: (N, L_out, lstm_output_dim)
            # attn_weights shape: (N, L_out, L_out) - weights for each query position

            # --- Aggregate Attention Output ---
            # Option 1: Global Average Pooling over the sequence length (L_out)
            aggregated_output = torch.mean(attn_output, dim=1) # -> (N, lstm_output_dim)
            # Option 2: Take the output of the last time step (less common with attention)
            # aggregated_output = attn_output[:, -1, :] # -> (N, lstm_output_dim)
            # Option 3: Weighted sum based on attention weights (more complex)
            # ...
        else:
            # If attention is not used, take the last output of the LSTM
            # Handle bidirectional case appropriately
            if self.bidirectional:
                 forward_last = lstm_out[:, -1, :self.hidden_dim_last_lstm]
                 backward_first = lstm_out[:, 0, self.hidden_dim_last_lstm:]
                 aggregated_output = torch.cat((forward_last, backward_first), dim=1) # -> (N, lstm_output_dim)
            else:
                 aggregated_output = lstm_out[:, -1, :] # -> (N, lstm_output_dim)


        # --- Final Classification Layer ---
        logits = self.fc(aggregated_output) # -> (N, output_dim)
        return logits

# ==============================================================================
# == Model Factory Function ==
# ==============================================================================
def get_model(config: Dict[str, Any], input_dim_sequence: int, input_dim_static: int) -> nn.Module:
    """
    Instantiates the appropriate model based on the configuration.

    Args:
        config (Dict[str, Any]): The main configuration dictionary.
        input_dim_sequence (int): Sequence input dimension for the model.
        input_dim_static (int): Static input dimension for the model.

    Returns:
        nn.Module: The instantiated PyTorch model.

    Raises:
        ValueError: If the model type in the config is unknown.
        Exception: If model instantiation fails for other reasons.
    """
    # Get model type from config, default to 'LSTM'
    model_type = safe_get(config, ['model_config', 'type'], 'LSTM').upper()
    model_config = safe_get(config, ['model_config'], {}) # Get the specific model config section
    output_dim = 1 # Assuming binary classification (outputting a single logit)

    log.info(f"Attempting to build model of type: {model_type}")

    if model_type == 'CNN-LSTM':
        try:
            model = StressCNNLSTM(
                input_dim_sequence=input_dim_sequence,
                input_dim_static=input_dim_static,
                model_config=model_config,
                output_dim=output_dim
            )
            log.info("StressCNNLSTM model built successfully.")
        except Exception as e:
            log.error(f"Failed to build StressCNNLSTM: {e}", exc_info=True)
            log.error("Falling back to StressLSTM.")
            model_type = 'LSTM' # Fallback to LSTM if CNN-LSTM fails

    # Default or fallback to LSTM
    if model_type == 'LSTM':
         try:
             # Ensure the LSTM config only contains relevant keys if falling back
             lstm_config = {
                 'lstm_layers': safe_get(model_config, ['lstm_layers'], [64]),
                 'dropout': safe_get(model_config, ['dropout'], 0.0),
                 'bidirectional': safe_get(model_config, ['bidirectional'], False)
             }
             model = StressLSTM(
                 input_dim_sequence=input_dim_sequence,
                 input_dim_static=input_dim_static,
                 model_config=lstm_config, # Pass only relevant LSTM config
                 output_dim=output_dim
             )
             log.info("StressLSTM model built successfully.")
         except Exception as e:
             log.critical(f"Failed to build StressLSTM: {e}", exc_info=True)
             raise # Re-raise critical error if even basic LSTM fails

    # Handle unknown model type
    if 'model' not in locals():
         raise ValueError(f"Unknown model type specified in config: '{model_type}'")

    return model
