# losses.py (Custom loss functions)
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F # Functional interface for losses, activations etc.

log = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for binary classification tasks.
    Focal Loss down-weights well-classified examples, focusing training on hard negatives.
    Reference: "Focal Loss for Dense Object Detection" - https://arxiv.org/abs/1708.02002

    Args:
        alpha (float, optional): Weighting factor for the positive class (often denoted as alpha_t).
                                 Value between 0 and 1. Set to -1 to disable alpha weighting.
                                 Defaults to 0.25 (as suggested in the paper).
        gamma (float, optional): Focusing parameter (>= 0). Higher values give more focus
                                 to hard examples. Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()

        # --- Input Validation ---
        if not (isinstance(gamma, (float, int)) and gamma >= 0):
             raise ValueError(f"FocalLoss gamma must be non-negative, got {gamma}")
        # Allow alpha = -1 to disable it
        if not (isinstance(alpha, (float, int)) and (0 <= alpha <= 1 or alpha == -1)):
            raise ValueError(f"FocalLoss alpha must be between 0 and 1 (or -1 to disable), got {alpha}")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode for FocalLoss: '{reduction}'. Choose 'none', 'mean', or 'sum'.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        log.info(f"Initialized FocalLoss: alpha={alpha if alpha != -1 else 'Disabled'}, gamma={gamma}, reduction='{reduction}'")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Focal Loss.

        Args:
            inputs (torch.Tensor): Logits predicted by the model (raw scores, before sigmoid).
                                   Expected shape (N,) or (N, 1) for binary classification.
            targets (torch.Tensor): Ground truth labels (0 or 1).
                                    Expected shape (N,) or (N, 1).

        Returns:
            torch.Tensor: The calculated focal loss, reduced according to self.reduction.
        """
        # Ensure shapes are compatible (N,)
        if inputs.ndim == targets.ndim + 1 and inputs.shape[-1] == 1:
            inputs = inputs.squeeze(-1)
        if targets.ndim == inputs.ndim + 1 and targets.shape[-1] == 1:
             targets = targets.squeeze(-1)
        if inputs.shape != targets.shape:
            raise ValueError(f"FocalLoss input shape ({inputs.shape}) and target shape ({targets.shape}) must match.")

        # Calculate Binary Cross Entropy loss without reduction, using logits for stability
        # targets.float() is necessary as BCE expects float targets
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')

        # Calculate pt = exp(-BCE_loss). This is the probability of the *correct* class
        # For positive targets (targets=1), pt = sigmoid(inputs)
        # For negative targets (targets=0), pt = 1 - sigmoid(inputs)
        # Using exp(-BCE_loss) calculates this directly and avoids potential sigmoid instability.
        pt = torch.exp(-BCE_loss)

        # Calculate the Focal Loss term: (1-pt)^gamma * BCE_loss
        # This down-weights easy examples (pt -> 1) and focuses on hard examples (pt -> 0)
        F_loss = (1 - pt)**self.gamma * BCE_loss

        # Apply alpha weighting (optional)
        if self.alpha != -1:
            # Create alpha weight tensor: alpha for positive class, (1-alpha) for negative class
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            # Ensure alpha_t is on the same device as the loss
            alpha_t = alpha_t.to(F_loss.device)
            # Apply the weighting
            F_loss = alpha_t * F_loss

        # Apply reduction based on the specified mode
        if self.reduction == 'mean':
            return F_loss.mean() # Average loss over the batch
        elif self.reduction == 'sum':
            return F_loss.sum() # Sum loss over the batch
        else: # 'none'
            return F_loss # Return loss for each element in the batch
