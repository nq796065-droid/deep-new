"""
losses.py - Loss Functions for Class Imbalance (Cross-Entropy, Focal Loss, Weighted CE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): Down-weights easy examples, focusing gradient on hard minority samples.
    - Default: Pure Focal Loss (alpha=None) focusing on sample difficulty.
    - Supports optional alpha tensor for alpha-balanced Focal Loss.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce)
        weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            weight = self.alpha.to(inputs.device)[targets] * weight
        return (weight * ce).mean()


def get_loss_fn(
    loss_type: str = "ce",
    class_counts: list = None,
    gamma: float = 2.0,
    use_alpha: bool = False,
    device: torch.device = None
):
    """
    Return loss function corresponding to experimental configuration.
    - 'ce': Standard Cross-Entropy.
    - 'focal': Focal Loss (gamma=2.0). Computes class-balanced alpha if use_alpha=True.
    - 'weighted_ce': Cost-sensitive Cross-Entropy with inverse class frequency weights.
    """
    dev = device or torch.device("cpu")

    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        alpha = None
        if use_alpha:
            if class_counts is None:
                raise ValueError("'class_counts' is required when use_alpha=True for Focal Loss.")
            counts = torch.tensor(class_counts, dtype=torch.float32)
            alpha = (counts.sum() / (len(counts) * counts))
            alpha = (alpha / alpha.mean()).to(dev)
        return FocalLoss(gamma=gamma, alpha=alpha)

    elif loss_type == "weighted_ce":
        if class_counts is None:
            raise ValueError("Parameter 'class_counts' is required for loss_type='weighted_ce'.")
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = counts.sum() / (len(counts) * counts)
        weights = (weights / weights.mean()).to(dev)
        return nn.CrossEntropyLoss(weight=weights)

    else:
        raise ValueError(
            f"Unsupported loss_type='{loss_type}'. "
            f"Valid options: 'ce', 'focal', 'weighted_ce'."
        )
