"""
losses.py - Ham mat mat xu ly mat can bang lop (Cross-Entropy, Focal Loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al.): Giam trong so mau de, tap trung vao mau kho va lop hiem."""
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


def get_loss_fn(loss_type: str = "ce", class_counts: list = None, gamma: float = 2.0, device: torch.device = None):
    """Tra ve ham loss tuong ung voi yeu cau thuc nghiem."""
    if loss_type == "focal":
        return FocalLoss(gamma=gamma)
    elif loss_type == "weighted_ce" and class_counts is not None:
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = counts.sum() / (len(counts) * counts)
        weights = (weights / weights.mean()).to(device or torch.device("cpu"))
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()
