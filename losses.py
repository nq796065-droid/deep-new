"""
losses.py - Hàm mất mát xử lý mất cân bằng lớp (Cross-Entropy, Focal Loss, Weighted CE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): Giảm trọng số các mẫu dễ, tập trung vào mẫu khó và lớp hiếm.
    - Mặc định: Focal Loss thuần (alpha=None) tập trung vào độ khó (modulating factor).
    - Hỗ trợ alpha tensor cho alpha-balanced Focal Loss khi cần.
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
    Trả về hàm loss tương ứng với cấu hình thực nghiệm.
    - 'ce': Cross-Entropy tiêu chuẩn.
    - 'focal': Focal Loss thuần (gamma=2.0). Nếu use_alpha=True thì tính alpha theo class_counts.
    - 'weighted_ce': Cross-Entropy có trọng số nghịch đảo tần suất lớp.
    """
    dev = device or torch.device("cpu")

    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        alpha = None
        if use_alpha:
            if class_counts is None:
                raise ValueError("Cần cung cấp 'class_counts' khi bật use_alpha=True cho Focal Loss.")
            counts = torch.tensor(class_counts, dtype=torch.float32)
            alpha = (counts.sum() / (len(counts) * counts))
            alpha = (alpha / alpha.mean()).to(dev)
        return FocalLoss(gamma=gamma, alpha=alpha)

    elif loss_type == "weighted_ce":
        if class_counts is None:
            raise ValueError("Tham số 'class_counts' là bắt buộc đối với loss_type='weighted_ce'.")
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = counts.sum() / (len(counts) * counts)
        weights = (weights / weights.mean()).to(dev)
        return nn.CrossEntropyLoss(weight=weights)

    else:
        raise ValueError(
            f"Không hỗ trợ loss_type='{loss_type}'. "
            f"Các lựa chọn hợp lệ: 'ce', 'focal', 'weighted_ce'."
        )
