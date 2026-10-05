"""
models/resnet.py - ResNet-50 Transfer Learning cho phan loai 7 lop ton thuong da.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SkinLesionModel(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def get_model(num_classes: int = 7, pretrained: bool = True) -> SkinLesionModel:
    return SkinLesionModel(num_classes=num_classes, pretrained=pretrained)
