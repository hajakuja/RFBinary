"""Model definitions for binary detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class VGG16Binary(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_features: bool = True) -> None:
        super().__init__()
        if pretrained:
            try:
                backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = models.vgg16(weights=None)
        else:
            backbone = models.vgg16(weights=None)

        if freeze_features:
            for p in backbone.features.parameters():
                p.requires_grad_(False)

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 2)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,H,W), (B,1,H,W), or (B,3,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected input ndim 3/4, got {x.shape}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
