"""Model definitions for binary detection."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

SUPPORTED_ARCHES = ("vgg16", "resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0")


def list_supported_arches() -> Tuple[str, ...]:
    return SUPPORTED_ARCHES


def _load_backbone(
    builder: Callable[..., nn.Module],
    weights_enum: object | None,
    pretrained: bool,
) -> nn.Module:
    if pretrained and weights_enum is not None:
        try:
            return builder(weights=weights_enum)
        except Exception:
            return builder(weights=None)
    return builder(weights=None)


def _set_binary_head(backbone: nn.Module, arch: str) -> None:
    if arch == "vgg16":
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 2)
        return

    if arch == "resnet18":
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        return

    if arch == "mobilenet_v3_small":
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 2)
        return

    if arch == "shufflenet_v2_x1_0":
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        return

    raise ValueError(f"Unsupported architecture: {arch}")


def _head_prefixes_for_arch(arch: str) -> Tuple[str, ...]:
    head_prefix: Dict[str, Tuple[str, ...]] = {
        "vgg16": ("classifier.",),
        "resnet18": ("fc.",),
        "mobilenet_v3_small": ("classifier.",),
        "shufflenet_v2_x1_0": ("fc.",),
    }
    if arch not in head_prefix:
        raise ValueError(f"Unsupported architecture: {arch}")
    return head_prefix[arch]


def _freeze_non_head(backbone: nn.Module, head_prefixes: Iterable[str]) -> None:
    prefixes = tuple(head_prefixes)
    for name, param in backbone.named_parameters():
        trainable = any(name.startswith(pfx) for pfx in prefixes)
        param.requires_grad_(trainable)


def _build_binary_backbone(arch: str, pretrained: bool) -> nn.Module:
    if arch == "vgg16":
        backbone = _load_backbone(
            models.vgg16,
            models.VGG16_Weights.IMAGENET1K_V1,
            pretrained=pretrained,
        )
    elif arch == "resnet18":
        backbone = _load_backbone(
            models.resnet18,
            models.ResNet18_Weights.IMAGENET1K_V1,
            pretrained=pretrained,
        )
    elif arch == "mobilenet_v3_small":
        backbone = _load_backbone(
            models.mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
            pretrained=pretrained,
        )
    elif arch == "shufflenet_v2_x1_0":
        backbone = _load_backbone(
            models.shufflenet_v2_x1_0,
            models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    _set_binary_head(backbone, arch=arch)
    return backbone


class BinaryImageClassifier(nn.Module):
    def __init__(
        self,
        arch: str = "vgg16",
        pretrained: bool = True,
        freeze_features: bool = True,
    ) -> None:
        super().__init__()
        if arch not in SUPPORTED_ARCHES:
            raise ValueError(f"Unsupported architecture: {arch}. Choose one of: {SUPPORTED_ARCHES}")

        self.arch = arch
        self.backbone = _build_binary_backbone(arch=arch, pretrained=pretrained)

        if freeze_features:
            _freeze_non_head(self.backbone, _head_prefixes_for_arch(arch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,H,W), (B,1,H,W), or (B,3,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected input ndim 3/4, got {x.shape}")
        if x.shape[1] not in {1, 3}:
            raise ValueError(f"Expected channel count 1 or 3, got {x.shape[1]}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def create_binary_model(
    arch: str = "vgg16",
    pretrained: bool = True,
    freeze_features: bool = True,
) -> BinaryImageClassifier:
    return BinaryImageClassifier(arch=arch, pretrained=pretrained, freeze_features=freeze_features)


class VGG16Binary(BinaryImageClassifier):
    def __init__(self, pretrained: bool = True, freeze_features: bool = True) -> None:
        super().__init__(arch="vgg16", pretrained=pretrained, freeze_features=freeze_features)
