"""Model definitions for binary detection."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class VGGSmallGAP(nn.Module):
    """Small VGG-style classifier with a Hailo-friendly GAP head."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(32, 64),
            self._conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(64, 128),
            self._conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(128, 192),
            self._conv_block(192, 192),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(192, 256),
            self._conv_block(256, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


def _build_vgg_small_gap(*, weights: object | None = None) -> nn.Module:
    _ = weights
    return VGGSmallGAP(num_classes=2)


ARCH_BUILDERS: Dict[str, tuple[Callable[..., nn.Module], object | None]] = {
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    "vgg13": (models.vgg13, models.VGG13_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "regnet_x_1_6gf": (models.regnet_x_1_6gf, models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1),
    "vgg_small_gap": (_build_vgg_small_gap, None),
}

CLASSIFIER_HEAD_ARCHES = {"vgg13", "vgg16", "mobilenet_v3_small"}
FC_HEAD_ARCHES = {"resnet18", "resnet34", "resnet50", "shufflenet_v2_x1_0", "regnet_x_1_6gf"}
FULLY_TRAINABLE_ARCHES = {"vgg_small_gap"}

SUPPORTED_ARCHES = tuple(ARCH_BUILDERS.keys())


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
    if arch in CLASSIFIER_HEAD_ARCHES:
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 2)
        return

    if arch in FC_HEAD_ARCHES:
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        return

    raise ValueError(f"Unsupported architecture: {arch}")


def _head_prefixes_for_arch(arch: str) -> Tuple[str, ...]:
    head_prefix: Dict[str, Tuple[str, ...]] = {
        "vgg13": ("classifier.",),
        "vgg16": ("classifier.",),
        "resnet18": ("fc.",),
        "mobilenet_v3_small": ("classifier.",),
        "shufflenet_v2_x1_0": ("fc.",),
        "resnet34": ("fc.",),
        "resnet50": ("fc.",),
        "regnet_x_1_6gf": ("fc.",),
        "vgg_small_gap": ("classifier.",),
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
    spec = ARCH_BUILDERS.get(arch)
    if spec is None:
        raise ValueError(f"Unsupported architecture: {arch}")
    builder, weights_enum = spec
    backbone = _load_backbone(builder, weights_enum, pretrained=pretrained)

    if arch != "vgg_small_gap":
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

        if freeze_features and arch not in FULLY_TRAINABLE_ARCHES:
            _freeze_non_head(self.backbone, _head_prefixes_for_arch(arch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,H,W), (B,1,H,W), or (B,3,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected input ndim 3/4, got {x.shape}")
        channels = x.shape[1]
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
        elif channels != 3:
            raise ValueError(f"Expected channel count 1 or 3, got {channels}")
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
