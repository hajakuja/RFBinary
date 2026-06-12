"""Model definitions for binary detection."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _conv_bn(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


class RepVGGBlock(nn.Module):
    """RepVGG block with train-time branches and deploy-time fused conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.deploy = bool(deploy)
        self.groups = int(groups)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.nonlinearity = nn.ReLU(inplace=True)

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_dense = _conv_bn(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            )
            self.rbr_1x1 = _conv_bn(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(x))

        identity = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + identity)

    @staticmethod
    def _pad_1x1_to_3x3(kernel: torch.Tensor | int) -> torch.Tensor | int:
        if isinstance(kernel, int):
            return kernel
        return F.pad(kernel, [1, 1, 1, 1])

    def _identity_kernel(self, branch: nn.BatchNorm2d) -> torch.Tensor:
        input_dim = self.in_channels // self.groups
        kernel = torch.zeros(
            (self.in_channels, input_dim, 3, 3),
            dtype=branch.weight.dtype,
            device=branch.weight.device,
        )
        for idx in range(self.in_channels):
            kernel[idx, idx % input_dim, 1, 1] = 1
        return kernel

    def _fuse_bn_tensor(self, branch: nn.Module | None) -> tuple[torch.Tensor | int, torch.Tensor | int]:
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            conv = branch[0]
            bn = branch[1]
            if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
                raise TypeError("Expected RepVGG branch to be Conv2d + BatchNorm2d")
            kernel = conv.weight
        elif isinstance(branch, nn.BatchNorm2d):
            bn = branch
            kernel = self._identity_kernel(bn)
        else:
            raise TypeError(f"Unsupported RepVGG branch type: {type(branch)!r}")

        std = torch.sqrt(bn.running_var + bn.eps)
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        bias = bn.bias - (bn.running_mean * bn.weight / std)
        return kernel * scale, bias

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel_3x3 + self._pad_1x1_to_3x3(kernel_1x1) + kernel_identity,
            bias_3x3 + bias_1x1 + bias_identity,
        )

    def switch_to_deploy(self) -> None:
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=3,
            stride=self.rbr_dense[0].stride,
            padding=1,
            groups=self.rbr_dense[0].groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel.detach().clone()
        self.rbr_reparam.bias.data = bias.detach().clone()
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        self.deploy = True


class RepVGG(nn.Module):
    """RepVGG image classifier for Hailo-friendly RF spectrogram exports."""

    def __init__(
        self,
        *,
        num_blocks: Sequence[int],
        width_multiplier: Sequence[float],
        num_classes: int = 2,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4:
            raise ValueError("RepVGG expects 4 stage block counts")
        if len(width_multiplier) != 4:
            raise ValueError("RepVGG expects 4 width multipliers")

        self.deploy = bool(deploy)
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(3, self.in_planes, stride=2, deploy=deploy)
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), int(num_blocks[0]), stride=2, deploy=deploy)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), int(num_blocks[1]), stride=2, deploy=deploy)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), int(num_blocks[2]), stride=2, deploy=deploy)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), int(num_blocks[3]), stride=2, deploy=deploy)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, out_channels: int, num_blocks: int, *, stride: int, deploy: bool) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(RepVGGBlock(self.in_planes, out_channels, stride=block_stride, deploy=deploy))
            self.in_planes = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.linear(x)

    def switch_to_deploy(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True


def _build_repvgg_a1(*, weights: object | None = None) -> nn.Module:
    _ = weights
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], num_classes=2)


def _build_repvgg_a2(*, weights: object | None = None) -> nn.Module:
    _ = weights
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75], num_classes=2)


REPVGG_LOCAL_ARCHES = {"repvgg_a1", "repvgg_a2"}
REPVGG_MODEL_ZOO_ARCHES = {"repvgg_a1_hmz", "repvgg_a2_hmz"}
REPVGG_BASE_ARCH_BY_ARCH = {
    "repvgg_a1": "repvgg_a1",
    "repvgg_a2": "repvgg_a2",
    "repvgg_a1_hmz": "repvgg_a1",
    "repvgg_a2_hmz": "repvgg_a2",
}
REPVGG_ARCHES = set(REPVGG_BASE_ARCH_BY_ARCH)
REPVGG_MODEL_ZOO_FILENAMES = {
    "repvgg_a1": "RepVGG-A1.onnx",
    "repvgg_a2": "RepVGG-A2.onnx",
    "repvgg_a1_hmz": "RepVGG-A1.onnx",
    "repvgg_a2_hmz": "RepVGG-A2.onnx",
}
CUSTOM_BINARY_ARCHES = {"vgg_small_gap", *REPVGG_ARCHES}


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
    "repvgg_a1": (_build_repvgg_a1, None),
    "repvgg_a2": (_build_repvgg_a2, None),
    "repvgg_a1_hmz": (_build_repvgg_a1, None),
    "repvgg_a2_hmz": (_build_repvgg_a2, None),
}

CLASSIFIER_HEAD_ARCHES = {"vgg13", "vgg16", "mobilenet_v3_small"}
FC_HEAD_ARCHES = {"resnet18", "resnet34", "resnet50", "shufflenet_v2_x1_0", "regnet_x_1_6gf"}
FULLY_TRAINABLE_ARCHES = {"vgg_small_gap", *REPVGG_ARCHES}

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
        "repvgg_a1": ("linear.",),
        "repvgg_a2": ("linear.",),
        "repvgg_a1_hmz": ("linear.",),
        "repvgg_a2_hmz": ("linear.",),
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

    if arch not in CUSTOM_BINARY_ARCHES:
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


def repvgg_model_zoo_onnx_path(arch: str, model_zoo_dir: str | Path) -> Path:
    filename = REPVGG_MODEL_ZOO_FILENAMES.get(arch)
    if filename is None:
        raise ValueError(f"RepVGG Model Zoo weights are only supported for {sorted(REPVGG_MODEL_ZOO_FILENAMES)}")
    return Path(model_zoo_dir) / filename


def _as_tensor_like(value: object, reference: torch.Tensor, *, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = value.copy()
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tuple(tensor.shape) != tuple(reference.shape):
        raise ValueError(f"Shape mismatch for {name}: expected {tuple(reference.shape)}, got {tuple(tensor.shape)}")
    return tensor


def _set_bn_passthrough_with_bias(bn: nn.BatchNorm2d, bias: torch.Tensor) -> None:
    bn.weight.data.fill_(1.0)
    bn.bias.data.copy_(bias)
    bn.running_mean.data.zero_()
    # With running_var + eps == 1, eval-mode BN becomes y = x + bias.
    bn.running_var.data.fill_(max(1.0 - float(bn.eps), 0.0))
    if hasattr(bn, "num_batches_tracked"):
        bn.num_batches_tracked.zero_()


def _zero_conv_bn(branch: nn.Sequential) -> None:
    conv = branch[0]
    bn = branch[1]
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("Expected RepVGG branch to be Conv2d + BatchNorm2d")
    conv.weight.data.zero_()
    _zero_bn(bn)


def _zero_bn(bn: nn.BatchNorm2d) -> None:
    bn.weight.data.zero_()
    bn.bias.data.zero_()
    bn.running_mean.data.zero_()
    bn.running_var.data.fill_(1.0)
    if hasattr(bn, "num_batches_tracked"):
        bn.num_batches_tracked.zero_()


def _load_repvgg_deploy_block(block: RepVGGBlock, weight: object, bias: object, *, prefix: str) -> int:
    with torch.no_grad():
        if hasattr(block, "rbr_reparam"):
            w = _as_tensor_like(weight, block.rbr_reparam.weight, name=f"{prefix}.rbr_reparam.weight")
            b = _as_tensor_like(bias, block.rbr_reparam.bias, name=f"{prefix}.rbr_reparam.bias")
            block.rbr_reparam.weight.data.copy_(w)
            block.rbr_reparam.bias.data.copy_(b)
            return 2

        conv = block.rbr_dense[0]
        bn = block.rbr_dense[1]
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            raise TypeError("Expected RepVGG dense branch to be Conv2d + BatchNorm2d")
        w = _as_tensor_like(weight, conv.weight, name=f"{prefix}.rbr_reparam.weight")
        b = _as_tensor_like(bias, bn.bias, name=f"{prefix}.rbr_reparam.bias")
        conv.weight.data.copy_(w)
        _set_bn_passthrough_with_bias(bn, b)
        _zero_conv_bn(block.rbr_1x1)
        if block.rbr_identity is not None:
            _zero_bn(block.rbr_identity)
        return 2


def _repvgg_backbone_from_model(model: nn.Module) -> tuple[str, RepVGG]:
    arch = str(getattr(model, "arch", ""))
    backbone = getattr(model, "backbone", model)
    if arch not in REPVGG_ARCHES or not isinstance(backbone, RepVGG):
        raise ValueError(f"RepVGG Model Zoo loading requires a RepVGG BinaryImageClassifier, got arch={arch!r}")
    return arch, backbone


def load_repvgg_deploy_state_dict(
    model: nn.Module,
    deploy_state: Mapping[str, object],
    *,
    source: str,
) -> Dict[str, object]:
    """Initialize a train-time RepVGG model from deploy-form RepVGG tensors.

    Hailo Model Zoo RepVGG ONNX files contain fused deploy-form Conv2d weights.
    RF checkpoints stay in train-form so export can still run `switch_to_deploy`;
    therefore the fused conv is loaded into the 3x3 branch while the 1x1 and
    identity branches are initialized to zero.
    """

    arch, backbone = _repvgg_backbone_from_model(model)
    loaded_tensors = 0
    missing: list[str] = []
    block_names: list[str] = []

    for name, module in backbone.named_modules():
        if not isinstance(module, RepVGGBlock):
            continue
        weight_name = f"{name}.rbr_reparam.weight"
        bias_name = f"{name}.rbr_reparam.bias"
        block_names.append(name)
        if weight_name not in deploy_state:
            missing.append(weight_name)
            continue
        if bias_name not in deploy_state:
            missing.append(bias_name)
            continue
        loaded_tensors += _load_repvgg_deploy_block(
            module,
            deploy_state[weight_name],
            deploy_state[bias_name],
            prefix=name,
        )

    if missing:
        raise ValueError(f"Missing RepVGG deploy tensors in {source}: {missing[:8]}")

    skipped_head_tensors = [name for name in ("linear.weight", "linear.bias") if name in deploy_state]
    return {
        "type": "repvgg_deploy_state_dict",
        "arch": arch,
        "base_arch": REPVGG_BASE_ARCH_BY_ARCH[arch],
        "source": str(source),
        "loaded_tensors": int(loaded_tensors),
        "loaded_blocks": int(len(block_names)),
        "skipped_head_tensors": skipped_head_tensors,
    }


def load_repvgg_model_zoo_onnx(model: nn.Module, onnx_path: str | Path) -> Dict[str, object]:
    """Load Hailo Model Zoo RepVGG ONNX backbone weights into a binary model."""

    try:
        import onnx
        from onnx import numpy_helper
    except Exception as exc:  # pragma: no cover - exercised only without optional onnx install
        raise RuntimeError("Loading RepVGG Model Zoo ONNX weights requires the `onnx` package") from exc

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"RepVGG Model Zoo ONNX file not found: {path}")
    graph = onnx.load(path).graph
    deploy_state = {initializer.name: numpy_helper.to_array(initializer) for initializer in graph.initializer}
    metadata = load_repvgg_deploy_state_dict(model, deploy_state, source=str(path))
    metadata.update(
        {
            "type": "hailo_model_zoo_repvgg_onnx",
            "onnx_path": str(path),
            "initializer_count": int(len(deploy_state)),
        }
    )
    return metadata


def prepare_model_for_export(model: nn.Module) -> list[str]:
    """Apply architecture-specific export transforms in-place and return their names."""
    arch = getattr(model, "arch", None)
    if arch in REPVGG_ARCHES:
        backbone = getattr(model, "backbone", None)
        if backbone is None or not hasattr(backbone, "switch_to_deploy"):
            raise ValueError(f"RepVGG export expected a deploy-convertible backbone for {arch}")
        backbone.switch_to_deploy()
        return ["repvgg_switch_to_deploy"]
    return []


class VGG16Binary(BinaryImageClassifier):
    def __init__(self, pretrained: bool = True, freeze_features: bool = True) -> None:
        super().__init__(arch="vgg16", pretrained=pretrained, freeze_features=freeze_features)
