import pytest
import torch

from rfbd.modeling import create_binary_model, list_supported_arches


def test_supported_arches_matches_hailo8_candidate_suite():
    assert list_supported_arches() == (
        "vgg16",
        "resnet18",
        "mobilenet_v3_small",
        "shufflenet_v2_x1_0",
        "vgg13",
        "resnet34",
        "resnet50",
        "regnet_x_1_6gf",
        "vgg_small_gap",
    )


@pytest.mark.parametrize("arch", list_supported_arches())
def test_model_factory_output_shape_and_channel_handling(arch: str):
    model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
    model.eval()

    x_hw = torch.randn(2, 224, 224)
    x_1c = torch.randn(2, 1, 224, 224)
    x_3c = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        y_hw = model(x_hw)
        y_1c = model(x_1c)
        y_3c = model(x_3c)

    assert tuple(y_hw.shape) == (2, 2)
    assert tuple(y_1c.shape) == (2, 2)
    assert tuple(y_3c.shape) == (2, 2)


def test_model_factory_rejects_invalid_channel_count():
    model = create_binary_model(arch="vgg16", pretrained=False, freeze_features=False)
    with pytest.raises(ValueError, match="channel count"):
        _ = model(torch.randn(2, 2, 224, 224))


def test_vgg_small_gap_has_small_gap_head_and_stays_trainable_when_freezing_requested():
    model = create_binary_model(arch="vgg_small_gap", pretrained=True, freeze_features=True)

    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert len(linear_layers) == 1
    assert linear_layers[0].in_features == 256
    assert linear_layers[0].out_features == 2
    assert max(layer.in_features * layer.out_features for layer in linear_layers) < 1_000_000
    assert any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules())
    assert all(param.requires_grad for param in model.parameters())
