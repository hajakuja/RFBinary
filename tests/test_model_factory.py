import pytest
import torch

from rfbd.modeling import create_binary_model, list_supported_arches


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
