import copy
from pathlib import Path

import onnx
import pytest
import torch

from rfbd.export import export_checkpoint
from rfbd.modeling import (
    REPVGG_MODEL_ZOO_FILENAMES,
    RepVGGBlock,
    create_binary_model,
    list_supported_arches,
    load_repvgg_deploy_state_dict,
    prepare_model_for_export,
    repvgg_model_zoo_onnx_path,
)


REPVGG_TEST_ARCHES = ["repvgg_a1", "repvgg_a2", "repvgg_a1_hmz", "repvgg_a2_hmz"]


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
        "repvgg_a1",
        "repvgg_a2",
        "repvgg_a1_hmz",
        "repvgg_a2_hmz",
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


@pytest.mark.parametrize("arch", REPVGG_TEST_ARCHES)
def test_repvgg_arches_remain_fully_trainable_with_freeze_features(arch: str):
    model = create_binary_model(arch=arch, pretrained=True, freeze_features=True)
    assert all(param.requires_grad for param in model.parameters())


@pytest.mark.parametrize("arch", REPVGG_TEST_ARCHES)
def test_repvgg_model_zoo_path_resolves_expected_filename(arch: str):
    root = Path("/weights")
    assert repvgg_model_zoo_onnx_path(arch, root) == root / REPVGG_MODEL_ZOO_FILENAMES[arch]


def test_repvgg_deploy_state_dict_initializes_train_form_backbone():
    seed_model = create_binary_model(arch="repvgg_a1", pretrained=False, freeze_features=False)
    deploy_model = copy.deepcopy(seed_model)
    target_model = copy.deepcopy(seed_model)
    deploy_model.eval()
    target_model.eval()
    assert prepare_model_for_export(deploy_model) == ["repvgg_switch_to_deploy"]

    metadata = load_repvgg_deploy_state_dict(
        target_model,
        deploy_model.backbone.state_dict(),
        source="unit-test",
    )
    assert metadata["type"] == "repvgg_deploy_state_dict"
    assert metadata["loaded_tensors"] == 44
    assert metadata["skipped_head_tensors"] == ["linear.weight", "linear.bias"]

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        target_logits = target_model(x)
        deploy_logits = deploy_model(x)
    assert torch.max(torch.abs(target_logits - deploy_logits)).item() < 1e-4

    assert prepare_model_for_export(target_model) == ["repvgg_switch_to_deploy"]
    with torch.no_grad():
        converted_logits = target_model(x)
    assert torch.max(torch.abs(converted_logits - deploy_logits)).item() < 1e-4


def test_repvgg_deploy_conversion_preserves_eval_logits_and_removes_training_branches():
    model = create_binary_model(arch="repvgg_a1", pretrained=False, freeze_features=False)
    model.eval()
    deploy_model = copy.deepcopy(model)
    deploy_model.eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        train_logits = model(x)
    assert prepare_model_for_export(deploy_model) == ["repvgg_switch_to_deploy"]
    with torch.no_grad():
        deploy_logits = deploy_model(x)

    assert torch.max(torch.abs(train_logits - deploy_logits)).item() < 1e-4
    blocks = [module for module in deploy_model.backbone.modules() if isinstance(module, RepVGGBlock)]
    assert blocks
    for block in blocks:
        assert hasattr(block, "rbr_reparam")
        assert not hasattr(block, "rbr_dense")
        assert not hasattr(block, "rbr_1x1")
        assert not hasattr(block, "rbr_identity")


@pytest.mark.parametrize("arch", ["repvgg_a1", "repvgg_a2", "repvgg_a1_hmz", "repvgg_a2_hmz"])
def test_repvgg_static_onnx_export_records_deploy_transform(tmp_path: Path, arch: str):
    model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
    checkpoint = tmp_path / f"{arch}_binary.pt"
    torch.save(
        {
            "arch": arch,
            "state_dict": model.state_dict(),
            "threshold": 0.5,
            "preprocessing_contract": {"resize_h": 32, "resize_w": 32},
            "label_contract": {"0": "no_drone", "1": "drone"},
        },
        checkpoint,
    )

    summary = export_checkpoint(
        checkpoint=checkpoint,
        out_dir=tmp_path / "export",
        arch=arch,
        formats=["onnx"],
        batch_size=1,
        onnx_opset=17,
        require_onnx=True,
        static_batch=True,
    )

    assert summary["export_transforms"] == ["repvgg_switch_to_deploy"]
    assert summary["onnx_metadata"]["exporter"] == "legacy_torchscript"
    assert summary["onnx_metadata"]["actual_default_opset"] == 17
    assert Path(summary["exported"]["onnx"]).exists()
    assert summary["export_contract"]["input_shape"] == [1, 1, 32, 32]
    assert summary["export_contract"]["output_shape"] == [1, 2]

    exported = onnx.load(summary["exported"]["onnx"])
    convs = [node for node in exported.graph.node if node.op_type == "Conv"]
    assert convs
    assert all(any(attr.name == "kernel_shape" for attr in node.attribute) for node in convs)
