import pytest

from rfbd.cli import build_parser
from rfbd.training import _resolve_repvgg_pretrained_onnx, resolve_device


def test_resolve_device_require_gpu_raises_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU is required but unavailable"):
        resolve_device(require_gpu=True)


def test_cli_parses_require_gpu_flags():
    parser = build_parser()

    train_args = parser.parse_args(
        [
            "train",
            "binary",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--arch",
            "vgg_small_gap",
            "--require-gpu",
        ]
    )
    assert train_args.require_gpu is True
    assert train_args.arch == "vgg_small_gap"

    repvgg_train_args = parser.parse_args(
        [
            "train",
            "binary",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--arch",
            "repvgg_a1",
            "--use-repvgg-model-zoo",
            "--repvgg-model-zoo-dir",
            "/tmp/repvgg",
        ]
    )
    assert repvgg_train_args.use_repvgg_model_zoo is True
    assert repvgg_train_args.repvgg_model_zoo_dir == "/tmp/repvgg"

    repvgg_hmz_train_args = parser.parse_args(
        [
            "train",
            "binary",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--arch",
            "repvgg_a1_hmz",
            "--repvgg-model-zoo-dir",
            "/tmp/repvgg",
        ]
    )
    assert _resolve_repvgg_pretrained_onnx(repvgg_hmz_train_args, "repvgg_a1_hmz") == "/tmp/repvgg/RepVGG-A1.onnx"

    legacy_args = parser.parse_args(
        [
            "train",
            "vgg-binary",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--require-gpu",
        ]
    )
    assert legacy_args.require_gpu is True

    eval_args = parser.parse_args(
        [
            "eval",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--arch",
            "resnet50",
            "--require-gpu",
        ]
    )
    assert eval_args.require_gpu is True
    assert eval_args.arch == "resnet50"

    repvgg_eval_args = parser.parse_args(
        [
            "eval",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--arch",
            "repvgg_a2",
            "--repvgg-pretrained-onnx",
            "/tmp/RepVGG-A2.onnx",
        ]
    )
    assert repvgg_eval_args.repvgg_pretrained_onnx == "/tmp/RepVGG-A2.onnx"

    export_args = parser.parse_args(
        [
            "export",
            "--checkpoint",
            "/tmp/model.pt",
            "--out-dir",
            "/tmp/export",
            "--format",
            "onnx",
            "--static-batch",
        ]
    )
    assert export_args.format == ["onnx"]
    assert export_args.static_batch is True

    hailo_args = parser.parse_args(
        [
            "hailo",
            "prepare",
            "--target",
            "hailo8",
            "--skip-existing",
            "--no-strict-far",
        ]
    )
    assert hailo_args.target == ["hailo8"]
    assert hailo_args.skip_existing is True
    assert hailo_args.include_strict_far is False

    hailo_compile_args = parser.parse_args(
        [
            "hailo",
            "compile",
            "--model-id",
            "vgg16_binary",
            "--runtime-metrics-json",
            "/tmp/runtime.json",
            "--hardware-results-json",
            "/tmp/hardware.json",
        ]
    )
    assert hailo_compile_args.model_id == ["vgg16_binary"]
    assert hailo_compile_args.runtime_metrics_json == "/tmp/runtime.json"
    assert hailo_compile_args.hardware_results_json == "/tmp/hardware.json"

    hailo_validate_args = parser.parse_args(
        [
            "hailo",
            "validate",
            "--model-id",
            "vgg16_binary",
            "--target",
            "hailo8",
            "--runtime-metrics-json",
            "/tmp/runtime.json",
        ]
    )
    assert hailo_validate_args.model_id == ["vgg16_binary"]
    assert hailo_validate_args.target == ["hailo8"]
    assert hailo_validate_args.runtime_metrics_json == "/tmp/runtime.json"
