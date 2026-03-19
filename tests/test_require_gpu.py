import pytest

from rfbd.cli import build_parser
from rfbd.training import resolve_device


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
            "mobilenet_v3_small",
            "--require-gpu",
        ]
    )
    assert train_args.require_gpu is True
    assert train_args.arch == "mobilenet_v3_small"

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
            "resnet18",
            "--require-gpu",
        ]
    )
    assert eval_args.require_gpu is True
    assert eval_args.arch == "resnet18"

    export_args = parser.parse_args(
        [
            "export",
            "--checkpoint",
            "/tmp/model.pt",
            "--out-dir",
            "/tmp/export",
            "--format",
            "onnx",
        ]
    )
    assert export_args.format == ["onnx"]
