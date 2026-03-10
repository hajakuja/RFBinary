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
            "vgg-binary",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--require-gpu",
        ]
    )
    assert train_args.require_gpu is True

    eval_args = parser.parse_args(
        [
            "eval",
            "--dataset-dir",
            "/tmp/ds",
            "--out-dir",
            "/tmp/out",
            "--require-gpu",
        ]
    )
    assert eval_args.require_gpu is True
