import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rfbd.export import export_checkpoint, run_export
from rfbd.modeling import create_binary_model


def test_onnx_export_logits_parity_when_ort_available(tmp_path: Path):
    ort = pytest.importorskip("onnxruntime")

    arch = "vgg_small_gap"
    model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
    model.eval()

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "arch": arch,
            "state_dict": model.state_dict(),
            "threshold": 0.5,
            "preprocessing_contract": {
                "segment_ms": 20,
                "nfft": 1024,
                "noverlap": 120,
                "resize_h": 224,
                "resize_w": 224,
                "log_power": True,
                "normalize": True,
            },
        },
        ckpt_path,
    )

    out_dir = tmp_path / "export"
    run_export(
        argparse.Namespace(
            checkpoint=str(ckpt_path),
            out_dir=str(out_dir),
            arch=None,
            format=["onnx"],
            batch_size=2,
            onnx_opset=17,
            require_onnx=False,
        )
    )

    summary_path = out_dir / "model_vgg_small_gap_export_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "onnx" in summary.get("skipped", {}):
        pytest.skip(f"ONNX export unavailable: {summary['skipped']['onnx']}")

    onnx_path = Path(summary["exported"]["onnx"])
    x = np.random.default_rng(13).standard_normal((2, 1, 224, 224), dtype=np.float32)

    with torch.no_grad():
        pt_logits = model(torch.from_numpy(x)).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0].name
    onnx_logits = session.run(None, {inp: x})[0]

    assert np.allclose(pt_logits, onnx_logits, rtol=1e-3, atol=1e-3)


def test_export_checkpoint_static_batch_omits_dynamic_axes(tmp_path: Path, monkeypatch):
    arch = "vgg_small_gap"
    model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
    model.eval()

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "arch": arch,
            "state_dict": model.state_dict(),
            "threshold": 0.5,
            "preprocessing_contract": {
                "segment_ms": 20,
                "nfft": 1024,
                "noverlap": 120,
                "resize_h": 224,
                "resize_w": 224,
                "log_power": True,
                "normalize": True,
            },
        },
        ckpt_path,
    )

    captured = {}

    def fake_torch_onnx_export(_model, _dummy, out_path, **kwargs):
        captured["dynamic_axes"] = kwargs.get("dynamic_axes")
        Path(out_path).write_bytes(b"fake-onnx")

    monkeypatch.setattr("rfbd.export.torch.onnx.export", fake_torch_onnx_export)

    summary = export_checkpoint(
        checkpoint=ckpt_path,
        out_dir=tmp_path / "export",
        formats=["onnx"],
        batch_size=1,
        static_batch=True,
        require_onnx=True,
    )

    assert captured["dynamic_axes"] is None
    assert summary["export_contract"]["onnx_dynamic_batch"] is False
    assert summary["export_contract"]["input_shape"] == [1, 1, 224, 224]
    assert summary["export_contract"]["output_shape"] == [1, 2]
