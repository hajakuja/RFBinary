import argparse
import json
from pathlib import Path

import torch

from rfbd.hailo import run_hailo_prepare


def _write_checkpoint(path: Path, arch: str, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "state_dict": {},
            "threshold": threshold,
            "preprocessing_contract": {
                "segment_ms": 20,
                "nfft": 1024,
                "noverlap": 120,
                "resize_h": 224,
                "resize_w": 224,
                "log_power": True,
                "normalize": True,
            },
            "label_contract": {
                "0": "no_drone",
                "1": "drone",
            },
        },
        path,
    )


def test_hailo_prepare_writes_exports_manifests_and_inventory(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    base_ckpt = repo_root / "data/experiments/g2_edge_suite/models/vgg_small_gap/vgg_small_gap_binary.pt"
    strict_ckpt = (
        repo_root
        / "data/experiments/g2_edge_suite/strict_far/models/resnet34/resnet34_binary_far003.pt"
    )
    _write_checkpoint(base_ckpt, arch="vgg_small_gap", threshold=0.75)
    _write_checkpoint(strict_ckpt, arch="resnet34", threshold=0.8)

    def fake_export_checkpoint(**kwargs):
        checkpoint = Path(kwargs["checkpoint"])
        out_dir = Path(kwargs["out_dir"])
        arch = str(kwargs["arch"])
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir / f"{checkpoint.stem}_{arch}.onnx"
        onnx_path.write_bytes(b"fake-onnx")
        summary = {
            "checkpoint": str(checkpoint),
            "arch": arch,
            "threshold": float(torch.load(checkpoint, map_location="cpu", weights_only=False).get("threshold", 0.5)),
            "preprocessing_contract": {
                "segment_ms": 20,
                "nfft": 1024,
                "noverlap": 120,
                "resize_h": 224,
                "resize_w": 224,
                "log_power": True,
                "normalize": True,
            },
            "label_contract": {
                "0": "no_drone",
                "1": "drone",
            },
            "export_contract": {
                "input_name": "input",
                "input_shape": [1, 1, 224, 224],
                "output_name": "logits",
                "output_shape": [1, 2],
                "onnx_dynamic_batch": False,
            },
            "exported": {
                "onnx": str(onnx_path),
            },
            "skipped": {},
        }
        summary_path = out_dir / f"{checkpoint.stem}_{arch}_export_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr("rfbd.hailo.export_checkpoint", fake_export_checkpoint)

    run_hailo_prepare(
        argparse.Namespace(
            repo_root=str(repo_root),
            target=["hailo8"],
            onnx_opset=17,
            skip_existing=False,
            include_strict_far=True,
            compiler_version=None,
            hailort_version=None,
        )
    )

    base_manifest = (
        repo_root
        / "data/experiments/g2_edge_suite/exports/vgg_small_gap/hailo/vgg_small_gap_binary.hailo8.manifest.json"
    )
    strict_manifest = (
        repo_root
        / "data/experiments/g2_edge_suite/strict_far/exports/resnet34/hailo/"
        / "resnet34_binary_far003.hailo8.manifest.json"
    )
    inventory_path = repo_root / "data/experiments/g2_edge_suite/hailo_compilation_inventory.json"
    calibration_readme = repo_root / "data/experiments/g2_edge_suite/hailo_calibration/README.md"

    assert base_manifest.exists()
    assert strict_manifest.exists()
    assert inventory_path.exists()
    assert calibration_readme.exists()

    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    assert base_payload["model_id"] == "vgg_small_gap_binary"
    assert base_payload["variant"] == "base"
    assert base_payload["target_arch"] == "hailo8"
    assert base_payload["hef_path"] == "data/experiments/g2_edge_suite/exports/vgg_small_gap/hailo/vgg_small_gap_binary.hailo8.hef"
    assert base_payload["source_onnx"] == "data/experiments/g2_edge_suite/exports/vgg_small_gap/vgg_small_gap_binary_vgg_small_gap.onnx"

    strict_payload = json.loads(strict_manifest.read_text(encoding="utf-8"))
    assert strict_payload["variant"] == "strict_far"
    assert strict_payload["target_arch"] == "hailo8"
    assert strict_payload["checkpoint_path"].startswith("data/experiments/g2_edge_suite/strict_far/models/")

    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory_payload["checkpoint_count"] == 2
    assert inventory_payload["targets"] == ["hailo8"]
    assert inventory_payload["unique_arches"] == ["resnet34", "vgg_small_gap"]


def test_hailo_prepare_skip_existing_reuses_summary(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "data/experiments/g2_edge_suite/models/resnet34/resnet34_binary.pt"
    _write_checkpoint(checkpoint, arch="resnet34", threshold=0.82)

    export_dir = repo_root / "data/experiments/g2_edge_suite/exports/resnet34"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "resnet34_binary_resnet34.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    summary_path = export_dir / "resnet34_binary_resnet34_export_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "arch": "resnet34",
                "threshold": 0.82,
                "preprocessing_contract": {
                    "segment_ms": 20,
                    "nfft": 1024,
                    "noverlap": 120,
                    "resize_h": 224,
                    "resize_w": 224,
                    "log_power": True,
                    "normalize": True,
                },
                "label_contract": {
                    "0": "no_drone",
                    "1": "drone",
                },
                "export_contract": {
                    "input_name": "input",
                    "input_shape": [1, 1, 224, 224],
                    "output_name": "logits",
                    "output_shape": [1, 2],
                    "onnx_dynamic_batch": False,
                },
                "exported": {
                    "onnx": str(onnx_path),
                },
                "skipped": {},
            }
        ),
        encoding="utf-8",
    )

    def fail_export_checkpoint(**kwargs):
        raise AssertionError("export_checkpoint should not run when --skip-existing can reuse the export summary")

    monkeypatch.setattr("rfbd.hailo.export_checkpoint", fail_export_checkpoint)

    run_hailo_prepare(
        argparse.Namespace(
            repo_root=str(repo_root),
            target=["hailo8"],
            onnx_opset=17,
            skip_existing=True,
            include_strict_far=False,
            compiler_version=None,
            hailort_version=None,
        )
    )

    manifest_path = repo_root / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.manifest.json"
    assert manifest_path.exists()
