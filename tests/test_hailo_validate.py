import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rfbd.hailo import (
    DEFAULT_FALLBACK_MODEL_ID,
    _default_host_contract,
    _model_script_lines,
    _prepare_hailo_workspace,
    _validate_single_target,
    build_hailo_calibration_set,
    measure_compute_spec_feature_stats,
    run_hailo_compile,
)


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


def _write_dataset_shard(path: Path, *, count_per_class: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = count_per_class * 2
    feat = np.zeros((total, 224, 224), dtype=np.float32)
    feat[:count_per_class] = 0.1
    feat[count_per_class:] = 0.9
    y = np.array([0] * count_per_class + [1] * count_per_class, dtype=np.int64)
    source = np.array(["demo"] * total, dtype="<U64")
    capture = np.array([f"cap_{i}" for i in range(total)], dtype="<U128")
    session = np.array(["s01"] * total, dtype="<U128")
    np.savez_compressed(path, feat=feat, y=y, source_domain=source, capture_id=capture, session_id=session)


def _fake_export_checkpoint(**kwargs):
    checkpoint_path = Path(kwargs["checkpoint"])
    out_dir = Path(kwargs["out_dir"])
    arch = str(kwargs["arch"])
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{checkpoint_path.stem}_{arch}.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    summary = {
        "checkpoint": str(checkpoint_path),
        "arch": arch,
        "threshold": float(torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("threshold", 0.5)),
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
    summary_path = out_dir / f"{checkpoint_path.stem}_{arch}_export_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary


@pytest.mark.parametrize("arch", ["vgg13", "vgg16"])
def test_measure_compute_spec_feature_stats_and_vgg_family_host_contract(tmp_path: Path, arch: str):
    dataset_dir = tmp_path / "dataset"
    _write_dataset_shard(dataset_dir / "binary_all_v1_shard_00000.npz", count_per_class=4)

    dataset_stats = {
        "min": 0.1,
        "max": 0.9,
    }
    runtime_contract = _default_host_contract(
        type("Spec", (), {"arch": arch})(),
        dataset_stats=dataset_stats,
    )
    output_path = tmp_path / "hailo_calibration" / f"{arch}_uint8.npy"
    calibration_npy, metadata_path = build_hailo_calibration_set(
        dataset_dir=dataset_dir,
        output_path=output_path,
        max_samples=6,
        seed=7,
        runtime_contract=runtime_contract,
    )
    calibration = np.load(calibration_npy)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    compute_spec_stats = measure_compute_spec_feature_stats(
        preprocessing_contract={
            "segment_ms": 20,
            "nfft": 128,
            "noverlap": 16,
            "resize_h": 224,
            "resize_w": 224,
            "log_power": True,
            "normalize": True,
        },
        sample_rate_sps=12_800.0,
        seed=7,
    )
    script_lines = _model_script_lines(type("Spec", (), {"arch": arch})(), runtime_contract)

    assert calibration.shape == (6, 224, 224, 1)
    assert calibration.dtype == np.uint8
    assert metadata["host_contract"]["host_input_dtype"] == "uint8"
    assert metadata["host_domain_stats"]["max"] > 200.0
    assert metadata["host_domain_stats"]["min"] >= 0.0
    assert metadata["host_domain_stats"]["max"] > metadata["host_domain_stats"]["min"]
    assert compute_spec_stats["aggregate_stats"]["shape"] == [6, 224, 224]
    assert any("normalization" in line for line in script_lines)
    assert "performance_param(compiler_optimization_level=max)" in script_lines


def test_validate_single_target_flags_flat_output_and_updates_manifest(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "data/experiments/g2_edge_suite/models/vgg13/vgg13_binary.pt"
    dataset_shard = repo_root / "data/datasets/binary_all_v1/binary_all_v1_shard_00000.npz"
    _write_checkpoint(checkpoint, arch="vgg13", threshold=0.77)
    _write_dataset_shard(dataset_shard, count_per_class=8)

    monkeypatch.setattr("rfbd.hailo.export_checkpoint", _fake_export_checkpoint)
    monkeypatch.setattr(
        "rfbd.hailo.measure_compute_spec_feature_stats",
        lambda **kwargs: {
            "sample_rate_sps": 20_971_520.0,
            "segment_samples": 419430,
            "aggregate_stats": {"shape": [6, 224, 224], "dtype": "float32", "count": 301056, "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2},
            "per_probe_stats": {},
            "feature_config": kwargs["preprocessing_contract"],
        },
    )

    workspace = _prepare_hailo_workspace(
        repo_root=repo_root,
        targets=("hailo8",),
        include_strict_far=False,
        onnx_opset=17,
        skip_existing_exports=False,
        compiler_version=None,
        hailort_version=None,
    )
    spec = workspace.specs[0]
    export_summary = workspace.export_summaries[spec.model_id]

    def fake_pytorch(_spec, frames):
        means = np.asarray(frames, dtype=np.float32).reshape(len(frames), -1).mean(axis=1)
        return np.stack([-means, means], axis=1).astype(np.float32)

    def fake_onnx(_onnx_path, frames):
        return fake_pytorch(spec, frames)

    def fake_hailo_emulation_stage(*, har_path, context_name, host_frames):
        _ = har_path, context_name, host_frames
        count = len(host_frames)
        return np.tile(np.array([[-0.58229166, 0.60577118]], dtype=np.float32), (count, 1))

    monkeypatch.setattr("rfbd.hailo._run_pytorch_stage", fake_pytorch)
    monkeypatch.setattr("rfbd.hailo._run_onnx_stage", fake_onnx)
    monkeypatch.setattr("rfbd.hailo._run_hailo_emulation_stage", fake_hailo_emulation_stage)
    monkeypatch.setattr(
        "rfbd.hailo._parse_performance_report",
        lambda spec, target_arch: {
            "compiler_context_count": 15,
            "multi_context": True,
            "compiler_estimated_fps": 25.5,
            "compiler_estimated_latency_ms": 39.2,
            "inter_context_bandwidth_mbps": 123.0,
            "target_latency_ms": 50.0,
        },
    )

    report_path = _validate_single_target(
        spec=spec,
        target_arch="hailo8",
        export_summary=export_summary,
        workspace=workspace,
        dataset_dir=repo_root / "data/datasets/binary_all_v1",
        real_probe_samples=8,
        seed=13,
        runtime_metrics_json=None,
        hardware_results_json=None,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (
            repo_root
            / "data/experiments/g2_edge_suite/exports/vgg13/hailo/vgg13_binary.hailo8.manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert report["flat_output"]["failed"] is True
    assert report["deployable"] is False
    assert report["deployment_decision"] == "rejected_quality"
    assert report["fallback_recommendation"] == DEFAULT_FALLBACK_MODEL_ID
    assert report["gates"]["manifest_contract"] is True
    assert report["observed_input_stats"]["dataset_feature_stats"]["min"] == pytest.approx(0.1)
    assert report["observed_input_stats"]["real_host_stats"]["dtype"] == "uint8"
    assert manifest["host_input_dtype"] == "uint8"
    assert manifest["deployable"] is False
    assert manifest["deployment_decision"] == "rejected_quality"
    assert manifest["validation_report"].endswith(".validation.json")
    assert manifest["observed_input_stats"]["real_host_stats"]["dtype"] == "uint8"


def test_validate_single_target_uses_compiler_latency_for_non_vgg(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "data/experiments/g2_edge_suite/models/resnet34/resnet34_binary.pt"
    dataset_shard = repo_root / "data/datasets/binary_all_v1/binary_all_v1_shard_00000.npz"
    _write_checkpoint(checkpoint, arch="resnet34", threshold=0.69)
    _write_dataset_shard(dataset_shard, count_per_class=8)

    monkeypatch.setattr("rfbd.hailo.export_checkpoint", _fake_export_checkpoint)
    monkeypatch.setattr(
        "rfbd.hailo.measure_compute_spec_feature_stats",
        lambda **kwargs: {
            "sample_rate_sps": 20_971_520.0,
            "segment_samples": 419430,
            "aggregate_stats": {"shape": [6, 224, 224], "dtype": "float32", "count": 301056, "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2},
            "per_probe_stats": {},
            "feature_config": kwargs["preprocessing_contract"],
        },
    )

    workspace = _prepare_hailo_workspace(
        repo_root=repo_root,
        targets=("hailo8",),
        include_strict_far=False,
        onnx_opset=17,
        skip_existing_exports=False,
        compiler_version=None,
        hailort_version=None,
    )
    spec = workspace.specs[0]
    export_summary = workspace.export_summaries[spec.model_id]

    def fake_stage(_spec_or_path, frames):
        means = np.asarray(frames, dtype=np.float32).reshape(len(frames), -1).mean(axis=1)
        return np.stack([-means, means], axis=1).astype(np.float32)

    def fake_hailo_emulation_stage(*, har_path, context_name, host_frames):
        _ = har_path, context_name
        reconstructed = np.asarray(host_frames, dtype=np.float32) / 255.0
        means = reconstructed.reshape(len(host_frames), -1).mean(axis=1)
        return np.stack([-means, means], axis=1).astype(np.float32)

    monkeypatch.setattr("rfbd.hailo._run_pytorch_stage", lambda spec, frames: fake_stage(spec, frames))
    monkeypatch.setattr("rfbd.hailo._run_onnx_stage", lambda onnx_path, frames: fake_stage(onnx_path, frames))
    monkeypatch.setattr("rfbd.hailo._run_hailo_emulation_stage", fake_hailo_emulation_stage)
    monkeypatch.setattr(
        "rfbd.hailo._parse_performance_report",
        lambda spec, target_arch: {
            "compiler_context_count": 1,
            "multi_context": False,
            "compiler_estimated_fps": 25.5,
            "compiler_estimated_latency_ms": 39.2,
            "inter_context_bandwidth_mbps": 12.0,
            "target_latency_ms": 50.0,
        },
    )

    report_path = _validate_single_target(
        spec=spec,
        target_arch="hailo8",
        export_summary=export_summary,
        workspace=workspace,
        dataset_dir=repo_root / "data/datasets/binary_all_v1",
        real_probe_samples=8,
        seed=13,
        runtime_metrics_json=None,
        hardware_results_json=None,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (
            repo_root
            / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert report["flat_output"]["failed"] is False
    assert report["deployable"] is True
    assert report["deployment_decision"] == "accepted_provisional_hailo"
    assert report["fallback_recommendation"] == DEFAULT_FALLBACK_MODEL_ID
    assert report["performance"]["latency_source"] == "compiler_estimate"
    assert report["performance"]["latency_is_provisional"] is True
    assert report["performance"]["selected_latency_ms"] == pytest.approx(39.2)
    assert report["gates"]["hardware_latency_confirmed"] is False
    assert manifest["host_input_dtype"] == "uint8"
    assert manifest["host_input_quantization"]["type"] == "linear_uint8"
    assert manifest["hailo_input_normalization"]["enabled"] is True
    assert manifest["deployable"] is True
    assert manifest["deployment_decision"] == "accepted_provisional_hailo"


def test_hailo_compile_skip_existing_resnet34_reuses_existing_validation(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "data/experiments/g2_edge_suite/models/resnet34/resnet34_binary.pt"
    dataset_shard = repo_root / "data/datasets/binary_all_v1/binary_all_v1_shard_00000.npz"
    hef_path = (
        repo_root
        / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.hef"
    )
    validation_report_path = (
        repo_root
        / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.validation.json"
    )
    _write_checkpoint(checkpoint, arch="resnet34", threshold=0.77)
    _write_dataset_shard(dataset_shard, count_per_class=8)
    hef_path.parent.mkdir(parents=True, exist_ok=True)
    spec_stub = type("Spec", (), {"arch": "resnet34"})()
    runtime_contract = _default_host_contract(spec_stub, dataset_stats={"min": 0.1, "max": 0.9})
    model_script_path = hef_path.with_suffix(".alls")
    model_script_path.write_text("\n".join(_model_script_lines(spec_stub, runtime_contract)) + "\n", encoding="utf-8")
    hef_path.write_bytes(b"hef-bytes")
    validation_report_path.write_text(
        json.dumps({"deployable": False, "runtime_contract": runtime_contract}),
        encoding="utf-8",
    )

    monkeypatch.setattr("rfbd.hailo.export_checkpoint", _fake_export_checkpoint)
    monkeypatch.setattr("rfbd.hailo._resolve_tool_binary", lambda explicit, candidates: explicit or "/fake/hailo")
    monkeypatch.setattr("rfbd.hailo._detect_tool_versions", lambda hailo_bin, hailortcli_bin: ("Hailo DFC", None))

    calls = {"count": 0}

    def fake_validate_single_target(**kwargs):
        calls["count"] += 1
        report_path = (
            repo_root
            / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.validation.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"deployable": False}), encoding="utf-8")
        manifest_path = (
            repo_root
            / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_report"] = "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.validation.json"
        manifest["deployable"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return report_path

    monkeypatch.setattr("rfbd.hailo._validate_single_target", fake_validate_single_target)

    run_hailo_compile(
        argparse.Namespace(
            repo_root=str(repo_root),
            target=["hailo8"],
            onnx_opset=17,
            include_strict_far=False,
            model_id=["resnet34_binary"],
            calibration_dataset_dir="data/datasets/binary_all_v1",
            calibration_npy=None,
            calibration_samples=8,
            real_probe_samples=8,
            seed=13,
            hailo_bin="/fake/hailo",
            hailortcli_bin=None,
            runtime_metrics_json=None,
            hardware_results_json=None,
            skip_existing=True,
            keep_going=False,
        )
    )

    assert validation_report_path.exists()
    assert validation_report_path.read_text(encoding="utf-8") == json.dumps(
        {"deployable": False, "runtime_contract": runtime_contract}
    )
    assert calls["count"] == 0


def test_hailo_compile_skip_existing_resnet34_backfills_missing_validation(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "data/experiments/g2_edge_suite/models/resnet34/resnet34_binary.pt"
    dataset_shard = repo_root / "data/datasets/binary_all_v1/binary_all_v1_shard_00000.npz"
    hef_path = (
        repo_root
        / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.hef"
    )
    _write_checkpoint(checkpoint, arch="resnet34", threshold=0.77)
    _write_dataset_shard(dataset_shard, count_per_class=8)
    hef_path.parent.mkdir(parents=True, exist_ok=True)
    spec_stub = type("Spec", (), {"arch": "resnet34"})()
    runtime_contract = _default_host_contract(spec_stub, dataset_stats={"min": 0.1, "max": 0.9})
    model_script_path = hef_path.with_suffix(".alls")
    model_script_path.write_text("\n".join(_model_script_lines(spec_stub, runtime_contract)) + "\n", encoding="utf-8")
    hef_path.write_bytes(b"hef-bytes")

    monkeypatch.setattr("rfbd.hailo.export_checkpoint", _fake_export_checkpoint)
    monkeypatch.setattr("rfbd.hailo._resolve_tool_binary", lambda explicit, candidates: explicit or "/fake/hailo")
    monkeypatch.setattr("rfbd.hailo._detect_tool_versions", lambda hailo_bin, hailortcli_bin: ("Hailo DFC", None))

    calls = {"count": 0}

    def fake_validate_single_target(**kwargs):
        calls["count"] += 1
        report_path = (
            repo_root
            / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.validation.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"deployable": False}), encoding="utf-8")
        manifest_path = (
            repo_root
            / "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_report"] = "data/experiments/g2_edge_suite/exports/resnet34/hailo/resnet34_binary.hailo8.validation.json"
        manifest["deployable"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return report_path

    monkeypatch.setattr("rfbd.hailo._validate_single_target", fake_validate_single_target)

    run_hailo_compile(
        argparse.Namespace(
            repo_root=str(repo_root),
            target=["hailo8"],
            onnx_opset=17,
            include_strict_far=False,
            model_id=["resnet34_binary"],
            calibration_dataset_dir="data/datasets/binary_all_v1",
            calibration_npy=None,
            calibration_samples=8,
            real_probe_samples=8,
            seed=13,
            hailo_bin="/fake/hailo",
            hailortcli_bin=None,
            runtime_metrics_json=None,
            hardware_results_json=None,
            skip_existing=True,
            keep_going=False,
        )
    )

    assert calls["count"] == 1
