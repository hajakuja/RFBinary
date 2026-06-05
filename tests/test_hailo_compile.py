import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rfbd.hailo import build_hailo_calibration_set, run_hailo_compile


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


def _write_constant_dataset_shard(path: Path, *, class_values: dict[int, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    labels = []
    for cls, values in class_values.items():
        for value in values:
            frames.append(np.full((224, 224), float(value), dtype=np.float32))
            labels.append(int(cls))
    feat = np.stack(frames, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    source = np.array(["demo"] * len(y), dtype="<U64")
    capture = np.array([f"cap_{i}" for i in range(len(y))], dtype="<U128")
    session = np.array(["s01"] * len(y), dtype="<U128")
    np.savez_compressed(path, feat=feat, y=y, source_domain=source, capture_id=capture, session_id=session)


def test_build_hailo_calibration_set_writes_balanced_nhwc_array(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _write_dataset_shard(dataset_dir / "binary_all_v1_shard_00000.npz", count_per_class=4)

    output_path = tmp_path / "hailo_calibration" / "rfbd_calibration_6.npy"
    calibration_npy, metadata_path = build_hailo_calibration_set(
        dataset_dir=dataset_dir,
        output_path=output_path,
        max_samples=6,
        seed=7,
    )

    calib = np.load(calibration_npy)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert calib.shape == (6, 224, 224, 1)
    assert calib.dtype == np.float32
    assert metadata["sample_count"] == 6
    assert metadata["selected_class_counts"] == {"0": 3, "1": 3} or metadata["selected_class_counts"] == {0: 3, 1: 3}


def test_build_hailo_calibration_set_samples_high_range_late_shards(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _write_constant_dataset_shard(
        dataset_dir / "binary_all_v1_shard_00000.npz",
        class_values={0: [0.001, 0.002, 0.003, 0.004], 1: [0.001, 0.002, 0.003, 0.004]},
    )
    _write_constant_dataset_shard(
        dataset_dir / "binary_all_v1_shard_00001.npz",
        class_values={0: [10.0, 11.0, 12.0, 13.0], 1: [20.0, 21.0, 22.0, 23.0]},
    )

    output_path = tmp_path / "hailo_calibration" / "rfbd_calibration_4.npy"
    calibration_npy, metadata_path = build_hailo_calibration_set(
        dataset_dir=dataset_dir,
        output_path=output_path,
        max_samples=4,
        seed=7,
    )

    calib = np.load(calibration_npy)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert calib.shape == (4, 224, 224, 1)
    assert float(calib.max()) >= 20.0
    assert metadata["selected_class_counts"] == {"0": 2, "1": 2} or metadata["selected_class_counts"] == {0: 2, 1: 2}


def test_hailo_compile_generates_vgg_small_gap_hef_and_updates_manifest(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "data/experiments/g2_edge_suite/models/vgg_small_gap/vgg_small_gap_binary.pt"
    _write_checkpoint(checkpoint, arch="vgg_small_gap", threshold=0.69)
    _write_dataset_shard(
        repo_root / "data/datasets/binary_all_v1/binary_all_v1_shard_00000.npz",
        count_per_class=8,
    )

    def fake_export_checkpoint(**kwargs):
        checkpoint_path = Path(kwargs["checkpoint"])
        out_dir = Path(kwargs["out_dir"])
        arch = str(kwargs["arch"])
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir / f"{checkpoint_path.stem}_{arch}.onnx"
        onnx_path.write_bytes(b"fake-onnx")
        summary = {
            "checkpoint": str(checkpoint_path),
            "arch": arch,
            "threshold": 0.69,
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

    def fake_resolve_tool_binary(explicit, candidates):
        return explicit or str(tmp_path / "fake-hailo")

    def fake_detect_tool_versions(hailo_bin, hailortcli_bin):
        return ("Hailo Dataflow Compiler v3.33.0", None)

    def fake_run_logged_command(command, *, log_path: Path, workdir: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")

        subcommand = command[1]
        if subcommand == "parser":
            har_path = Path(command[command.index("--har-path") + 1])
            har_path.write_bytes(b"native-har")
            augmented_path = Path(command[command.index("--augmented-path") + 1])
            augmented_path.write_text("augmented", encoding="utf-8")
            parsing_report_path = Path(command[command.index("--parsing-report-path") + 1])
            parsing_report_path.write_text("{}", encoding="utf-8")
        elif subcommand == "optimize":
            optimized_har_path = Path(command[command.index("--output-har-path") + 1])
            optimized_har_path.write_bytes(b"optimized-har")
        elif subcommand == "compiler":
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            optimized_har_path = Path(command[2])
            generated_hef = output_dir / optimized_har_path.with_suffix(".hef").name
            generated_hef.write_bytes(b"hef-bytes")
            compiled_har_path = Path(command[command.index("--output-har-path") + 1])
            compiled_har_path.write_bytes(b"compiled-har")
        else:
            raise AssertionError(f"Unexpected Hailo subcommand: {subcommand}")

    def fake_validate_single_target(**kwargs):
        spec = kwargs["spec"]
        target_arch = kwargs["target_arch"]
        report_path = (
            repo_root
            / f"data/experiments/g2_edge_suite/exports/{spec.arch}/hailo/{spec.model_id}.{target_arch}.validation.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"deployable": False}), encoding="utf-8")
        manifest_path = (
            repo_root
            / f"data/experiments/g2_edge_suite/exports/{spec.arch}/hailo/{spec.model_id}.{target_arch}.manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_report"] = (
            f"data/experiments/g2_edge_suite/exports/{spec.arch}/hailo/{spec.model_id}.{target_arch}.validation.json"
        )
        manifest["deployable"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return report_path

    monkeypatch.setattr("rfbd.hailo.export_checkpoint", fake_export_checkpoint)
    monkeypatch.setattr("rfbd.hailo._resolve_tool_binary", fake_resolve_tool_binary)
    monkeypatch.setattr("rfbd.hailo._detect_tool_versions", fake_detect_tool_versions)
    monkeypatch.setattr("rfbd.hailo._run_logged_command", fake_run_logged_command)
    monkeypatch.setattr("rfbd.hailo._validate_single_target", fake_validate_single_target)

    run_hailo_compile(
        argparse.Namespace(
            repo_root=str(repo_root),
            target=["hailo8"],
            onnx_opset=17,
            include_strict_far=False,
            model_id=[],
            calibration_dataset_dir="data/datasets/binary_all_v1",
            calibration_npy=None,
            calibration_samples=8,
            real_probe_samples=8,
            seed=13,
            hailo_bin="/fake/hailo",
            hailortcli_bin=None,
            runtime_metrics_json=None,
            hardware_results_json=None,
            skip_existing=False,
            keep_going=False,
        )
    )

    hef_path = (
        repo_root
        / "data/experiments/g2_edge_suite/exports/vgg_small_gap/hailo/"
        / "vgg_small_gap_binary.hailo8.hef"
    )
    manifest_path = (
        repo_root
        / "data/experiments/g2_edge_suite/exports/vgg_small_gap/hailo/"
        / "vgg_small_gap_binary.hailo8.manifest.json"
    )
    commands_path = (
        repo_root
        / "data/experiments/g2_edge_suite/exports/vgg_small_gap/hailo/"
        / "vgg_small_gap_binary.hailo8.commands.json"
    )

    assert hef_path.exists()
    assert commands_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = json.loads(commands_path.read_text(encoding="utf-8"))["commands"]
    optimize_command = next(item["command"] for item in commands if item["stage"] == "optimize")
    assert manifest["compiled"] is True
    assert manifest["compiler_version"] == "Hailo Dataflow Compiler v3.33.0"
    assert manifest["calibration_npy"] == "data/experiments/g2_edge_suite/hailo_calibration/rfbd_linear_uint8_calibration_8.npy"
    assert manifest["host_input_dtype"] == "uint8"
    assert manifest["host_input_layout"] == "NHWC"
    assert manifest["host_input_quantization"]["type"] == "linear_uint8"
    assert manifest["hailo_input_normalization"]["enabled"] is True
    assert manifest["model_script"].endswith(".alls")
    assert "--model-script" in optimize_command
    assert manifest["deployable"] is False
    assert manifest["validation_report"].endswith(".validation.json")
    assert manifest["native_har"].endswith(".native.har")
    assert manifest["optimized_har"].endswith(".optimized.har")
    assert manifest["compiled_har"].endswith(".compiled.har")
    assert manifest["parser_log"].endswith(".parser.log")
    assert manifest["optimize_log"].endswith(".optimize.log")
    assert manifest["compiler_log"].endswith(".compiler.log")
