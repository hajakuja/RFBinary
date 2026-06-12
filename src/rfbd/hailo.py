"""Hailo preparation, compilation, and validation utilities."""

from __future__ import annotations

import argparse
import heapq
import importlib
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .contracts import FeatureConfig
from .export import export_checkpoint
from .features import compute_spec_feature, samples_per_segment
from .io import iter_npz_files, load_json, load_npz_shard, save_json
from .modeling import REPVGG_ARCHES


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = Path("data/experiments/g2_edge_suite")
BASE_MODEL_ROOT = SUITE_ROOT / "models"
BASE_EXPORT_ROOT = SUITE_ROOT / "exports"
STRICT_MODEL_ROOT = SUITE_ROOT / "strict_far" / "models"
STRICT_EXPORT_ROOT = SUITE_ROOT / "strict_far" / "exports"
CALIBRATION_ROOT = SUITE_ROOT / "hailo_calibration"
DEFAULT_CALIBRATION_DATASET = Path("data/datasets/binary_all_v1")
SUPPORTED_TARGET_ARCHS = ("hailo8", "hailo8l")
DEFAULT_TARGET_ARCHS = ("hailo8",)
DEFAULT_CALIBRATION_SAMPLES = 1024
DEFAULT_VGG_FAMILY_CALIBRATION_SAMPLES = 1024
DEFAULT_VGG_FAMILY_RANGE_SAMPLES = 256
DEFAULT_VALIDATION_REAL_SAMPLES = 32
DEFAULT_SYNTHETIC_PROBE_NAMES = (
    "zeros",
    "ones",
    "mid_gray",
    "checkerboard",
    "random_0",
    "random_1",
)
DEFAULT_HAILO_LATENCY_TARGET_MS = 50.0
DEFAULT_FALLBACK_MODEL_ID = "shufflenet_v2_x1_0_binary"
VGG_FAMILY_ARCHES = frozenset({"vgg13", "vgg16"})
FLAT_LOGIT_SPREAD_EPS = 1e-3
FLAT_P_DRONE_VARIANCE_EPS = 1e-6
ORDERING_SPEARMAN_MIN = 0.9
HAILO_EMULATION_BATCH_SIZE = 8

CALIBRATION_README = """# Hailo Calibration Assets

Place representative `224x224` single-channel PSD frames for Hailo quantization here.

Recommended contents:
- `.npy` calibration frames exported from existing `RFBinaryDetect` preprocessing output
- a small metadata note describing frame sources and generation date
- optional helper script to regenerate the calibration set from feature data
"""


@dataclass(frozen=True)
class HailoCheckpointSpec:
    model_id: str
    arch: str
    variant: str
    checkpoint_path: Path
    export_dir: Path
    hailo_dir: Path


@dataclass(frozen=True)
class HailoWorkspace:
    repo_root: Path
    specs: Sequence[HailoCheckpointSpec]
    export_summaries: Dict[str, Dict[str, object]]
    manifest_paths: Dict[tuple[str, str], Path]
    calibration_dir: Path
    inventory_path: Path
    targets: Sequence[str]


def _repo_relative(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _summary_path(spec: HailoCheckpointSpec) -> Path:
    return spec.export_dir / f"{spec.checkpoint_path.stem}_{spec.arch}_export_summary.json"


def _manifest_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.manifest.json"


def _hef_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.hef"


def _native_har_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.native.har"


def _optimized_har_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.optimized.har"


def _compiled_har_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.compiled.har"


def _augmented_onnx_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.augmented.onnx"


def _parsing_report_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.parsing_report.json"


def _parser_log_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.parser.log"


def _optimize_log_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.optimize.log"


def _compiler_log_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.compiler.log"


def _commands_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.commands.json"


def _compile_output_dir(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.compiler_output"


def _model_script_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.alls"


def _probe_corpus_path(spec: HailoCheckpointSpec) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.probe_corpus.npz"


def _probe_corpus_metadata_path(spec: HailoCheckpointSpec) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.probe_corpus.metadata.json"


def _validation_report_path(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}.validation.json"


def _runtime_contract_from_payload(payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "host_input_dtype": payload.get("host_input_dtype", "float32"),
        "host_input_layout": payload.get("host_input_layout", "NHWC"),
        "host_input_range": payload.get("host_input_range", {"min": 0.0, "max": 1.0}),
        "host_input_quantization": payload.get(
            "host_input_quantization",
            {"type": "none", "source_min": 0.0, "source_max": 1.0},
        ),
        "hailo_input_normalization": payload.get(
            "hailo_input_normalization",
            {"enabled": False, "mean": [0.0], "std": [1.0]},
        ),
    }


def _runtime_contracts_match(left: Dict[str, object], right: Dict[str, object]) -> bool:
    return _runtime_contract_from_payload(left) == _runtime_contract_from_payload(right)


def _existing_model_script_matches(
    *,
    spec: HailoCheckpointSpec,
    target_arch: str,
    runtime_contract: Dict[str, object],
) -> bool:
    expected_lines = _model_script_lines(spec, runtime_contract)
    script_path = _model_script_path(spec, target_arch)
    if not expected_lines:
        return not script_path.exists()
    expected_text = "\n".join(expected_lines) + "\n"
    if not script_path.exists():
        return False
    return script_path.read_text(encoding="utf-8") == expected_text


def _existing_hef_is_reusable(
    *,
    spec: HailoCheckpointSpec,
    target_arch: str,
    runtime_contract: Dict[str, object],
) -> bool:
    hef_path = _hef_path(spec, target_arch)
    if not hef_path.exists():
        return False
    hef_mtime = hef_path.stat().st_mtime
    for dependency_path in (
        _optimized_har_path(spec, target_arch),
        _commands_path(spec, target_arch),
        _model_script_path(spec, target_arch),
    ):
        if dependency_path.exists() and dependency_path.stat().st_mtime > hef_mtime:
            return False
    return _existing_model_script_matches(spec=spec, target_arch=target_arch, runtime_contract=runtime_contract)


def _reusable_validation_report_path(
    spec: HailoCheckpointSpec,
    target_arch: str,
    runtime_contract: Dict[str, object],
) -> Path | None:
    report_path = _validation_report_path(spec, target_arch)
    if not report_path.exists():
        return None

    report_mtime = report_path.stat().st_mtime
    for artifact_path in (
        _hef_path(spec, target_arch),
        _compiled_har_path(spec, target_arch),
        _commands_path(spec, target_arch),
        _model_script_path(spec, target_arch),
    ):
        if artifact_path.exists() and artifact_path.stat().st_mtime > report_mtime:
            return None

    report = load_json(report_path, default={})
    report_contract = report.get("runtime_contract", {}) if isinstance(report, dict) else {}
    if not _runtime_contracts_match(runtime_contract, report_contract):
        return None
    return report_path


def _refresh_manifest_runtime_contract(manifest_path: Path, runtime_contract: Dict[str, object]) -> None:
    manifest = load_json(manifest_path, default={})
    manifest.update(_runtime_contract_from_payload(runtime_contract))
    save_json(manifest_path, manifest)


def _calibration_artifacts_match(
    *,
    output_path: Path,
    metadata_path: Path,
    runtime_contract: Dict[str, object],
    sample_count: int,
) -> bool:
    if not output_path.exists() or not metadata_path.exists():
        return False

    metadata = load_json(metadata_path, default={})
    if not isinstance(metadata, dict):
        return False

    actual_count = metadata.get("sample_count", metadata.get("requested_max_samples"))
    if actual_count is not None and int(actual_count) != int(sample_count):
        return False

    shape = metadata.get("shape")
    if isinstance(shape, list) and shape and int(shape[0]) != int(sample_count):
        return False

    host_contract = metadata.get("host_contract", {})
    if not _runtime_contracts_match(runtime_contract, host_contract if isinstance(host_contract, dict) else {}):
        return False

    quant = runtime_contract.get("host_input_quantization", {})
    if str(quant.get("type", "none")) == "linear_uint8":
        if metadata.get("dtype") != "uint8":
            return False
        host_stats = metadata.get("host_domain_stats", {})
        if not isinstance(host_stats, dict):
            return False
        host_min = host_stats.get("min")
        host_max = host_stats.get("max")
        if host_min is None or host_max is None:
            return False
        # A valid uint8 calibration set must exercise the upper host range; stale
        # VGG calibration files previously topped out around 6 and poisoned quantization.
        if float(host_min) < 0.0 or float(host_max) > 255.0 or float(host_max) < 200.0:
            return False

    return True


def _feature_stats_cache_path(repo_root: Path, dataset_dir: Path) -> Path:
    stem = dataset_dir.name or "dataset"
    return repo_root / CALIBRATION_ROOT / f"{stem}_feature_stats.json"


def _uses_vgg_family_contract(spec: HailoCheckpointSpec) -> bool:
    return spec.arch in VGG_FAMILY_ARCHES


def _linear_uint8_host_contract(*, source_min: float, source_max: float) -> Dict[str, object]:
    source_range = max(float(source_max) - float(source_min), 1e-12)
    quant_scale = 255.0 / source_range
    return {
        "host_input_dtype": "uint8",
        "host_input_layout": "NHWC",
        "host_input_range": {"min": 0.0, "max": 255.0},
        "host_input_quantization": {
            "type": "linear_uint8",
            "source_min": float(source_min),
            "source_max": float(source_max),
        },
        "hailo_input_normalization": {
            "enabled": True,
            "mean": [float(-float(source_min) * quant_scale)],
            "std": [float(quant_scale)],
            "reconstructs_original_scale": True,
        },
    }


def _vgg_family_calibration_path(repo_root: Path, spec: HailoCheckpointSpec, sample_count: int) -> Path:
    return repo_root / CALIBRATION_ROOT / f"{spec.model_id}_uint8_calibration_{sample_count}.npy"


def _ensure_hailo_suite_site_packages() -> None:
    roots = sorted(Path("/root/hailo_ai_sw_suite").glob("hailo_venv/lib/python*/site-packages"))
    for site_packages in reversed(roots):
        site_text = str(site_packages)
        if site_text in sys.path:
            sys.path.remove(site_text)
        sys.path.insert(0, site_text)


def _purge_hailo_conflicting_modules() -> None:
    prefixes = (
        "google.protobuf",
        "google._upb",
        "onnx",
        "onnxruntime",
    )
    for key in list(sys.modules):
        if key == "google" or any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[key]


def _optional_import_onnxruntime():
    _ensure_hailo_suite_site_packages()
    try:
        return importlib.import_module("onnxruntime")
    except Exception:
        try:
            return importlib.import_module("onnxruntime")
        except Exception:
            return None


def _optional_import_hailo_sdk() -> tuple[Any | None, Any | None]:
    _ensure_hailo_suite_site_packages()
    _purge_hailo_conflicting_modules()
    try:
        module = importlib.import_module("hailo_sdk_client")
        return module.ClientRunner, module.InferenceContext
    except Exception:
        try:
            module = importlib.import_module("hailo_sdk_client")
            return module.ClientRunner, module.InferenceContext
        except Exception:
            return None, None


def _write_calibration_readme(calibration_dir: Path) -> None:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    readme_path = calibration_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(CALIBRATION_README, encoding="utf-8")


def discover_hailo_checkpoints(repo_root: str | Path, include_strict_far: bool = True) -> List[HailoCheckpointSpec]:
    root = Path(repo_root)
    scan_roots: List[tuple[str, Path, Path]] = [
        ("base", BASE_MODEL_ROOT, BASE_EXPORT_ROOT),
    ]
    if include_strict_far:
        scan_roots.append(("strict_far", STRICT_MODEL_ROOT, STRICT_EXPORT_ROOT))

    specs: List[HailoCheckpointSpec] = []
    for variant, model_root_rel, export_root_rel in scan_roots:
        model_root = root / model_root_rel
        if not model_root.exists():
            continue

        for checkpoint_path in sorted(model_root.glob("*/*.pt")):
            arch = checkpoint_path.parent.name
            export_dir = root / export_root_rel / arch
            hailo_dir = export_dir / "hailo"
            specs.append(
                HailoCheckpointSpec(
                    model_id=checkpoint_path.stem,
                    arch=arch,
                    variant=variant,
                    checkpoint_path=checkpoint_path,
                    export_dir=export_dir,
                    hailo_dir=hailo_dir,
                )
            )

    if not specs:
        raise FileNotFoundError(f"No Hailo compilation checkpoints found under {root}")

    return specs


def _load_existing_export_summary(summary_path: Path) -> Dict[str, object]:
    if not summary_path.exists():
        return {}
    payload = load_json(summary_path)
    if not payload:
        return {}
    onnx_path = payload.get("exported", {}).get("onnx")
    if not onnx_path:
        return {}
    if not Path(str(onnx_path)).exists():
        return {}
    if payload.get("arch") in REPVGG_ARCHES:
        export_contract = payload.get("export_contract", {})
        expected_opset = int(export_contract.get("onnx_opset", 17)) if isinstance(export_contract, dict) else 17
        metadata = payload.get("onnx_metadata", {})
        if not isinstance(metadata, dict):
            return {}
        if metadata.get("actual_default_opset") != expected_opset:
            return {}
        if expected_opset < 18 and metadata.get("exporter") != "legacy_torchscript":
            return {}
    return payload


def _export_for_hailo(
    spec: HailoCheckpointSpec,
    *,
    skip_existing: bool,
    onnx_opset: int,
) -> Dict[str, object]:
    spec.export_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _summary_path(spec)
    if skip_existing:
        existing = _load_existing_export_summary(summary_path)
        if existing:
            return existing

    return export_checkpoint(
        checkpoint=spec.checkpoint_path,
        out_dir=spec.export_dir,
        arch=spec.arch,
        formats=["onnx"],
        batch_size=1,
        onnx_opset=onnx_opset,
        require_onnx=True,
        static_batch=True,
    )


def _array_stats(arr: np.ndarray) -> Dict[str, object]:
    x = np.asarray(arr)
    if x.size == 0:
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    x64 = x.astype(np.float64, copy=False)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "count": int(x.size),
        "min": float(x64.min()),
        "max": float(x64.max()),
        "mean": float(x64.mean()),
        "std": float(x64.std()),
    }


def _select_balanced_feature_frames(
    *,
    dataset_dir: str | Path,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    dataset_root = Path(dataset_dir)
    shards = iter_npz_files(dataset_root)
    if not shards:
        raise FileNotFoundError(f"No NPZ shards found under {dataset_root}")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    rng = np.random.default_rng(seed)
    target_take = {0: max_samples // 2, 1: max_samples - (max_samples // 2)}
    random_buffers: Dict[int, List[tuple[int, np.ndarray]]] = {0: [], 1: []}
    high_energy_heaps: Dict[int, List[tuple[float, int, np.ndarray]]] = {0: [], 1: []}
    class_seen = {0: 0, 1: 0}
    source_shards: List[str] = []
    serial = 0

    for shard_path in shards:
        shard = load_npz_shard(shard_path)
        feat = np.asarray(shard["feat"], dtype=np.float32)
        y = np.asarray(shard["y"], dtype=np.int64)
        if feat.ndim != 3:
            raise ValueError(f"Expected shard features shaped (N,H,W), got {feat.shape} in {shard_path}")

        source_shards.append(str(shard_path))
        for cls in (0, 1):
            cls_idx = np.flatnonzero(y == cls)
            if cls_idx.size == 0:
                continue
            cls_feat = feat[cls_idx]
            scores = np.max(np.abs(cls_feat.reshape(len(cls_feat), -1)), axis=1)
            random_cap = max(1, target_take[cls])
            high_cap = max(1, target_take[cls])
            for frame, score_value in zip(cls_feat, scores):
                frame_copy = np.asarray(frame, dtype=np.float32).copy()
                frame_serial = serial
                serial += 1
                class_seen[cls] += 1

                if len(random_buffers[cls]) < random_cap:
                    random_buffers[cls].append((frame_serial, frame_copy))
                else:
                    replace_at = int(rng.integers(0, class_seen[cls]))
                    if replace_at < random_cap:
                        random_buffers[cls][replace_at] = (frame_serial, frame_copy)

                heap_entry = (float(score_value), frame_serial, frame_copy)
                if len(high_energy_heaps[cls]) < high_cap:
                    heapq.heappush(high_energy_heaps[cls], heap_entry)
                elif float(score_value) > high_energy_heaps[cls][0][0]:
                    heapq.heapreplace(high_energy_heaps[cls], heap_entry)

    selected_frames: List[np.ndarray] = []
    selected_labels: List[np.ndarray] = []
    used_counts = {0: 0, 1: 0}

    for cls in (0, 1):
        target = target_take[cls]
        if class_seen[cls] < target:
            continue

        class_selected: List[np.ndarray] = []
        used_serials: set[int] = set()
        high_entries = sorted(high_energy_heaps[cls], key=lambda item: (item[0], item[1]), reverse=True)
        high_take = min(len(high_entries), max(1, target // 2))
        for _, frame_serial, frame in high_entries[:high_take]:
            if frame_serial not in used_serials:
                class_selected.append(frame)
                used_serials.add(frame_serial)

        random_order = rng.permutation(len(random_buffers[cls]))
        for idx in random_order:
            if len(class_selected) >= target:
                break
            frame_serial, frame = random_buffers[cls][int(idx)]
            if frame_serial in used_serials:
                continue
            class_selected.append(frame)
            used_serials.add(frame_serial)

        for _, frame_serial, frame in high_entries:
            if len(class_selected) >= target:
                break
            if frame_serial in used_serials:
                continue
            class_selected.append(frame)
            used_serials.add(frame_serial)

        take = min(len(class_selected), target)
        if take > 0:
            selected_frames.append(np.stack(class_selected[:take], axis=0))
            selected_labels.append(np.full((take,), cls, dtype=np.int64))
            used_counts[cls] = take

    chosen_count = sum(used_counts.values())
    shortfall = max_samples - chosen_count

    if shortfall > 0:
        raise ValueError(
            f"Unable to build balanced feature sample set of {max_samples} samples from {dataset_root}; "
            f"only gathered {max_samples - shortfall}"
        )

    frames = np.concatenate(selected_frames, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(selected_labels, axis=0).astype(np.int64, copy=False)
    perm = rng.permutation(len(frames))
    frames = frames[perm]
    labels = labels[perm]
    metadata = {
        "dataset_dir": str(dataset_root),
        "sample_count": int(len(frames)),
        "shape": list(frames.shape),
        "dtype": str(frames.dtype),
        "selected_class_counts": {str(k): int(v) for k, v in used_counts.items()},
        "source_shards": source_shards,
        "seed": int(seed),
    }
    return frames, labels, metadata


def _select_up_to_balanced_feature_frames(
    *,
    dataset_dir: str | Path,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    attempt = max(2, int(max_samples))
    last_error: Exception | None = None
    while attempt >= 2:
        try:
            return _select_balanced_feature_frames(dataset_dir=dataset_dir, max_samples=attempt, seed=seed)
        except ValueError as exc:
            last_error = exc
            next_attempt = attempt // 2
            if next_attempt == attempt:
                next_attempt -= 1
            attempt = max(0, next_attempt)
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to select representative feature frames from {dataset_dir}")


def measure_dataset_feature_stats(
    *,
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Dict[str, object]:
    dataset_root = Path(dataset_dir)
    output = Path(output_path) if output_path is not None else None
    if output is not None and output.exists() and not force:
        return load_json(output)

    shards = iter_npz_files(dataset_root)
    if not shards:
        raise FileNotFoundError(f"No NPZ shards found under {dataset_root}")

    global_min = np.inf
    global_max = -np.inf
    total_values = 0
    total_samples = 0
    sum_values = 0.0
    sum_sq_values = 0.0
    label_counts = {0: 0, 1: 0}
    source_shards: List[str] = []

    for shard_path in shards:
        shard = load_npz_shard(shard_path)
        feat = np.asarray(shard["feat"], dtype=np.float32)
        y = np.asarray(shard["y"], dtype=np.int64)
        if feat.ndim != 3:
            raise ValueError(f"Expected shard features shaped (N,H,W), got {feat.shape} in {shard_path}")

        source_shards.append(str(shard_path))
        feat64 = feat.astype(np.float64, copy=False)
        global_min = min(global_min, float(feat64.min()))
        global_max = max(global_max, float(feat64.max()))
        total_values += int(feat64.size)
        total_samples += int(feat64.shape[0])
        sum_values += float(feat64.sum())
        sum_sq_values += float(np.square(feat64).sum())
        for cls in (0, 1):
            label_counts[cls] += int(np.sum(y == cls))

    mean = sum_values / max(1, total_values)
    variance = max(0.0, (sum_sq_values / max(1, total_values)) - (mean * mean))
    payload = {
        "dataset_dir": str(dataset_root),
        "sample_count": int(total_samples),
        "value_count": int(total_values),
        "min": float(global_min),
        "max": float(global_max),
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "source_shards": source_shards,
    }
    if output is not None:
        save_json(output, payload)
    return payload


def _default_host_contract(spec: HailoCheckpointSpec, dataset_stats: Dict[str, object] | None = None) -> Dict[str, object]:
    if dataset_stats:
        source_min = min(0.0, float(dataset_stats.get("min", 0.0) or 0.0))
        source_max = max(float(dataset_stats.get("max", 1.0) or 1.0), 1.0)
        return _linear_uint8_host_contract(source_min=source_min, source_max=source_max)

    return {
        "host_input_dtype": "float32",
        "host_input_layout": "NHWC",
        "host_input_range": {"min": 0.0, "max": 1.0},
        "host_input_quantization": {"type": "none", "source_min": 0.0, "source_max": 1.0},
        "hailo_input_normalization": {"enabled": False, "mean": [0.0], "std": [1.0]},
    }


def _feature_config_from_contract(contract: Dict[str, object]) -> FeatureConfig:
    return FeatureConfig(
        segment_ms=int(contract.get("segment_ms", 20)),
        nfft=int(contract.get("nfft", 1024)),
        noverlap=int(contract.get("noverlap", 120)),
        resize_h=int(contract.get("resize_h", 224)),
        resize_w=int(contract.get("resize_w", 224)),
        log_power=bool(contract.get("log_power", True)),
        normalize=bool(contract.get("normalize", True)),
    )


def measure_compute_spec_feature_stats(
    *,
    preprocessing_contract: Dict[str, object],
    sample_rate_sps: float = 20_971_520.0,
    seed: int = 13,
) -> Dict[str, object]:
    cfg = _feature_config_from_contract(preprocessing_contract)
    seg_len = samples_per_segment(sample_rate_sps, cfg.segment_ms)
    t = np.arange(seg_len, dtype=np.float32) / float(sample_rate_sps)
    rng0 = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed + 1)

    signal_bank = {
        "zeros": np.zeros((seg_len,), dtype=np.float32),
        "ones": np.ones((seg_len,), dtype=np.float32),
        "sine_100khz": np.sin(2.0 * np.pi * 100_000.0 * t).astype(np.float32),
        "sine_1mhz": np.sin(2.0 * np.pi * 1_000_000.0 * t).astype(np.float32),
        "random_0": rng0.standard_normal(seg_len).astype(np.float32),
        "random_1": rng1.standard_normal(seg_len).astype(np.float32),
    }

    features: Dict[str, np.ndarray] = {}
    for name, segment in signal_bank.items():
        features[name] = compute_spec_feature(segment, sample_rate_sps, cfg)

    stacked = np.stack(list(features.values()), axis=0).astype(np.float32, copy=False)
    return {
        "sample_rate_sps": float(sample_rate_sps),
        "segment_samples": int(seg_len),
        "feature_config": {
            "segment_ms": cfg.segment_ms,
            "nfft": cfg.nfft,
            "noverlap": cfg.noverlap,
            "resize_h": cfg.resize_h,
            "resize_w": cfg.resize_w,
            "log_power": cfg.log_power,
            "normalize": cfg.normalize,
        },
        "aggregate_stats": _array_stats(stacked),
        "per_probe_stats": {name: _array_stats(feat) for name, feat in features.items()},
    }


def derive_vgg_family_runtime_contract(
    *,
    spec: HailoCheckpointSpec,
    dataset_dir: str | Path,
    dataset_stats: Dict[str, object] | None,
    preprocessing_contract: Dict[str, object],
    sample_count: int = DEFAULT_VGG_FAMILY_RANGE_SAMPLES,
    seed: int = 13,
) -> tuple[Dict[str, object], Dict[str, object]]:
    frames, _, sample_meta = _select_up_to_balanced_feature_frames(
        dataset_dir=dataset_dir,
        max_samples=sample_count,
        seed=seed,
    )
    sample_stats = _array_stats(frames)
    compute_spec_stats = measure_compute_spec_feature_stats(
        preprocessing_contract=preprocessing_contract,
        seed=seed,
    )
    sample_min = float(sample_stats.get("min", 0.0) or 0.0)
    sample_max = float(sample_stats.get("max", 0.0) or 0.0)
    compute_max = float(compute_spec_stats.get("aggregate_stats", {}).get("max", 0.0) or 0.0)
    global_max = float((dataset_stats or {}).get("max", 0.0) or 0.0)

    source_min = min(0.0, sample_min)
    source_max = max(sample_max, compute_max, 1.0)
    runtime_contract = _default_host_contract(
        spec,
        dataset_stats={
            "min": source_min,
            "max": source_max,
        },
    )
    source_stats = {
        "global_dataset_feature_stats": dataset_stats,
        "representative_sample_source": sample_meta,
        "representative_sample_stats": sample_stats,
        "compute_spec_feature_stats": compute_spec_stats,
        "runtime_contract_source_min": float(source_min),
        "runtime_contract_source_max": float(source_max),
        "global_to_runtime_scale_ratio": None if source_max <= 0 else float(global_max / source_max),
    }
    return runtime_contract, source_stats


def build_hailo_manifest(
    spec: HailoCheckpointSpec,
    export_summary: Dict[str, object],
    *,
    target_arch: str,
    repo_root: Path,
    calibration_dir: Path,
    compiler_version: str | None = None,
    hailort_version: str | None = None,
    runtime_contract: Dict[str, object] | None = None,
) -> Dict[str, object]:
    export_contract = export_summary.get("export_contract", {})
    onnx_path = Path(str(export_summary.get("exported", {}).get("onnx", "")))
    runtime_contract = runtime_contract or _default_host_contract(spec, dataset_stats=None)

    return {
        "model_id": spec.model_id,
        "arch": spec.arch,
        "variant": spec.variant,
        "checkpoint_path": _repo_relative(repo_root, spec.checkpoint_path),
        "source_onnx": _repo_relative(repo_root, onnx_path),
        "hef_path": _repo_relative(repo_root, _hef_path(spec, target_arch)),
        "target_arch": target_arch,
        "input_name": export_contract.get("input_name", "input"),
        "input_shape": export_contract.get("input_shape", [1, 1, 224, 224]),
        "output_name": export_contract.get("output_name", "logits"),
        "output_shape": export_contract.get("output_shape", [1, 2]),
        "threshold": float(export_summary.get("threshold", 0.5)),
        "preprocessing_contract": export_summary.get("preprocessing_contract", {}),
        "label_contract": export_summary.get("label_contract", {}),
        "calibration_source": _repo_relative(repo_root, calibration_dir),
        "compiler_version": compiler_version,
        "hailort_version": hailort_version,
        "postprocess": "host_softmax_threshold",
        "host_input_dtype": runtime_contract["host_input_dtype"],
        "host_input_layout": runtime_contract["host_input_layout"],
        "host_input_range": runtime_contract["host_input_range"],
        "host_input_quantization": runtime_contract["host_input_quantization"],
        "hailo_input_normalization": runtime_contract["hailo_input_normalization"],
        "validation_report": None,
        "deployable": False,
        "deployment_decision": "unvalidated",
        "observed_input_stats": None,
    }


def _inventory_path(repo_root: Path) -> Path:
    return repo_root / SUITE_ROOT / "hailo_compilation_inventory.json"


def _build_inventory_payload(
    specs: Sequence[HailoCheckpointSpec],
    manifests: Iterable[Path],
    *,
    repo_root: Path,
    targets: Sequence[str],
    calibration_dir: Path,
) -> Dict[str, object]:
    manifest_paths = [Path(p) for p in manifests]
    by_checkpoint: Dict[str, List[str]] = {}
    for manifest_path in manifest_paths:
        stem = manifest_path.name.split(".")[0]
        by_checkpoint.setdefault(stem, []).append(_repo_relative(repo_root, manifest_path) or str(manifest_path))

    records = []
    for spec in specs:
        records.append(
            {
                "model_id": spec.model_id,
                "arch": spec.arch,
                "variant": spec.variant,
                "checkpoint_path": _repo_relative(repo_root, spec.checkpoint_path),
                "export_dir": _repo_relative(repo_root, spec.export_dir),
                "hailo_dir": _repo_relative(repo_root, spec.hailo_dir),
                "export_summary": _repo_relative(repo_root, _summary_path(spec)),
                "manifest_paths": sorted(by_checkpoint.get(spec.model_id, [])),
            }
        )

    return {
        "repo_root": str(repo_root),
        "targets": list(targets),
        "calibration_dir": _repo_relative(repo_root, calibration_dir),
        "checkpoint_count": len(specs),
        "unique_arches": sorted({spec.arch for spec in specs}),
        "records": records,
    }


def _merge_existing_manifest_artifacts(
    *,
    manifest_path: Path,
    payload: Dict[str, object],
) -> Dict[str, object]:
    if not manifest_path.exists():
        return payload

    existing = load_json(manifest_path, default={})
    if not isinstance(existing, dict) or not existing:
        return payload

    preserved_fields = (
        "calibration_npy",
        "calibration_metadata",
        "native_har",
        "optimized_har",
        "compiled_har",
        "augmented_onnx",
        "parsing_report",
        "parser_log",
        "optimize_log",
        "compiler_log",
        "commands_path",
        "compiler_output_dir",
        "model_script",
        "compiled",
        "validation_report",
        "deployable",
        "deployment_decision",
        "validation_probe_corpus",
        "feature_stats",
        "observed_input_stats",
        "fallback_recommendation",
    )
    for field in preserved_fields:
        if field in existing and existing[field] is not None:
            payload[field] = existing[field]
    return payload


def _rehydrate_manifest_from_disk(
    *,
    spec: HailoCheckpointSpec,
    target_arch: str,
    repo_root: Path,
    payload: Dict[str, object],
) -> Dict[str, object]:
    native_har_path = _native_har_path(spec, target_arch)
    optimized_har_path = _optimized_har_path(spec, target_arch)
    compiled_har_path = _compiled_har_path(spec, target_arch)
    augmented_onnx_path = _augmented_onnx_path(spec, target_arch)
    parsing_report_path = _parsing_report_path(spec, target_arch)
    parser_log_path = _parser_log_path(spec, target_arch)
    optimize_log_path = _optimize_log_path(spec, target_arch)
    compiler_log_path = _compiler_log_path(spec, target_arch)
    commands_path = _commands_path(spec, target_arch)
    compiler_output_dir = _compile_output_dir(spec, target_arch)
    model_script_path = _model_script_path(spec, target_arch)
    final_hef_path = _hef_path(spec, target_arch)
    validation_report_path = _validation_report_path(spec, target_arch)

    if final_hef_path.exists():
        payload["compiled"] = True
    for field, path in (
        ("native_har", native_har_path),
        ("optimized_har", optimized_har_path),
        ("compiled_har", compiled_har_path),
        ("augmented_onnx", augmented_onnx_path),
        ("parsing_report", parsing_report_path),
        ("parser_log", parser_log_path),
        ("optimize_log", optimize_log_path),
        ("compiler_log", compiler_log_path),
        ("commands_path", commands_path),
        ("model_script", model_script_path),
    ):
        if path.exists():
            payload[field] = _repo_relative(repo_root, path)
    if compiler_output_dir.exists():
        payload["compiler_output_dir"] = _repo_relative(repo_root, compiler_output_dir)

    if validation_report_path.exists():
        payload["validation_report"] = _repo_relative(repo_root, validation_report_path)
        report = load_json(validation_report_path, default={})
        if isinstance(report, dict):
            runtime_contract = report.get("runtime_contract")
            if isinstance(runtime_contract, dict):
                payload.update(_runtime_contract_from_payload(runtime_contract))
            if "deployable" in report:
                payload["deployable"] = bool(report["deployable"])
            if "deployment_decision" in report:
                payload["deployment_decision"] = report["deployment_decision"]
            if "fallback_recommendation" in report:
                payload["fallback_recommendation"] = report["fallback_recommendation"]
            if "observed_input_stats" in report:
                payload["observed_input_stats"] = report["observed_input_stats"]
    return payload


def _prepare_hailo_workspace(
    *,
    repo_root: Path,
    targets: Sequence[str],
    include_strict_far: bool,
    onnx_opset: int,
    skip_existing_exports: bool,
    compiler_version: str | None,
    hailort_version: str | None,
) -> HailoWorkspace:
    specs = discover_hailo_checkpoints(repo_root, include_strict_far=include_strict_far)
    calibration_dir = repo_root / CALIBRATION_ROOT
    _write_calibration_readme(calibration_dir)

    export_summaries: Dict[str, Dict[str, object]] = {}
    manifest_paths: Dict[tuple[str, str], Path] = {}
    saved_manifest_paths: List[Path] = []
    dataset_contract_cache: Dict[str, Dict[str, object]] = {}

    for spec in specs:
        export_summary = _export_for_hailo(
            spec,
            skip_existing=skip_existing_exports,
            onnx_opset=onnx_opset,
        )
        export_summaries[spec.model_id] = export_summary

        spec.hailo_dir.mkdir(parents=True, exist_ok=True)
        runtime_contract = None
        if _uses_vgg_family_contract(spec):
            default_dataset_dir = repo_root / DEFAULT_CALIBRATION_DATASET
            dataset_key = str(default_dataset_dir.resolve())
            if iter_npz_files(default_dataset_dir):
                dataset_stats = dataset_contract_cache.get(dataset_key)
                if dataset_stats is None:
                    dataset_stats = measure_dataset_feature_stats(
                        dataset_dir=default_dataset_dir,
                        output_path=_feature_stats_cache_path(repo_root, default_dataset_dir),
                        force=False,
                    )
                    dataset_contract_cache[dataset_key] = dataset_stats
                runtime_contract, _ = derive_vgg_family_runtime_contract(
                    spec=spec,
                    dataset_dir=default_dataset_dir,
                    dataset_stats=dataset_stats,
                    preprocessing_contract=export_summary.get("preprocessing_contract", {}),
                    sample_count=DEFAULT_VGG_FAMILY_RANGE_SAMPLES,
                )
        for target_arch in targets:
            manifest_path = _manifest_path(spec, target_arch)
            payload = build_hailo_manifest(
                spec,
                export_summary,
                target_arch=target_arch,
                repo_root=repo_root,
                calibration_dir=calibration_dir,
                compiler_version=compiler_version,
                hailort_version=hailort_version,
                runtime_contract=runtime_contract,
            )
            payload = _merge_existing_manifest_artifacts(manifest_path=manifest_path, payload=payload)
            payload = _rehydrate_manifest_from_disk(
                spec=spec,
                target_arch=target_arch,
                repo_root=repo_root,
                payload=payload,
            )
            save_json(manifest_path, payload)
            manifest_paths[(spec.model_id, target_arch)] = manifest_path
            saved_manifest_paths.append(manifest_path)

    inventory_path = _inventory_path(repo_root)
    inventory_payload = _build_inventory_payload(
        specs,
        saved_manifest_paths,
        repo_root=repo_root,
        targets=targets,
        calibration_dir=calibration_dir,
    )
    save_json(inventory_path, inventory_payload)

    return HailoWorkspace(
        repo_root=repo_root,
        specs=specs,
        export_summaries=export_summaries,
        manifest_paths=manifest_paths,
        calibration_dir=calibration_dir,
        inventory_path=inventory_path,
        targets=targets,
    )


def _resolve_tool_binary(explicit: str | None, candidates: Sequence[str]) -> str:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path)
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        raise FileNotFoundError(f"Requested tool was not found: {explicit}")

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise FileNotFoundError(f"Unable to locate any of the expected tool binaries: {candidates}")


def _default_hailo_candidates() -> List[str]:
    return [
        str(Path("/root/hailo_ai_sw_suite/hailo_venv/bin/hailo")),
        "hailo",
    ]


def _default_hailortcli_candidates() -> List[str]:
    return [
        str(Path("/root/hailo_ai_sw_suite/hailo_venv/bin/hailortcli")),
        "hailortcli",
    ]


def _version_line(text: str) -> str | None:
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        if not line.startswith("[info]"):
            return line
    stripped = text.strip()
    return stripped or None


def _detect_tool_versions(hailo_bin: str, hailortcli_bin: str | None) -> tuple[str | None, str | None]:
    compiler_version = None
    hailort_version = None

    compiler_res = subprocess.run(
        [hailo_bin, "--version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    compiler_text = "\n".join(part for part in [compiler_res.stdout, compiler_res.stderr] if part)
    compiler_version = _version_line(compiler_text)

    if hailortcli_bin:
        hailort_res = subprocess.run(
            [hailortcli_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        hailort_text = "\n".join(part for part in [hailort_res.stdout, hailort_res.stderr] if part)
        hailort_version = _version_line(hailort_text)

    return compiler_version, hailort_version


def _apply_host_transform(frames: np.ndarray, runtime_contract: Dict[str, object]) -> np.ndarray:
    x = np.asarray(frames, dtype=np.float32)
    quant = runtime_contract.get("host_input_quantization", {})
    qtype = str(quant.get("type", "none"))
    if qtype == "linear_uint8":
        source_min = float(quant.get("source_min", 0.0))
        source_max = float(quant.get("source_max", 1.0))
        source_range = max(source_max - source_min, 1e-12)
        scaled = np.rint(((x - source_min) / source_range) * 255.0)
        scaled = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
        return scaled[..., None]
    return x.astype(np.float32, copy=False)[..., None]


def build_hailo_calibration_set(
    *,
    dataset_dir: str | Path,
    output_path: str | Path,
    max_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    seed: int = 13,
    runtime_contract: Dict[str, object] | None = None,
) -> tuple[Path, Path]:
    frames, labels, metadata = _select_balanced_feature_frames(
        dataset_dir=dataset_dir,
        max_samples=max_samples,
        seed=seed,
    )
    runtime_contract = runtime_contract or {
        "host_input_dtype": "float32",
        "host_input_layout": "NHWC",
        "host_input_range": {"min": 0.0, "max": 1.0},
        "host_input_quantization": {"type": "none", "source_min": 0.0, "source_max": 1.0},
        "hailo_input_normalization": {"enabled": False, "mean": [0.0], "std": [1.0]},
    }
    host_frames = _apply_host_transform(frames, runtime_contract)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, host_frames)

    metadata_path = output.with_suffix(".metadata.json")
    payload = {
        **metadata,
        "output_path": str(output),
        "shape": list(host_frames.shape),
        "dtype": str(host_frames.dtype),
        "selected_class_counts": metadata["selected_class_counts"],
        "requested_max_samples": int(max_samples),
        "host_contract": runtime_contract,
        "model_domain_stats": _array_stats(frames),
        "host_domain_stats": _array_stats(host_frames),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }
    save_json(metadata_path, payload)
    return output, metadata_path


def _make_synthetic_model_probes(
    *,
    runtime_contract: Dict[str, object],
    seed: int,
) -> tuple[np.ndarray, List[str]]:
    quant = runtime_contract.get("host_input_quantization", {})
    source_min = float(quant.get("source_min", 0.0))
    source_max = float(quant.get("source_max", 1.0))
    source_range = max(source_max - source_min, 1e-12)
    rng0 = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed + 1)
    checker = (np.indices((224, 224)).sum(axis=0) % 2).astype(np.float32)

    frames = [
        np.zeros((224, 224), dtype=np.float32),
        np.full((224, 224), source_max, dtype=np.float32),
        np.full((224, 224), source_min + (0.5 * source_range), dtype=np.float32),
        source_min + (checker * source_range),
        rng0.random((224, 224), dtype=np.float32) * source_range + source_min,
        rng1.random((224, 224), dtype=np.float32) * source_range + source_min,
    ]
    return np.stack(frames, axis=0).astype(np.float32, copy=False), list(DEFAULT_SYNTHETIC_PROBE_NAMES)


def build_hailo_probe_corpus(
    *,
    spec: HailoCheckpointSpec,
    dataset_dir: str | Path,
    runtime_contract: Dict[str, object],
    real_samples: int = DEFAULT_VALIDATION_REAL_SAMPLES,
    seed: int = 13,
) -> tuple[Path, Path]:
    synthetic_model_frames, synthetic_names = _make_synthetic_model_probes(
        runtime_contract=runtime_contract,
        seed=seed,
    )
    synthetic_host_frames = _apply_host_transform(synthetic_model_frames, runtime_contract)

    real_model_frames, real_labels, real_meta = _select_balanced_feature_frames(
        dataset_dir=dataset_dir,
        max_samples=real_samples,
        seed=seed,
    )
    real_host_frames = _apply_host_transform(real_model_frames, runtime_contract)

    probe_path = _probe_corpus_path(spec)
    metadata_path = _probe_corpus_metadata_path(spec)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probe_path,
        synthetic_model=synthetic_model_frames.astype(np.float32, copy=False),
        synthetic_host=synthetic_host_frames,
        synthetic_names=np.asarray(synthetic_names, dtype="<U64"),
        real_model=real_model_frames.astype(np.float32, copy=False),
        real_host=real_host_frames,
        real_labels=real_labels.astype(np.int64, copy=False),
    )
    metadata = {
        "probe_path": str(probe_path),
        "runtime_contract": runtime_contract,
        "synthetic_names": synthetic_names,
        "synthetic_model_stats": _array_stats(synthetic_model_frames),
        "synthetic_host_stats": _array_stats(synthetic_host_frames),
        "real_model_stats": _array_stats(real_model_frames),
        "real_host_stats": _array_stats(real_host_frames),
        "real_label_counts": {str(k): int(v) for k, v in zip(*np.unique(real_labels, return_counts=True))},
        "real_source": real_meta,
        "seed": int(seed),
    }
    save_json(metadata_path, metadata)
    return probe_path, metadata_path


def _run_logged_command(command: Sequence[str], *, log_path: Path, workdir: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            list(command),
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(command))


def _expected_compiler_output_hef(optimized_har_path: Path) -> Path:
    return optimized_har_path.with_suffix(".hef")


def _model_script_lines(spec: HailoCheckpointSpec, runtime_contract: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    if spec.arch == "mobilenet_v3_small":
        lines.append(
            "pre_quantization_optimization(global_avgpool_reduction, layers=avgpool1, division_factors=[4, 4])"
        )
    norm = runtime_contract.get("hailo_input_normalization", {})
    if norm.get("enabled"):
        means = ", ".join(f"{float(v):.12g}" for v in norm.get("mean", [0.0]))
        stds = ", ".join(f"{float(v):.12g}" for v in norm.get("std", [1.0]))
        lines.append(f"normalization1 = normalization([{means}], [{stds}])")
    if _uses_vgg_family_contract(spec):
        lines.append("performance_param(compiler_optimization_level=max)")
    return lines


def _write_model_script(
    spec: HailoCheckpointSpec,
    target_arch: str,
    runtime_contract: Dict[str, object],
) -> Path | None:
    lines = _model_script_lines(spec, runtime_contract)
    if not lines:
        return None
    path = _model_script_path(spec, target_arch)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_performance_report(spec: HailoCheckpointSpec, target_arch: str) -> Dict[str, object]:
    compiler_log_path = _compiler_log_path(spec, target_arch)
    core_log_path = _compile_output_dir(spec, target_arch) / "hailo_sdk.core.log"
    compiler_text = compiler_log_path.read_text(encoding="utf-8", errors="replace") if compiler_log_path.exists() else ""
    core_text = core_log_path.read_text(encoding="utf-8", errors="replace") if core_log_path.exists() else ""

    context_count = None
    fps = None
    inter_context_bandwidth_mbps = None

    context_match = re.findall(r"Found valid partition to (\d+) contexts", compiler_text)
    if context_match:
        context_count = int(context_match[-1])

    fps_match = re.findall(r"with ([0-9.eE+\-]+) FPS", core_text)
    if fps_match:
        fps = float(fps_match[-1])

    bandwidth_match = re.findall(r"Bandwidth of inter context tensors: ([0-9.eE+\-]+) Mbps", core_text)
    if bandwidth_match:
        inter_context_bandwidth_mbps = float(bandwidth_match[-1])

    estimated_latency_ms = None if not fps or fps <= 0 else float(1000.0 / fps)
    return {
        "compiler_context_count": context_count,
        "multi_context": bool(context_count and context_count > 1),
        "compiler_estimated_fps": fps,
        "compiler_estimated_latency_ms": estimated_latency_ms,
        "inter_context_bandwidth_mbps": inter_context_bandwidth_mbps,
        "target_latency_ms": DEFAULT_HAILO_LATENCY_TARGET_MS,
    }


def _softmax_p_drone(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    probs = ex / np.sum(ex, axis=1, keepdims=True)
    return probs[:, 1].astype(np.float32, copy=False)


def _coerce_logits_batch(output: Any, expected_count: int) -> np.ndarray:
    if isinstance(output, dict):
        if not output:
            raise ValueError("Received empty output mapping from inference stage")
        output = next(iter(output.values()))
    elif isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    arr = np.asarray(output)
    if arr.size != expected_count * 2 and not (arr.ndim >= 1 and arr.shape[-1] == 2):
        raise ValueError(f"Unexpected logits output shape {arr.shape}, expected count {expected_count}")
    arr = arr.reshape(-1, 2)
    if len(arr) < expected_count:
        raise ValueError(f"Expected at least {expected_count} rows of logits, got {len(arr)}")
    return arr[:expected_count].astype(np.float32, copy=False)


def _run_pytorch_stage(
    spec: HailoCheckpointSpec,
    model_frames: np.ndarray,
) -> np.ndarray:
    import torch

    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    from .modeling import create_binary_model

    model = create_binary_model(
        arch=str(checkpoint.get("arch", spec.arch)),
        pretrained=False,
        freeze_features=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(model_frames, dtype=np.float32)))
    return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _run_onnx_stage(
    onnx_path: Path,
    model_frames: np.ndarray,
) -> np.ndarray | None:
    ort = _optional_import_onnxruntime()
    if ort is None:
        return None
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rows: List[np.ndarray] = []
    for frame in np.asarray(model_frames, dtype=np.float32):
        x = frame[None, None, :, :]
        logits = session.run([output_name], {input_name: x})[0]
        rows.append(np.asarray(logits, dtype=np.float32).reshape(1, 2))
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 2), dtype=np.float32)


def _run_hailo_emulation_stage(
    *,
    har_path: Path,
    context_name: str,
    host_frames: np.ndarray,
) -> np.ndarray | None:
    if not har_path.exists():
        return None
    ClientRunner, InferenceContext = _optional_import_hailo_sdk()
    if ClientRunner is None or InferenceContext is None:
        return None

    ctx_enum = getattr(InferenceContext, context_name)
    frames = np.asarray(host_frames)
    if len(frames) == 0:
        return np.empty((0, 2), dtype=np.float32)

    def run_batched() -> np.ndarray:
        runner = ClientRunner(har=str(har_path))
        rows: List[np.ndarray] = []
        with runner.infer_context(ctx_enum) as ctx:
            for start in range(0, len(frames), HAILO_EMULATION_BATCH_SIZE):
                chunk = frames[start : start + HAILO_EMULATION_BATCH_SIZE]
                output = runner.infer(ctx, chunk, batch_size=len(chunk))
                rows.append(_coerce_logits_batch(output, expected_count=len(chunk)))
        return np.concatenate(rows, axis=0)

    def run_per_frame() -> np.ndarray:
        runner = ClientRunner(har=str(har_path))
        rows: List[np.ndarray] = []
        with runner.infer_context(ctx_enum) as ctx:
            for frame in frames:
                output = runner.infer(ctx, frame[None, ...], batch_size=1)
                rows.append(_coerce_logits_batch(output, expected_count=1))
        return np.concatenate(rows, axis=0)

    try:
        return run_batched()
    except Exception:
        # Some Hailo SDK emulation contexts only accept batch_size=1. Keep those
        # paths working, but prefer batching because per-frame calls retain a lot
        # of TensorFlow/Hailo graph state during validation.
        return run_per_frame()


def _load_runtime_metrics(runtime_metrics_json: Path | None) -> Dict[str, object] | None:
    if runtime_metrics_json is None:
        return None
    if not runtime_metrics_json.exists():
        raise FileNotFoundError(f"Runtime metrics JSON not found: {runtime_metrics_json}")
    payload = load_json(runtime_metrics_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime metrics JSON must decode to an object: {runtime_metrics_json}")
    return payload


def _load_hardware_results(hardware_results_json: Path | None) -> Dict[str, object] | None:
    if hardware_results_json is None:
        return None
    if not hardware_results_json.exists():
        raise FileNotFoundError(f"Hardware results JSON not found: {hardware_results_json}")
    payload = load_json(hardware_results_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Hardware results JSON must decode to an object: {hardware_results_json}")
    return payload


def _resolve_latency_gate(
    *,
    runtime_metrics: Dict[str, object] | None,
    compiler_estimated_latency_ms: float | None,
) -> Dict[str, object]:
    hardware_latency_values: List[float] = []
    if runtime_metrics is not None:
        for key in ("mean_latency_ms", "p95_latency_ms", "latency_ms"):
            value = runtime_metrics.get(key)
            if value is not None:
                hardware_latency_values.append(float(value))

    if hardware_latency_values:
        selected_latency_ms = max(hardware_latency_values)
        runtime_pass = selected_latency_ms < DEFAULT_HAILO_LATENCY_TARGET_MS
        return {
            "selected_latency_ms": float(selected_latency_ms),
            "measured_latency_ms": float(selected_latency_ms),
            "latency_source": "hardware_runtime_metrics",
            "latency_is_provisional": False,
            "hardware_latency_confirmed": True,
            "runtime_gate_status": "pass" if runtime_pass else "fail",
            "runtime_pass": bool(runtime_pass),
        }

    if compiler_estimated_latency_ms is not None:
        selected_latency_ms = float(compiler_estimated_latency_ms)
        runtime_pass = selected_latency_ms < DEFAULT_HAILO_LATENCY_TARGET_MS
        return {
            "selected_latency_ms": float(selected_latency_ms),
            "measured_latency_ms": None,
            "latency_source": "compiler_estimate",
            "latency_is_provisional": True,
            "hardware_latency_confirmed": False,
            "runtime_gate_status": "pass" if runtime_pass else "fail",
            "runtime_pass": bool(runtime_pass),
        }

    return {
        "selected_latency_ms": None,
        "measured_latency_ms": None,
        "latency_source": None,
        "latency_is_provisional": True,
        "hardware_latency_confirmed": False,
        "runtime_gate_status": "pending_runtime_metrics",
        "runtime_pass": False,
    }


def _deployment_decision(
    *,
    quality_pass: bool,
    latency_available: bool,
    hardware_latency_confirmed: bool,
    runtime_pass: bool,
    deployable: bool,
) -> str:
    if deployable:
        if hardware_latency_confirmed:
            return "accepted_live_hailo"
        return "accepted_provisional_hailo"
    if not quality_pass:
        return "rejected_quality"
    if not latency_available:
        return "pending_runtime_metrics"
    if not runtime_pass:
        return "rejected_runtime"
    return "rejected"


def _pairwise_drift(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    diff = np.abs(ref - cand)
    p_ref = _softmax_p_drone(ref.astype(np.float32))
    p_cand = _softmax_p_drone(cand.astype(np.float32))
    p_diff = np.abs(p_ref.astype(np.float64) - p_cand.astype(np.float64))
    return {
        "max_abs_logit": float(diff.max()),
        "mean_abs_logit": float(diff.mean()),
        "max_abs_p_drone": float(p_diff.max()),
        "mean_abs_p_drone": float(p_diff.mean()),
    }


def _spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        return None
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    if np.std(rank_x) < 1e-12 or np.std(rank_y) < 1e-12:
        return 0.0
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _build_stage_report(logits: np.ndarray) -> Dict[str, object]:
    p_drone = _softmax_p_drone(logits)
    return {
        "logits": np.asarray(logits, dtype=np.float32).tolist(),
        "p_drone": p_drone.astype(np.float32).tolist(),
        "logits_stats": _array_stats(np.asarray(logits, dtype=np.float32)),
        "p_drone_stats": _array_stats(np.asarray(p_drone, dtype=np.float32)),
    }


def _validate_single_target(
    *,
    spec: HailoCheckpointSpec,
    target_arch: str,
    export_summary: Dict[str, object],
    workspace: HailoWorkspace,
    dataset_dir: Path,
    real_probe_samples: int,
    seed: int,
    runtime_metrics_json: Path | None,
    hardware_results_json: Path | None,
) -> Path:
    dataset_stats_path = _feature_stats_cache_path(workspace.repo_root, dataset_dir)
    dataset_stats = measure_dataset_feature_stats(
        dataset_dir=dataset_dir,
        output_path=dataset_stats_path,
        force=False,
    )
    if _uses_vgg_family_contract(spec):
        runtime_contract, contract_source_stats = derive_vgg_family_runtime_contract(
            spec=spec,
            dataset_dir=dataset_dir,
            dataset_stats=dataset_stats,
            preprocessing_contract=export_summary.get("preprocessing_contract", {}),
            sample_count=DEFAULT_VGG_FAMILY_RANGE_SAMPLES,
            seed=seed,
        )
        compute_spec_stats = contract_source_stats["compute_spec_feature_stats"]
    else:
        compute_spec_stats = measure_compute_spec_feature_stats(
            preprocessing_contract=export_summary.get("preprocessing_contract", {}),
            seed=seed,
        )
        runtime_contract = _default_host_contract(spec, dataset_stats=dataset_stats)
        contract_source_stats = {
            "global_dataset_feature_stats": dataset_stats,
            "compute_spec_feature_stats": compute_spec_stats,
        }
    probe_path, probe_metadata_path = build_hailo_probe_corpus(
        spec=spec,
        dataset_dir=dataset_dir,
        runtime_contract=runtime_contract,
        real_samples=real_probe_samples,
        seed=seed,
    )
    with np.load(probe_path, allow_pickle=False) as data:
        synthetic_model = np.asarray(data["synthetic_model"], dtype=np.float32)
        synthetic_host = np.asarray(data["synthetic_host"])
        synthetic_names = [str(v) for v in data["synthetic_names"].tolist()]
        real_model = np.asarray(data["real_model"], dtype=np.float32)
        real_host = np.asarray(data["real_host"])
        real_labels = np.asarray(data["real_labels"], dtype=np.int64)

    stage_reports: Dict[str, Dict[str, object]] = {}

    pytorch_synth = _run_pytorch_stage(spec, synthetic_model)
    pytorch_real = _run_pytorch_stage(spec, real_model)
    stage_reports["pytorch"] = {
        "available": True,
        "synthetic": _build_stage_report(pytorch_synth),
        "real": _build_stage_report(pytorch_real),
    }

    onnx_path = Path(str(export_summary.get("exported", {}).get("onnx", "")))
    onnx_synth = _run_onnx_stage(onnx_path, synthetic_model) if onnx_path.exists() else None
    onnx_real = _run_onnx_stage(onnx_path, real_model) if onnx_path.exists() else None
    stage_reports["onnx"] = {
        "available": onnx_synth is not None and onnx_real is not None,
        "synthetic": _build_stage_report(onnx_synth) if onnx_synth is not None else None,
        "real": _build_stage_report(onnx_real) if onnx_real is not None else None,
    }

    native_har = _native_har_path(spec, target_arch)
    optimized_har = _optimized_har_path(spec, target_arch)
    sdk_native_synth = _run_hailo_emulation_stage(
        har_path=native_har,
        context_name="SDK_NATIVE",
        host_frames=synthetic_model[..., None].astype(np.float32, copy=False),
    )
    sdk_native_real = _run_hailo_emulation_stage(
        har_path=native_har,
        context_name="SDK_NATIVE",
        host_frames=real_model[..., None].astype(np.float32, copy=False),
    )
    stage_reports["sdk_native"] = {
        "available": sdk_native_synth is not None and sdk_native_real is not None,
        "synthetic": _build_stage_report(sdk_native_synth) if sdk_native_synth is not None else None,
        "real": _build_stage_report(sdk_native_real) if sdk_native_real is not None else None,
    }

    sdk_fp_synth = _run_hailo_emulation_stage(
        har_path=optimized_har,
        context_name="SDK_FP_OPTIMIZED",
        host_frames=synthetic_host,
    )
    sdk_fp_real = _run_hailo_emulation_stage(
        har_path=optimized_har,
        context_name="SDK_FP_OPTIMIZED",
        host_frames=real_host,
    )
    stage_reports["sdk_fp_optimized"] = {
        "available": sdk_fp_synth is not None and sdk_fp_real is not None,
        "synthetic": _build_stage_report(sdk_fp_synth) if sdk_fp_synth is not None else None,
        "real": _build_stage_report(sdk_fp_real) if sdk_fp_real is not None else None,
    }

    sdk_quant_synth = _run_hailo_emulation_stage(
        har_path=optimized_har,
        context_name="SDK_QUANTIZED",
        host_frames=synthetic_host,
    )
    sdk_quant_real = _run_hailo_emulation_stage(
        har_path=optimized_har,
        context_name="SDK_QUANTIZED",
        host_frames=real_host,
    )
    stage_reports["sdk_quantized"] = {
        "available": sdk_quant_synth is not None and sdk_quant_real is not None,
        "synthetic": _build_stage_report(sdk_quant_synth) if sdk_quant_synth is not None else None,
        "real": _build_stage_report(sdk_quant_real) if sdk_quant_real is not None else None,
    }

    hardware_results = _load_hardware_results(hardware_results_json)
    if hardware_results is not None:
        hardware_synth = np.asarray(hardware_results.get("synthetic", {}).get("logits", []), dtype=np.float32)
        hardware_real = np.asarray(hardware_results.get("real", {}).get("logits", []), dtype=np.float32)
        stage_reports["hardware"] = {
            "available": hardware_synth.size > 0 and hardware_real.size > 0,
            "synthetic": _build_stage_report(hardware_synth.reshape(len(synthetic_names), 2))
            if hardware_synth.size
            else None,
            "real": _build_stage_report(hardware_real.reshape(len(real_model), 2)) if hardware_real.size else None,
        }

    drift: Dict[str, Dict[str, object]] = {"synthetic": {}, "real": {}}
    available_stage_names = [name for name, payload in stage_reports.items() if payload.get("available")]
    for idx, left_name in enumerate(available_stage_names):
        left = stage_reports[left_name]
        for right_name in available_stage_names[idx + 1 :]:
            right = stage_reports[right_name]
            key = f"{left_name}__{right_name}"
            drift["synthetic"][key] = _pairwise_drift(
                np.asarray(left["synthetic"]["logits"], dtype=np.float32),
                np.asarray(right["synthetic"]["logits"], dtype=np.float32),
            )
            drift["real"][key] = _pairwise_drift(
                np.asarray(left["real"]["logits"], dtype=np.float32),
                np.asarray(right["real"]["logits"], dtype=np.float32),
            )

    artifact_stage_name = "hardware"
    if "hardware" not in stage_reports or not stage_reports["hardware"].get("available"):
        artifact_stage_name = "sdk_quantized"
    if not stage_reports.get(artifact_stage_name, {}).get("available"):
        artifact_stage_name = "sdk_fp_optimized"

    artifact_available = stage_reports.get(artifact_stage_name, {}).get("available", False)
    if artifact_available:
        artifact_synth_logits = np.asarray(stage_reports[artifact_stage_name]["synthetic"]["logits"], dtype=np.float32)
        artifact_synth_p = np.asarray(stage_reports[artifact_stage_name]["synthetic"]["p_drone"], dtype=np.float32)
        artifact_real_p = np.asarray(stage_reports[artifact_stage_name]["real"]["p_drone"], dtype=np.float32)
        pytorch_real_p = np.asarray(stage_reports["pytorch"]["real"]["p_drone"], dtype=np.float32)
        logit_spread = np.ptp(artifact_synth_logits, axis=0)
        p_variance = float(np.var(artifact_synth_p.astype(np.float64)))
        spearman = _spearman_rank_correlation(pytorch_real_p, artifact_real_p)
        flat_failed = bool(np.all(logit_spread < FLAT_LOGIT_SPREAD_EPS) or p_variance < FLAT_P_DRONE_VARIANCE_EPS)
        ordering_passed = spearman is not None and spearman >= ORDERING_SPEARMAN_MIN
    else:
        logit_spread = np.asarray([0.0, 0.0], dtype=np.float32)
        p_variance = 0.0
        spearman = None
        flat_failed = True
        ordering_passed = False

    manifest_path = workspace.manifest_paths[(spec.model_id, target_arch)]
    manifest = load_json(manifest_path, default={})
    required_contract_fields = [
        "host_input_dtype",
        "host_input_layout",
        "host_input_range",
        "host_input_quantization",
        "hailo_input_normalization",
    ]
    manifest_contract_pass = all(field in manifest for field in required_contract_fields)

    runtime_metrics = _load_runtime_metrics(runtime_metrics_json)
    performance = _parse_performance_report(spec, target_arch)
    performance["runtime_metrics"] = runtime_metrics
    latency_gate = _resolve_latency_gate(
        runtime_metrics=runtime_metrics,
        compiler_estimated_latency_ms=performance.get("compiler_estimated_latency_ms"),
    )
    performance.update(latency_gate)
    runtime_pass = bool(latency_gate["runtime_pass"])

    quality_pass = bool(manifest_contract_pass and not flat_failed and ordering_passed)
    deployable = bool(quality_pass and runtime_pass)
    deployment_decision = _deployment_decision(
        quality_pass=quality_pass,
        latency_available=latency_gate["selected_latency_ms"] is not None,
        hardware_latency_confirmed=bool(latency_gate["hardware_latency_confirmed"]),
        runtime_pass=runtime_pass,
        deployable=deployable,
    )
    fallback_recommendation = (
        None if deployable and latency_gate["hardware_latency_confirmed"] else DEFAULT_FALLBACK_MODEL_ID
    )
    observed_input_stats = {
        "dataset_feature_stats": dataset_stats,
        "compute_spec_feature_stats": compute_spec_stats,
        "runtime_contract_source_stats": contract_source_stats,
        "synthetic_model_stats": _array_stats(synthetic_model),
        "synthetic_host_stats": _array_stats(synthetic_host),
        "real_model_stats": _array_stats(real_model),
        "real_host_stats": _array_stats(real_host),
    }

    report = {
        "model_id": spec.model_id,
        "arch": spec.arch,
        "variant": spec.variant,
        "target_arch": target_arch,
        "threshold": float(export_summary.get("threshold", 0.5)),
        "runtime_contract": runtime_contract,
        "dataset_feature_stats": dataset_stats,
        "compute_spec_feature_stats": compute_spec_stats,
        "observed_input_stats": observed_input_stats,
        "probe_corpus": {
            "path": _repo_relative(workspace.repo_root, probe_path),
            "metadata_path": _repo_relative(workspace.repo_root, probe_metadata_path),
            "synthetic_names": synthetic_names,
            "real_label_counts": {str(k): int(v) for k, v in zip(*np.unique(real_labels, return_counts=True))},
        },
        "stages": stage_reports,
        "drift": drift,
        "flat_output": {
            "stage": artifact_stage_name,
            "available": bool(artifact_available),
            "synthetic_p_drone_variance": float(p_variance),
            "synthetic_logit_spread": [float(v) for v in np.asarray(logit_spread).tolist()],
            "failed": bool(flat_failed),
        },
        "ordering": {
            "stage": artifact_stage_name,
            "spearman_vs_pytorch": spearman,
            "passed": bool(ordering_passed),
        },
        "performance": performance,
        "gates": {
            "manifest_contract": bool(manifest_contract_pass),
            "flat_output": not bool(flat_failed),
            "ordering": bool(ordering_passed),
            "quality_pass": bool(quality_pass),
            "runtime_pass": bool(runtime_pass),
            "hardware_latency_confirmed": bool(latency_gate["hardware_latency_confirmed"]),
            "overall_pass": bool(deployable),
        },
        "deployable": bool(deployable),
        "deployment_decision": deployment_decision,
        "fallback_recommendation": fallback_recommendation,
    }

    report_path = _validation_report_path(spec, target_arch)
    save_json(report_path, report)

    manifest.update(runtime_contract)
    manifest.update(
        {
            "validation_report": _repo_relative(workspace.repo_root, report_path),
            "deployable": bool(deployable),
            "deployment_decision": deployment_decision,
            "validation_probe_corpus": _repo_relative(workspace.repo_root, probe_path),
            "feature_stats": _repo_relative(workspace.repo_root, dataset_stats_path),
            "observed_input_stats": observed_input_stats,
            "fallback_recommendation": fallback_recommendation,
        }
    )
    save_json(manifest_path, manifest)
    return report_path


def _resolve_calibration_assets(
    *,
    spec: HailoCheckpointSpec,
    repo_root: Path,
    dataset_dir: Path,
    requested_calibration_npy: Path | None,
    requested_samples: int,
    seed: int,
    dataset_stats_cache: Dict[str, Dict[str, object]],
    generic_cache: Dict[str, tuple[Path, Path]],
) -> tuple[Path, Path, Dict[str, object]]:
    if requested_calibration_npy is not None:
        metadata_path = requested_calibration_npy.with_suffix(".metadata.json")
        if not metadata_path.exists():
            save_json(
                metadata_path,
                {
                    "dataset_dir": None,
                    "output_path": str(requested_calibration_npy),
                    "sample_count": None,
                    "shape": None,
                    "dtype": None,
                    "source_shards": [],
                    "requested_max_samples": None,
                    "selected_class_counts": {},
                    "seed": seed,
                },
            )
        manifest = load_json(_manifest_path(spec, DEFAULT_TARGET_ARCHS[0]), default={})
        runtime_contract = _runtime_contract_from_payload(manifest)
        return requested_calibration_npy, metadata_path, runtime_contract

    if _uses_vgg_family_contract(spec):
        dataset_key = str(dataset_dir.resolve())
        dataset_stats = dataset_stats_cache.get(dataset_key)
        if dataset_stats is None:
            dataset_stats = measure_dataset_feature_stats(
                dataset_dir=dataset_dir,
                output_path=_feature_stats_cache_path(repo_root, dataset_dir),
                force=False,
            )
            dataset_stats_cache[dataset_key] = dataset_stats
        export_summary = load_json(_summary_path(spec))
        runtime_contract, _ = derive_vgg_family_runtime_contract(
            spec=spec,
            dataset_dir=dataset_dir,
            dataset_stats=dataset_stats,
            preprocessing_contract=export_summary.get("preprocessing_contract", {}),
            sample_count=min(
                max(int(requested_samples), DEFAULT_VGG_FAMILY_RANGE_SAMPLES),
                DEFAULT_VGG_FAMILY_CALIBRATION_SAMPLES,
            ),
            seed=seed,
        )
        sample_count = max(int(requested_samples), DEFAULT_VGG_FAMILY_CALIBRATION_SAMPLES)
        output_path = _vgg_family_calibration_path(repo_root, spec, sample_count)
        metadata_path = output_path.with_suffix(".metadata.json")
        if not _calibration_artifacts_match(
            output_path=output_path,
            metadata_path=metadata_path,
            runtime_contract=runtime_contract,
            sample_count=sample_count,
        ):
            if output_path.exists():
                print(f"Rebuilding stale calibration set: {output_path}")
            build_hailo_calibration_set(
                dataset_dir=dataset_dir,
                output_path=output_path,
                max_samples=sample_count,
                seed=seed,
                runtime_contract=runtime_contract,
            )
        return output_path, metadata_path, runtime_contract

    dataset_key = str(dataset_dir.resolve())
    dataset_stats = dataset_stats_cache.get(dataset_key)
    if dataset_stats is None:
        dataset_stats = measure_dataset_feature_stats(
            dataset_dir=dataset_dir,
            output_path=_feature_stats_cache_path(repo_root, dataset_dir),
            force=False,
        )
        dataset_stats_cache[dataset_key] = dataset_stats
    runtime_contract = _default_host_contract(spec, dataset_stats=dataset_stats)
    quant_type = str(runtime_contract.get("host_input_quantization", {}).get("type", "none"))
    quant = runtime_contract.get("host_input_quantization", {})
    generic_key = (
        f"{dataset_dir.resolve()}::{requested_samples}::{quant_type}::"
        f"{quant.get('source_min', 0.0)}::{quant.get('source_max', 1.0)}"
    )
    cached = generic_cache.get(generic_key)
    if cached is None:
        output_path = repo_root / CALIBRATION_ROOT / f"rfbd_{quant_type}_calibration_{requested_samples}.npy"
        metadata_path = output_path.with_suffix(".metadata.json")
        if not _calibration_artifacts_match(
            output_path=output_path,
            metadata_path=metadata_path,
            runtime_contract=runtime_contract,
            sample_count=requested_samples,
        ):
            if output_path.exists():
                print(f"Rebuilding stale calibration set: {output_path}")
            build_hailo_calibration_set(
                dataset_dir=dataset_dir,
                output_path=output_path,
                max_samples=requested_samples,
                seed=seed,
                runtime_contract=runtime_contract,
            )
        cached = (output_path, metadata_path)
        generic_cache[generic_key] = cached
    return cached[0], cached[1], runtime_contract


def _compile_single_target(
    *,
    spec: HailoCheckpointSpec,
    target_arch: str,
    export_summary: Dict[str, object],
    workspace: HailoWorkspace,
    hailo_bin: str,
    calibration_npy: Path,
    calibration_metadata: Path,
    runtime_contract: Dict[str, object],
    compiler_version: str | None,
    hailort_version: str | None,
) -> Path:
    onnx_path = Path(str(export_summary.get("exported", {}).get("onnx", "")))
    if not onnx_path.exists():
        raise FileNotFoundError(f"Expected ONNX export missing for {spec.model_id}: {onnx_path}")

    output_name = str(export_summary.get("export_contract", {}).get("output_name", "logits"))
    native_har_path = _native_har_path(spec, target_arch)
    optimized_har_path = _optimized_har_path(spec, target_arch)
    compiled_har_path = _compiled_har_path(spec, target_arch)
    augmented_onnx_path = _augmented_onnx_path(spec, target_arch)
    parsing_report_path = _parsing_report_path(spec, target_arch)
    parser_log_path = _parser_log_path(spec, target_arch)
    optimize_log_path = _optimize_log_path(spec, target_arch)
    compiler_log_path = _compiler_log_path(spec, target_arch)
    commands_path = _commands_path(spec, target_arch)
    compiler_output_dir = _compile_output_dir(spec, target_arch)
    final_hef_path = _hef_path(spec, target_arch)
    model_script_path = _write_model_script(spec, target_arch, runtime_contract)

    spec.hailo_dir.mkdir(parents=True, exist_ok=True)
    compiler_output_dir.mkdir(parents=True, exist_ok=True)

    parse_net_name = f"{spec.model_id}_{target_arch}"
    parser_cmd = [
        hailo_bin,
        "parser",
        "onnx",
        str(onnx_path),
        "--net-name",
        parse_net_name,
        "--har-path",
        str(native_har_path),
        "--hw-arch",
        target_arch,
        "-y",
        "--parsing-report-path",
        str(parsing_report_path),
        "--augmented-path",
        str(augmented_onnx_path),
        "--end-node-names",
        output_name,
    ]
    optimize_cmd = [
        hailo_bin,
        "optimize",
        str(native_har_path),
        "--hw-arch",
        target_arch,
        "--calib-set-path",
        str(calibration_npy),
        "--output-har-path",
        str(optimized_har_path),
    ]
    compiler_cmd = [
        hailo_bin,
        "compiler",
        str(optimized_har_path),
        "--hw-arch",
        target_arch,
        "--output-dir",
        str(compiler_output_dir),
        "--output-har-path",
        str(compiled_har_path),
    ]
    if model_script_path is not None:
        optimize_cmd.extend(["--model-script", str(model_script_path)])
        compiler_cmd.extend(["--model-script", str(model_script_path)])

    _run_logged_command(parser_cmd, log_path=parser_log_path, workdir=workspace.repo_root)
    _run_logged_command(optimize_cmd, log_path=optimize_log_path, workdir=workspace.repo_root)

    preexisting_hefs = {p.resolve() for p in compiler_output_dir.glob("*.hef")}
    _run_logged_command(compiler_cmd, log_path=compiler_log_path, workdir=workspace.repo_root)
    post_hefs = sorted(compiler_output_dir.glob("*.hef"), key=lambda p: p.stat().st_mtime)
    new_hefs = [p for p in post_hefs if p.resolve() not in preexisting_hefs]
    compiled_hef_path = new_hefs[-1] if new_hefs else None
    if compiled_hef_path is None:
        expected = _expected_compiler_output_hef(optimized_har_path)
        if expected.exists():
            compiled_hef_path = expected
    if compiled_hef_path is None:
        raise FileNotFoundError(f"No HEF was produced by Hailo compiler for {spec.model_id} {target_arch}")

    compiled_hef_path.replace(final_hef_path)

    command_records = [
        {
            "stage": "parser",
            "command": parser_cmd,
            "log_path": _repo_relative(workspace.repo_root, parser_log_path),
        },
        {
            "stage": "optimize",
            "command": optimize_cmd,
            "log_path": _repo_relative(workspace.repo_root, optimize_log_path),
        },
        {
            "stage": "compiler",
            "command": compiler_cmd,
            "log_path": _repo_relative(workspace.repo_root, compiler_log_path),
        },
    ]
    save_json(commands_path, {"commands": command_records})

    manifest_path = workspace.manifest_paths[(spec.model_id, target_arch)]
    manifest = load_json(manifest_path, default={})
    manifest.update(runtime_contract)
    manifest.update(
        {
            "model_id": spec.model_id,
            "arch": spec.arch,
            "variant": spec.variant,
            "target_arch": target_arch,
            "source_onnx": _repo_relative(workspace.repo_root, onnx_path),
            "hef_path": _repo_relative(workspace.repo_root, final_hef_path),
            "compiler_version": compiler_version,
            "hailort_version": hailort_version,
            "calibration_source": _repo_relative(workspace.repo_root, workspace.calibration_dir),
            "calibration_npy": _repo_relative(workspace.repo_root, calibration_npy),
            "calibration_metadata": _repo_relative(workspace.repo_root, calibration_metadata),
            "native_har": _repo_relative(workspace.repo_root, native_har_path),
            "optimized_har": _repo_relative(workspace.repo_root, optimized_har_path),
            "compiled_har": _repo_relative(workspace.repo_root, compiled_har_path) if compiled_har_path.exists() else None,
            "augmented_onnx": _repo_relative(workspace.repo_root, augmented_onnx_path) if augmented_onnx_path.exists() else None,
            "parsing_report": _repo_relative(workspace.repo_root, parsing_report_path) if parsing_report_path.exists() else None,
            "parser_log": _repo_relative(workspace.repo_root, parser_log_path),
            "optimize_log": _repo_relative(workspace.repo_root, optimize_log_path),
            "compiler_log": _repo_relative(workspace.repo_root, compiler_log_path),
            "commands_path": _repo_relative(workspace.repo_root, commands_path),
            "compiler_output_dir": _repo_relative(workspace.repo_root, compiler_output_dir),
            "model_script": _repo_relative(workspace.repo_root, model_script_path),
            "compiled": True,
            "postprocess": "host_softmax_threshold",
        }
    )
    save_json(manifest_path, manifest)
    return final_hef_path


def _mark_compile_failure(
    *,
    spec: HailoCheckpointSpec,
    target_arch: str,
    workspace: HailoWorkspace,
    runtime_contract: Dict[str, object],
    error: Exception,
) -> None:
    manifest_path = workspace.manifest_paths[(spec.model_id, target_arch)]
    manifest = load_json(manifest_path, default={})
    manifest.update(runtime_contract)
    manifest.update(
        {
            "compiled": False,
            "deployable": False,
            "deployment_decision": "compile_failed",
            "compile_error": str(error),
            "parser_log": _repo_relative(workspace.repo_root, _parser_log_path(spec, target_arch)),
            "optimize_log": _repo_relative(workspace.repo_root, _optimize_log_path(spec, target_arch)),
            "compiler_log": _repo_relative(workspace.repo_root, _compiler_log_path(spec, target_arch)),
            "model_script": _repo_relative(workspace.repo_root, _model_script_path(spec, target_arch))
            if _model_script_path(spec, target_arch).exists()
            else None,
        }
    )
    save_json(manifest_path, manifest)


def run_hailo_prepare(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    targets = tuple(args.target or DEFAULT_TARGET_ARCHS)
    workspace = _prepare_hailo_workspace(
        repo_root=repo_root,
        targets=targets,
        include_strict_far=args.include_strict_far,
        onnx_opset=args.onnx_opset,
        skip_existing_exports=args.skip_existing,
        compiler_version=args.compiler_version,
        hailort_version=args.hailort_version,
    )

    print(
        f"Prepared Hailo workspace for {len(workspace.specs)} checkpoints "
        f"across {len(targets)} targets ({len(workspace.manifest_paths)} manifests)."
    )
    print(f"Calibration directory: {workspace.calibration_dir}")
    print(f"Inventory written: {workspace.inventory_path}")


def run_hailo_validate(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    targets = tuple(args.target or DEFAULT_TARGET_ARCHS)
    workspace = _prepare_hailo_workspace(
        repo_root=repo_root,
        targets=targets,
        include_strict_far=args.include_strict_far,
        onnx_opset=args.onnx_opset,
        skip_existing_exports=True,
        compiler_version=args.compiler_version,
        hailort_version=args.hailort_version,
    )

    selected_model_ids = set(args.model_id or [])
    specs = [spec for spec in workspace.specs if not selected_model_ids or spec.model_id in selected_model_ids]
    if selected_model_ids and not specs:
        raise ValueError(f"No requested model ids were found: {sorted(selected_model_ids)}")

    reports: List[Path] = []
    for spec in specs:
        export_summary = workspace.export_summaries[spec.model_id]
        for target_arch in targets:
            report_path = _validate_single_target(
                spec=spec,
                target_arch=target_arch,
                export_summary=export_summary,
                workspace=workspace,
                dataset_dir=repo_root / args.validation_dataset_dir,
                real_probe_samples=args.real_probe_samples,
                seed=args.seed,
                runtime_metrics_json=Path(args.runtime_metrics_json).resolve() if args.runtime_metrics_json else None,
                hardware_results_json=Path(args.hardware_results_json).resolve() if args.hardware_results_json else None,
            )
            reports.append(report_path)
            print(f"Validation report written: {report_path}")

    print(f"Validation reports: {len(reports)}")


def run_hailo_compile(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    targets = tuple(args.target or DEFAULT_TARGET_ARCHS)
    hailo_bin = _resolve_tool_binary(args.hailo_bin, _default_hailo_candidates())

    hailortcli_bin = None
    try:
        hailortcli_bin = _resolve_tool_binary(args.hailortcli_bin, _default_hailortcli_candidates())
    except FileNotFoundError:
        hailortcli_bin = None

    compiler_version, hailort_version = _detect_tool_versions(hailo_bin, hailortcli_bin)
    workspace = _prepare_hailo_workspace(
        repo_root=repo_root,
        targets=targets,
        include_strict_far=args.include_strict_far,
        onnx_opset=args.onnx_opset,
        skip_existing_exports=True,
        compiler_version=compiler_version,
        hailort_version=hailort_version,
    )

    requested_calibration_npy = Path(args.calibration_npy).resolve() if args.calibration_npy else None
    selected_model_ids = set(args.model_id or [])
    specs = [spec for spec in workspace.specs if not selected_model_ids or spec.model_id in selected_model_ids]
    if selected_model_ids and not specs:
        raise ValueError(f"No requested model ids were found: {sorted(selected_model_ids)}")

    compiled_hefs: List[Path] = []
    failures: List[str] = []
    dataset_stats_cache: Dict[str, Dict[str, object]] = {}
    generic_calibration_cache: Dict[str, tuple[Path, Path]] = {}
    validation_reports: List[Path] = []

    for spec in specs:
        export_summary = workspace.export_summaries[spec.model_id]
        calibration_npy, calibration_metadata, runtime_contract = _resolve_calibration_assets(
            spec=spec,
            repo_root=repo_root,
            dataset_dir=repo_root / args.calibration_dataset_dir,
            requested_calibration_npy=requested_calibration_npy,
            requested_samples=args.calibration_samples,
            seed=args.seed,
            dataset_stats_cache=dataset_stats_cache,
            generic_cache=generic_calibration_cache,
        )
        for target_arch in targets:
            _refresh_manifest_runtime_contract(workspace.manifest_paths[(spec.model_id, target_arch)], runtime_contract)

        targets_requiring_compile = [
            target_arch
            for target_arch in targets
            if not (
                args.skip_existing
                and _existing_hef_is_reusable(
                    spec=spec,
                    target_arch=target_arch,
                    runtime_contract=runtime_contract,
                )
            )
        ]
        if not targets_requiring_compile:
            for target_arch in targets:
                final_hef_path = _hef_path(spec, target_arch)
                existing_report_path = _reusable_validation_report_path(spec, target_arch, runtime_contract)
                if existing_report_path is not None:
                    print(f"Skipping existing HEF and validation: {final_hef_path}")
                    compiled_hefs.append(final_hef_path)
                    validation_reports.append(existing_report_path)
                    continue

                print(f"Skipping existing HEF: {final_hef_path}")
                compiled_hefs.append(final_hef_path)
                report_path = _validate_single_target(
                    spec=spec,
                    target_arch=target_arch,
                    export_summary=export_summary,
                    workspace=workspace,
                    dataset_dir=repo_root / args.calibration_dataset_dir,
                    real_probe_samples=args.real_probe_samples,
                    seed=args.seed,
                    runtime_metrics_json=Path(args.runtime_metrics_json).resolve()
                    if args.runtime_metrics_json
                    else None,
                    hardware_results_json=Path(args.hardware_results_json).resolve()
                    if args.hardware_results_json
                    else None,
                )
                validation_reports.append(report_path)
            continue

        for target_arch in targets:
            final_hef_path = _hef_path(spec, target_arch)
            if args.skip_existing and final_hef_path.exists():
                if _existing_hef_is_reusable(
                    spec=spec,
                    target_arch=target_arch,
                    runtime_contract=runtime_contract,
                ):
                    existing_report_path = _reusable_validation_report_path(spec, target_arch, runtime_contract)
                    if existing_report_path is not None:
                        print(f"Skipping existing HEF and validation: {final_hef_path}")
                        compiled_hefs.append(final_hef_path)
                        validation_reports.append(existing_report_path)
                        continue

                    print(f"Skipping existing HEF: {final_hef_path}")
                    compiled_hefs.append(final_hef_path)
                    report_path = _validate_single_target(
                        spec=spec,
                        target_arch=target_arch,
                        export_summary=export_summary,
                        workspace=workspace,
                        dataset_dir=repo_root / args.calibration_dataset_dir,
                        real_probe_samples=args.real_probe_samples,
                        seed=args.seed,
                        runtime_metrics_json=Path(args.runtime_metrics_json).resolve()
                        if args.runtime_metrics_json
                        else None,
                        hardware_results_json=Path(args.hardware_results_json).resolve()
                        if args.hardware_results_json
                        else None,
                    )
                    validation_reports.append(report_path)
                    continue

                print(f"Existing HEF is stale for {spec.model_id} [{target_arch}]; recompiling")

            print(f"Compiling {spec.model_id} for {target_arch}")
            try:
                hef_path = _compile_single_target(
                    spec=spec,
                    target_arch=target_arch,
                    export_summary=export_summary,
                    workspace=workspace,
                    hailo_bin=hailo_bin,
                    calibration_npy=calibration_npy,
                    calibration_metadata=calibration_metadata,
                    runtime_contract=runtime_contract,
                    compiler_version=compiler_version,
                    hailort_version=hailort_version,
                )
                compiled_hefs.append(hef_path)
                report_path = _validate_single_target(
                    spec=spec,
                    target_arch=target_arch,
                    export_summary=export_summary,
                    workspace=workspace,
                    dataset_dir=repo_root / args.calibration_dataset_dir,
                    real_probe_samples=args.real_probe_samples,
                    seed=args.seed,
                    runtime_metrics_json=Path(args.runtime_metrics_json).resolve()
                    if args.runtime_metrics_json
                    else None,
                    hardware_results_json=Path(args.hardware_results_json).resolve()
                    if args.hardware_results_json
                    else None,
                )
                validation_reports.append(report_path)
            except Exception as exc:
                _mark_compile_failure(
                    spec=spec,
                    target_arch=target_arch,
                    workspace=workspace,
                    runtime_contract=runtime_contract,
                    error=exc,
                )
                message = f"{spec.model_id} [{target_arch}] failed: {exc}"
                failures.append(message)
                print(message, file=sys.stderr)
                if not args.keep_going:
                    raise

    if validation_reports:
        print(f"Validation reports: {len(validation_reports)}")
    print(f"Compiled HEFs: {len(compiled_hefs)}")
    if failures:
        print(f"Failures: {len(failures)}", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise RuntimeError("One or more Hailo compilation targets failed")


def add_hailo_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_hailo = subparsers.add_parser("hailo", help="Prepare, compile, and validate Hailo artifacts")
    hailo_subparsers = p_hailo.add_subparsers(dest="hailo_command", required=True)

    p_prepare = hailo_subparsers.add_parser(
        "prepare",
        help="Export static-batch ONNX artifacts and write Hailo manifest stubs.",
    )
    p_prepare.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    p_prepare.add_argument("--target", action="append", choices=SUPPORTED_TARGET_ARCHS, default=[])
    p_prepare.add_argument("--onnx-opset", type=int, default=17)
    p_prepare.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Reuse an existing ONNX export summary when the exported ONNX file still exists.",
    )
    p_prepare.add_argument(
        "--no-strict-far",
        action="store_false",
        dest="include_strict_far",
        default=True,
        help="Only prepare the base architecture checkpoints.",
    )
    p_prepare.add_argument("--compiler-version", type=str, default=None)
    p_prepare.add_argument("--hailort-version", type=str, default=None)
    p_prepare.set_defaults(func=run_hailo_prepare)

    p_compile = hailo_subparsers.add_parser(
        "compile",
        help="Generate calibration data and compile exported ONNX models into HEF artifacts.",
    )
    p_compile.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    p_compile.add_argument("--target", action="append", choices=SUPPORTED_TARGET_ARCHS, default=[])
    p_compile.add_argument("--onnx-opset", type=int, default=17)
    p_compile.add_argument(
        "--no-strict-far",
        action="store_false",
        dest="include_strict_far",
        default=True,
        help="Only compile the base architecture checkpoints.",
    )
    p_compile.add_argument("--model-id", action="append", default=[])
    p_compile.add_argument(
        "--calibration-dataset-dir",
        type=str,
        default=str(DEFAULT_CALIBRATION_DATASET),
        help="Dataset directory containing NPZ shards with `feat` and `y` arrays.",
    )
    p_compile.add_argument(
        "--calibration-npy",
        type=str,
        default=None,
        help="Use an existing calibration .npy instead of generating one from NPZ shards.",
    )
    p_compile.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Representative PSD frames to include in the generated calibration array.",
    )
    p_compile.add_argument(
        "--real-probe-samples",
        type=int,
        default=DEFAULT_VALIDATION_REAL_SAMPLES,
        help="Balanced real probe frames to include in validation reports.",
    )
    p_compile.add_argument("--seed", type=int, default=13)
    p_compile.add_argument("--hailo-bin", type=str, default=None)
    p_compile.add_argument("--hailortcli-bin", type=str, default=None)
    p_compile.add_argument(
        "--runtime-metrics-json",
        type=str,
        default=None,
        help="Optional JSON file with on-device Hailo-8 latency metrics used for the runtime gate.",
    )
    p_compile.add_argument(
        "--hardware-results-json",
        type=str,
        default=None,
        help="Optional JSON file with externally measured HEF logits for synthetic and real probes.",
    )
    p_compile.add_argument("--skip-existing", action="store_true", default=False)
    p_compile.add_argument("--keep-going", action="store_true", default=False)
    p_compile.set_defaults(func=run_hailo_compile)

    p_validate = hailo_subparsers.add_parser(
        "validate",
        help="Validate parity, flat-output behavior, and deployability for compiled Hailo artifacts.",
    )
    p_validate.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    p_validate.add_argument("--target", action="append", choices=SUPPORTED_TARGET_ARCHS, default=[])
    p_validate.add_argument("--onnx-opset", type=int, default=17)
    p_validate.add_argument(
        "--no-strict-far",
        action="store_false",
        dest="include_strict_far",
        default=True,
        help="Only validate the base architecture checkpoints.",
    )
    p_validate.add_argument("--model-id", action="append", default=[])
    p_validate.add_argument(
        "--validation-dataset-dir",
        type=str,
        default=str(DEFAULT_CALIBRATION_DATASET),
        help="Dataset directory containing NPZ shards used for probe generation and feature stats.",
    )
    p_validate.add_argument(
        "--real-probe-samples",
        type=int,
        default=DEFAULT_VALIDATION_REAL_SAMPLES,
        help="Balanced real probe frames to include in validation reports.",
    )
    p_validate.add_argument("--seed", type=int, default=13)
    p_validate.add_argument(
        "--runtime-metrics-json",
        type=str,
        default=None,
        help="Optional JSON file with on-device latency metrics used for the runtime gate.",
    )
    p_validate.add_argument(
        "--hardware-results-json",
        type=str,
        default=None,
        help="Optional JSON file with externally measured HEF logits for synthetic/real probes.",
    )
    p_validate.add_argument("--compiler-version", type=str, default=None)
    p_validate.add_argument("--hailort-version", type=str, default=None)
    p_validate.set_defaults(func=run_hailo_validate)
