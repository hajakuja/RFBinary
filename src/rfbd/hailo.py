"""Hailo preparation and compilation utilities for multi-model workflows."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .export import export_checkpoint
from .io import iter_npz_files, load_json, load_npz_shard, save_json


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = Path("data/experiments/g2_edge_suite")
BASE_MODEL_ROOT = SUITE_ROOT / "models"
BASE_EXPORT_ROOT = SUITE_ROOT / "exports"
STRICT_MODEL_ROOT = SUITE_ROOT / "strict_far" / "models"
STRICT_EXPORT_ROOT = SUITE_ROOT / "strict_far" / "exports"
CALIBRATION_ROOT = SUITE_ROOT / "hailo_calibration"
DEFAULT_CALIBRATION_DATASET = Path("data/datasets/binary_all_v1")
DEFAULT_TARGET_ARCHS = ("hailo8", "hailo8l")
DEFAULT_CALIBRATION_SAMPLES = 256

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


def _stage_prefix(spec: HailoCheckpointSpec, target_arch: str) -> Path:
    return spec.hailo_dir / f"{spec.checkpoint_path.stem}.{target_arch}"


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


def build_hailo_manifest(
    spec: HailoCheckpointSpec,
    export_summary: Dict[str, object],
    *,
    target_arch: str,
    repo_root: Path,
    calibration_dir: Path,
    compiler_version: str | None = None,
    hailort_version: str | None = None,
) -> Dict[str, object]:
    export_contract = export_summary.get("export_contract", {})
    onnx_path = Path(str(export_summary.get("exported", {}).get("onnx", "")))

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

    for spec in specs:
        export_summary = _export_for_hailo(
            spec,
            skip_existing=skip_existing_exports,
            onnx_opset=onnx_opset,
        )
        export_summaries[spec.model_id] = export_summary

        spec.hailo_dir.mkdir(parents=True, exist_ok=True)
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


def build_hailo_calibration_set(
    *,
    dataset_dir: str | Path,
    output_path: str | Path,
    max_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    seed: int = 13,
) -> tuple[Path, Path]:
    dataset_root = Path(dataset_dir)
    shards = iter_npz_files(dataset_root)
    if not shards:
        raise FileNotFoundError(f"No calibration NPZ shards found under {dataset_root}")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    rng = np.random.default_rng(seed)
    per_class_cap = max_samples
    class_buffers: Dict[int, List[np.ndarray]] = {0: [], 1: []}
    class_counts = {0: 0, 1: 0}
    source_shards: List[str] = []

    for shard_path in shards:
        shard = load_npz_shard(shard_path)
        feat = np.asarray(shard["feat"], dtype=np.float32)
        y = np.asarray(shard["y"], dtype=np.int64)
        if feat.ndim != 3:
            raise ValueError(f"Expected shard features shaped (N,H,W), got {feat.shape} in {shard_path}")

        source_shards.append(str(shard_path))
        for cls in (0, 1):
            if class_counts[cls] >= per_class_cap:
                continue
            cls_idx = np.flatnonzero(y == cls)
            if cls_idx.size == 0:
                continue
            rng.shuffle(cls_idx)
            need = min(per_class_cap - class_counts[cls], int(cls_idx.size))
            if need <= 0:
                continue
            take_idx = cls_idx[:need]
            class_buffers[cls].append(feat[take_idx])
            class_counts[cls] += need

        if class_counts[0] >= per_class_cap and class_counts[1] >= per_class_cap:
            break

    class_arrays: Dict[int, np.ndarray] = {}
    for cls in (0, 1):
        if class_buffers[cls]:
            class_arrays[cls] = np.concatenate(class_buffers[cls], axis=0)
        else:
            class_arrays[cls] = np.empty((0, 224, 224), dtype=np.float32)

    target_take = {
        0: max_samples // 2,
        1: max_samples - (max_samples // 2),
    }
    selected: List[np.ndarray] = []
    used_counts = {0: 0, 1: 0}
    for cls in (0, 1):
        take = min(int(len(class_arrays[cls])), target_take[cls])
        if take > 0:
            selected.append(class_arrays[cls][:take])
            used_counts[cls] = take

    chosen_count = sum(used_counts.values())
    shortfall = max_samples - chosen_count
    if shortfall > 0:
        extras: List[np.ndarray] = []
        for cls in (0, 1):
            extras_arr = class_arrays[cls][used_counts[cls] :]
            if len(extras_arr):
                extras.append(extras_arr)
        if extras:
            extra_pool = np.concatenate(extras, axis=0)
            perm = rng.permutation(len(extra_pool))
            extra_pool = extra_pool[perm]
            take = min(shortfall, int(len(extra_pool)))
            if take > 0:
                selected.append(extra_pool[:take])
                shortfall -= take
    if shortfall > 0:
        raise ValueError(
            f"Unable to build calibration set of {max_samples} samples from {dataset_root}; "
            f"only gathered {max_samples - shortfall}"
        )

    calib_nchw = np.concatenate(selected, axis=0).astype(np.float32, copy=False)
    perm = rng.permutation(len(calib_nchw))
    calib_nchw = calib_nchw[perm]
    calib_nhwc = calib_nchw[..., None]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, calib_nhwc)

    metadata_path = output.with_suffix(".metadata.json")
    metadata = {
        "dataset_dir": str(dataset_root),
        "output_path": str(output),
        "sample_count": int(calib_nhwc.shape[0]),
        "shape": list(calib_nhwc.shape),
        "dtype": str(calib_nhwc.dtype),
        "source_shards": source_shards,
        "requested_max_samples": int(max_samples),
        "selected_class_counts": used_counts,
        "min_value": float(calib_nhwc.min()),
        "max_value": float(calib_nhwc.max()),
        "seed": int(seed),
    }
    save_json(metadata_path, metadata)
    return output, metadata_path


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


def _model_script_lines(spec: HailoCheckpointSpec) -> List[str]:
    lines: List[str] = []
    if spec.arch == "mobilenet_v3_small":
        lines.append(
            "pre_quantization_optimization(global_avgpool_reduction, layers=avgpool1, division_factors=[4, 4])"
        )
    return lines


def _write_model_script(spec: HailoCheckpointSpec, target_arch: str) -> Path | None:
    lines = _model_script_lines(spec)
    if not lines:
        return None
    path = _model_script_path(spec, target_arch)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _compile_single_target(
    *,
    spec: HailoCheckpointSpec,
    target_arch: str,
    export_summary: Dict[str, object],
    workspace: HailoWorkspace,
    hailo_bin: str,
    calibration_npy: Path,
    calibration_metadata: Path,
    compiler_version: str | None,
    hailort_version: str | None,
) -> Path:
    onnx_path = Path(str(export_summary.get("exported", {}).get("onnx", "")))
    if not onnx_path.exists():
        raise FileNotFoundError(f"Expected ONNX export missing for {spec.model_id}: {onnx_path}")

    output_name = str(export_summary.get("export_contract", {}).get("output_name", "logits"))
    stage_prefix = _stage_prefix(spec, target_arch)
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
    model_script_path = _write_model_script(spec, target_arch)

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

    calibration_npy = Path(args.calibration_npy).resolve() if args.calibration_npy else None
    calibration_metadata = calibration_npy.with_suffix(".metadata.json") if calibration_npy else None
    if calibration_npy is None:
        calibration_npy, calibration_metadata = build_hailo_calibration_set(
            dataset_dir=repo_root / args.calibration_dataset_dir,
            output_path=repo_root / CALIBRATION_ROOT / f"rfbd_calibration_{args.calibration_samples}.npy",
            max_samples=args.calibration_samples,
            seed=args.seed,
        )
    else:
        if not calibration_npy.exists():
            raise FileNotFoundError(f"Calibration NPY does not exist: {calibration_npy}")
        if calibration_metadata is None or not calibration_metadata.exists():
            calibration_metadata = calibration_npy.with_suffix(".metadata.json")
            if not calibration_metadata.exists():
                save_json(
                    calibration_metadata,
                    {
                        "dataset_dir": None,
                        "output_path": str(calibration_npy),
                        "sample_count": None,
                        "shape": None,
                        "dtype": None,
                        "source_shards": [],
                        "requested_max_samples": None,
                        "selected_class_counts": {},
                        "seed": args.seed,
                    },
                )

    selected_model_ids = set(args.model_id or [])
    specs = [
        spec for spec in workspace.specs if not selected_model_ids or spec.model_id in selected_model_ids
    ]
    if selected_model_ids and not specs:
        raise ValueError(f"No requested model ids were found: {sorted(selected_model_ids)}")

    compiled_hefs: List[Path] = []
    failures: List[str] = []
    for spec in specs:
        export_summary = workspace.export_summaries[spec.model_id]
        for target_arch in targets:
            final_hef_path = _hef_path(spec, target_arch)
            if args.skip_existing and final_hef_path.exists():
                print(f"Skipping existing HEF: {final_hef_path}")
                compiled_hefs.append(final_hef_path)
                continue

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
                    compiler_version=compiler_version,
                    hailort_version=hailort_version,
                )
                compiled_hefs.append(hef_path)
            except Exception as exc:
                message = f"{spec.model_id} [{target_arch}] failed: {exc}"
                failures.append(message)
                print(message, file=sys.stderr)
                if not args.keep_going:
                    raise

    print(f"Calibration NPY: {calibration_npy}")
    print(f"Compiled HEFs: {len(compiled_hefs)}")
    if failures:
        print(f"Failures: {len(failures)}", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise RuntimeError("One or more Hailo compilation targets failed")


def add_hailo_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_hailo = subparsers.add_parser("hailo", help="Prepare and compile Hailo artifacts")
    hailo_subparsers = p_hailo.add_subparsers(dest="hailo_command", required=True)

    p_prepare = hailo_subparsers.add_parser(
        "prepare",
        help="Export static-batch ONNX artifacts and write Hailo manifest stubs.",
    )
    p_prepare.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    p_prepare.add_argument("--target", action="append", choices=DEFAULT_TARGET_ARCHS, default=[])
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
    p_prepare.add_argument(
        "--compiler-version",
        type=str,
        default=None,
        help="Optional recorded Hailo compiler version for manifest stubs.",
    )
    p_prepare.add_argument(
        "--hailort-version",
        type=str,
        default=None,
        help="Optional recorded HailoRT version for manifest stubs.",
    )
    p_prepare.set_defaults(func=run_hailo_prepare)

    p_compile = hailo_subparsers.add_parser(
        "compile",
        help="Generate calibration data and compile exported ONNX models into HEF artifacts.",
    )
    p_compile.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    p_compile.add_argument("--target", action="append", choices=DEFAULT_TARGET_ARCHS, default=[])
    p_compile.add_argument("--onnx-opset", type=int, default=17)
    p_compile.add_argument(
        "--no-strict-far",
        action="store_false",
        dest="include_strict_far",
        default=True,
        help="Only compile the base architecture checkpoints.",
    )
    p_compile.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Limit compilation to one or more specific model ids.",
    )
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
        help="Use an existing NHWC calibration .npy instead of generating one from NPZ shards.",
    )
    p_compile.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of representative PSD frames to include in the generated calibration array.",
    )
    p_compile.add_argument("--seed", type=int, default=13)
    p_compile.add_argument(
        "--hailo-bin",
        type=str,
        default=None,
        help="Path to the `hailo` CLI. Defaults to the local Hailo suite venv if present.",
    )
    p_compile.add_argument(
        "--hailortcli-bin",
        type=str,
        default=None,
        help="Optional path to `hailortcli` for version detection.",
    )
    p_compile.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip a target when the final HEF already exists.",
    )
    p_compile.add_argument(
        "--keep-going",
        action="store_true",
        default=False,
        help="Continue compiling remaining targets after a failure and report all failures at the end.",
    )
    p_compile.set_defaults(func=run_hailo_compile)
