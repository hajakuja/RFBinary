"""Batch orchestration for no-drone raw IQ ingestion with remote read-only safety."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .contracts import MANIFEST_COLUMNS
from .extract import run_extract
from .io import load_json, save_json


FILENAME_RE = re.compile(
    r"^(?P<capture_id>[^_]+)_(?P<timestamp>\d{8}T\d{6}Z)_(?P<center>\d+)Hz_(?P<sps>\d+)sps_(?P<gain>\d+)g_(?P<session>s\d+)\.s16$"
)


@dataclass(frozen=True)
class SourceFile:
    rel_path: str
    abs_path: str
    size: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except Exception:
        return False


def _guard_delete_path(path: Path, source_dir: Path) -> None:
    if _is_under(path, source_dir):
        raise RuntimeError(f"Refusing delete under source mount: {path}")


def snapshot_source_dir(source_dir: Path) -> List[SourceFile]:
    files = []
    for p in sorted(source_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(source_dir))
            files.append(SourceFile(rel_path=rel, abs_path=str(p), size=int(p.stat().st_size)))
    return files


def _snapshot_payload(files: Sequence[SourceFile]) -> Dict[str, Any]:
    return {
        "count": len(files),
        "total_bytes": int(sum(f.size for f in files)),
        "files": [{"rel_path": f.rel_path, "size": f.size} for f in files],
    }


def _snapshot_signature(files: Sequence[SourceFile]) -> Dict[str, int]:
    return {f.rel_path: int(f.size) for f in files}


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


def _parse_filename_metadata(name: str) -> Dict[str, str]:
    m = FILENAME_RE.match(name)
    if not m:
        return {}
    g = m.groupdict()
    return {
        "capture_id": g["capture_id"],
        "timestamp": datetime.strptime(g["timestamp"], "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "center_freq_hz": g["center"],
        "sample_rate_sps": g["sps"],
        "gain_db": g["gain"],
        "session_id": g["session"],
    }


def clean_and_validate_manifest(
    manifest_path: Path,
    source_dir: Path,
    source_files: Sequence[SourceFile],
    default_label: str = "no_drone",
    default_source_domain: str = "custom_bg",
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    _validate_required_columns(df)

    if df["capture_id"].duplicated().any():
        dups = df.loc[df["capture_id"].duplicated(), "capture_id"].tolist()
        raise ValueError(f"Duplicate capture_id values in manifest: {dups[:10]}")

    source_by_name: Dict[str, List[SourceFile]] = {}
    for sf in source_files:
        source_by_name.setdefault(Path(sf.rel_path).name, []).append(sf)

    cleaned_rows: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        row = dict(row)
        raw_path = Path(str(row.get("file_path", "")))
        file_name = raw_path.name

        resolved: Path | None = None
        if raw_path.exists() and _is_under(raw_path, source_dir):
            resolved = raw_path
        elif file_name in source_by_name:
            matches = source_by_name[file_name]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous source match for file name {file_name}: {len(matches)} candidates")
            resolved = Path(matches[0].abs_path)

        if resolved is None or not resolved.exists():
            raise FileNotFoundError(f"Manifest file not found in source directory: {row.get('file_path')}")

        fname_meta = _parse_filename_metadata(file_name)
        for key, value in fname_meta.items():
            if pd.isna(row.get(key)) or str(row.get(key)).strip() == "":
                row[key] = value

        if pd.isna(row.get("label")) or str(row.get("label")).strip() == "":
            row["label"] = default_label
        if pd.isna(row.get("source_domain")) or str(row.get("source_domain")).strip() == "":
            row["source_domain"] = default_source_domain
        if pd.isna(row.get("environment")) or str(row.get("environment")).strip() == "":
            row["environment"] = "unknown"

        row["file_path"] = str(resolved)
        cleaned_rows.append(row)

    clean_df = pd.DataFrame(cleaned_rows)
    _validate_required_columns(clean_df)

    # Final existence and uniqueness checks.
    for p in clean_df["file_path"].tolist():
        if not Path(p).exists():
            raise FileNotFoundError(f"Cleaned manifest path does not exist: {p}")
    if clean_df["capture_id"].duplicated().any():
        raise ValueError("Duplicate capture_id values after cleaning")

    return clean_df


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _save_completed_ids(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(set(ids))) + "\n", encoding="utf-8")


def _append_ledger(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _resolve_runtime_paths(args: argparse.Namespace, source_dir: Path) -> Tuple[str, Path, Path]:
    run_id = str(getattr(args, "run_id", "") or source_dir.name).strip()
    if not run_id:
        raise ValueError("Could not derive run_id. Pass --run-id explicitly.")

    staging_dir_arg = getattr(args, "staging_dir", None)
    features_dir_arg = getattr(args, "features_dir", None)
    prefix_arg = getattr(args, "prefix", None)

    staging_dir = (
        Path(staging_dir_arg)
        if staging_dir_arg
        else Path("/root/RFBinaryDetect/data/staging") / run_id
    )
    features_dir = (
        Path(features_dir_arg)
        if features_dir_arg
        else Path("/root/RFBinaryDetect/data/no_drone_features") / run_id
    )
    prefix = str(prefix_arg or run_id)
    return prefix, staging_dir, features_dir


def _state_identity_payload(source_dir: Path, manifest_path: Path, run_id: str) -> Dict[str, str]:
    return {
        "source_dir": str(source_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "run_id": run_id,
    }


def _enforce_state_identity(state_identity_path: Path, incoming: Dict[str, str], allow_reuse: bool) -> None:
    if state_identity_path.exists():
        existing = load_json(state_identity_path)
        keys = ("source_dir", "manifest_path", "run_id")
        mismatch = any(str(existing.get(k, "")) != str(incoming.get(k, "")) for k in keys)
        if mismatch and not allow_reuse:
            raise RuntimeError(
                "State directory identity mismatch. Existing state points to "
                f"{existing}, incoming is {incoming}. Use a new state/features directory, "
                "or pass --allow-state-reuse to override."
            )
    save_json(state_identity_path, incoming)


def _build_batches(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    batches = []
    for i in range(0, len(df), batch_size):
        batches.append(df.iloc[i : i + batch_size].copy())
    return batches


def _copy_batch_to_staging(batch_df: pd.DataFrame, staging_batch_dir: Path, source_dir: Path) -> Tuple[pd.DataFrame, int]:
    staging_batch_dir.mkdir(parents=True, exist_ok=True)
    copied_bytes = 0

    out_df = batch_df.copy()
    staged_paths = []
    for p_str in out_df["file_path"].tolist():
        src = Path(str(p_str))
        if not _is_under(src, source_dir):
            raise RuntimeError(f"Source file is outside source_dir guard: {src}")
        dst = staging_batch_dir / src.name
        shutil.copy2(src, dst)
        copied_bytes += int(dst.stat().st_size)
        staged_paths.append(str(dst))

    out_df["file_path"] = staged_paths
    return out_df, copied_bytes


def _delete_staged_files(batch_manifest_df: pd.DataFrame, source_dir: Path) -> int:
    reclaimed = 0
    for p_str in batch_manifest_df["file_path"].tolist():
        p = Path(str(p_str))
        if p.exists():
            _guard_delete_path(p, source_dir)
            reclaimed += int(p.stat().st_size)
            p.unlink()
    return reclaimed


def run_no_drone_batches(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    manifest_path = Path(args.manifest)
    prefix, staging_dir, features_dir = _resolve_runtime_paths(args, source_dir)
    run_id = str(getattr(args, "run_id", "") or source_dir.name).strip()
    state_dir = Path(args.state_dir) if args.state_dir else features_dir / "state"
    allow_state_reuse = bool(getattr(args, "allow_state_reuse", False))

    features_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    source_pre_path = state_dir / "source_snapshot_pre.json"
    source_post_path = state_dir / "source_snapshot_post.json"
    manifest_clean_path = state_dir / "manifest_clean.csv"
    ledger_path = state_dir / "batch_ledger.jsonl"
    completed_path = state_dir / "completed_capture_ids.txt"
    failed_path = state_dir / "failed_batches.json"
    final_report_path = state_dir / "final_report.json"
    state_identity_path = state_dir / "state_identity.json"

    _enforce_state_identity(
        state_identity_path,
        _state_identity_payload(source_dir, manifest_path, run_id),
        allow_reuse=allow_state_reuse,
    )

    pre_files = snapshot_source_dir(source_dir)
    pre_payload = _snapshot_payload(pre_files)
    if not source_pre_path.exists():
        save_json(source_pre_path, pre_payload)

    clean_df = clean_and_validate_manifest(manifest_path, source_dir, pre_files)
    clean_df.to_csv(manifest_clean_path, index=False)

    completed = _load_completed_ids(completed_path)
    failed_state = load_json(failed_path, default={"failed_batches": []})
    failed_batches = failed_state.get("failed_batches", [])
    failed_capture_ids = set()
    for item in failed_batches:
        for cid in item.get("capture_ids", []):
            failed_capture_ids.add(str(cid))

    pending_df = clean_df[~clean_df["capture_id"].isin(completed)].copy()
    if not args.retry_failed:
        pending_df = pending_df[~pending_df["capture_id"].isin(failed_capture_ids)].copy()

    pending_df = pending_df.reset_index(drop=True)
    batches = _build_batches(pending_df, batch_size=int(args.batch_size))

    total_copied = 0
    total_reclaimed = 0
    completed_this_run = 0
    failed_this_run = 0
    started_at = time.time()

    for batch_idx, batch_df in enumerate(batches):
        batch_id = f"batch_{batch_idx:05d}"
        batch_capture_ids = batch_df["capture_id"].astype(str).tolist()
        batch_started = time.time()
        staging_batch_dir = staging_dir / batch_id
        batch_manifest_path = state_dir / f"{batch_id}_manifest.csv"

        try:
            copy_started = time.time()
            batch_local_df, copied_bytes = _copy_batch_to_staging(batch_df, staging_batch_dir, source_dir)
            copy_seconds = time.time() - copy_started
            total_copied += copied_bytes

            batch_local_df.to_csv(batch_manifest_path, index=False)

            extract_started = time.time()
            extract_args = argparse.Namespace(
                adapter="custom-iq",
                manifest=str(batch_manifest_path),
                dronerf_root=None,
                out_dir=str(features_dir),
                prefix=prefix,
                workers=int(args.workers),
                shard_size=int(args.shard_size),
                resume=True,
                segment_ms=int(args.segment_ms),
                nfft=int(args.nfft),
                noverlap=int(args.noverlap),
                resize_h=int(args.resize_h),
                resize_w=int(args.resize_w),
                no_log_power=bool(args.no_log_power),
                no_normalize=bool(args.no_normalize),
                cache_dir=None,
                build_cache=False,
                dronerf_band="L",
                dronerf_sample_rate_sps=40_000_000.0,
            )
            run_extract(extract_args)
            extract_seconds = time.time() - extract_started

            summary_path = features_dir / f"{prefix}_extract_summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"Expected extract summary not found: {summary_path}")
            summary = load_json(summary_path)
            samples_written = int(summary.get("samples_written", 0))
            if samples_written <= 0:
                raise RuntimeError("Extraction completed but reported zero samples written")

            reclaimed = _delete_staged_files(batch_local_df, source_dir)
            total_reclaimed += reclaimed

            # Remove empty staging subdir.
            try:
                staging_batch_dir.rmdir()
            except OSError:
                pass

            completed.update(batch_capture_ids)
            _save_completed_ids(completed_path, completed)
            completed_this_run += len(batch_capture_ids)

            entry = {
                "ts": _utc_now(),
                "batch_id": batch_id,
                "status": "completed",
                "capture_ids": batch_capture_ids,
                "files": len(batch_capture_ids),
                "copy_seconds": copy_seconds,
                "extract_seconds": extract_seconds,
                "batch_seconds": time.time() - batch_started,
                "copied_bytes": copied_bytes,
                "reclaimed_bytes": reclaimed,
                "samples_written": samples_written,
            }
            _append_ledger(ledger_path, entry)

        except Exception as exc:
            failed_this_run += 1
            failure = {
                "ts": _utc_now(),
                "batch_id": batch_id,
                "status": "failed",
                "capture_ids": batch_capture_ids,
                "error": str(exc),
            }
            failed_batches.append(failure)
            save_json(failed_path, {"failed_batches": failed_batches})
            _append_ledger(ledger_path, failure)

            if not args.continue_on_failure:
                raise

    # Post snapshot and immutability check.
    post_files = snapshot_source_dir(source_dir)
    save_json(source_post_path, _snapshot_payload(post_files))

    if _snapshot_signature(pre_files) != _snapshot_signature(post_files):
        raise RuntimeError("Source snapshot mismatch detected after run. Aborting due to immutability violation.")

    report = {
        "ts": _utc_now(),
        "source_dir": str(source_dir),
        "manifest": str(manifest_path),
        "manifest_clean": str(manifest_clean_path),
        "features_dir": str(features_dir),
        "staging_dir": str(staging_dir),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "run_id": run_id,
        "prefix": prefix,
        "pending_batches": len(batches),
        "completed_capture_ids_total": len(completed),
        "completed_capture_ids_this_run": completed_this_run,
        "failed_batches_this_run": failed_this_run,
        "failed_batches_total": len(failed_batches),
        "copied_bytes_this_run": total_copied,
        "reclaimed_bytes_this_run": total_reclaimed,
        "elapsed_seconds": time.time() - started_at,
        "source_immutable_verified": True,
    }
    save_json(final_report_path, report)
    print(json.dumps(report, indent=2))


def add_no_drone_batches_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "process-no-drone-batches",
        help="Copy no-drone raw files in batches, extract features locally, and clean staged local files",
    )

    p.add_argument(
        "--source-dir",
        type=str,
        default="/root/gargoyle/EDGE SYSTEM/RF SUBSYSTEM/SW/no-drone-data/nd_run_01",
    )
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--run-id", type=str, default=None, help="Optional run identifier; defaults to basename(source-dir)")
    p.add_argument(
        "--staging-dir",
        type=str,
        default=None,
        help="Local staging directory; defaults to /root/RFBinaryDetect/data/staging/<run-id>",
    )
    p.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Output feature directory; defaults to /root/RFBinaryDetect/data/no_drone_features/<run-id>",
    )
    p.add_argument("--state-dir", type=str, default=None)

    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--prefix", type=str, default=None, help="Shard prefix; defaults to <run-id>")

    p.add_argument("--segment-ms", type=int, default=20)
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--noverlap", type=int, default=120)
    p.add_argument("--resize-h", type=int, default=224)
    p.add_argument("--resize-w", type=int, default=224)
    p.add_argument("--no-log-power", action="store_true", default=False)
    p.add_argument("--no-normalize", action="store_true", default=False)

    p.add_argument("--retry-failed", action="store_true", default=False)
    p.add_argument("--continue-on-failure", action="store_true", default=False)
    p.add_argument(
        "--allow-state-reuse",
        action="store_true",
        default=False,
        help="Allow reusing an existing state directory with different source/manifest identity",
    )
    p.set_defaults(func=run_no_drone_batches)
