"""Feature extraction command implementation."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm

from .contracts import ExtractConfig, FeatureConfig
from .features import (
    compute_spec_feature,
    load_csv_signal,
    load_custom_iq,
    parse_dronerf_binary_label,
    samples_per_segment,
    split_segments_1d,
)
from .io import ShardWriter, iter_npz_files, load_json, load_npz_shard, read_manifest, save_json


def _feature_cfg_from_dict(cfg: Dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        segment_ms=int(cfg["segment_ms"]),
        nfft=int(cfg["nfft"]),
        noverlap=int(cfg["noverlap"]),
        resize_h=int(cfg["resize_h"]),
        resize_w=int(cfg["resize_w"]),
        log_power=bool(cfg["log_power"]),
        normalize=bool(cfg["normalize"]),
    )


def _extract_custom_task(task: Dict[str, Any], cfg: Dict[str, Any], tmp_dir: str) -> Dict[str, Any]:
    fcfg = _feature_cfg_from_dict(cfg)
    iq = load_custom_iq(task["file_path"])
    signal_real = np.real(iq).astype(np.float32, copy=False)

    sps = float(task["sample_rate_sps"])
    seg_len = samples_per_segment(sps, fcfg.segment_ms)

    feats: List[np.ndarray] = []
    for seg in split_segments_1d(signal_real, seg_len):
        feats.append(compute_spec_feature(seg, sps, fcfg))

    if feats:
        feat = np.stack(feats, axis=0).astype(np.float32, copy=False)
        n = feat.shape[0]
        y = np.full((n,), int(task["label"]), dtype=np.int64)
        domain = np.full((n,), str(task["source_domain"]), dtype="<U64")
        capture = np.full((n,), str(task["capture_id"]), dtype="<U128")
        session = np.full((n,), str(task["session_id"]), dtype="<U128")
    else:
        feat = np.empty((0, fcfg.resize_h, fcfg.resize_w), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        domain = np.empty((0,), dtype="<U64")
        capture = np.empty((0,), dtype="<U128")
        session = np.empty((0,), dtype="<U128")

    tmp_path = Path(tmp_dir) / f"{task['task_key']}.npz"
    np.savez_compressed(
        tmp_path,
        feat=feat,
        y=y,
        source_domain=domain,
        capture_id=capture,
        session_id=session,
    )
    return {"task_key": task["task_key"], "tmp_path": str(tmp_path), "count": int(len(y))}


def _extract_dronerf_task(task: Dict[str, Any], cfg: Dict[str, Any], tmp_dir: str) -> Dict[str, Any]:
    fcfg = _feature_cfg_from_dict(cfg)
    cache_dir = cfg.get("cache_dir")
    build_cache = bool(cfg.get("build_cache", False))

    high = load_csv_signal(task["high_path"], cache_dir=cache_dir, build_cache=build_cache)
    low = load_csv_signal(task["low_path"], cache_dir=cache_dir, build_cache=build_cache)

    n = min(len(high), len(low))
    high = high[:n]
    low = low[:n]

    band = str(cfg.get("dronerf_band", "L")).upper()
    sig = high if band == "H" else low

    sps = float(cfg.get("dronerf_sample_rate_sps", 40_000_000.0))
    seg_len = samples_per_segment(sps, fcfg.segment_ms)

    feats: List[np.ndarray] = []
    for seg in split_segments_1d(sig, seg_len):
        feats.append(compute_spec_feature(seg, sps, fcfg))

    label = int(task["label"])
    if feats:
        feat = np.stack(feats, axis=0).astype(np.float32, copy=False)
        n_seg = feat.shape[0]
        y = np.full((n_seg,), label, dtype=np.int64)
        domain = np.full((n_seg,), "dronerf", dtype="<U64")
        capture = np.full((n_seg,), str(task["capture_id"]), dtype="<U128")
        session = np.full((n_seg,), str(task["session_id"]), dtype="<U128")
    else:
        feat = np.empty((0, fcfg.resize_h, fcfg.resize_w), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        domain = np.empty((0,), dtype="<U64")
        capture = np.empty((0,), dtype="<U128")
        session = np.empty((0,), dtype="<U128")

    tmp_path = Path(tmp_dir) / f"{task['task_key']}.npz"
    np.savez_compressed(
        tmp_path,
        feat=feat,
        y=y,
        source_domain=domain,
        capture_id=capture,
        session_id=session,
    )
    return {"task_key": task["task_key"], "tmp_path": str(tmp_path), "count": int(len(y))}


def _worker(task: Dict[str, Any], cfg: Dict[str, Any], tmp_dir: str) -> Dict[str, Any]:
    if task["adapter"] == "custom-iq":
        return _extract_custom_task(task, cfg, tmp_dir)
    if task["adapter"] == "legacy-dronerf":
        return _extract_dronerf_task(task, cfg, tmp_dir)
    raise ValueError(f"Unsupported adapter: {task['adapter']}")


def _build_custom_tasks(manifest_path: str) -> List[Dict[str, Any]]:
    records = read_manifest(manifest_path)
    tasks = []
    for i, rec in enumerate(records):
        tasks.append(
            {
                "adapter": "custom-iq",
                "task_key": f"custom_{i:07d}_{rec.capture_id}",
                "capture_id": rec.capture_id,
                "session_id": rec.session_id,
                "file_path": str(rec.file_path),
                "label": rec.label,
                "source_domain": rec.source_domain,
                "sample_rate_sps": rec.sample_rate_sps,
            }
        )
    return tasks


def _pair_dronerf_files(high_files: List[Path], low_files: List[Path]) -> List[Dict[str, Any]]:
    def key_of(name: str) -> str:
        stem = Path(name).stem
        stem = stem.replace("H_", "_").replace("L_", "_")
        return stem

    high_map = {key_of(p.name): p for p in high_files}
    low_map = {key_of(p.name): p for p in low_files}

    keys = sorted(set(high_map.keys()) & set(low_map.keys()))
    tasks = []
    for idx, k in enumerate(keys):
        high = high_map[k]
        low = low_map[k]
        label = parse_dronerf_binary_label(low.name)
        task_key = f"dronerf_{idx:07d}_{low.stem}"
        tasks.append(
            {
                "adapter": "legacy-dronerf",
                "task_key": task_key,
                "high_path": str(high),
                "low_path": str(low),
                "label": int(label),
                "capture_id": low.stem,
                "session_id": low.stem,
            }
        )
    return tasks


def _build_dronerf_tasks(dronerf_root: str) -> List[Dict[str, Any]]:
    root = Path(dronerf_root)
    high_dir = root / "High"
    low_dir = root / "Low"

    if not high_dir.exists() or not low_dir.exists():
        raise FileNotFoundError(f"DroneRF root must contain High/ and Low/: {root}")

    high_files = sorted([p for p in high_dir.iterdir() if p.suffix.lower() == ".csv"])
    low_files = sorted([p for p in low_dir.iterdir() if p.suffix.lower() == ".csv"])
    if not high_files or not low_files:
        raise ValueError("No DroneRF CSV files found under High/Low directories")

    return _pair_dronerf_files(high_files, low_files)


def _infer_next_shard_index(out_dir: Path, prefix: str) -> int:
    files = sorted(out_dir.glob(f"{prefix}_shard_*.npz"))
    if not files:
        return 0
    last = files[-1].stem
    idx = int(last.split("_")[-1])
    return idx + 1


def run_extract(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExtractConfig(
        segment_ms=args.segment_ms,
        nfft=args.nfft,
        noverlap=args.noverlap,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        log_power=not args.no_log_power,
        normalize=not args.no_normalize,
        workers=args.workers,
        shard_size=args.shard_size,
        out_dir=out_dir,
        resume=args.resume,
        adapter=args.adapter,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        build_cache=bool(args.build_cache),
        dronerf_band=args.dronerf_band,
        dronerf_sample_rate_sps=args.dronerf_sample_rate_sps,
    )

    if cfg.adapter == "custom-iq":
        if not args.manifest:
            raise ValueError("--manifest is required for adapter=custom-iq")
        tasks = _build_custom_tasks(args.manifest)
    elif cfg.adapter == "legacy-dronerf":
        if not args.dronerf_root:
            raise ValueError("--dronerf-root is required for adapter=legacy-dronerf")
        tasks = _build_dronerf_tasks(args.dronerf_root)
    else:
        raise ValueError(f"Unsupported adapter: {cfg.adapter}")

    if not tasks:
        print("No extraction tasks found.")
        return

    prefix = args.prefix
    state_path = out_dir / f"{prefix}_extract_state.json"
    state = load_json(state_path, default={"done_keys": [], "next_shard_index": None, "processed": 0})
    done_keys = set(state.get("done_keys", [])) if cfg.resume else set()

    start_index = state.get("next_shard_index")
    if start_index is None:
        start_index = _infer_next_shard_index(out_dir, prefix) if cfg.resume else 0

    writer = ShardWriter(out_dir=out_dir, prefix=prefix, shard_size=cfg.shard_size, start_index=start_index)

    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    remaining = [t for t in tasks if t["task_key"] not in done_keys]
    if not remaining:
        print("All tasks already processed; nothing to do.")
        return

    cfg_dict = asdict(cfg)
    if cfg.cache_dir is not None:
        cfg_dict["cache_dir"] = str(cfg.cache_dir)

    pbar = tqdm(total=len(remaining), desc="extract", unit="file")
    summaries: List[Dict[str, Any]] = []

    def _consume_result(res: Dict[str, Any]) -> None:
        nonlocal done_keys, state
        shard = load_npz_shard(res["tmp_path"])
        writer.add_batch(
            shard["feat"],
            shard["y"],
            shard["source_domain"],
            shard["capture_id"],
            shard["session_id"],
        )
        Path(res["tmp_path"]).unlink(missing_ok=True)

        done_keys.add(res["task_key"])
        state = {
            "done_keys": sorted(done_keys),
            "next_shard_index": writer.next_shard_index,
            "processed": int(len(done_keys)),
        }
        save_json(state_path, state)
        summaries.append(res)

    if cfg.workers <= 1:
        for task in remaining:
            res = _worker(task, cfg_dict, str(tmp_dir))
            _consume_result(res)
            pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = [ex.submit(_worker, task, cfg_dict, str(tmp_dir)) for task in remaining]
            for fut in as_completed(futures):
                res = fut.result()
                _consume_result(res)
                pbar.update(1)

    pbar.close()
    writer.finalize()

    summary = {
        "adapter": cfg.adapter,
        "tasks_total": len(tasks),
        "tasks_processed_this_run": len(remaining),
        "samples_written": writer.total_written,
        "output_dir": str(out_dir),
        "prefix": prefix,
        "next_shard_index": writer.next_shard_index,
    }
    save_json(out_dir / f"{prefix}_extract_summary.json", summary)
    print(f"Extraction complete: wrote {writer.total_written} samples to {out_dir}")


def add_extract_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("extract", help="Extract SPEC ARR features into contract shards")
    p.add_argument("--adapter", choices=["custom-iq", "legacy-dronerf"], required=True)
    p.add_argument("--manifest", type=str, default=None, help="Raw manifest CSV (custom-iq adapter)")
    p.add_argument("--dronerf-root", type=str, default=None, help="DroneRF root containing High/ and Low/")

    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="extract")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--resume", action="store_true", default=False)

    p.add_argument("--segment-ms", type=int, default=20)
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--noverlap", type=int, default=120)
    p.add_argument("--resize-h", type=int, default=224)
    p.add_argument("--resize-w", type=int, default=224)
    p.add_argument("--no-log-power", action="store_true", default=False)
    p.add_argument("--no-normalize", action="store_true", default=False)

    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--build-cache", action="store_true", default=False)
    p.add_argument("--dronerf-band", choices=["H", "L"], default="L")
    p.add_argument("--dronerf-sample-rate-sps", type=float, default=40_000_000.0)

    p.set_defaults(func=run_extract)
