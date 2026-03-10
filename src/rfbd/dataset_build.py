"""Dataset merging command for binary drone/no-drone training."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .io import ShardWriter, iter_npz_files, load_npz_shard, save_json, summarize_labels


def _resize_stack(feat: np.ndarray, h: int = 224, w: int = 224) -> np.ndarray:
    feat = np.asarray(feat)
    if feat.ndim != 3:
        raise ValueError(f"Expected 3D feature array (N,H,W), got {feat.shape}")
    if feat.shape[1] == h and feat.shape[2] == w:
        return feat.astype(np.float32, copy=False)

    out = np.empty((feat.shape[0], h, w), dtype=np.float32)
    for i in range(feat.shape[0]):
        out[i] = cv2.resize(feat[i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _load_legacy_dronedetect(dir_path: Path, h: int, w: int) -> List[Dict[str, np.ndarray]]:
    files = sorted([p for p in dir_path.iterdir() if p.suffix.lower() == ".npy"])
    batches: List[Dict[str, np.ndarray]] = []

    for f in tqdm(files, desc="load dronedetect", unit="file"):
        payload = np.load(f, allow_pickle=True).item()
        raw_feat = np.asarray(payload.get("feat"))
        if raw_feat.ndim != 3 or raw_feat.shape[0] == 0:
            warnings.warn(f"Skipping invalid dronedetect file {f.name}: feat shape {raw_feat.shape}")
            continue

        feat = _resize_stack(raw_feat.astype(np.float32, copy=False), h=h, w=w)
        n = feat.shape[0]
        y = np.ones((n,), dtype=np.int64)
        source = np.full((n,), "dronedetect", dtype="<U64")
        capture = np.array([f"{f.stem}:{i}" for i in range(n)], dtype="<U128")
        session = np.full((n,), f.stem, dtype="<U128")
        batches.append(
            {
                "feat": feat,
                "y": y,
                "source_domain": source,
                "capture_id": capture,
                "session_id": session,
            }
        )
    return batches


def _load_legacy_dronerf(dir_path: Path, h: int, w: int) -> List[Dict[str, np.ndarray]]:
    files = sorted([p for p in dir_path.iterdir() if p.suffix.lower() == ".npy"])
    batches: List[Dict[str, np.ndarray]] = []

    for f in tqdm(files, desc="load dronerf", unit="file"):
        payload = np.load(f, allow_pickle=True).item()
        raw_feat = np.asarray(payload.get("feat"))
        if raw_feat.ndim != 3 or raw_feat.shape[0] == 0:
            warnings.warn(f"Skipping invalid dronerf file {f.name}: feat shape {raw_feat.shape}")
            continue

        feat = _resize_stack(raw_feat.astype(np.float32, copy=False), h=h, w=w)
        bi = np.asarray(payload.get("bi"))
        if bi.ndim != 1:
            bi = bi.reshape(-1)
        y = (bi.astype(np.int64) > 0).astype(np.int64)

        n = feat.shape[0]
        if len(y) != n:
            m = min(n, len(y))
            feat = feat[:m]
            y = y[:m]
            n = m
        if n <= 0:
            warnings.warn(f"Skipping empty dronerf file after alignment: {f.name}")
            continue

        source = np.full((n,), "dronerf", dtype="<U64")
        capture = np.array([f"{f.stem}:{i}" for i in range(n)], dtype="<U128")
        session = np.full((n,), f.stem, dtype="<U128")
        batches.append(
            {
                "feat": feat,
                "y": y,
                "source_domain": source,
                "capture_id": capture,
                "session_id": session,
            }
        )
    return batches


def _load_contract_shards(dir_path: Path) -> List[Dict[str, np.ndarray]]:
    files = iter_npz_files(dir_path)
    batches = [load_npz_shard(p) for p in files]
    return batches


def run_dataset_build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(out_dir=out_dir, prefix=args.prefix, shard_size=args.shard_size)

    total_sources: Dict[str, int] = {}

    def _append_batches(batches: List[Dict[str, np.ndarray]]) -> None:
        for b in batches:
            writer.add_batch(
                b["feat"],
                b["y"],
                b["source_domain"],
                b["capture_id"],
                b["session_id"],
            )
            uniq, cnt = np.unique(b["source_domain"], return_counts=True)
            for u, c in zip(uniq, cnt):
                total_sources[str(u)] = total_sources.get(str(u), 0) + int(c)

    if args.dronedetect_arr_dir:
        _append_batches(_load_legacy_dronedetect(Path(args.dronedetect_arr_dir), h=args.resize_h, w=args.resize_w))

    if args.dronerf_arr_dir:
        _append_batches(_load_legacy_dronerf(Path(args.dronerf_arr_dir), h=args.resize_h, w=args.resize_w))

    if args.custom_shards_dir:
        _append_batches(_load_contract_shards(Path(args.custom_shards_dir)))

    if args.extra_shard_dir:
        _append_batches(_load_contract_shards(Path(args.extra_shard_dir)))

    writer.finalize()

    # Summarize final merged labels without loading large feature tensors.
    labels = []
    for shard in iter_npz_files(out_dir):
        with np.load(shard, allow_pickle=False) as data:
            labels.append(np.asarray(data["y"], dtype=np.int64))
    y_all = np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)

    summary = {
        "out_dir": str(out_dir),
        "prefix": args.prefix,
        "samples_written": int(writer.total_written),
        "label_counts": summarize_labels(y_all),
        "source_counts": total_sources,
    }
    save_json(out_dir / f"{args.prefix}_summary.json", summary)
    print(f"Merged dataset written to {out_dir}. Samples: {writer.total_written}")


def add_dataset_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_dataset = subparsers.add_parser("dataset", help="Dataset operations")
    ds_sub = p_dataset.add_subparsers(dest="dataset_cmd", required=True)

    p_build = ds_sub.add_parser("build", help="Build merged binary dataset shards")
    p_build.add_argument("--dronedetect-arr-dir", type=str, default=None)
    p_build.add_argument("--dronerf-arr-dir", type=str, default=None)
    p_build.add_argument("--custom-shards-dir", type=str, default=None)
    p_build.add_argument("--extra-shard-dir", type=str, default=None)

    p_build.add_argument("--out-dir", type=str, required=True)
    p_build.add_argument("--prefix", type=str, default="dataset")
    p_build.add_argument("--shard-size", type=int, default=1024)
    p_build.add_argument("--resize-h", type=int, default=224)
    p_build.add_argument("--resize-w", type=int, default=224)

    p_build.set_defaults(func=run_dataset_build)
