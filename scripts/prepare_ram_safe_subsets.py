#!/usr/bin/env python3
"""Prepare RAM-safe curated subsets for binary training on constrained hosts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir exists: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _dronedetect_count(path: Path) -> int:
    payload = np.load(path, allow_pickle=True).item()
    feat = np.asarray(payload.get("feat"))
    if feat.ndim != 3 or feat.shape[0] == 0:
        return 0
    return int(feat.shape[0])


def _build_dronedetect_subset(
    src_dir: Path,
    out_dir: Path,
    target_samples: int,
    seed: int,
    mode: str,
) -> dict:
    files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() == ".npy"])
    if not files:
        raise FileNotFoundError(f"No .npy files found in {src_dir}")

    valid: List[Tuple[Path, int]] = []
    skipped: List[str] = []
    for f in files:
        n = _dronedetect_count(f)
        if n <= 0:
            skipped.append(f.name)
            continue
        valid.append((f, n))

    if not valid:
        raise RuntimeError(f"No valid dronedetect files in {src_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(valid)).tolist()

    selected: List[Tuple[Path, int]] = []
    samples = 0
    for idx in order:
        item = valid[idx]
        selected.append(item)
        samples += item[1]
        if samples >= target_samples:
            break

    for src, _ in selected:
        _link_or_copy(src, out_dir / src.name, mode=mode)

    summary = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "target_samples": int(target_samples),
        "selected_samples": int(samples),
        "selected_files": int(len(selected)),
        "skipped_invalid_files": skipped,
    }
    (out_dir / "dronedetect_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _npz_y_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as z:
        return int(len(z["y"]))


def _build_no_drone_subset(
    src_dir: Path,
    out_dir: Path,
    target_samples: int,
    mode: str,
) -> dict:
    files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() == ".npz"])
    if not files:
        raise FileNotFoundError(f"No .npz files found in {src_dir}")

    counts = [max(1, _npz_y_count(p)) for p in files]
    nominal = int(np.median(np.asarray(counts, dtype=np.int64)))
    n_pick = max(1, int(math.ceil(float(target_samples) / float(nominal))))
    tall_idx = [i for i, c in enumerate(counts) if c >= max(1, nominal // 2)]
    pool_idx = tall_idx if len(tall_idx) >= n_pick else list(range(len(files)))
    n_pick = min(n_pick, len(pool_idx))

    ranked_idx = sorted(pool_idx, key=lambda i: (counts[i], -i), reverse=True)
    chosen = sorted(set(ranked_idx[:n_pick]))

    samples = 0
    emitted = 0
    used = set(chosen)
    for i in chosen:
        src = files[i]
        _link_or_copy(src, out_dir / f"no_drone_subset_shard_{emitted:05d}.npz", mode=mode)
        emitted += 1
        samples += _npz_y_count(src)

    if samples < target_samples:
        for i, src in enumerate(files):
            if i in used:
                continue
            _link_or_copy(src, out_dir / f"no_drone_subset_shard_{emitted:05d}.npz", mode=mode)
            emitted += 1
            samples += _npz_y_count(src)
            if samples >= target_samples:
                break

    summary = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "target_samples": int(target_samples),
        "selected_samples": int(samples),
        "selected_shards": int(emitted),
        "nominal_shard_samples": int(nominal),
    }
    (out_dir / "no_drone_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Build curated RAM-safe subset directories.")
    p.add_argument("--dronedetect-src", type=str, required=True)
    p.add_argument("--dronedetect-out", type=str, required=True)
    p.add_argument("--dronedetect-target-samples", type=int, default=20_000)

    p.add_argument("--no-drone-src", type=str, required=True)
    p.add_argument("--no-drone-out", type=str, required=True)
    p.add_argument("--no-drone-target-samples", type=int, default=8_192)

    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true", default=False)
    args = p.parse_args()

    drde_src = Path(args.dronedetect_src)
    drde_out = Path(args.dronedetect_out)
    nd_src = Path(args.no_drone_src)
    nd_out = Path(args.no_drone_out)

    _reset_dir(drde_out, overwrite=args.overwrite)
    _reset_dir(nd_out, overwrite=args.overwrite)

    drde_summary = _build_dronedetect_subset(
        drde_src,
        drde_out,
        target_samples=int(args.dronedetect_target_samples),
        seed=int(args.seed),
        mode=args.link_mode,
    )
    nd_summary = _build_no_drone_subset(
        nd_src,
        nd_out,
        target_samples=int(args.no_drone_target_samples),
        mode=args.link_mode,
    )

    payload = {"dronedetect": drde_summary, "no_drone": nd_summary}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
