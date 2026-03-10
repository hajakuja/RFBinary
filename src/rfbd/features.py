"""Feature extraction primitives used by CLI extract/build paths."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd
from scipy import signal

from .contracts import FeatureConfig


def normalize_0_1(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn + eps)).astype(np.float32, copy=False)


def compute_spec_feature(segment: np.ndarray, sample_rate_sps: float, cfg: FeatureConfig) -> np.ndarray:
    nperseg = min(cfg.nfft, int(len(segment)))
    noverlap = min(cfg.noverlap, max(0, nperseg - 1))
    _, _, sxx = signal.spectrogram(
        segment,
        fs=sample_rate_sps,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    feat = sxx.astype(np.float32, copy=False)
    if cfg.log_power:
        feat = 10.0 * np.log10(feat + 1e-12)
    if cfg.normalize:
        feat = normalize_0_1(feat)

    # cv2.resize uses (width, height).
    feat = cv2.resize(feat, (cfg.resize_w, cfg.resize_h), interpolation=cv2.INTER_LINEAR)
    return feat.astype(np.float32, copy=False)


def split_segments_1d(x: np.ndarray, samples_per_segment: int) -> Iterator[np.ndarray]:
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be > 0")
    n = len(x) // samples_per_segment
    for i in range(n):
        j0 = i * samples_per_segment
        j1 = j0 + samples_per_segment
        yield x[j0:j1]


def load_custom_iq(path: str | Path) -> np.ndarray:
    """Load raw IQ file into complex64 1D numpy array.

    Supported:
    - `.npy` complex arrays or shape (N,2) I/Q float arrays
    - `.npz` first array key or `iq`
    - `.s16`/`.bin` interleaved int16 I,Q
    """

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npy":
        arr = np.load(p)
        return _to_complex(arr)

    if suffix == ".npz":
        with np.load(p) as data:
            if "iq" in data.files:
                arr = data["iq"]
            else:
                arr = data[data.files[0]]
        return _to_complex(arr)

    if suffix in {".s16", ".bin", ".iq"}:
        raw = np.fromfile(p, dtype=np.int16)
        if raw.size < 2:
            return np.empty((0,), dtype=np.complex64)
        raw = raw[: raw.size - (raw.size % 2)]
        iq = raw.reshape(-1, 2).astype(np.float32, copy=False)
        return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64, copy=False)

    raise ValueError(f"Unsupported custom IQ format: {p}")


def _to_complex(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False).reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return (arr[:, 0].astype(np.float32) + 1j * arr[:, 1].astype(np.float32)).astype(
            np.complex64,
            copy=False,
        )
    if arr.ndim == 1:
        return arr.astype(np.complex64, copy=False)
    raise ValueError(f"Cannot interpret array shape {arr.shape} as IQ")


def load_csv_signal(path: str | Path, cache_dir: Optional[str | Path] = None, build_cache: bool = False) -> np.ndarray:
    """Load 1D float signal from DroneRF CSV with optional `.npy` cache."""

    p = Path(path)
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{p.name}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    # pandas is still robust for very large one-column CSV files.
    arr = pd.read_csv(p, header=None, dtype=np.float32).values.reshape(-1)

    if cache_path is not None and build_cache:
        np.save(cache_path, arr)

    return arr.astype(np.float32, copy=False)


def parse_dronerf_binary_label(low_file_name: str) -> int:
    first = str(low_file_name)[0]
    if first not in {"0", "1"}:
        # Any non-zero leading binary code means drone present.
        try:
            return 1 if int(first) > 0 else 0
        except ValueError:
            return 0
    return int(first)


def samples_per_segment(sample_rate_sps: float, segment_ms: int) -> int:
    return int((segment_ms / 1000.0) * sample_rate_sps)
