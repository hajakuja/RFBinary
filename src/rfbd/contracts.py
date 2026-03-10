"""Data contracts and typed config objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MANIFEST_COLUMNS = [
    "capture_id",
    "file_path",
    "label",
    "source_domain",
    "center_freq_hz",
    "sample_rate_sps",
    "gain_db",
    "session_id",
    "timestamp",
    "environment",
]


SHARD_KEYS = ["feat", "y", "source_domain", "capture_id", "session_id"]


@dataclass(frozen=True)
class ManifestRecord:
    capture_id: str
    file_path: Path
    label: int
    source_domain: str
    center_freq_hz: float
    sample_rate_sps: float
    gain_db: float
    session_id: str
    timestamp: str
    environment: str


@dataclass(frozen=True)
class FeatureConfig:
    segment_ms: int = 20
    nfft: int = 1024
    noverlap: int = 120
    resize_h: int = 224
    resize_w: int = 224
    log_power: bool = True
    normalize: bool = True


@dataclass(frozen=True)
class ExtractConfig(FeatureConfig):
    workers: int = 4
    shard_size: int = 512
    out_dir: Path = Path("./out")
    resume: bool = True
    adapter: str = "custom-iq"
    cache_dir: Optional[Path] = None
    build_cache: bool = False
    dronerf_band: str = "L"
    dronerf_sample_rate_sps: float = 40_000_000.0
