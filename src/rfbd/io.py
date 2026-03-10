"""I/O helpers for manifests and shard files."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .contracts import MANIFEST_COLUMNS, SHARD_KEYS, ManifestRecord
from .labels import normalize_label


def read_manifest(path: str | Path) -> List[ManifestRecord]:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    missing = [col for col in MANIFEST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    records: List[ManifestRecord] = []
    for row in df.to_dict(orient="records"):
        records.append(
            ManifestRecord(
                capture_id=str(row["capture_id"]),
                file_path=Path(str(row["file_path"])),
                label=normalize_label(row["label"]),
                source_domain=str(row["source_domain"]),
                center_freq_hz=float(row["center_freq_hz"]),
                sample_rate_sps=float(row["sample_rate_sps"]),
                gain_db=float(row["gain_db"]),
                session_id=str(row["session_id"]),
                timestamp=str(row["timestamp"]),
                environment=str(row["environment"]),
            )
        )
    return records


def iter_npz_files(path: str | Path, suffix: str = ".npz") -> List[Path]:
    p = Path(path)
    if not p.exists():
        return []
    return sorted([q for q in p.iterdir() if q.is_file() and q.suffix == suffix])


def load_npz_shard(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}

    missing = [k for k in SHARD_KEYS if k not in out]
    if missing:
        raise ValueError(f"Shard {path} missing keys: {missing}")
    return out


class ShardWriter:
    """Buffered writer for contract-compliant `.npz` feature shards."""

    def __init__(
        self,
        out_dir: str | Path,
        prefix: str,
        shard_size: int,
        start_index: int = 0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = int(shard_size)
        self.next_shard_index = int(start_index)

        self._feat: List[np.ndarray] = []
        self._y: List[np.ndarray] = []
        self._domain: List[np.ndarray] = []
        self._capture_id: List[np.ndarray] = []
        self._session_id: List[np.ndarray] = []
        self.total_written = 0

    def _buffer_len(self) -> int:
        if not self._y:
            return 0
        return int(sum(len(x) for x in self._y))

    def add_batch(
        self,
        feat: np.ndarray,
        y: np.ndarray,
        source_domain: np.ndarray,
        capture_id: np.ndarray,
        session_id: np.ndarray,
    ) -> None:
        if len(y) == 0:
            return

        self._feat.append(np.asarray(feat, dtype=np.float32))
        self._y.append(np.asarray(y, dtype=np.int64))
        self._domain.append(np.asarray(source_domain, dtype="<U64"))
        self._capture_id.append(np.asarray(capture_id, dtype="<U128"))
        self._session_id.append(np.asarray(session_id, dtype="<U128"))

        while self._buffer_len() >= self.shard_size:
            self.flush(limit=self.shard_size)

    def _concat(self) -> Dict[str, np.ndarray]:
        return {
            "feat": np.concatenate(self._feat, axis=0),
            "y": np.concatenate(self._y, axis=0),
            "source_domain": np.concatenate(self._domain, axis=0),
            "capture_id": np.concatenate(self._capture_id, axis=0),
            "session_id": np.concatenate(self._session_id, axis=0),
        }

    def flush(self, limit: Optional[int] = None) -> Optional[Path]:
        if self._buffer_len() == 0:
            return None

        merged = self._concat()
        n_total = len(merged["y"])
        n_take = n_total if limit is None else min(int(limit), n_total)

        write_batch = {k: v[:n_take] for k, v in merged.items()}
        out_path = self.out_dir / f"{self.prefix}_shard_{self.next_shard_index:05d}.npz"
        np.savez_compressed(out_path, **write_batch)
        self.next_shard_index += 1
        self.total_written += n_take

        if n_take == n_total:
            self._feat = []
            self._y = []
            self._domain = []
            self._capture_id = []
            self._session_id = []
        else:
            self._feat = [merged["feat"][n_take:]]
            self._y = [merged["y"][n_take:]]
            self._domain = [merged["source_domain"][n_take:]]
            self._capture_id = [merged["capture_id"][n_take:]]
            self._session_id = [merged["session_id"][n_take:]]

        return out_path

    def finalize(self) -> Optional[Path]:
        return self.flush(limit=None)


def save_json(path: str | Path, payload: Dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path, default: Optional[Dict] = None) -> Dict:
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_labels(labels: np.ndarray) -> Dict[str, int]:
    uniq, cnt = np.unique(labels.astype(np.int64), return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(uniq, cnt)}
