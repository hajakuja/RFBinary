import argparse
import csv
from pathlib import Path

import numpy as np

from rfbd.extract import run_extract
from rfbd.io import iter_npz_files, load_npz_shard


def _write_manifest(path: Path, iq_file: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
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
        )
        w.writerow(
            [
                "c1",
                str(iq_file),
                "no_drone",
                "custom_bg",
                2437000000,
                1000,
                40,
                "s1",
                "2026-01-01T00:00:00Z",
                "indoor_quiet",
            ]
        )


def _run(tmp_path: Path, out_name: str):
    sps = 1000
    t = np.arange(0, 2000) / sps
    iq = (np.sin(2 * np.pi * 20 * t) + 1j * np.cos(2 * np.pi * 30 * t)).astype(np.complex64)
    iq_file = tmp_path / f"signal_{out_name}.npy"
    np.save(iq_file, iq)

    manifest = tmp_path / f"manifest_{out_name}.csv"
    _write_manifest(manifest, iq_file)

    out_dir = tmp_path / out_name
    args = argparse.Namespace(
        adapter="custom-iq",
        manifest=str(manifest),
        dronerf_root=None,
        out_dir=str(out_dir),
        prefix="extract",
        workers=1,
        shard_size=16,
        resume=False,
        segment_ms=20,
        nfft=64,
        noverlap=16,
        resize_h=224,
        resize_w=224,
        no_log_power=False,
        no_normalize=False,
        cache_dir=None,
        build_cache=False,
        dronerf_band="L",
        dronerf_sample_rate_sps=40000000.0,
    )
    run_extract(args)
    shards = iter_npz_files(out_dir)
    assert shards
    return load_npz_shard(shards[0])


def test_extract_custom_iq_shape_dtype_and_deterministic(tmp_path: Path):
    a = _run(tmp_path, "run_a")
    b = _run(tmp_path, "run_b")

    assert a["feat"].dtype == np.float32
    assert a["feat"].ndim == 3
    assert a["feat"].shape[1:] == (224, 224)
    assert np.array_equal(a["y"], np.zeros_like(a["y"]))

    assert a["feat"].shape == b["feat"].shape
    assert np.allclose(a["feat"], b["feat"], atol=1e-6)
