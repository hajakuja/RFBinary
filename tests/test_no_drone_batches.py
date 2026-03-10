import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfbd.no_drone_batches import (
    clean_and_validate_manifest,
    run_no_drone_batches,
    snapshot_source_dir,
)


def _write_s16(path: Path, n_complex: int = 2000) -> None:
    i = np.random.randint(-1000, 1000, size=n_complex, dtype=np.int16)
    q = np.random.randint(-1000, 1000, size=n_complex, dtype=np.int16)
    interleaved = np.empty((n_complex * 2,), dtype=np.int16)
    interleaved[0::2] = i
    interleaved[1::2] = q
    interleaved.tofile(path)


def _manifest_row(file_name: str, file_path: str) -> dict:
    return {
        "capture_id": file_name.split("_")[0],
        "file_path": file_path,
        "label": "no_drone",
        "source_domain": "custom_bg",
        "center_freq_hz": "",
        "sample_rate_sps": "",
        "gain_db": "",
        "session_id": "",
        "timestamp": "",
        "environment": "",
    }


def test_clean_manifest_fills_missing_and_resolves_path(tmp_path: Path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    name = "bg00001_20260306T214943Z_2437000000Hz_20971520sps_35g_s01.s16"
    _write_s16(source_dir / name)

    manifest = tmp_path / "manifest.csv"
    df = pd.DataFrame([_manifest_row(name, f"/nonexistent/path/{name}")])
    df.to_csv(manifest, index=False)

    source_files = snapshot_source_dir(source_dir)
    clean_df = clean_and_validate_manifest(manifest, source_dir, source_files)

    assert clean_df.iloc[0]["file_path"] == str(source_dir / name)
    assert int(clean_df.iloc[0]["center_freq_hz"]) == 2437000000
    assert int(clean_df.iloc[0]["sample_rate_sps"]) == 20971520
    assert int(clean_df.iloc[0]["gain_db"]) == 35
    assert clean_df.iloc[0]["session_id"] == "s01"
    assert clean_df.iloc[0]["timestamp"] == "2026-03-06T21:49:43Z"


def test_batch_happy_path_extract_and_cleanup(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    names = [
        "bg00001_20260306T214943Z_2437000000Hz_1000sps_35g_s01.s16",
        "bg00002_20260306T215819Z_2437000000Hz_1000sps_50g_s01.s16",
    ]
    for n in names:
        _write_s16(source_dir / n, n_complex=2000)

    manifest = tmp_path / "manifest.csv"
    rows = [_manifest_row(n, f"/wrong/prefix/{n}") for n in names]
    pd.DataFrame(rows).to_csv(manifest, index=False)

    staging = tmp_path / "staging"
    features = tmp_path / "features"
    state = tmp_path / "state"

    args = argparse.Namespace(
        source_dir=str(source_dir),
        manifest=str(manifest),
        staging_dir=str(staging),
        features_dir=str(features),
        state_dir=str(state),
        batch_size=1,
        workers=1,
        shard_size=16,
        prefix="nd_run_01",
        segment_ms=20,
        nfft=64,
        noverlap=16,
        resize_h=224,
        resize_w=224,
        no_log_power=False,
        no_normalize=False,
        retry_failed=False,
        continue_on_failure=False,
    )

    before = {p.name: p.stat().st_size for p in source_dir.iterdir()}
    run_no_drone_batches(args)
    after = {p.name: p.stat().st_size for p in source_dir.iterdir()}

    assert before == after
    assert any(features.glob("nd_run_01_shard_*.npz"))

    completed_path = state / "completed_capture_ids.txt"
    completed = [x for x in completed_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert set(completed) == {"bg00001", "bg00002"}

    staged_files = list(staging.rglob("*.s16"))
    assert not staged_files


def test_extraction_failure_records_failed_and_keeps_staged(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    name = "bg00003_20260306T220242Z_2437000000Hz_1000sps_35g_s01.s16"
    _write_s16(source_dir / name, n_complex=200)

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([_manifest_row(name, str(source_dir / name))]).to_csv(manifest, index=False)

    def _boom(_):
        raise RuntimeError("extract failed")

    monkeypatch.setattr("rfbd.no_drone_batches.run_extract", _boom)

    args = argparse.Namespace(
        source_dir=str(source_dir),
        manifest=str(manifest),
        staging_dir=str(tmp_path / "staging"),
        features_dir=str(tmp_path / "features"),
        state_dir=str(tmp_path / "state"),
        batch_size=1,
        workers=1,
        shard_size=8,
        prefix="nd_run_01",
        segment_ms=20,
        nfft=64,
        noverlap=16,
        resize_h=224,
        resize_w=224,
        no_log_power=False,
        no_normalize=False,
        retry_failed=False,
        continue_on_failure=True,
    )

    run_no_drone_batches(args)

    failed_json = tmp_path / "state" / "failed_batches.json"
    payload = json_load(failed_json)
    assert payload["failed_batches"]

    staged = list((tmp_path / "staging").rglob("*.s16"))
    assert staged


def test_resume_skips_completed_batches(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    name = "bg00004_20260306T220725Z_2437000000Hz_1000sps_35g_s01.s16"
    _write_s16(source_dir / name, n_complex=2000)

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([_manifest_row(name, str(source_dir / name))]).to_csv(manifest, index=False)

    base_args = dict(
        source_dir=str(source_dir),
        manifest=str(manifest),
        staging_dir=str(tmp_path / "staging"),
        features_dir=str(tmp_path / "features"),
        state_dir=str(tmp_path / "state"),
        batch_size=1,
        workers=1,
        shard_size=8,
        prefix="nd_run_01",
        segment_ms=20,
        nfft=64,
        noverlap=16,
        resize_h=224,
        resize_w=224,
        no_log_power=False,
        no_normalize=False,
        retry_failed=False,
        continue_on_failure=False,
    )

    run_no_drone_batches(argparse.Namespace(**base_args))

    def _should_not_run(_):
        raise RuntimeError("extract should not run when all captures completed")

    monkeypatch.setattr("rfbd.no_drone_batches.run_extract", _should_not_run)
    run_no_drone_batches(argparse.Namespace(**base_args))


def test_dynamic_defaults_isolate_state_between_source_dirs(tmp_path: Path):
    run_id_01 = f"pytest_{tmp_path.name}_run1"
    run_id_02 = f"pytest_{tmp_path.name}_run2"
    source_01 = tmp_path / run_id_01
    source_02 = tmp_path / run_id_02
    source_01.mkdir()
    source_02.mkdir()

    name_01 = "bg10001_20260306T220725Z_2437000000Hz_1000sps_35g_s01.s16"
    name_02 = "bg20001_20260306T220825Z_2437000000Hz_1000sps_35g_s01.s16"
    _write_s16(source_01 / name_01, n_complex=2000)
    _write_s16(source_02 / name_02, n_complex=2000)

    manifest_01 = tmp_path / "manifest_01.csv"
    manifest_02 = tmp_path / "manifest_02.csv"
    pd.DataFrame([_manifest_row(name_01, str(source_01 / name_01))]).to_csv(manifest_01, index=False)
    pd.DataFrame([_manifest_row(name_02, str(source_02 / name_02))]).to_csv(manifest_02, index=False)

    args_01 = argparse.Namespace(
        source_dir=str(source_01),
        manifest=str(manifest_01),
        run_id=None,
        staging_dir=None,
        features_dir=None,
        state_dir=None,
        batch_size=1,
        workers=1,
        shard_size=8,
        prefix=None,
        segment_ms=20,
        nfft=64,
        noverlap=16,
        resize_h=224,
        resize_w=224,
        no_log_power=False,
        no_normalize=False,
        retry_failed=False,
        continue_on_failure=False,
        allow_state_reuse=False,
    )
    args_02 = argparse.Namespace(**{**vars(args_01), "source_dir": str(source_02), "manifest": str(manifest_02)})

    out_01 = Path("/root/RFBinaryDetect/data/no_drone_features") / run_id_01
    out_02 = Path("/root/RFBinaryDetect/data/no_drone_features") / run_id_02
    shutil.rmtree(out_01, ignore_errors=True)
    shutil.rmtree(out_02, ignore_errors=True)

    run_no_drone_batches(args_01)
    run_no_drone_batches(args_02)

    assert any(out_01.glob(f"{run_id_01}_shard_*.npz"))
    assert any(out_02.glob(f"{run_id_02}_shard_*.npz"))


def test_state_identity_mismatch_errors_without_override(tmp_path: Path):
    source_a = tmp_path / "src_a"
    source_b = tmp_path / "src_b"
    source_a.mkdir()
    source_b.mkdir()

    name_a = "bg30001_20260306T220725Z_2437000000Hz_1000sps_35g_s01.s16"
    name_b = "bg30002_20260306T220825Z_2437000000Hz_1000sps_35g_s01.s16"
    _write_s16(source_a / name_a, n_complex=2000)
    _write_s16(source_b / name_b, n_complex=2000)

    manifest_a = tmp_path / "manifest_a.csv"
    manifest_b = tmp_path / "manifest_b.csv"
    pd.DataFrame([_manifest_row(name_a, str(source_a / name_a))]).to_csv(manifest_a, index=False)
    pd.DataFrame([_manifest_row(name_b, str(source_b / name_b))]).to_csv(manifest_b, index=False)

    shared_state = tmp_path / "shared_state"
    base = dict(
        source_dir=str(source_a),
        manifest=str(manifest_a),
        run_id=None,
        staging_dir=str(tmp_path / "staging"),
        features_dir=str(tmp_path / "features"),
        state_dir=str(shared_state),
        batch_size=1,
        workers=1,
        shard_size=8,
        prefix=None,
        segment_ms=20,
        nfft=64,
        noverlap=16,
        resize_h=224,
        resize_w=224,
        no_log_power=False,
        no_normalize=False,
        retry_failed=False,
        continue_on_failure=False,
        allow_state_reuse=False,
    )

    run_no_drone_batches(argparse.Namespace(**base))
    with pytest.raises(RuntimeError, match="State directory identity mismatch"):
        run_no_drone_batches(
            argparse.Namespace(**{**base, "source_dir": str(source_b), "manifest": str(manifest_b)})
        )


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
