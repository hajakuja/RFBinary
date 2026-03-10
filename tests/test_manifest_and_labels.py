import csv
from pathlib import Path

import numpy as np

from rfbd.io import read_manifest
from rfbd.labels import ID_TO_LABEL, LABEL_TO_ID, label_name, normalize_label


def test_label_contract_roundtrip():
    assert LABEL_TO_ID["no_drone"] == 0
    assert LABEL_TO_ID["drone"] == 1
    assert ID_TO_LABEL[0] == "no_drone"
    assert ID_TO_LABEL[1] == "drone"
    assert normalize_label("drone") == 1
    assert normalize_label("no_drone") == 0
    assert normalize_label("none") == 0
    assert label_name(1) == "drone"


def test_manifest_schema_parse(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
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
        writer.writerow(
            [
                "cap1",
                str(tmp_path / "x.npy"),
                "no_drone",
                "custom_bg",
                5735000000,
                20971520,
                50,
                "s01",
                "2026-01-01T00:00:00Z",
                "indoor_busy",
            ]
        )

    recs = read_manifest(manifest)
    assert len(recs) == 1
    assert recs[0].capture_id == "cap1"
    assert recs[0].label == 0
    assert int(recs[0].sample_rate_sps) == 20971520
