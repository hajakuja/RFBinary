import argparse
from pathlib import Path
import torch

import numpy as np

from rfbd.dataset_build import run_dataset_build
from rfbd.eval import run_eval
from rfbd.io import iter_npz_files, load_npz_shard
from rfbd.training import run_train


def _make_legacy_detect_file(path: Path, n: int = 8):
    feat = np.random.rand(n, 224, 224).astype(np.float32)
    payload = {
        "feat": feat,
        "drones": np.array(["AIR"] * n),
        "conds": np.array(["ON"] * n),
        "ints": np.array(["11"] * n),
    }
    np.save(path, payload)


def _make_custom_shard(path: Path, n: int = 8):
    feat = np.random.rand(n, 224, 224).astype(np.float32)
    y = np.zeros((n,), dtype=np.int64)
    source = np.array(["custom_bg"] * n)
    cap = np.array([f"c{i}" for i in range(n)])
    sess = np.array(["s01"] * n)
    np.savez_compressed(path, feat=feat, y=y, source_domain=source, capture_id=cap, session_id=sess)


def _make_dronerf_legacy(path: Path, n: int = 8):
    feat = np.random.rand(n, 224, 224).astype(np.float32)
    bi = np.array([0, 1] * (n // 2), dtype=np.int64)
    payload = {"feat": feat, "bi": bi, "drones": bi, "modes": bi}
    np.save(path, payload)


def test_end_to_end_smoke(tmp_path: Path):
    drde = tmp_path / "drde"
    drde.mkdir()
    drde_file = drde / "BOTH_SPEC_AIR_ON_1024.npy"
    _make_legacy_detect_file(drde_file, n=10)

    drrf = tmp_path / "drrf"
    drrf.mkdir()
    drrf_file = drrf / "SPEC_1024_0.npy"
    _make_dronerf_legacy(drrf_file, n=10)

    custom = tmp_path / "custom"
    custom.mkdir()
    _make_custom_shard(custom / "extract_shard_00000.npz", n=10)

    merged = tmp_path / "merged"
    run_dataset_build(
        argparse.Namespace(
            dronedetect_arr_dir=str(drde),
            dronerf_arr_dir=str(drrf),
            custom_shards_dir=str(custom),
            extra_shard_dir=None,
            out_dir=str(merged),
            prefix="dataset",
            shard_size=32,
            resize_h=224,
            resize_w=224,
        )
    )

    shards = iter_npz_files(merged)
    assert shards
    merged_sample_count = sum(len(load_npz_shard(s)["y"]) for s in shards)
    assert merged_sample_count == 30

    models = tmp_path / "models"
    run_train(
        argparse.Namespace(
            dataset_dir=str(merged),
            out_dir=str(models),
            model_name="mobilenet_smoke.pt",
            arch="mobilenet_v3_small",
            exclude_domain=[],
            batch_size=4,
            epochs=0,
            learning_rate=1e-3,
            weight_decay=1e-4,
            val_fraction=0.2,
            target_far=0.2,
            seed=7,
            no_pretrained=True,
            unfreeze_features=False,
        )
    )
    ckpt_path = models / "mobilenet_smoke.pt"
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["arch"] == "mobilenet_v3_small"
    assert "preprocessing_contract" in ckpt
    assert "label_contract" in ckpt

    eval_dir = tmp_path / "eval"
    run_eval(
        argparse.Namespace(
            dataset_dir=str(merged),
            out_dir=str(eval_dir),
            arch="mobilenet_v3_small",
            custom_domain="custom_bg",
            custom_session_fraction=0.5,
            batch_size=4,
            epochs=0,
            learning_rate=1e-3,
            weight_decay=1e-4,
            val_fraction=0.2,
            target_far=0.2,
            seed=7,
            no_pretrained=True,
            unfreeze_features=False,
        )
    )
    assert (eval_dir / "domain_holdout_report.json").exists()
