"""Domain-held-out evaluation command."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .io import save_json
from .metrics import summarize_binary
from .modeling import list_supported_arches
from .training import (
    DEFAULT_REPVGG_MODEL_ZOO_DIR,
    TrainConfig,
    _resolve_repvgg_pretrained_onnx,
    label_contract,
    load_dataset_arrays,
    predict_proba,
    preprocessing_contract,
    resolve_device,
    stratified_val_split,
    train_binary_indexed,
)


def _train_test_by_domain(data: Dict[str, np.ndarray], domain: str) -> Tuple[np.ndarray, np.ndarray]:
    test_mask = data["source_domain"] == domain
    train_mask = ~test_mask
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def _train_test_custom_session_holdout(
    data: Dict[str, np.ndarray],
    custom_domain: str = "custom_bg",
    session_fraction: float = 0.2,
    seed: int = 13,
) -> Tuple[np.ndarray, np.ndarray]:
    domain = data["source_domain"]
    sessions = data["session_id"]

    custom_sessions = np.unique(sessions[domain == custom_domain])
    if len(custom_sessions) == 0:
        raise ValueError(f"No sessions found for custom domain '{custom_domain}'")

    rng = np.random.default_rng(seed)
    shuf = custom_sessions.copy()
    rng.shuffle(shuf)

    n_test = max(1, int(round(len(shuf) * session_fraction)))
    test_sessions = set(shuf[:n_test].tolist())

    test_mask = np.array(
        [
            (d == custom_domain and s in test_sessions)
            for d, s in zip(domain.tolist(), sessions.tolist())
        ],
        dtype=bool,
    )
    train_mask = ~test_mask
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def _validate_split(y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, split_name: str) -> None:
    if len(test_idx) == 0:
        raise ValueError(f"Split '{split_name}' has empty test set")
    if len(train_idx) == 0:
        raise ValueError(f"Split '{split_name}' has empty training set")
    if len(np.unique(y[train_idx])) < 2:
        raise ValueError(f"Split '{split_name}' training set must contain both classes")


def run_eval(args: argparse.Namespace) -> None:
    arch = getattr(args, "arch", "vgg16")
    cfg = TrainConfig(
        arch=arch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        freeze_features=not args.unfreeze_features,
        target_far=args.target_far,
        seed=args.seed,
        repvgg_pretrained_onnx=_resolve_repvgg_pretrained_onnx(args=args, arch=arch),
    )
    device = resolve_device(require_gpu=bool(getattr(args, "require_gpu", False)))

    data = load_dataset_arrays(args.dataset_dir)
    X = np.asarray(data["feat"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = []

    if np.any(data["source_domain"] == "dronedetect"):
        splits.append(("holdout_dronedetect",) + _train_test_by_domain(data, "dronedetect"))

    if np.any(data["source_domain"] == "dronerf"):
        splits.append(("holdout_dronerf",) + _train_test_by_domain(data, "dronerf"))

    if np.any(data["source_domain"] == args.custom_domain):
        splits.append(
            (
                "holdout_custom_bg_sessions",
                *_train_test_custom_session_holdout(
                    data,
                    custom_domain=args.custom_domain,
                    session_fraction=args.custom_session_fraction,
                    seed=args.seed,
                ),
            )
        )

    if not splits:
        raise ValueError("No valid holdout split could be formed from dataset domains")

    report = {
        "dataset_dir": args.dataset_dir,
        "arch": cfg.arch,
        "splits": {},
        "config": asdict(cfg),
        "preprocessing_contract": preprocessing_contract(),
        "label_contract": label_contract(),
    }

    for split_name, train_idx, test_idx in splits:
        _validate_split(y, train_idx, test_idx, split_name)

        y_train_all = y[train_idx]
        tr_rel_idx, val_rel_idx = stratified_val_split(y_train_all, val_fraction=args.val_fraction, seed=args.seed)
        tr_idx = train_idx[tr_rel_idx]
        val_idx = train_idx[val_rel_idx]

        model, threshold, val_metrics = train_binary_indexed(X, y, tr_idx, val_idx, cfg, device=device)

        y_test = y[test_idx]
        prob_test = predict_proba(
            model,
            X,
            batch_size=max(cfg.batch_size, 128),
            device=device,
            indices=test_idx,
        )
        test_summary = summarize_binary(y_test, prob_test, threshold)

        ckpt_path = out_dir / f"{cfg.arch}_binary_{split_name}.pt"
        torch.save(
            {
                "arch": cfg.arch,
                "state_dict": model.state_dict(),
                "threshold": float(threshold),
                "split": split_name,
                "config": asdict(cfg),
                "val_metrics": val_metrics,
                "preprocessing_contract": preprocessing_contract(),
                "label_contract": label_contract(),
                "pretrained_init": getattr(model, "pretrained_init", None),
            },
            ckpt_path,
        )

        report["splits"][split_name] = {
            "checkpoint": str(ckpt_path),
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "pretrained_init": getattr(model, "pretrained_init", None),
            "test_metrics": {
                "roc_auc": float(test_summary.roc_auc),
                "pr_auc": float(test_summary.pr_auc),
                "f1": float(test_summary.f1),
                "precision": float(test_summary.precision),
                "recall": float(test_summary.recall),
                "far": float(test_summary.far),
                "confusion_matrix": test_summary.confusion_matrix,
                "test_size": int(len(test_idx)),
            },
        }

        print(
            f"[{cfg.arch}:{split_name}] roc_auc={test_summary.roc_auc:.4f} "
            f"pr_auc={test_summary.pr_auc:.4f} f1={test_summary.f1:.4f} far={test_summary.far:.4f}"
        )
        del model, prob_test, y_test, y_train_all
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(out_dir / "domain_holdout_report.json", report)
    print(f"Evaluation report written to {out_dir / 'domain_holdout_report.json'}")


def add_eval_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("eval", help="Domain-held-out evaluation")
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--arch", type=str, default="vgg16", choices=list_supported_arches())

    p.add_argument("--custom-domain", type=str, default="custom_bg")
    p.add_argument("--custom-session-fraction", type=float, default=0.2)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--target-far", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--no-pretrained", action="store_true", default=False)
    p.add_argument("--unfreeze-features", action="store_true", default=False)
    p.add_argument(
        "--use-repvgg-model-zoo",
        action="store_true",
        default=False,
        help="Initialize RepVGG backbones from downloaded Hailo Model Zoo ONNX deploy weights.",
    )
    p.add_argument(
        "--repvgg-pretrained-onnx",
        type=str,
        default=None,
        help="Explicit RepVGG-A1/A2 ONNX file to use as a pretrained deploy-form backbone initializer.",
    )
    p.add_argument(
        "--repvgg-model-zoo-dir",
        type=str,
        default=str(DEFAULT_REPVGG_MODEL_ZOO_DIR),
        help="Directory containing RepVGG-A1.onnx and RepVGG-A2.onnx for --use-repvgg-model-zoo or *_hmz arches.",
    )
    p.add_argument(
        "--require-gpu",
        action="store_true",
        default=False,
        help="Fail fast if CUDA GPU is unavailable.",
    )

    p.set_defaults(func=run_eval)
