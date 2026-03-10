"""Training and prediction utilities for binary VGG pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from .io import iter_npz_files, load_npz_shard, save_json
from .metrics import summarize_binary, tune_threshold
from .modeling import VGG16Binary


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    pretrained: bool = True
    freeze_features: bool = True
    target_far: float = 0.05
    seed: int = 13


def load_dataset_arrays(dataset_dir: str | Path) -> Dict[str, np.ndarray]:
    files = iter_npz_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No NPZ shards found in {dataset_dir}")

    acc = {"feat": [], "y": [], "source_domain": [], "capture_id": [], "session_id": []}
    for f in files:
        d = load_npz_shard(f)
        for k in acc:
            acc[k].append(d[k])

    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def stratified_val_split(y: np.ndarray, val_fraction: float, seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    val_idx = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        val_idx.append(cls_idx[:n_val])

    val_idx = np.concatenate(val_idx, axis=0)
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = idx[train_mask]
    return train_idx, val_idx


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, train: bool, seed: int) -> DataLoader:
    x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    ds = TensorDataset(x_t, y_t)

    if train:
        cls, cnt = np.unique(y, return_counts=True)
        weights = {int(c): 1.0 / float(n) for c, n in zip(cls, cnt)}
        sample_w = np.array([weights[int(v)] for v in y], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)

    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def resolve_device(require_gpu: bool = False) -> torch.device:
    cuda_init_error: Exception | None = None
    if torch.cuda.is_available():
        try:
            # Some environments report CUDA available but cannot execute kernels.
            _ = torch.zeros(1).to("cuda")
            return torch.device("cuda")
        except Exception as exc:
            cuda_init_error = exc

    if require_gpu:
        detail = f" CUDA init error: {cuda_init_error}" if cuda_init_error is not None else ""
        raise RuntimeError(
            "CUDA GPU is required but unavailable. "
            f"Run GPU preflight (`nvidia-smi` and strict torch CUDA probe) and retry.{detail}"
        )
    return torch.device("cpu")


def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 256, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = resolve_device(require_gpu=False)

    loader = _to_loader(X, np.zeros((len(X),), dtype=np.int64), batch_size=batch_size, train=False, seed=0)
    model.eval()
    probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p1 = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p1.detach().cpu().numpy())

    return np.concatenate(probs, axis=0) if probs else np.empty((0,), dtype=np.float32)


def train_vgg_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> Tuple[nn.Module, float, Dict[str, float]]:
    if device is None:
        device = resolve_device(require_gpu=False)
    model = VGG16Binary(pretrained=cfg.pretrained, freeze_features=cfg.freeze_features).to(device)

    cls, cnt = np.unique(y_train, return_counts=True)
    class_weight = np.ones((2,), dtype=np.float32)
    for c, n in zip(cls, cnt):
        class_weight[int(c)] = len(y_train) / max(1, 2 * n)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_loader = _to_loader(X_train, y_train, batch_size=cfg.batch_size, train=True, seed=cfg.seed)

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"train epoch {epoch+1}/{cfg.epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if losses:
            print(f"epoch {epoch+1}: train loss={np.mean(losses):.4f}")

    val_prob = predict_proba(model, X_val, batch_size=max(cfg.batch_size, 128), device=device)
    threshold = tune_threshold(y_val, val_prob, target_far=cfg.target_far)
    val_summary = summarize_binary(y_val, val_prob, threshold)
    return model, threshold, {
        "val_roc_auc": float(val_summary.roc_auc),
        "val_pr_auc": float(val_summary.pr_auc),
        "val_f1": float(val_summary.f1),
        "val_far": float(val_summary.far),
    }


def _filter_domains(data: Dict[str, np.ndarray], exclude_domains: List[str]) -> Dict[str, np.ndarray]:
    if not exclude_domains:
        return data
    exclude_set = set(exclude_domains)
    mask = np.array([d not in exclude_set for d in data["source_domain"]], dtype=bool)
    return {k: v[mask] for k, v in data.items()}


def run_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        freeze_features=not args.unfreeze_features,
        target_far=args.target_far,
        seed=args.seed,
    )
    device = resolve_device(require_gpu=bool(getattr(args, "require_gpu", False)))

    data = load_dataset_arrays(args.dataset_dir)
    data = _filter_domains(data, args.exclude_domain or [])

    X = data["feat"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("Training set must contain both labels (0 and 1)")

    train_idx, val_idx = stratified_val_split(y, val_fraction=args.val_fraction, seed=args.seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model, threshold, val_metrics = train_vgg_binary(X_train, y_train, X_val, y_val, cfg, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / args.model_name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": float(threshold),
            "config": asdict(cfg),
            "val_metrics": val_metrics,
        },
        ckpt_path,
    )

    save_json(
        out_dir / f"{Path(args.model_name).stem}_summary.json",
        {
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
        },
    )
    print(f"Saved model checkpoint to {ckpt_path}")


def add_train_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_train = subparsers.add_parser("train", help="Model training commands")
    train_sub = p_train.add_subparsers(dest="train_cmd", required=True)

    p = train_sub.add_parser("vgg-binary", help="Train VGG16 binary classifier")
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="vgg_binary.pt")
    p.add_argument("--exclude-domain", action="append", default=[])

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
        "--require-gpu",
        action="store_true",
        default=False,
        help="Fail fast if CUDA GPU is unavailable.",
    )

    p.set_defaults(func=run_train)
