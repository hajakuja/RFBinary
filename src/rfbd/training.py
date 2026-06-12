"""Training and prediction utilities for binary classification models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .contracts import FeatureConfig
from .io import iter_npz_files, load_npz_shard, save_json
from .labels import ID_TO_LABEL
from .metrics import summarize_binary, tune_threshold
from .modeling import (
    REPVGG_ARCHES,
    REPVGG_MODEL_ZOO_ARCHES,
    create_binary_model,
    list_supported_arches,
    load_repvgg_model_zoo_onnx,
    repvgg_model_zoo_onnx_path,
)


DEFAULT_REPVGG_MODEL_ZOO_DIR = Path("data/pretrained/hailo_model_zoo/repvgg/extracted")


@dataclass
class TrainConfig:
    arch: str = "vgg16"
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    pretrained: bool = True
    freeze_features: bool = True
    target_far: float = 0.05
    seed: int = 13
    repvgg_pretrained_onnx: str | None = None


def preprocessing_contract() -> Dict[str, object]:
    return asdict(FeatureConfig())


def label_contract() -> Dict[str, str]:
    return {str(int(k)): str(v) for k, v in ID_TO_LABEL.items()}


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


class IndexedArrayDataset(Dataset):
    """Dataset wrapper that keeps one shared feature array and split indices."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = None if y is None else np.asarray(y, dtype=np.int64)
        if indices is None:
            self.indices = np.arange(len(self.X), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        x = torch.from_numpy(self.X[idx])
        if self.y is None:
            return x
        return x, int(self.y[idx])


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


def _to_loader(
    X: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    train: bool,
    seed: int,
    indices: np.ndarray | None = None,
) -> DataLoader:
    _ = seed
    ds = IndexedArrayDataset(X=X, y=y, indices=indices)

    if train:
        if ds.y is None:
            raise ValueError("Training loader requires labels")
        subset_y = ds.y[ds.indices]
        cls, cnt = np.unique(subset_y, return_counts=True)
        weights = {int(c): 1.0 / float(n) for c, n in zip(cls, cnt)}
        sample_w = np.array([weights[int(v)] for v in subset_y], dtype=np.float64)
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


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: torch.device | None = None,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    if device is None:
        device = resolve_device(require_gpu=False)

    loader = _to_loader(X, y=None, batch_size=batch_size, train=False, seed=0, indices=indices)
    model.eval()
    probs = []

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p1 = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p1.detach().cpu().numpy())

    return np.concatenate(probs, axis=0) if probs else np.empty((0,), dtype=np.float32)


def train_binary_indexed(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> Tuple[nn.Module, float, Dict[str, float]]:
    if device is None:
        device = resolve_device(require_gpu=False)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    model = create_binary_model(
        arch=cfg.arch,
        pretrained=cfg.pretrained,
        freeze_features=cfg.freeze_features,
    )
    if cfg.repvgg_pretrained_onnx:
        pretrained_init = load_repvgg_model_zoo_onnx(model, cfg.repvgg_pretrained_onnx)
        setattr(model, "pretrained_init", pretrained_init)
        print(
            f"Loaded {cfg.arch} pretrained backbone from {cfg.repvgg_pretrained_onnx} "
            f"({pretrained_init['loaded_tensors']} tensors, skipped {pretrained_init['skipped_head_tensors']})"
        )
    model = model.to(device)

    y_train = y[train_idx]
    y_val = y[val_idx]

    cls, cnt = np.unique(y_train, return_counts=True)
    class_weight = np.ones((2,), dtype=np.float32)
    for c, n in zip(cls, cnt):
        class_weight[int(c)] = len(y_train) / max(1, 2 * n)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_loader = _to_loader(
        X,
        y=y,
        batch_size=cfg.batch_size,
        train=True,
        seed=cfg.seed,
        indices=train_idx,
    )

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

    val_prob = predict_proba(
        model,
        X,
        batch_size=max(cfg.batch_size, 128),
        device=device,
        indices=val_idx,
    )
    threshold = tune_threshold(y_val, val_prob, target_far=cfg.target_far)
    val_summary = summarize_binary(y_val, val_prob, threshold)
    return model, threshold, {
        "val_roc_auc": float(val_summary.roc_auc),
        "val_pr_auc": float(val_summary.pr_auc),
        "val_f1": float(val_summary.f1),
        "val_far": float(val_summary.far),
    }


def train_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> Tuple[nn.Module, float, Dict[str, float]]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    train_idx = np.arange(len(y_train), dtype=np.int64)
    val_idx = np.arange(len(y_train), len(y), dtype=np.int64)
    return train_binary_indexed(X, y, train_idx, val_idx, cfg, device=device)


def train_vgg_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> Tuple[nn.Module, float, Dict[str, float]]:
    cfg_vgg = TrainConfig(**{**asdict(cfg), "arch": "vgg16"})
    return train_binary(X_train, y_train, X_val, y_val, cfg_vgg, device=device)


def _filter_domains(data: Dict[str, np.ndarray], exclude_domains: List[str]) -> Dict[str, np.ndarray]:
    if not exclude_domains:
        return data
    exclude_set = set(exclude_domains)
    mask = np.array([d not in exclude_set for d in data["source_domain"]], dtype=bool)
    return {k: v[mask] for k, v in data.items()}


def _build_cfg(args: argparse.Namespace, arch: str) -> TrainConfig:
    return TrainConfig(
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


def _resolve_repvgg_pretrained_onnx(args: argparse.Namespace, arch: str) -> str | None:
    explicit = getattr(args, "repvgg_pretrained_onnx", None)
    use_model_zoo = bool(getattr(args, "use_repvgg_model_zoo", False))
    auto_model_zoo = arch in REPVGG_MODEL_ZOO_ARCHES
    if explicit and use_model_zoo:
        raise ValueError("Use either --repvgg-pretrained-onnx or --use-repvgg-model-zoo, not both")
    if explicit:
        if arch not in REPVGG_ARCHES:
            raise ValueError("--repvgg-pretrained-onnx is only valid with RepVGG architectures")
        return str(Path(explicit))
    if not use_model_zoo and not auto_model_zoo:
        return None
    if arch not in REPVGG_ARCHES:
        raise ValueError("--use-repvgg-model-zoo is only valid with RepVGG architectures")
    model_zoo_dir = Path(getattr(args, "repvgg_model_zoo_dir", DEFAULT_REPVGG_MODEL_ZOO_DIR))
    return str(repvgg_model_zoo_onnx_path(arch, model_zoo_dir))


def _run_train_impl(args: argparse.Namespace, arch: str) -> None:
    cfg = _build_cfg(args=args, arch=arch)
    device = resolve_device(require_gpu=bool(getattr(args, "require_gpu", False)))

    data = load_dataset_arrays(args.dataset_dir)
    data = _filter_domains(data, args.exclude_domain or [])

    X = np.asarray(data["feat"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("Training set must contain both labels (0 and 1)")

    train_idx, val_idx = stratified_val_split(y, val_fraction=args.val_fraction, seed=args.seed)
    model, threshold, val_metrics = train_binary_indexed(X, y, train_idx, val_idx, cfg, device=device)
    pretrained_init = getattr(model, "pretrained_init", None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / args.model_name

    payload = {
        "arch": cfg.arch,
        "state_dict": model.state_dict(),
        "threshold": float(threshold),
        "config": asdict(cfg),
        "val_metrics": val_metrics,
        "preprocessing_contract": preprocessing_contract(),
        "label_contract": label_contract(),
        "pretrained_init": pretrained_init,
    }
    torch.save(payload, ckpt_path)

    save_json(
        out_dir / f"{Path(args.model_name).stem}_summary.json",
        {
            "arch": cfg.arch,
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "preprocessing_contract": preprocessing_contract(),
            "pretrained_init": pretrained_init,
        },
    )
    print(f"Saved model checkpoint to {ckpt_path}")


def run_train(args: argparse.Namespace) -> None:
    arch = getattr(args, "arch", "vgg16")
    _run_train_impl(args=args, arch=arch)


def run_train_vgg_alias(args: argparse.Namespace) -> None:
    _run_train_impl(args=args, arch="vgg16")


def _add_shared_train_args(p: argparse.ArgumentParser, default_model_name: str) -> None:
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default=default_model_name)
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


def add_train_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_train = subparsers.add_parser("train", help="Model training commands")
    train_sub = p_train.add_subparsers(dest="train_cmd", required=True)

    p_binary = train_sub.add_parser("binary", help="Train binary classifier with selectable architecture")
    _add_shared_train_args(p_binary, default_model_name="binary_model.pt")
    p_binary.add_argument("--arch", type=str, default="vgg16", choices=list_supported_arches())
    p_binary.set_defaults(func=run_train)

    # Backward-compatible alias.
    p_vgg = train_sub.add_parser("vgg-binary", help="Train VGG16 binary classifier (compat alias)")
    _add_shared_train_args(p_vgg, default_model_name="vgg_binary.pt")
    p_vgg.set_defaults(func=run_train_vgg_alias)
