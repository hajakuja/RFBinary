"""Model export commands for deployment artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch

from .contracts import FeatureConfig
from .io import save_json
from .labels import ID_TO_LABEL
from .modeling import create_binary_model, list_supported_arches


SUPPORTED_EXPORT_FORMATS = ("torchscript", "onnx")


def _resolve_arch(args_arch: str | None, checkpoint_payload: Dict) -> str:
    arch = args_arch or checkpoint_payload.get("arch") or "vgg16"
    if arch not in list_supported_arches():
        raise ValueError(f"Unsupported architecture in checkpoint/args: {arch}")
    return arch


def _resolve_contract(checkpoint_payload: Dict) -> Dict[str, object]:
    contract = checkpoint_payload.get("preprocessing_contract")
    if isinstance(contract, dict):
        return contract
    return asdict(FeatureConfig())


def _dummy_input(contract: Dict[str, object], batch_size: int) -> torch.Tensor:
    h = int(contract.get("resize_h", 224))
    w = int(contract.get("resize_w", 224))
    return torch.randn(batch_size, 1, h, w, dtype=torch.float32)


def run_export(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint is missing 'state_dict': {ckpt_path}")

    arch = _resolve_arch(args.arch, ckpt)
    contract = _resolve_contract(ckpt)
    dummy = _dummy_input(contract=contract, batch_size=args.batch_size)

    model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem

    formats = args.format or list(SUPPORTED_EXPORT_FORMATS)
    exported: Dict[str, str] = {}
    skipped: Dict[str, str] = {}

    if "torchscript" in formats:
        ts_path = out_dir / f"{stem}_{arch}.torchscript.pt"
        traced = torch.jit.trace(model, dummy)
        traced.save(str(ts_path))
        exported["torchscript"] = str(ts_path)
        print(f"Exported TorchScript: {ts_path}")

    if "onnx" in formats:
        onnx_path = out_dir / f"{stem}_{arch}.onnx"
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
                opset_version=args.onnx_opset,
                do_constant_folding=True,
            )
            exported["onnx"] = str(onnx_path)
            print(f"Exported ONNX: {onnx_path}")
        except Exception as exc:
            skipped["onnx"] = f"{type(exc).__name__}: {exc}"
            print(f"ONNX export skipped: {skipped['onnx']}")
            if args.require_onnx:
                raise RuntimeError(skipped["onnx"]) from exc

    summary = {
        "checkpoint": str(ckpt_path),
        "arch": arch,
        "threshold": float(ckpt.get("threshold", 0.5)),
        "config": ckpt.get("config", {}),
        "preprocessing_contract": contract,
        "label_contract": ckpt.get("label_contract")
        or {str(int(k)): str(v) for k, v in ID_TO_LABEL.items()},
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = out_dir / f"{stem}_{arch}_export_summary.json"
    save_json(summary_path, summary)
    print(f"Export summary written: {summary_path}")


def add_export_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("export", help="Export trained checkpoint to deployment formats")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--arch", type=str, default=None, choices=list_supported_arches())
    p.add_argument("--format", action="append", choices=SUPPORTED_EXPORT_FORMATS, default=[])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--onnx-opset", type=int, default=17)
    p.add_argument(
        "--require-onnx",
        action="store_true",
        default=False,
        help="Fail if ONNX export cannot be produced.",
    )
    p.set_defaults(func=run_export)
