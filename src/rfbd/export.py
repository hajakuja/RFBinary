"""Model export commands for deployment artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable

import torch

from .contracts import FeatureConfig
from .io import save_json
from .labels import ID_TO_LABEL
from .modeling import create_binary_model, list_supported_arches


SUPPORTED_EXPORT_FORMATS = ("torchscript", "onnx")
DEFAULT_INPUT_NAME = "input"
DEFAULT_OUTPUT_NAME = "logits"


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


def _build_export_contract(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    *,
    input_name: str,
    output_name: str,
    onnx_opset: int,
    static_batch: bool,
) -> Dict[str, object]:
    with torch.no_grad():
        logits = model(dummy)

    return {
        "input_name": input_name,
        "input_shape": [int(v) for v in dummy.shape],
        "output_name": output_name,
        "output_shape": [int(v) for v in logits.shape],
        "batch_size": int(dummy.shape[0]),
        "onnx_opset": int(onnx_opset),
        "onnx_dynamic_batch": not static_batch,
    }


def load_export_context(
    checkpoint: str | Path,
    *,
    arch: str | None = None,
    batch_size: int = 1,
) -> Dict[str, object]:
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint is missing 'state_dict': {ckpt_path}")

    arch = _resolve_arch(arch, ckpt)
    contract = _resolve_contract(ckpt)
    dummy = _dummy_input(contract=contract, batch_size=batch_size)

    model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {
        "checkpoint_path": ckpt_path,
        "checkpoint_payload": ckpt,
        "arch": arch,
        "contract": contract,
        "dummy": dummy,
        "model": model,
        "threshold": float(ckpt.get("threshold", 0.5)),
        "config": ckpt.get("config", {}),
        "label_contract": ckpt.get("label_contract")
        or {str(int(k)): str(v) for k, v in ID_TO_LABEL.items()},
    }


def export_checkpoint(
    *,
    checkpoint: str | Path,
    out_dir: str | Path,
    arch: str | None = None,
    formats: Iterable[str] = SUPPORTED_EXPORT_FORMATS,
    batch_size: int = 1,
    onnx_opset: int = 17,
    require_onnx: bool = False,
    static_batch: bool = False,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Dict[str, object]:
    ctx = load_export_context(checkpoint, arch=arch, batch_size=batch_size)
    ckpt_path = Path(ctx["checkpoint_path"])
    model = ctx["model"]
    dummy = ctx["dummy"]
    arch = str(ctx["arch"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem

    formats = list(formats or SUPPORTED_EXPORT_FORMATS)
    exported: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    export_contract = _build_export_contract(
        model,
        dummy,
        input_name=input_name,
        output_name=output_name,
        onnx_opset=onnx_opset,
        static_batch=static_batch,
    )

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
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes=None
                if static_batch
                else {
                    input_name: {0: "batch_size"},
                    output_name: {0: "batch_size"},
                },
                opset_version=onnx_opset,
                do_constant_folding=True,
            )
            exported["onnx"] = str(onnx_path)
            print(f"Exported ONNX: {onnx_path}")
        except Exception as exc:
            skipped["onnx"] = f"{type(exc).__name__}: {exc}"
            print(f"ONNX export skipped: {skipped['onnx']}")
            if require_onnx:
                raise RuntimeError(skipped["onnx"]) from exc

    summary = {
        "checkpoint": str(ckpt_path),
        "arch": arch,
        "threshold": float(ctx["threshold"]),
        "config": ctx["config"],
        "preprocessing_contract": ctx["contract"],
        "label_contract": ctx["label_contract"],
        "export_contract": export_contract,
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = out_dir / f"{stem}_{arch}_export_summary.json"
    save_json(summary_path, summary)
    print(f"Export summary written: {summary_path}")
    return summary


def run_export(args: argparse.Namespace) -> None:
    export_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        arch=args.arch,
        formats=args.format or list(SUPPORTED_EXPORT_FORMATS),
        batch_size=args.batch_size,
        onnx_opset=args.onnx_opset,
        require_onnx=args.require_onnx,
        static_batch=args.static_batch,
    )


def add_export_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("export", help="Export trained checkpoint to deployment formats")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--arch", type=str, default=None, choices=list_supported_arches())
    p.add_argument("--format", action="append", choices=SUPPORTED_EXPORT_FORMATS, default=[])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--onnx-opset", type=int, default=17)
    p.add_argument(
        "--static-batch",
        action="store_true",
        default=False,
        help="Export ONNX with a fixed batch dimension instead of dynamic batch axes.",
    )
    p.add_argument(
        "--require-onnx",
        action="store_true",
        default=False,
        help="Fail if ONNX export cannot be produced.",
    )
    p.set_defaults(func=run_export)
