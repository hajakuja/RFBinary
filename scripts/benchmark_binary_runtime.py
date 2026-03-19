#!/usr/bin/env python3
"""Benchmark binary model inference latency for PyTorch and optional ONNX runtime."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from rfbd.modeling import create_binary_model


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _cpu_load_worker(stop_flag: mp.Event, size: int) -> None:
    rng = np.random.default_rng(13)
    x = rng.standard_normal((size,), dtype=np.float32)
    while not stop_flag.is_set():
        x = np.tanh(x * 1.0001 + 0.001).astype(np.float32, copy=False)


def _start_load_processes(num_workers: int, vec_size: int) -> Tuple[mp.Event, List[mp.Process]]:
    stop = mp.Event()
    procs: List[mp.Process] = []
    for _ in range(num_workers):
        p = mp.Process(target=_cpu_load_worker, args=(stop, vec_size), daemon=True)
        p.start()
        procs.append(p)
    return stop, procs


def _stop_load_processes(stop: mp.Event, procs: List[mp.Process]) -> None:
    stop.set()
    for p in procs:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()


def _summarize_latencies_ms(lat_ms: List[float], batch_size: int) -> Dict[str, float]:
    if not lat_ms:
        return {
            "samples": 0,
            "batch_size": int(batch_size),
            "latency_ms_mean": float("nan"),
            "latency_ms_p50": float("nan"),
            "latency_ms_p95": float("nan"),
            "throughput_fps": float("nan"),
        }

    total_s = float(np.sum(np.asarray(lat_ms, dtype=np.float64)) / 1000.0)
    total_frames = len(lat_ms) * int(batch_size)
    return {
        "samples": int(len(lat_ms)),
        "batch_size": int(batch_size),
        "latency_ms_mean": float(np.mean(lat_ms)),
        "latency_ms_p50": _percentile(lat_ms, 50.0),
        "latency_ms_p95": _percentile(lat_ms, 95.0),
        "throughput_fps": float(total_frames / max(total_s, 1e-9)),
    }


def _load_checkpoint(path: Path, arch_override: str | None = None) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint missing state_dict: {path}")
    arch = arch_override or ckpt.get("arch") or "vgg16"
    model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _benchmark_torch(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)

        lat_ms: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

    return _summarize_latencies_ms(lat_ms=lat_ms, batch_size=int(x.shape[0]))


def _benchmark_onnx(
    onnx_model: Path,
    x: np.ndarray,
    warmup: int,
    iters: int,
    ort_threads: int,
) -> Dict[str, float]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(f"onnxruntime unavailable: {exc}") from exc

    sess_opts = ort.SessionOptions()
    if ort_threads > 0:
        sess_opts.intra_op_num_threads = int(ort_threads)
        sess_opts.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(onnx_model),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    inp_name = session.get_inputs()[0].name

    for _ in range(warmup):
        _ = session.run(None, {inp_name: x})

    lat_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = session.run(None, {inp_name: x})
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    return _summarize_latencies_ms(lat_ms=lat_ms, batch_size=int(x.shape[0]))


def _run_with_optional_load(
    fn,
    with_load: bool,
    load_workers: int,
    load_vec_size: int,
):
    stop = None
    procs: List[mp.Process] = []
    if with_load and load_workers > 0:
        stop, procs = _start_load_processes(num_workers=load_workers, vec_size=load_vec_size)
        time.sleep(0.3)
    try:
        return fn()
    finally:
        if stop is not None:
            _stop_load_processes(stop=stop, procs=procs)


def run(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    model, ckpt = _load_checkpoint(ckpt_path, arch_override=args.arch)
    contract = ckpt.get("preprocessing_contract", {})
    h = int(contract.get("resize_h", args.height))
    w = int(contract.get("resize_w", args.width))
    batch_size = int(args.batch_size)

    x_torch = torch.randn(batch_size, 1, h, w, dtype=torch.float32)
    x_np = x_torch.numpy()

    rows: List[Dict[str, object]] = []

    for load_state in ("unloaded", "loaded"):
        with_load = load_state == "loaded"
        torch_stats = _run_with_optional_load(
            fn=lambda: _benchmark_torch(
                model=model,
                x=x_torch,
                warmup=args.warmup,
                iters=args.iterations,
            ),
            with_load=with_load,
            load_workers=args.load_workers,
            load_vec_size=args.load_vec_size,
        )
        rows.append(
            {
                "runtime": "pytorch",
                "load_state": load_state,
                **torch_stats,
            }
        )

    if args.onnx_model:
        onnx_path = Path(args.onnx_model)
        for load_state in ("unloaded", "loaded"):
            with_load = load_state == "loaded"
            try:
                onnx_stats = _run_with_optional_load(
                    fn=lambda: _benchmark_onnx(
                        onnx_model=onnx_path,
                        x=x_np,
                        warmup=args.warmup,
                        iters=args.iterations,
                        ort_threads=args.ort_threads,
                    ),
                    with_load=with_load,
                    load_workers=args.load_workers,
                    load_vec_size=args.load_vec_size,
                )
                rows.append(
                    {
                        "runtime": "onnx",
                        "load_state": load_state,
                        **onnx_stats,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "runtime": "onnx",
                        "load_state": load_state,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "arch": args.arch or ckpt.get("arch") or "vgg16",
        "input_shape": [batch_size, 1, h, w],
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "load_workers": int(args.load_workers),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote runtime benchmark JSON: {out_json}")
    print(f"Wrote runtime benchmark CSV: {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark PyTorch/ONNX binary model latency on CPU")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--onnx-model", type=str, default=None)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)

    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--ort-threads", type=int, default=1)

    default_workers = max(1, min(3, (os.cpu_count() or 4) - 1))
    p.add_argument("--load-workers", type=int, default=default_workers)
    p.add_argument("--load-vec-size", type=int, default=8192)

    p.add_argument("--out-json", type=str, required=True)
    p.add_argument("--out-csv", type=str, required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
