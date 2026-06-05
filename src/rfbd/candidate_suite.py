"""Candidate-suite leaderboard helpers for Hailo-8-first model selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_BASE_ARCHES = (
    "shufflenet_v2_x1_0",
    "resnet34",
    "resnet50",
    "regnet_x_1_6gf",
    "vgg_small_gap",
)
LEGACY_REFERENCE_ARCHES = ("vgg13", "vgg16", "mobilenet_v3_small", "resnet18")
LEGACY_VGG_ARCHES = frozenset({"vgg13", "vgg16"})
HAILO_TARGET = "hailo8"
LATENCY_TARGET_MS = 50.0
BASELINE_ARCH = "vgg16"


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _numeric(value: object) -> float | None:
    if isinstance(value, (float, int)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return None


def _best_loaded_runtime(rows: Iterable[Dict[str, object]]) -> Dict[str, float | str] | None:
    best: Dict[str, float | str] | None = None
    for row in rows:
        if row.get("load_state") != "loaded" or "error" in row:
            continue
        p95 = _numeric(row.get("latency_ms_p95"))
        if p95 is None:
            continue
        candidate = {
            "runtime": str(row.get("runtime", "")),
            "latency_ms_p95": p95,
            "latency_ms_p50": _numeric(row.get("latency_ms_p50")) if _numeric(row.get("latency_ms_p50")) is not None else float("nan"),
            "throughput_fps": _numeric(row.get("throughput_fps")) if _numeric(row.get("throughput_fps")) is not None else float("nan"),
        }
        if best is None or p95 < float(best["latency_ms_p95"]):
            best = candidate
    return best


def _average_split_metrics(split_metrics: Dict[str, object]) -> tuple[float, float]:
    fars: List[float] = []
    recalls: List[float] = []
    for split in split_metrics.values():
        if not isinstance(split, dict):
            continue
        metrics = split.get("test_metrics", {})
        if not isinstance(metrics, dict):
            continue
        far = _numeric(metrics.get("far"))
        recall = _numeric(metrics.get("recall"))
        if far is not None:
            fars.append(far)
        if recall is not None:
            recalls.append(recall)
    if not fars or not recalls:
        return float("nan"), float("nan")
    return float(sum(fars) / len(fars)), float(sum(recalls) / len(recalls))


def _split_gate_against_baseline(
    *,
    split_metrics: Dict[str, object],
    baseline_split_metrics: Dict[str, object],
) -> tuple[bool, bool]:
    far_ok = True
    recall_ok = True
    for split_name, split_data in split_metrics.items():
        if not isinstance(split_data, dict):
            continue
        cand_metrics = split_data.get("test_metrics", {})
        base_metrics = {}
        if isinstance(baseline_split_metrics.get(split_name), dict):
            base_metrics = baseline_split_metrics[split_name].get("test_metrics", {})

        cand_far = _numeric(cand_metrics.get("far"))
        cand_recall = _numeric(cand_metrics.get("recall"))
        base_far = _numeric(base_metrics.get("far"))
        base_recall = _numeric(base_metrics.get("recall"))

        if cand_far is None or base_far is None or cand_far > base_far + 0.005:
            far_ok = False
        if cand_recall is None or base_recall is None or cand_recall < base_recall - 0.03:
            recall_ok = False

    return bool(far_ok), bool(recall_ok)


def _hailo_artifact_paths(suite_root: Path, arch: str) -> tuple[Path, Path]:
    hailo_dir = suite_root / "exports" / arch / "hailo"
    stem = f"{arch}_binary.{HAILO_TARGET}"
    return hailo_dir / f"{stem}.manifest.json", hailo_dir / f"{stem}.validation.json"


def build_hailo8_leaderboard(
    suite_root: str | Path,
    arches: Sequence[str] = DEFAULT_BASE_ARCHES,
) -> Dict[str, object]:
    suite_root = Path(suite_root)
    eval_arches = tuple(dict.fromkeys([*arches, BASELINE_ARCH]))
    eval_reports = {arch: _load_json(suite_root / "eval" / arch / "domain_holdout_report.json") for arch in eval_arches}
    baseline_split_metrics = eval_reports.get(BASELINE_ARCH, {}).get("splits", {})
    if not isinstance(baseline_split_metrics, dict):
        baseline_split_metrics = {}

    records: List[Dict[str, object]] = []
    for arch in arches:
        eval_report = eval_reports.get(arch, {})
        bench_report = _load_json(suite_root / "bench" / f"{arch}_runtime.json")
        manifest_path, validation_path = _hailo_artifact_paths(suite_root, arch)
        manifest = _load_json(manifest_path)
        validation = _load_json(validation_path)

        split_metrics = eval_report.get("splits", {})
        if not isinstance(split_metrics, dict):
            split_metrics = {}
        avg_far, avg_recall = _average_split_metrics(split_metrics)
        gate_far, gate_recall = _split_gate_against_baseline(
            split_metrics=split_metrics,
            baseline_split_metrics=baseline_split_metrics,
        )

        best_loaded = _best_loaded_runtime(bench_report.get("rows", [])) if isinstance(bench_report, dict) else None
        performance = validation.get("performance", {}) if isinstance(validation, dict) else {}
        gates = validation.get("gates", {}) if isinstance(validation, dict) else {}

        hailo_latency_ms = _numeric(performance.get("selected_latency_ms"))
        compile_success = bool(manifest.get("compiled"))
        validation_success = bool(validation)
        flat_output_pass = bool(gates.get("flat_output"))
        manifest_contract_pass = bool(gates.get("manifest_contract"))
        ordering_pass = bool(gates.get("ordering"))
        deployable = bool(validation.get("deployable"))
        gate_speed = hailo_latency_ms is not None and hailo_latency_ms < LATENCY_TARGET_MS
        vgg_family_stop = arch in LEGACY_VGG_ARCHES and (not flat_output_pass or not gate_speed)

        record = {
            "arch": arch,
            "loaded_runtime": (best_loaded or {}).get("runtime"),
            "loaded_p50_ms": (best_loaded or {}).get("latency_ms_p50", float("nan")),
            "loaded_p95_ms": (best_loaded or {}).get("latency_ms_p95", float("nan")),
            "loaded_throughput_fps": (best_loaded or {}).get("throughput_fps", float("nan")),
            "avg_far": avg_far,
            "avg_recall": avg_recall,
            "hailo_compile_success": compile_success,
            "hailo_validation_success": validation_success,
            "hailo_manifest_contract": manifest_contract_pass,
            "hailo_flat_output_pass": flat_output_pass,
            "hailo_ordering_pass": ordering_pass,
            "hailo_latency_ms": hailo_latency_ms if hailo_latency_ms is not None else float("nan"),
            "hailo_latency_source": performance.get("latency_source"),
            "hailo_latency_provisional": bool(performance.get("latency_is_provisional", True)),
            "hailo_hardware_latency_confirmed": bool(performance.get("hardware_latency_confirmed", False)),
            "hailo_deployable": deployable,
            "deployment_decision": validation.get("deployment_decision"),
            "fallback_recommendation": validation.get("fallback_recommendation"),
            "vgg_family_stop": vgg_family_stop,
            "gate_far": gate_far,
            "gate_recall": gate_recall,
            "gate_speed": gate_speed,
            "gate_pass": bool(
                compile_success
                and validation_success
                and manifest_contract_pass
                and flat_output_pass
                and ordering_pass
                and deployable
                and gate_far
                and gate_recall
                and gate_speed
            ),
        }
        records.append(record)

    records_sorted = sorted(
        records,
        key=lambda row: (
            not bool(row["gate_pass"]),
            -float(row["avg_recall"]) if _numeric(row["avg_recall"]) is not None else float("inf"),
            float(row["avg_far"]) if _numeric(row["avg_far"]) is not None else float("inf"),
            float(row["hailo_latency_ms"]) if _numeric(row["hailo_latency_ms"]) is not None else float("inf"),
            str(row["arch"]),
        ),
    )

    strict_candidates = [
        row["arch"]
        for row in records_sorted
        if row["arch"] != BASELINE_ARCH and row["gate_pass"] and not row["vgg_family_stop"]
    ][:2]

    return {
        "target_arch": HAILO_TARGET,
        "latency_target_ms": LATENCY_TARGET_MS,
        "baseline_arch": BASELINE_ARCH,
        "legacy_reference_arches": list(LEGACY_REFERENCE_ARCHES),
        "records": records_sorted,
        "strict_candidates": strict_candidates,
    }


def write_hailo8_leaderboard(
    suite_root: str | Path,
    arches: Sequence[str] = DEFAULT_BASE_ARCHES,
) -> Dict[str, object]:
    suite_root = Path(suite_root)
    leaderboard = build_hailo8_leaderboard(suite_root=suite_root, arches=arches)
    leaderboard_dir = suite_root / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    (leaderboard_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    fields = [
        "arch",
        "loaded_runtime",
        "loaded_p50_ms",
        "loaded_p95_ms",
        "loaded_throughput_fps",
        "avg_far",
        "avg_recall",
        "hailo_compile_success",
        "hailo_validation_success",
        "hailo_flat_output_pass",
        "hailo_latency_ms",
        "hailo_latency_source",
        "hailo_latency_provisional",
        "hailo_hardware_latency_confirmed",
        "hailo_deployable",
        "gate_far",
        "gate_recall",
        "gate_speed",
        "gate_pass",
    ]
    with (leaderboard_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in leaderboard["records"]:
            writer.writerow({field: row.get(field) for field in fields})

    (leaderboard_dir / "strict_far_candidates.txt").write_text(
        "\n".join(str(v) for v in leaderboard["strict_candidates"]) + ("\n" if leaderboard["strict_candidates"] else ""),
        encoding="utf-8",
    )
    return leaderboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Hailo-8-first leaderboard for RFBinaryDetect candidate models.")
    parser.add_argument("--suite-root", type=str, required=True)
    parser.add_argument("--arch", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    arches = tuple(args.arch or DEFAULT_BASE_ARCHES)
    leaderboard = write_hailo8_leaderboard(suite_root=args.suite_root, arches=arches)
    print("leaderboard_json", Path(args.suite_root) / "leaderboard" / "leaderboard.json")
    print("leaderboard_csv", Path(args.suite_root) / "leaderboard" / "leaderboard.csv")
    print("strict_far_candidates", leaderboard["strict_candidates"])


if __name__ == "__main__":
    main()
