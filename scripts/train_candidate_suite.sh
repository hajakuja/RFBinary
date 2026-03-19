#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/RFBinaryDetect}"
PY="${PY:-${ROOT}/.venv/bin/python}"

DATASET_DIR="${DATASET_DIR:-${ROOT}/data/datasets/binary_all_v1}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/experiments/g2_edge_suite}"

MODEL_ROOT="${OUT_ROOT}/models"
EVAL_ROOT="${OUT_ROOT}/eval"
EXPORT_ROOT="${OUT_ROOT}/exports"
BENCH_ROOT="${OUT_ROOT}/bench"
LEADERBOARD_ROOT="${OUT_ROOT}/leaderboard"
STRICT_ROOT="${OUT_ROOT}/strict_far"

ARCHES=(
  "vgg16"
  "resnet18"
  "mobilenet_v3_small"
  "shufflenet_v2_x1_0"
)

EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TARGET_FAR="${TARGET_FAR:-0.05}"
SEED="${SEED:-13}"
STRICT_TARGET_FAR="${STRICT_TARGET_FAR:-0.03}"
BENCH_ITERS="${BENCH_ITERS:-120}"
BENCH_WARMUP="${BENCH_WARMUP:-20}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"

mkdir -p "${MODEL_ROOT}" "${EVAL_ROOT}" "${EXPORT_ROOT}" "${BENCH_ROOT}" "${LEADERBOARD_ROOT}" "${STRICT_ROOT}"

echo "Running candidate suite in ${OUT_ROOT}"
echo "Dataset: ${DATASET_DIR}"

GPU_FLAG=()
if [[ "${REQUIRE_GPU}" == "1" ]]; then
  GPU_FLAG+=(--require-gpu)
fi

for arch in "${ARCHES[@]}"; do
  echo ""
  echo "=== [${arch}] train ==="
  arch_model_dir="${MODEL_ROOT}/${arch}"
  arch_eval_dir="${EVAL_ROOT}/${arch}"
  arch_export_dir="${EXPORT_ROOT}/${arch}"
  mkdir -p "${arch_model_dir}" "${arch_eval_dir}" "${arch_export_dir}"

  "${PY}" -m rfbd.cli train binary \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${arch_model_dir}" \
    --model-name "${arch}_binary.pt" \
    --arch "${arch}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-fraction "${VAL_FRACTION}" \
    --target-far "${TARGET_FAR}" \
    --seed "${SEED}" \
    "${GPU_FLAG[@]}"

  echo "=== [${arch}] eval ==="
  "${PY}" -m rfbd.cli eval \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${arch_eval_dir}" \
    --arch "${arch}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-fraction "${VAL_FRACTION}" \
    --target-far "${TARGET_FAR}" \
    --custom-domain custom_bg \
    --custom-session-fraction 0.2 \
    --seed "${SEED}" \
    "${GPU_FLAG[@]}"

  echo "=== [${arch}] export ==="
  "${PY}" -m rfbd.cli export \
    --checkpoint "${arch_model_dir}/${arch}_binary.pt" \
    --out-dir "${arch_export_dir}" \
    --format torchscript \
    --format onnx || true

  onnx_path="${arch_export_dir}/${arch}_binary_${arch}.onnx"
  bench_args=(
    "--checkpoint" "${arch_model_dir}/${arch}_binary.pt"
    "--arch" "${arch}"
    "--warmup" "${BENCH_WARMUP}"
    "--iterations" "${BENCH_ITERS}"
    "--out-json" "${BENCH_ROOT}/${arch}_runtime.json"
    "--out-csv" "${BENCH_ROOT}/${arch}_runtime.csv"
  )
  if [[ -f "${onnx_path}" ]]; then
    bench_args+=("--onnx-model" "${onnx_path}")
  fi

  echo "=== [${arch}] benchmark ==="
  "${PY}" "${ROOT}/scripts/benchmark_binary_runtime.py" "${bench_args[@]}"
done

echo ""
echo "=== Build leaderboard and choose strict-FAR candidates ==="
SUITE_ROOT="${OUT_ROOT}" "${PY}" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

suite_root = Path(os.environ["SUITE_ROOT"])
arches = ["vgg16", "resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]

def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def best_loaded_runtime(rows):
    best = None
    for row in rows:
        if row.get("load_state") != "loaded":
            continue
        if "error" in row:
            continue
        p95 = row.get("latency_ms_p95")
        if p95 is None or not isinstance(p95, (float, int)) or math.isnan(float(p95)):
            continue
        if best is None or float(p95) < best["latency_ms_p95"]:
            best = {
                "runtime": row.get("runtime"),
                "latency_ms_p95": float(p95),
                "latency_ms_p50": float(row.get("latency_ms_p50", float("nan"))),
                "throughput_fps": float(row.get("throughput_fps", float("nan"))),
            }
    return best

records = []
for arch in arches:
    eval_report = load_json(suite_root / "eval" / arch / "domain_holdout_report.json")
    bench_report = load_json(suite_root / "bench" / f"{arch}_runtime.json")

    split_metrics = eval_report.get("splits", {})
    avg_far = float("nan")
    avg_recall = float("nan")
    if split_metrics:
        fars = []
        recalls = []
        for split in split_metrics.values():
            tm = split.get("test_metrics", {})
            fars.append(float(tm.get("far", float("nan"))))
            recalls.append(float(tm.get("recall", float("nan"))))
        avg_far = float(sum(fars) / len(fars))
        avg_recall = float(sum(recalls) / len(recalls))

    best_loaded = best_loaded_runtime(bench_report.get("rows", []))
    records.append(
        {
            "arch": arch,
            "avg_far": avg_far,
            "avg_recall": avg_recall,
            "loaded_runtime": (best_loaded or {}).get("runtime"),
            "loaded_p95_ms": (best_loaded or {}).get("latency_ms_p95", float("nan")),
            "loaded_p50_ms": (best_loaded or {}).get("latency_ms_p50", float("nan")),
            "loaded_throughput_fps": (best_loaded or {}).get("throughput_fps", float("nan")),
            "splits": split_metrics,
        }
    )

base = next((r for r in records if r["arch"] == "vgg16"), None)
base_splits = base.get("splits", {}) if base else {}

for rec in records:
    far_ok = True
    recall_ok = True
    for split_name, split_data in rec["splits"].items():
        cand_tm = split_data.get("test_metrics", {})
        base_tm = base_splits.get(split_name, {}).get("test_metrics", {})
        base_far = float(base_tm.get("far", 0.0))
        base_recall = float(base_tm.get("recall", 1.0))
        cand_far = float(cand_tm.get("far", 1.0))
        cand_recall = float(cand_tm.get("recall", 0.0))
        if cand_far > base_far + 0.005:
            far_ok = False
        if cand_recall < base_recall - 0.03:
            recall_ok = False

    speed_ok = (
        isinstance(rec["loaded_p95_ms"], (float, int))
        and not math.isnan(float(rec["loaded_p95_ms"]))
        and float(rec["loaded_p95_ms"]) <= 250.0
        and float(rec.get("loaded_throughput_fps", float("nan"))) >= 4.0
    )
    rec["gate_far"] = bool(far_ok)
    rec["gate_recall"] = bool(recall_ok)
    rec["gate_speed"] = bool(speed_ok)
    rec["gate_pass"] = bool(far_ok and recall_ok and speed_ok)

records_sorted = sorted(
    records,
    key=lambda r: float("inf")
    if not isinstance(r.get("loaded_p95_ms"), (float, int)) or math.isnan(float(r["loaded_p95_ms"]))
    else float(r["loaded_p95_ms"]),
)

leaderboard_dir = suite_root / "leaderboard"
leaderboard_dir.mkdir(parents=True, exist_ok=True)

(leaderboard_dir / "leaderboard.json").write_text(
    json.dumps({"records": records_sorted}, indent=2),
    encoding="utf-8",
)

csv_path = leaderboard_dir / "leaderboard.csv"
fields = [
    "arch",
    "loaded_runtime",
    "loaded_p50_ms",
    "loaded_p95_ms",
    "loaded_throughput_fps",
    "avg_far",
    "avg_recall",
    "gate_far",
    "gate_recall",
    "gate_speed",
    "gate_pass",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in records_sorted:
        writer.writerow({k: row.get(k) for k in fields})

strict_candidates = [
    row["arch"]
    for row in records_sorted
    if row["arch"] != "vgg16" and isinstance(row.get("loaded_p95_ms"), (float, int)) and not math.isnan(float(row["loaded_p95_ms"]))
][:2]

(leaderboard_dir / "strict_far_candidates.txt").write_text("\n".join(strict_candidates) + "\n", encoding="utf-8")
print("leaderboard_json", leaderboard_dir / "leaderboard.json")
print("leaderboard_csv", csv_path)
print("strict_far_candidates", strict_candidates)
PY

mapfile -t strict_candidates < <(head -n 2 "${LEADERBOARD_ROOT}/strict_far_candidates.txt" | sed '/^\s*$/d')

for arch in "${strict_candidates[@]}"; do
  echo ""
  echo "=== [${arch}] strict-FAR train/eval (target_far=${STRICT_TARGET_FAR}) ==="
  strict_model_dir="${STRICT_ROOT}/models/${arch}"
  strict_eval_dir="${STRICT_ROOT}/eval/${arch}"
  mkdir -p "${strict_model_dir}" "${strict_eval_dir}"

  "${PY}" -m rfbd.cli train binary \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${strict_model_dir}" \
    --model-name "${arch}_binary_far003.pt" \
    --arch "${arch}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-fraction "${VAL_FRACTION}" \
    --target-far "${STRICT_TARGET_FAR}" \
    --seed "${SEED}" \
    "${GPU_FLAG[@]}"

  "${PY}" -m rfbd.cli eval \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${strict_eval_dir}" \
    --arch "${arch}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-fraction "${VAL_FRACTION}" \
    --target-far "${STRICT_TARGET_FAR}" \
    --custom-domain custom_bg \
    --custom-session-fraction 0.2 \
    --seed "${SEED}" \
    "${GPU_FLAG[@]}"
done

echo ""
echo "=== Strict-FAR summary ==="
SUITE_ROOT="${OUT_ROOT}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

suite_root = Path(os.environ["SUITE_ROOT"])
strict_root = suite_root / "strict_far" / "eval"

payload = {"strict_far_reports": {}}
if strict_root.exists():
    for arch_dir in sorted([p for p in strict_root.iterdir() if p.is_dir()]):
        rep = arch_dir / "domain_holdout_report.json"
        if rep.exists():
            payload["strict_far_reports"][arch_dir.name] = json.loads(rep.read_text(encoding="utf-8"))

out = suite_root / "leaderboard" / "strict_far_report.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("strict_far_report", out)
PY

echo "Done. Leaderboard: ${LEADERBOARD_ROOT}/leaderboard.json"
