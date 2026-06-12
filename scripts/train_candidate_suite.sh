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
  "shufflenet_v2_x1_0"
  "resnet34"
  "resnet50"
  "regnet_x_1_6gf"
  "vgg_small_gap"
  "repvgg_a1_hmz"
  "repvgg_a2_hmz"
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
SKIP_EXISTING="${SKIP_EXISTING:-1}"
HAILO_TARGET="${HAILO_TARGET:-hailo8}"
HAILO_REAL_PROBE_SAMPLES="${HAILO_REAL_PROBE_SAMPLES:-32}"
HAILO_KEEP_GOING="${HAILO_KEEP_GOING:-0}"
HAILO_BIN="${HAILO_BIN:-}"
HAILORTCLI_BIN="${HAILORTCLI_BIN:-}"
HAILO_RUNTIME_METRICS_JSON="${HAILO_RUNTIME_METRICS_JSON:-}"
HAILO_HARDWARE_RESULTS_JSON="${HAILO_HARDWARE_RESULTS_JSON:-}"
USE_REPVGG_MODEL_ZOO="${USE_REPVGG_MODEL_ZOO:-0}"
REPVGG_MODEL_ZOO_DIR="${REPVGG_MODEL_ZOO_DIR:-${ROOT}/data/pretrained/hailo_model_zoo/repvgg/extracted}"

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
  model_ckpt="${arch_model_dir}/${arch}_binary.pt"
  eval_report="${arch_eval_dir}/domain_holdout_report.json"
  export_summary="${arch_export_dir}/${arch}_binary_${arch}_export_summary.json"
  bench_json="${BENCH_ROOT}/${arch}_runtime.json"
  mkdir -p "${arch_model_dir}" "${arch_eval_dir}" "${arch_export_dir}"
  repvgg_pretrain_args=()
  if [[ "${USE_REPVGG_MODEL_ZOO}" == "1" && ( "${arch}" == "repvgg_a1" || "${arch}" == "repvgg_a2" || "${arch}" == "repvgg_a1_hmz" || "${arch}" == "repvgg_a2_hmz" ) ]]; then
    repvgg_pretrain_args+=(--use-repvgg-model-zoo --repvgg-model-zoo-dir "${REPVGG_MODEL_ZOO_DIR}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${model_ckpt}" ]]; then
    echo "Skipping train; found ${model_ckpt}"
  else
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
      "${repvgg_pretrain_args[@]}" \
      "${GPU_FLAG[@]}"
  fi

  echo "=== [${arch}] eval ==="
  if [[ "${SKIP_EXISTING}" == "1" && -f "${eval_report}" ]]; then
    echo "Skipping eval; found ${eval_report}"
  else
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
      "${repvgg_pretrain_args[@]}" \
      "${GPU_FLAG[@]}"
  fi

  echo "=== [${arch}] export ==="
  if [[ "${SKIP_EXISTING}" == "1" && -f "${export_summary}" ]]; then
    echo "Skipping export; found ${export_summary}"
  else
    "${PY}" -m rfbd.cli export \
      --checkpoint "${model_ckpt}" \
      --out-dir "${arch_export_dir}" \
      --format torchscript \
      --format onnx || true
  fi

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
  if [[ "${SKIP_EXISTING}" == "1" && -f "${bench_json}" ]]; then
    echo "Skipping benchmark; found ${bench_json}"
  else
    "${PY}" "${ROOT}/scripts/benchmark_binary_runtime.py" "${bench_args[@]}"
  fi
done

echo ""
echo "=== [base] Hailo-8 compile/validate ==="
hailo_base_args=(
  -m rfbd.cli hailo compile
  --repo-root "${ROOT}"
  --target "${HAILO_TARGET}"
  --no-strict-far
  --calibration-dataset-dir "${DATASET_DIR}"
  --real-probe-samples "${HAILO_REAL_PROBE_SAMPLES}"
  --seed "${SEED}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  hailo_base_args+=(--skip-existing)
fi
if [[ "${HAILO_KEEP_GOING}" == "1" ]]; then
  hailo_base_args+=(--keep-going)
fi
if [[ -n "${HAILO_BIN}" ]]; then
  hailo_base_args+=(--hailo-bin "${HAILO_BIN}")
fi
if [[ -n "${HAILORTCLI_BIN}" ]]; then
  hailo_base_args+=(--hailortcli-bin "${HAILORTCLI_BIN}")
fi
if [[ -n "${HAILO_RUNTIME_METRICS_JSON}" ]]; then
  hailo_base_args+=(--runtime-metrics-json "${HAILO_RUNTIME_METRICS_JSON}")
fi
if [[ -n "${HAILO_HARDWARE_RESULTS_JSON}" ]]; then
  hailo_base_args+=(--hardware-results-json "${HAILO_HARDWARE_RESULTS_JSON}")
fi
for arch in "${ARCHES[@]}"; do
  hailo_base_args+=(--model-id "${arch}_binary")
done
"${PY}" "${hailo_base_args[@]}"

echo ""
echo "=== Build Hailo-8-first leaderboard and choose strict-FAR candidates ==="
leaderboard_args=(-m rfbd.candidate_suite --suite-root "${OUT_ROOT}")
for arch in "${ARCHES[@]}"; do
  leaderboard_args+=(--arch "${arch}")
done
"${PY}" "${leaderboard_args[@]}"

mapfile -t strict_candidates < <(head -n 2 "${LEADERBOARD_ROOT}/strict_far_candidates.txt" | sed '/^\s*$/d')

for arch in "${strict_candidates[@]}"; do
  echo ""
  echo "=== [${arch}] strict-FAR train/eval (target_far=${STRICT_TARGET_FAR}) ==="
  strict_model_dir="${STRICT_ROOT}/models/${arch}"
  strict_eval_dir="${STRICT_ROOT}/eval/${arch}"
  strict_model_ckpt="${strict_model_dir}/${arch}_binary_far003.pt"
  strict_eval_report="${strict_eval_dir}/domain_holdout_report.json"
  mkdir -p "${strict_model_dir}" "${strict_eval_dir}"
  repvgg_pretrain_args=()
  if [[ "${USE_REPVGG_MODEL_ZOO}" == "1" && ( "${arch}" == "repvgg_a1" || "${arch}" == "repvgg_a2" || "${arch}" == "repvgg_a1_hmz" || "${arch}" == "repvgg_a2_hmz" ) ]]; then
    repvgg_pretrain_args+=(--use-repvgg-model-zoo --repvgg-model-zoo-dir "${REPVGG_MODEL_ZOO_DIR}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${strict_model_ckpt}" ]]; then
    echo "Skipping strict-FAR train; found ${strict_model_ckpt}"
  else
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
      "${repvgg_pretrain_args[@]}" \
      "${GPU_FLAG[@]}"
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${strict_eval_report}" ]]; then
    echo "Skipping strict-FAR eval; found ${strict_eval_report}"
  else
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
      "${repvgg_pretrain_args[@]}" \
      "${GPU_FLAG[@]}"
  fi
done

if [[ "${#strict_candidates[@]}" -gt 0 ]]; then
  echo ""
  echo "=== [strict-FAR] Hailo-8 compile/validate ==="
  hailo_strict_args=(
    -m rfbd.cli hailo compile
    --repo-root "${ROOT}"
    --target "${HAILO_TARGET}"
    --calibration-dataset-dir "${DATASET_DIR}"
    --real-probe-samples "${HAILO_REAL_PROBE_SAMPLES}"
    --seed "${SEED}"
  )
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    hailo_strict_args+=(--skip-existing)
  fi
  if [[ "${HAILO_KEEP_GOING}" == "1" ]]; then
    hailo_strict_args+=(--keep-going)
  fi
  if [[ -n "${HAILO_BIN}" ]]; then
    hailo_strict_args+=(--hailo-bin "${HAILO_BIN}")
  fi
  if [[ -n "${HAILORTCLI_BIN}" ]]; then
    hailo_strict_args+=(--hailortcli-bin "${HAILORTCLI_BIN}")
  fi
  if [[ -n "${HAILO_RUNTIME_METRICS_JSON}" ]]; then
    hailo_strict_args+=(--runtime-metrics-json "${HAILO_RUNTIME_METRICS_JSON}")
  fi
  if [[ -n "${HAILO_HARDWARE_RESULTS_JSON}" ]]; then
    hailo_strict_args+=(--hardware-results-json "${HAILO_HARDWARE_RESULTS_JSON}")
  fi
  for arch in "${strict_candidates[@]}"; do
    hailo_strict_args+=(--model-id "${arch}_binary_far003")
  done
  "${PY}" "${hailo_strict_args[@]}"
fi

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
