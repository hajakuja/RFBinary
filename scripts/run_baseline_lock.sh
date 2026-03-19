#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/RFBinaryDetect}"
PY="${PY:-${ROOT}/.venv/bin/python}"

DATASET_DIR="${DATASET_DIR:-${ROOT}/data/datasets/binary_all_v1}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/baseline/g2_vgg16}"

MODEL_DIR="${OUT_ROOT}/models"
EVAL_DIR="${OUT_ROOT}/eval"
EXPORT_DIR="${OUT_ROOT}/exports"
BENCH_DIR="${OUT_ROOT}/bench"

EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TARGET_FAR="${TARGET_FAR:-0.05}"
SEED="${SEED:-13}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"

mkdir -p "${MODEL_DIR}" "${EVAL_DIR}" "${EXPORT_DIR}" "${BENCH_DIR}"

GPU_FLAG=()
if [[ "${REQUIRE_GPU}" == "1" ]]; then
  GPU_FLAG+=(--require-gpu)
fi

echo "[1/4] Train baseline vgg16"
"${PY}" -m rfbd.cli train binary \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${MODEL_DIR}" \
  --model-name "vgg16_binary_baseline.pt" \
  --arch vgg16 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --val-fraction "${VAL_FRACTION}" \
  --target-far "${TARGET_FAR}" \
  --seed "${SEED}" \
  "${GPU_FLAG[@]}"

echo "[2/4] Eval baseline vgg16"
"${PY}" -m rfbd.cli eval \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${EVAL_DIR}" \
  --arch vgg16 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --val-fraction "${VAL_FRACTION}" \
  --target-far "${TARGET_FAR}" \
  --custom-domain custom_bg \
  --custom-session-fraction 0.2 \
  --seed "${SEED}" \
  "${GPU_FLAG[@]}"

echo "[3/4] Export baseline artifacts"
"${PY}" -m rfbd.cli export \
  --checkpoint "${MODEL_DIR}/vgg16_binary_baseline.pt" \
  --out-dir "${EXPORT_DIR}" \
  --format torchscript \
  --format onnx || true

echo "[4/4] Benchmark baseline runtime"
onnx_model="${EXPORT_DIR}/vgg16_binary_baseline_vgg16.onnx"
bench_args=(
  "--checkpoint" "${MODEL_DIR}/vgg16_binary_baseline.pt"
  "--arch" "vgg16"
  "--out-json" "${BENCH_DIR}/vgg16_runtime.json"
  "--out-csv" "${BENCH_DIR}/vgg16_runtime.csv"
)
if [[ -f "${onnx_model}" ]]; then
  bench_args+=("--onnx-model" "${onnx_model}")
fi
"${PY}" "${ROOT}/scripts/benchmark_binary_runtime.py" "${bench_args[@]}"

echo "Baseline artifacts written to ${OUT_ROOT}"
