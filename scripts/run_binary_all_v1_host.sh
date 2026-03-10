#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/RFBinaryDetect}"
PY="${ROOT}/.venv/bin/python"

DRDE_SRC="${DRDE_SRC:-/root/Features/ARR_SPEC_1024_20}"
DRRF_L_SRC="${DRRF_L_SRC:-/root/Features_DroneRF/ARR_SPEC_L_1024_20}"
DRRF_H_SRC="${DRRF_H_SRC:-/root/Features_DroneRF/ARR_SPEC_H_1024_20}"
NO_DRONE_MERGED="${NO_DRONE_MERGED:-${ROOT}/data/merged/no_drone_merged}"

DRDE_CURATED="${DRDE_CURATED:-${ROOT}/data/curated/dronedetect_20k}"
NO_DRONE_CURATED="${NO_DRONE_CURATED:-${ROOT}/data/curated/no_drone_8k}"
DRRF_H_CONTRACT="${DRRF_H_CONTRACT:-${ROOT}/data/merged/dronerf_h_contract}"

DATASET_DIR="${DATASET_DIR:-${ROOT}/data/datasets/binary_all_v1}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/data/models/binary_all_v1}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/data/eval/binary_all_v1}"

run_privileged() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

check_gpu() {
  nvidia-smi >/dev/null 2>&1 || return 1
  "${PY}" - <<'PY'
import sys
import torch
ok = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
probe_err = None
if ok:
    try:
        _ = torch.zeros(1, device="cuda")
    except Exception as exc:
        probe_err = repr(exc)
        ok = False
print(
    f"torch_cuda_available={torch.cuda.is_available()} "
    f"device_count={torch.cuda.device_count()} "
    f"cuda_probe_error={probe_err}"
)
sys.exit(0 if ok else 1)
PY
}

echo "[1/6] GPU preflight"
if ! check_gpu; then
  echo "GPU preflight failed. Attempting no-reboot recovery..."
  run_privileged pkill -9 nvidia-persistenced || true
  run_privileged nvidia-persistenced --user nvidia-persistenced || true
  run_privileged nvidia-modprobe -u -c=0 || true
fi

check_gpu

echo "[2/6] Prepare curated RAM-safe subsets"
"${PY}" "${ROOT}/scripts/prepare_ram_safe_subsets.py" \
  --dronedetect-src "${DRDE_SRC}" \
  --dronedetect-out "${DRDE_CURATED}" \
  --dronedetect-target-samples 20000 \
  --no-drone-src "${NO_DRONE_MERGED}" \
  --no-drone-out "${NO_DRONE_CURATED}" \
  --no-drone-target-samples 8192 \
  --seed 13 \
  --link-mode symlink \
  --overwrite

echo "[3/6] Build DroneRF H contract shards (if missing)"
if [[ ! -f "${DRRF_H_CONTRACT}/dronerf_h_contract_summary.json" ]]; then
  "${PY}" -m rfbd.cli dataset build \
    --dronerf-arr-dir "${DRRF_H_SRC}" \
    --out-dir "${DRRF_H_CONTRACT}" \
    --prefix dronerf_h_contract \
    --shard-size 2048
fi

echo "[4/6] Build final merged dataset"
rm -rf "${DATASET_DIR}" "${MODEL_DIR}" "${EVAL_DIR}"
mkdir -p "${DATASET_DIR}" "${MODEL_DIR}" "${EVAL_DIR}"

"${PY}" -m rfbd.cli dataset build \
  --dronedetect-arr-dir "${DRDE_CURATED}" \
  --dronerf-arr-dir "${DRRF_L_SRC}" \
  --custom-shards-dir "${NO_DRONE_CURATED}" \
  --extra-shard-dir "${DRRF_H_CONTRACT}" \
  --out-dir "${DATASET_DIR}" \
  --prefix binary_all_v1 \
  --shard-size 2048

DATASET_SUMMARY_PATH="${DATASET_DIR}/binary_all_v1_summary.json" "${PY}" - <<'PY'
import json
import os
from pathlib import Path
summary_path = Path(os.environ["DATASET_SUMMARY_PATH"])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
labels = summary.get("label_counts", {})
sources = summary.get("source_counts", {})
samples = int(summary.get("samples_written", 0))
assert "0" in labels and "1" in labels, f"Expected both labels in {labels}"
for key in ("dronedetect", "dronerf", "custom_bg"):
    assert key in sources, f"Missing source '{key}' in {sources}"
assert 30000 <= samples <= 35000, f"Expected 30k-35k samples, got {samples}"
print("dataset_summary_ok", summary)
PY

echo "[5/6] Train vgg-binary (GPU required)"
if ! check_gpu; then
  echo "GPU check failed before training. Attempting recovery..."
  run_privileged pkill -9 nvidia-persistenced || true
  run_privileged nvidia-persistenced --user nvidia-persistenced || true
  run_privileged nvidia-modprobe -u -c=0 || true
  check_gpu
fi

"${PY}" -m rfbd.cli train vgg-binary \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${MODEL_DIR}" \
  --model-name vgg_binary_all_v1.pt \
  --epochs 8 \
  --batch-size 64 \
  --val-fraction 0.1 \
  --target-far 0.05 \
  --seed 13 \
  --require-gpu

echo "[6/6] Eval domain holdout (GPU required)"
if ! check_gpu; then
  echo "GPU check failed before eval. Attempting recovery..."
  run_privileged pkill -9 nvidia-persistenced || true
  run_privileged nvidia-persistenced --user nvidia-persistenced || true
  run_privileged nvidia-modprobe -u -c=0 || true
  check_gpu
fi

"${PY}" -m rfbd.cli eval \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${EVAL_DIR}" \
  --epochs 8 \
  --batch-size 64 \
  --val-fraction 0.1 \
  --target-far 0.05 \
  --custom-domain custom_bg \
  --custom-session-fraction 0.2 \
  --seed 13 \
  --require-gpu

echo "Done."
echo "Model: ${MODEL_DIR}/vgg_binary_all_v1.pt"
echo "Eval report: ${EVAL_DIR}/domain_holdout_report.json"
