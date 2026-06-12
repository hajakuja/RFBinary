#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/RFBinaryDetect
PY=$ROOT/.venv/bin/python
DATASET_DIR=$ROOT/data/datasets/binary_all_v1
OUT_ROOT=$ROOT/data/experiments/g2_edge_suite
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_HAILO_RECOMPILE="${FORCE_HAILO_RECOMPILE:-0}"
HAILO_KEEP_GOING="${HAILO_KEEP_GOING:-1}"
HAILO_ONE_AT_A_TIME="${HAILO_ONE_AT_A_TIME:-1}"
HAILO_REQUIRE_GPU="${HAILO_REQUIRE_GPU:-1}"
HAILO_VENV="${HAILO_VENV:-/root/hailo_ai_sw_suite/hailo_venv}"
USE_REPVGG_MODEL_ZOO="${USE_REPVGG_MODEL_ZOO:-0}"
REPVGG_MODEL_ZOO_DIR="${REPVGG_MODEL_ZOO_DIR:-$ROOT/data/pretrained/hailo_model_zoo/repvgg/extracted}"
ARCHES="${ARCHES-vgg_small_gap resnet34 resnet50 regnet_x_1_6gf repvgg_a1_hmz repvgg_a2_hmz}"
read -r -a ARCH_LIST <<< "$ARCHES"

if [[ -x "$HAILO_VENV/bin/python" ]]; then
  hailo_cuda_libs="$("$HAILO_VENV/bin/python" - <<'PY'
import glob
import os
import site

paths = []
for root in site.getsitepackages():
    paths.extend(sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib"))))
print(":".join(paths))
PY
)"
  if [[ -n "$hailo_cuda_libs" ]]; then
    export LD_LIBRARY_PATH="$hailo_cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

  if [[ "$HAILO_REQUIRE_GPU" == "1" ]]; then
    "$HAILO_VENV/bin/python" - <<'PY'
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("Hailo TensorFlow GPUs:", [gpu.name for gpu in gpus])
sys.exit(0 if gpus else 1)
PY
  fi
fi

for arch in "${ARCH_LIST[@]}"; do
  eval_report="$OUT_ROOT/eval/$arch/domain_holdout_report.json"
  export_summary="$OUT_ROOT/exports/$arch/${arch}_binary_${arch}_export_summary.json"
  onnx_model="$OUT_ROOT/exports/$arch/${arch}_binary_${arch}.onnx"
  bench_json="$OUT_ROOT/bench/${arch}_runtime.json"
  repvgg_pretrain_args=()
  if [[ "$USE_REPVGG_MODEL_ZOO" == "1" && ( "$arch" == "repvgg_a1" || "$arch" == "repvgg_a2" || "$arch" == "repvgg_a1_hmz" || "$arch" == "repvgg_a2_hmz" ) ]]; then
    repvgg_pretrain_args+=(--use-repvgg-model-zoo --repvgg-model-zoo-dir "$REPVGG_MODEL_ZOO_DIR")
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$eval_report" ]]; then
    echo "Skipping eval for $arch; found $eval_report"
  else
    "$PY" -m rfbd.cli eval \
      --dataset-dir "$DATASET_DIR" \
      --out-dir "$OUT_ROOT/eval/$arch" \
      --arch "$arch" \
      --epochs 8 \
      --batch-size 64 \
      --val-fraction 0.1 \
      --target-far 0.05 \
      --custom-domain custom_bg \
      --custom-session-fraction 0.2 \
      --seed 13 \
      "${repvgg_pretrain_args[@]}"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$export_summary" && -f "$onnx_model" ]]; then
    echo "Skipping export for $arch; found $export_summary"
  else
    "$PY" -m rfbd.cli export \
      --checkpoint "$OUT_ROOT/models/$arch/${arch}_binary.pt" \
      --out-dir "$OUT_ROOT/exports/$arch" \
      --format torchscript \
      --format onnx \
      --static-batch
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$bench_json" ]]; then
    echo "Skipping benchmark for $arch; found $bench_json"
  else
    "$PY" "$ROOT/scripts/benchmark_binary_runtime.py" \
      --checkpoint "$OUT_ROOT/models/$arch/${arch}_binary.pt" \
      --arch "$arch" \
      --onnx-model "$onnx_model" \
      --warmup 20 \
      --iterations 120 \
      --out-json "$OUT_ROOT/bench/${arch}_runtime.json" \
      --out-csv "$OUT_ROOT/bench/${arch}_runtime.csv"
  fi
done

run_hailo_compile() {
  local -a model_args=("$@")
  local -a hailo_compile_args=(
    "$PY" -m rfbd.cli hailo compile
    --repo-root "$ROOT"
    --target hailo8
    --no-strict-far
    --calibration-dataset-dir data/datasets/binary_all_v1
    "${model_args[@]}"
  )

  if [[ "$FORCE_HAILO_RECOMPILE" != "1" && "$SKIP_EXISTING" == "1" ]]; then
    hailo_compile_args+=(--skip-existing)
  fi
  if [[ "$HAILO_KEEP_GOING" == "1" ]]; then
    hailo_compile_args+=(--keep-going)
  fi
  "${hailo_compile_args[@]}"
}

if [[ ${#ARCH_LIST[@]} -eq 0 ]]; then
  echo "Skipping Hailo compile; all requested models already have HEFs and validation reports"
elif [[ "$HAILO_ONE_AT_A_TIME" == "1" ]]; then
  echo "Checking Hailo artifacts one architecture at a time; this avoids Hailo/TensorFlow memory retention across models"
  for arch in "${ARCH_LIST[@]}"; do
    echo "Running Hailo compile for $arch in an isolated Python process"
    run_hailo_compile --model-id "${arch}_binary"
  done
else
  echo "Checking Hailo artifacts; reusable HEFs/reports will be skipped by rfbd.cli"
  hailo_model_args=()
  for arch in "${ARCH_LIST[@]}"; do
    hailo_model_args+=(--model-id "${arch}_binary")
  done
  run_hailo_compile "${hailo_model_args[@]}"
fi
