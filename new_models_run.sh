ROOT=/root/RFBinaryDetect
PY=$ROOT/.venv/bin/python
DATASET_DIR=$ROOT/data/datasets/binary_all_v1
OUT_ROOT=$ROOT/data/experiments/g2_edge_suite

ARCHES="${ARCHES:-vgg_small_gap resnet34 resnet50 regnet_x_1_6gf repvgg_a1_hmz repvgg_a2_hmz}"
USE_REPVGG_MODEL_ZOO="${USE_REPVGG_MODEL_ZOO:-0}"
REPVGG_MODEL_ZOO_DIR="${REPVGG_MODEL_ZOO_DIR:-$ROOT/data/pretrained/hailo_model_zoo/repvgg/extracted}"
read -r -a ARCH_LIST <<< "$ARCHES"

for arch in "${ARCH_LIST[@]}"; do
  repvgg_pretrain_args=()
  if [[ "$USE_REPVGG_MODEL_ZOO" == "1" && ( "$arch" == "repvgg_a1" || "$arch" == "repvgg_a2" || "$arch" == "repvgg_a1_hmz" || "$arch" == "repvgg_a2_hmz" ) ]]; then
    repvgg_pretrain_args+=(--use-repvgg-model-zoo --repvgg-model-zoo-dir "$REPVGG_MODEL_ZOO_DIR")
  fi
  "$PY" -m rfbd.cli train binary \
    --dataset-dir "$DATASET_DIR" \
    --out-dir "$OUT_ROOT/models/$arch" \
    --model-name "${arch}_binary.pt" \
    --arch "$arch" \
    --epochs 8 \
    --batch-size 64 \
    --val-fraction 0.1 \
    --target-far 0.05 \
    --seed 13 \
    "${repvgg_pretrain_args[@]}"
done
