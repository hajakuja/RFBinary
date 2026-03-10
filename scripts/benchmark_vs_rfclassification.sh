#!/usr/bin/env bash
set -euo pipefail

# Manual benchmark helper.
# Compares the new extractor against RFClassification/run_dronerf_feat.py on the same DroneRF subset.
# This script does not modify RFClassification.

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <dronerf_root> <out_new> <out_old>"
  echo "example: $0 /data/DroneRF /tmp/rfbd_new /tmp/rfclass_old"
  exit 1
fi

DRONERF_ROOT="$1"
OUT_NEW="$2"
OUT_OLD="$3"

mkdir -p "$OUT_NEW" "$OUT_OLD"

echo "Running new extractor..."
/usr/bin/time -p rfbd extract \
  --adapter legacy-dronerf \
  --dronerf-root "$DRONERF_ROOT" \
  --out-dir "$OUT_NEW" \
  --prefix bench \
  --workers 4 \
  --segment-ms 20 \
  --nfft 1024 \
  --noverlap 120 \
  --resize-h 224 \
  --resize-w 224

echo "Run RFClassification baseline manually with matching subset/settings and capture elapsed time in $OUT_OLD/."
echo "Then compare throughput (samples/sec). Target: new pipeline >= 3x baseline."
