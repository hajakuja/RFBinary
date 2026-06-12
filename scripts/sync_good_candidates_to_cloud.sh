#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/RFBinaryDetect}"
PY="${PY:-${ROOT}/.venv/bin/python}"
SRC="${SRC:-${ROOT}/data/experiments/g2_edge_suite}"
DEST="${DEST:-/root/gargoyle/EDGE SYSTEM/RF SUBSYSTEM/SW/Training/g2_edge_suite}"

ACTIVE_ARCHES="${ACTIVE_ARCHES:-shufflenet_v2_x1_0 resnet34 resnet50 regnet_x_1_6gf vgg_small_gap repvgg_a1_hmz repvgg_a2_hmz}"
LEGACY_REMOVE_ARCHES="${LEGACY_REMOVE_ARCHES:-vgg13 vgg16 mobilenet_v3_small resnet18}"
read -r -a ACTIVE_ARCH_LIST <<< "${ACTIVE_ARCHES}"
read -r -a LEGACY_REMOVE_ARCH_LIST <<< "${LEGACY_REMOVE_ARCHES}"

mkdir -p "${DEST}/models" "${DEST}/eval" "${DEST}/exports" "${DEST}/bench" "${DEST}/hailo_calibration" "${DEST}/leaderboard"

included_arches=()
skipped_arches=()
for arch in "${ACTIVE_ARCH_LIST[@]}"; do
  stem="${arch}_binary"
  manifest="${SRC}/exports/${arch}/hailo/${stem}.hailo8.manifest.json"
  hef="${SRC}/exports/${arch}/hailo/${stem}.hailo8.hef"
  if [[ ! -f "${manifest}" || ! -f "${hef}" ]]; then
    skipped_arches+=("${arch}:missing_manifest_or_hef")
    rm -rf "${DEST}/models/${arch}" "${DEST}/eval/${arch}" "${DEST}/exports/${arch}"
    rm -f "${DEST}/bench/${arch}_runtime.csv" "${DEST}/bench/${arch}_runtime.json"
    continue
  fi

  compiled="$("${PY}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1])).get("compiled"))).lower())' "${manifest}")"
  decision="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("deployment_decision") or "")' "${manifest}")"
  if [[ "${compiled}" != "true" || "${decision}" == "compile_failed" || "${decision}" == "stale_contract" ]]; then
    skipped_arches+=("${arch}:${decision:-not_compiled}")
    rm -rf "${DEST}/models/${arch}" "${DEST}/eval/${arch}" "${DEST}/exports/${arch}"
    rm -f "${DEST}/bench/${arch}_runtime.csv" "${DEST}/bench/${arch}_runtime.json"
    continue
  fi

  for section in models eval exports; do
    if [[ -d "${SRC}/${section}/${arch}" ]]; then
      mkdir -p "${DEST}/${section}/${arch}"
      rsync -a --delete "${SRC}/${section}/${arch}/" "${DEST}/${section}/${arch}/"
    fi
  done
  rsync -a --delete "${SRC}/bench/${arch}_runtime."* "${DEST}/bench/" 2>/dev/null || true
  included_arches+=("${arch}")
done

for arch in "${LEGACY_REMOVE_ARCH_LIST[@]}"; do
  rm -rf "${DEST}/models/${arch}" "${DEST}/eval/${arch}" "${DEST}/exports/${arch}"
  rm -f "${DEST}/bench/${arch}_runtime.csv" "${DEST}/bench/${arch}_runtime.json"
done
rm -rf "${DEST}/strict_far"

rsync -a "${SRC}/hailo_calibration/README.md" "${DEST}/hailo_calibration/" 2>/dev/null || true
rsync -a "${SRC}/hailo_calibration/binary_all_v1_feature_stats.json" "${DEST}/hailo_calibration/" 2>/dev/null || true
rsync -a "${SRC}/hailo_calibration/rfbd_linear_uint8_calibration_1024.npy" "${DEST}/hailo_calibration/" 2>/dev/null || true
rsync -a "${SRC}/hailo_calibration/rfbd_linear_uint8_calibration_1024.metadata.json" "${DEST}/hailo_calibration/" 2>/dev/null || true
rm -f "${DEST}/hailo_calibration/vgg13_binary_uint8_calibration_1024.npy" \
      "${DEST}/hailo_calibration/vgg13_binary_uint8_calibration_1024.metadata.json" \
      "${DEST}/hailo_calibration/vgg16_binary_uint8_calibration_1024.npy" \
      "${DEST}/hailo_calibration/vgg16_binary_uint8_calibration_1024.metadata.json"

"${PY}" - "${SRC}" "${DEST}" "${ACTIVE_ARCHES}" "${LEGACY_REMOVE_ARCHES}" "${included_arches[*]}" "${skipped_arches[*]}" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

src = Path(sys.argv[1])
dest = Path(sys.argv[2])
active_arches = sys.argv[3].split()
legacy_arches = sys.argv[4].split()
included = sys.argv[5].split()
skipped = sys.argv[6].split()
records = []
for arch in included:
    stem = f"{arch}_binary"
    manifest_path = src / "exports" / arch / "hailo" / f"{stem}.hailo8.manifest.json"
    validation_path = src / "exports" / arch / "hailo" / f"{stem}.hailo8.validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    performance = validation.get("performance", {}) if isinstance(validation, dict) else {}
    records.append(
        {
            "arch": arch,
            "model_id": stem,
            "compiled": bool(manifest.get("compiled")),
            "deployable": bool(manifest.get("deployable")),
            "deployment_decision": manifest.get("deployment_decision"),
            "hef_path": f"exports/{arch}/hailo/{stem}.hailo8.hef",
            "validation_report": f"exports/{arch}/hailo/{stem}.hailo8.validation.json" if validation_path.exists() else None,
            "latency_ms": performance.get("selected_latency_ms"),
            "latency_source": performance.get("latency_source"),
        }
    )

leaderboard = dest / "leaderboard"
leaderboard.mkdir(parents=True, exist_ok=True)
for path in leaderboard.iterdir():
    if path.is_file():
        path.unlink()
payload = {
    "synced_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": str(src),
    "destination": str(dest),
    "active_arches": active_arches,
    "included_arches": included,
    "skipped_arches": skipped,
    "legacy_removed_arches": legacy_arches,
    "notes": [
        "vgg_small_gap is synced only when a valid non-stale Hailo-8 HEF exists.",
        "vgg13/vgg16 are intentionally removed from the cloud copy because their Hailo artifacts are not deployable.",
        "shufflenet_v2_x1_0 remains the fallback until a newer candidate passes hardware Hailo-8 validation.",
    ],
    "records": records,
}
(leaderboard / "good_candidates_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
with (leaderboard / "good_candidates_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    fieldnames = [
        "arch",
        "model_id",
        "compiled",
        "deployable",
        "deployment_decision",
        "hef_path",
        "validation_report",
        "latency_ms",
        "latency_source",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
PY

echo "Synced Hailo candidate arches: ${included_arches[*]:-(none)}"
echo "Skipped arches: ${skipped_arches[*]:-(none)}"
echo "Destination: ${DEST}"
