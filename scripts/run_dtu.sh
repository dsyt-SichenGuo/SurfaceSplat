#!/usr/bin/env bash
# SurfaceSplat on the 15 DTU test scans (5 input views).
#   bash scripts/run_dtu.sh <dtu_root> <output_root>
# <dtu_root> contains dtu_scan<id>/ (image/, mask/, cameras_sphere.npz) and the GT ObsMask/, Points/.
set -e
DTU_ROOT=$1
OUT=${2:-output/dtu}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCANS=${SCANS:-"24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"}

mkdir -p "$OUT"
for id in $SCANS; do
    bash "$ROOT/scripts/run_scene.sh" dtu "$DTU_ROOT/dtu_scan$id" scan$id "$OUT" 2>&1 | tee "$OUT/scan$id.log"
done
python "$ROOT/tools/summarize.py" "$OUT"
