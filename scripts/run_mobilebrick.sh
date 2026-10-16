#!/usr/bin/env bash
# SurfaceSplat on the 18 MobileBrick test scenes (10 input views).
#   bash scripts/run_mobilebrick.sh <mobilebrick_test_root> <output_root>
# <mobilebrick_test_root>/<scene>/ contains image/, mask/, cameras.npz and the GT mesh/, visibility_mask.npy.
set -e
MB_ROOT=$1
OUT=${2:-output/mobilebrick}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCENES=${SCENES:-"aston audi beetles big_ben boat bridge cabin camera castle colosseum convertible ferrari jeep london_bus motorcycle porsche satellite space_shuttle"}

mkdir -p "$OUT"
for scene in $SCENES; do
    bash "$ROOT/scripts/run_scene.sh" mobilebrick "$MB_ROOT/$scene" $scene "$OUT" 2>&1 | tee "$OUT/$scene.log"
done
python "$ROOT/tools/summarize.py" "$OUT"
