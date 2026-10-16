#!/usr/bin/env bash
# Full SurfaceSplat pipeline for one scene.
#
#   bash scripts/run_scene.sh <dtu|mobilebrick> <scene_dir> <scene_name> <output_root>
#
#   e.g. bash scripts/run_scene.sh dtu data/DTU/dtu_scan24 scan24 output/dtu
#        bash scripts/run_scene.sh mobilebrick data/MobileBrick/test/aston aston output/mobilebrick
#
# Run it inside the `surfacesplat` environment with CUDA_HOME set (see README).
# Optional:
#   VOXURF_PYTHON  interpreter for the SDF stages and point sampling (default: python)
#   GS_PYTHON      interpreter for 3DGS                              (default: python)
#   NORMAL_PYTHON  interpreter for Metric3D normal estimation       (default: python)
#   NORMAL_DIR     monocular normals (default: <scene_dir>/normals, estimated if missing)
#   DTU_EVAL_DIR   DTU GT root with ObsMask/ and Points/ (default: parent of <scene_dir>)
#   NOVEL_METHOD   spline | perturb | sphere (default: spline)
#   NUM_NOVEL      number of 3DGS novel views (default: 10)
#   GS_MASK        1: train 3DGS with the object masks (default: 1 on DTU, 0 on MobileBrick)
set -e

DATASET=$1
DATADIR=$(realpath "$2")
SCENE=$3
OUT=$(realpath -m "$4")
if [ -z "$DATASET" ] || [ -z "$DATADIR" ] || [ -z "$SCENE" ] || [ -z "$4" ]; then
    echo "usage: bash scripts/run_scene.sh <dtu|mobilebrick> <scene_dir> <scene_name> <output_root>"; exit 1
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VOXURF_PYTHON=${VOXURF_PYTHON:-python}
GS_PYTHON=${GS_PYTHON:-python}
NORMAL_PYTHON=${NORMAL_PYTHON:-python}
NORMAL_DIR=${NORMAL_DIR:-$DATADIR/normals}
NOVEL_METHOD=${NOVEL_METHOD:-spline}
NUM_NOVEL=${NUM_NOVEL:-10}

case $DATASET in
    dtu)         CAMERAS=cameras_sphere.npz; NUM_IMG=5;  EVAL_GT=${DTU_EVAL_DIR:-$(dirname "$DATADIR")}; GS_MASK=${GS_MASK:-1} ;;
    mobilebrick) CAMERAS=cameras.npz;        NUM_IMG=10; EVAL_GT=$DATADIR; GS_MASK=${GS_MASK:-0} ;;
    *) echo "unknown dataset $DATASET (dtu | mobilebrick)"; exit 1 ;;
esac
EXP=$OUT/$SCENE
mkdir -p "$EXP"
echo "==> $DATASET / $SCENE : $DATADIR -> $EXP"

# 0. monocular normals (Metric3D)
if [ ! -d "$NORMAL_DIR" ]; then
    echo "==> [0/5] estimating monocular normals"
    $NORMAL_PYTHON "$ROOT/tools/estimate_normals.py" --datadir "$DATADIR" --out_dir "$NORMAL_DIR"
fi

# 1. coarse SDF with normal supervision -> coarse mesh
echo "==> [1/5] coarse SDF stage"
cd "$ROOT/voxurf"
VOXURF_ARGS="--scene $SCENE --datadir $DATADIR --basedir $OUT"
$VOXURF_PYTHON run.py --config configs/$DATASET/coarse.py $VOXURF_ARGS --normal_dir "$NORMAL_DIR" --no_reload
$VOXURF_PYTHON run.py --config configs/$DATASET/coarse.py $VOXURF_ARGS --mesh_from_sdf

# 2. mesh cleaning + depth-based point sampling
echo "==> [2/5] sampling points from the coarse mesh"
$VOXURF_PYTHON "$ROOT/tools/sample_points.py" --mesh "$EXP/coarse/meshes/${SCENE}_mesh_normalized.ply" \
    --datadir "$DATADIR" --cameras_name $CAMERAS --num_img $NUM_IMG --out "$EXP/points.ply"

# 3. 3DGS initialized with the sampled points -> novel views
echo "==> [3/5] 3DGS"
cd "$ROOT/gaussian_splatting"
$GS_PYTHON train.py -s "$DATADIR" -m "$EXP/gs" --cameras_name $CAMERAS --num_img $NUM_IMG \
    --init_ply "$EXP/points.ply" -r 1 --disable_viewer --quiet $([ "$GS_MASK" = 1 ] && echo --mask)
$GS_PYTHON render.py -m "$EXP/gs" --novel_method $NOVEL_METHOD --num_novel $NUM_NOVEL --quiet
$GS_PYTHON metrics.py -m "$EXP/gs" || echo "warning: novel view metrics failed (LPIPS weights are downloaded on first use)"

# 4. fine SDF on the input views + 3DGS novel views -> final mesh + evaluation
echo "==> [4/5] fine SDF stage"
cd "$ROOT/voxurf"
$VOXURF_PYTHON run.py --config configs/$DATASET/fine.py $VOXURF_ARGS --gs_dir "$EXP/gs/novel_views" --no_reload
echo "==> [5/5] final mesh + evaluation"
$VOXURF_PYTHON run.py --config configs/$DATASET/fine.py $VOXURF_ARGS --mesh_from_sdf --eval_gt_dir "$EVAL_GT"

echo "==> done: mesh $EXP/fine/meshes/${SCENE}_mesh.ply"
