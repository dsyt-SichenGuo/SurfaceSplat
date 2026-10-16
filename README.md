<div align="center">

# SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting

Zihui Gao<sup>1,3*</sup>, &nbsp;
[Jia-Wang Bian](https://scholar.google.com/citations?user=zeGz5JcAAAAJ&hl=en)<sup>2*</sup>, &nbsp;
[Guosheng Lin](https://scholar.google.com/citations?user=ZudEhvcAAAAJ&hl=en)<sup>3</sup>, &nbsp;
[Hao Chen](https://stan-haochen.github.io/)<sup>1†</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>ByteDance Seed, &nbsp;
<sup>3</sup>[Nanyang Technological University](https://www.ntu.edu.sg/), &nbsp;

**ICCV2025**
</div>
<div align="center">
    <a href="[PROJECT_PAGE_LINK_HERE]">
        <img src="teaser.png" width="90%">
    </a>
</div>


## 🚀 Overview
<div align="center">
    <a href="[PROJECT_PAGE_LINK_HERE]">
        <img src="main.png" width="90%">
    </a>
</div>


## 📖 Description

Surface reconstruction and novel view rendering from sparse-view images are challenging. 
Signed Distance Function (SDF)-based methods struggle with fine details, while 3D Gaussian Splatting (3DGS)-based approaches lack global geometry coherence. 
We propose a novel hybrid method that combines the strengths of both approaches: SDF captures coarse geometry to enhance 3DGS-based rendering, while newly rendered images from 3DGS refine the details of SDF for accurate surface reconstruction. As a result, our method surpasses state-of-the-art approaches in surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets.

The pipeline has four stages:

1. **Coarse SDF** (Voxurf, 96³ grid, 10k iters) trained on the sparse views with a monocular normal loss (Metric3D).
2. **Mesh cleaning + point sampling**: keep the largest connected component of the coarse mesh, render its depth into
   the input views and back-project randomly sampled pixels into a colored point cloud.
3. **3DGS** initialized with these points (7k iters), then renders novel views from expanded camera poses
   (spline pose interpolation by default).
4. **Fine SDF** (256³ grid, 20k iters) initialized from the coarse SDF and trained on the input views + the 3DGS
   novel views, giving the final mesh.


## 🚩 Plan
- [x] Release the training code.


## 🛠️ Getting Started

### Repository layout

```
SurfaceSplat/
├── voxurf/                 # SDF stages (coarse / fine), mesh extraction and evaluation
│   ├── run.py
│   ├── configs/{dtu,mobilebrick}/{coarse,fine}.py
│   └── lib/                # Voxurf models, data loader, CUDA kernels, DTU / MobileBrick evaluation
├── gaussian_splatting/     # 3DGS stage
│   ├── train.py  render.py  metrics.py
│   ├── scene/novel_views.py            # pose expansion (spline / perturb / sphere)
│   └── submodules/                     # modified diff-gaussian-rasterization, simple-knn
├── tools/
│   ├── estimate_normals.py # Metric3D monocular normals
│   ├── sample_points.py    # mesh cleaning + depth-based point sampling
│   └── summarize.py        # collect per-scene results
└── scripts/
    ├── run_scene.sh        # full pipeline for one scene
    ├── run_dtu.sh          # 15 DTU scans, 5 input views
    └── run_mobilebrick.sh  # 18 MobileBrick scenes, 10 input views
```

### Installation

We recommend a **single environment** for all stages (SDF, 3DGS, point sampling and Metric3D normals).
It needs a CUDA toolkit that matches PyTorch (CUDA 11.8 below) and a gcc supported by that toolkit (gcc <= 11).

```bash
conda create -n surfacesplat python=3.10 -y
conda activate surfacesplat
conda install -c conda-forge "libstdcxx-ng>=12" -y  # the CUDA kernels built with a recent gcc need GLIBCXX_3.4.29+

# PyTorch 2.1 + CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -r requirements.txt
pip install "setuptools<70" wheel                 # torch 2.1 extensions still need pkg_resources
pip install --no-build-isolation mmcv==1.7.2      # only used by Metric3D (pure python "lite" build)

# CUDA extensions: the 3DGS rasterizer (modified, also returns depth and accumulated opacity) and simple-knn
export CUDA_HOME=/usr/local/cuda-11.8      # adapt to your system
pip install --no-build-isolation gaussian_splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation gaussian_splatting/submodules/simple-knn
```

If `pip` reports *"Defaulting to user installation because normal site-packages is not writeable"* (this can
happen when the conda environment lives on a network file system), add `--no-user` to the `pip install` commands
so that the packages go into the environment and not into `~/.local`.

The Voxurf CUDA kernels (`voxurf/lib/cuda`) are compiled just-in-time the first time `voxurf/run.py` is
started, so `CUDA_HOME` must also be set when running the pipeline. They are cached in `~/.cache/torch_extensions`
(or `$TORCH_EXTENSIONS_DIR`).

`tools/sample_points.py` renders depth with pyrender. On a headless server it uses EGL
(`PYOPENGL_PLATFORM=egl`, set by the script), which requires an NVIDIA driver with EGL support.

`tools/estimate_normals.py` downloads Metric3D through `torch.hub` (GitHub access is needed once; code and
weights are cached in `$TORCH_HOME/hub`, default `~/.cache/torch/hub`). On a machine without internet access,
copy that cache from another machine and point `TORCH_HOME` to it.

<details>
<summary>Separate environments</summary>

The stages only communicate through files, so they can also run in different environments.
`scripts/run_scene.sh` picks the interpreter of each stage from `VOXURF_PYTHON` (SDF stages and point sampling),
`GS_PYTHON` (3DGS) and `NORMAL_PYTHON` (Metric3D), all defaulting to `python`. The experiments of the paper
used PyTorch 1.8 / CUDA 11.1 for the SDF stages (as in Voxurf), PyTorch 1.12 / CUDA 11.6 for 3DGS
(`gaussian_splatting/environment.yml`) and PyTorch 2.0 for Metric3D.
</details>

### Data

Both datasets use the IDR / NeuS scene format.

**DTU**: the preprocessed DTU of [NeuS](https://github.com/Totoro97/NeuS) / [Voxurf](https://github.com/wutong16/Voxurf),
plus the official evaluation data (`ObsMask/`, `Points/stl/`):
```
DTU/
├── dtu_scan24/{image/*.png, mask/*.png, cameras_sphere.npz}
├── ...
├── ObsMask/ObsMask24_10.mat, Plane24.mat, ...
└── Points/stl/stl024_total.ply, ...
```

**MobileBrick**: the official [MobileBrick](https://github.com/ActiveVisionLab/MobileBrick) test split:
```
MobileBrick/test/aston/{image/*.jpg, mask/*.png, cameras.npz, mesh/gt_mesh.ply, visibility_mask.npy}
```

**Sparse views.** Every stage uses the same `num_img` views evenly spaced over the image sequence
(`np.linspace(0, N-1, num_img)`): 5 views on DTU, 10 views on MobileBrick. All remaining views are used as test
views for novel view synthesis.

**Monocular normals** are estimated once per scene (`<scene>/normals/*.npy`, one `(3, H, W)` camera-space map per image).
`run_scene.sh` does this automatically if the folder does not exist:
```bash
python tools/estimate_normals.py --datadir DTU/dtu_scan24
```

### Running

With the environment activated (and `CUDA_HOME` set), run the whole pipeline for one scene:

```bash
bash scripts/run_scene.sh dtu         DTU/dtu_scan24          scan24 output/dtu
bash scripts/run_scene.sh mobilebrick MobileBrick/test/aston  aston  output/mobilebrick
```

or a whole benchmark (results are collected in `<output_root>/results.csv`):

```bash
bash scripts/run_dtu.sh         DTU               output/dtu
bash scripts/run_mobilebrick.sh MobileBrick/test  output/mobilebrick
```

**Dataset-specific settings.** The defaults of the configs and of `run_scene.sh` follow the paper experiments:

| | DTU | MobileBrick |
|---|---|---|
| input views | 5 | 10 |
| 3DGS novel views (`NUM_NOVEL`) | 10 | 10 |

Optional environment variables of `run_scene.sh`: `NOVEL_METHOD` (`spline` | `perturb` | `sphere`, default `spline`),
`NUM_NOVEL` (default 10), `NORMAL_DIR`, `DTU_EVAL_DIR`, and `CUDA_VISIBLE_DEVICES`.

<details>
<summary>Step-by-step commands (what <code>run_scene.sh</code> does)</summary>

```bash
SCENE=scan24; DATA=DTU/dtu_scan24; OUT=output/dtu
# 1. coarse SDF with normal supervision, then the coarse mesh
cd voxurf
python run.py --config configs/dtu/coarse.py --scene $SCENE --datadir $DATA --basedir $OUT --normal_dir $DATA/normals --no_reload
python run.py --config configs/dtu/coarse.py --scene $SCENE --datadir $DATA --basedir $OUT --mesh_from_sdf
cd ..
# 2. mesh cleaning + depth-based point sampling
python tools/sample_points.py --mesh $OUT/$SCENE/coarse/meshes/${SCENE}_mesh_normalized.ply \
    --datadir $DATA --cameras_name cameras_sphere.npz --num_img 5 --out $OUT/$SCENE/points.ply
# 3. 3DGS: train, render test views + expanded novel views, novel view metrics
cd gaussian_splatting
python train.py -s $DATA -m $OUT/$SCENE/gs --cameras_name cameras_sphere.npz --num_img 5 \
    --init_ply $OUT/$SCENE/points.ply -r 1 --disable_viewer
python render.py -m $OUT/$SCENE/gs --novel_method spline --num_novel 10
python metrics.py -m $OUT/$SCENE/gs
cd ..
# 4. fine SDF on input + novel views, then the final mesh and its evaluation
cd voxurf
python run.py --config configs/dtu/fine.py --scene $SCENE --datadir $DATA --basedir $OUT --gs_dir $OUT/$SCENE/gs/novel_views --no_reload
python run.py --config configs/dtu/fine.py --scene $SCENE --datadir $DATA --basedir $OUT --mesh_from_sdf --eval_gt_dir DTU
```
(relative paths above are relative to the repository root; use absolute paths when running from the sub-folders)
</details>

### Outputs

```
<output_root>/<scene>/
├── coarse/meshes/<scene>_mesh.ply             # coarse mesh (world frame)
├── points.ply                                 # sampled points used to initialize 3DGS
├── gs/
│   ├── test/ours_7000/{renders,gt,masks_gt}/  # held-out views
│   ├── novel_views/{renders,masks,camera_pose_new.npy}   # expanded views for the fine stage
│   └── results.json                           # PSNR / PSNR-F / SSIM / LPIPS
└── fine/meshes/
    ├── <scene>_mesh.ply                       # final mesh (world frame)
    └── results.json                           # DTU: chamfer | MobileBrick: chamfer, P/R/F1 @ 2.5mm and 5mm
```

## 🙏 Acknowledgements

This code is built on [Voxurf](https://github.com/wutong16/Voxurf),
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
[Metric3D](https://github.com/YvanYin/Metric3D). The DTU evaluation is adapted from
[DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and the MobileBrick evaluation follows
[MobileBrick](https://github.com/ActiveVisionLab/MobileBrick). Please also respect their licenses
(`voxurf/LICENSE`, `gaussian_splatting/LICENSE.md`).

## 🎫 License

For academic usage, this project is licensed under [the 2-clause BSD License](LICENSE). For commercial inquiries, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 📝 Citation

```bibtex
@inproceedings{gao2025surfacesplat,
  author    = {Gao, Zihui and Bian, Jia-Wang and Lin, Guosheng and Chen, Hao and Shen, Chunhua},
  title     = {SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting},
  booktitle = {2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  doi       = {10.1109/ICCV51701.2025.02649}
}
```
