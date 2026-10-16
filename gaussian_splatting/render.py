#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# SurfaceSplat rendering:
#   * held-out test views  -> <model>/test/ours_<iter>/{renders,gt,masks_gt}   (novel view metrics)
#   * expanded novel views -> <model>/novel_views/{renders,masks,camera_pose_new.npy} (fine SDF stage)
#
#   python render.py -m <out>/scan24/gs --novel_method spline --num_novel 10
#

import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from scene import Scene
from scene.novel_views import generate_novel_cameras
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args

MASK_THRES = 0.9


def render_test_set(out_dir, views, gaussians, pipeline, background):
    for sub in ['renders', 'gt', 'masks_gt']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    for view in tqdm(views, desc="Rendering test views"):
        rendering = render(view, gaussians, pipeline, background)["render"]
        mask_gt = torch.tensor(view.mask_gt >= MASK_THRES, dtype=torch.float32)[None].repeat(3, 1, 1)
        name = view.image_name + ".png"
        torchvision.utils.save_image(rendering, os.path.join(out_dir, 'renders', name))
        torchvision.utils.save_image(view.original_image[0:3], os.path.join(out_dir, 'gt', name))
        torchvision.utils.save_image(mask_gt, os.path.join(out_dir, 'masks_gt', name))


def render_novel_views(out_dir, views, c2ws, gaussians, pipeline, background):
    for sub in ['renders', 'masks']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    np.save(os.path.join(out_dir, 'camera_pose_new.npy'), c2ws)
    for idx, view in enumerate(tqdm(views, desc="Rendering novel views")):
        out = render(view, gaussians, pipeline, background)
        mask = (out["uncertainty"] >= MASK_THRES).float().repeat(3, 1, 1)  # accumulated opacity
        torchvision.utils.save_image(out["render"], os.path.join(out_dir, 'renders', '{0:05d}.png'.format(idx)))
        torchvision.utils.save_image(mask, os.path.join(out_dir, 'masks', '{0:05d}.png'.format(idx)))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_test:
            render_test_set(os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter)),
                            scene.getTestCameras(), gaussians, pipeline, background)

        if not args.skip_novel:
            # training cameras are kept in the order of the input sequence (shuffle=False)
            views, c2ws = generate_novel_cameras(scene.getTrainCameras(), args.novel_method,
                                                 args.num_novel, args.novel_seed)
            out_dir = args.novel_dir or os.path.join(dataset.model_path, "novel_views")
            render_novel_views(out_dir, views, c2ws, gaussians, pipeline, background)
            print("Saved {} {} novel views to {}".format(len(views), args.novel_method, out_dir))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_novel", action="store_true")
    parser.add_argument("--novel_method", default="spline", choices=["spline", "perturb", "sphere"],
                        help="pose expansion strategy for the fine SDF stage")
    parser.add_argument("--num_novel", default=10, type=int, help="number of expanded views")
    parser.add_argument("--novel_seed", default=0, type=int)
    parser.add_argument("--novel_dir", default="", help="default: <model>/novel_views")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)
