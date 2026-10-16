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
# SurfaceSplat: scenes are read directly from the IDR / NeuS format used by the SDF stages
# (image/, mask/, cameras npz), in the same normalized frame and with the same sparse views.

import os
from glob import glob
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from utils.graphics_utils import getWorld2View2, focal2fov
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    mask_gt: np.array
    width: int
    height: int
    FovY: np.array = None
    FovX: np.array = None
    fx: np.array = None
    fy: np.array = None
    cx: np.array = None
    cy: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def select_views(n_images, num_img):
    """Indices of the sparse input views (identical to voxurf/lib/load_data.py)."""
    return np.linspace(0, n_images - 1, num_img, dtype=int)


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K, R, t = out[0], out[1], out[2]
    K = K / K[2, 2]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T, translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readSparseCameras(path, cameras_name):
    """All cameras of a scene, with normalized OpenCV camera-to-world poses."""
    image_paths = sorted(glob(os.path.join(path, 'image', '*.png')))
    if len(image_paths) == 0:
        image_paths = sorted(glob(os.path.join(path, 'image', '*.jpg')))
    assert len(image_paths) > 0, f'no images found in {path}/image'
    mask_paths = sorted(glob(os.path.join(path, 'mask', '*.png')))
    camera_dict = np.load(os.path.join(path, cameras_name))

    cam_infos = []
    for idx, image_path in enumerate(image_paths):
        P = camera_dict['world_mat_%d' % idx].astype(np.float32) @ camera_dict['scale_mat_%d' % idx].astype(np.float32)
        K, c2w = load_K_Rt_from_P(P[:3, :4])
        w2c = np.linalg.inv(c2w)
        R = c2w[:3, :3]  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)
        width, height = image.size
        if len(mask_paths) == len(image_paths):
            mask_gt = cv2.imread(mask_paths[idx])[:, :, 1] / 256.0
        else:
            mask_gt = np.ones((height, width), dtype=np.float64)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        cam_infos.append(CameraInfo(
            uid=idx, R=R, T=T, FovY=focal2fov(fy, height), FovX=focal2fov(fx, width),
            fx=fx, fy=fy, cx=cx, cy=cy, image=image, image_path=image_path,
            image_name=os.path.splitext(os.path.basename(image_path))[0], mask_gt=mask_gt,
            width=width, height=height))
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readSparseSceneInfo(path, cameras_name, num_img, init_ply):
    """Sparse input views for training, all remaining views for evaluation.

    The Gaussians are initialized from `init_ply`, the point cloud sampled from the
    cleaned coarse mesh (tools/sample_points.py).
    """
    cam_infos = readSparseCameras(path, cameras_name)
    train_ids = select_views(len(cam_infos), num_img)
    train_cam_infos = [cam_infos[i] for i in train_ids]
    test_cam_infos = [c for i, c in enumerate(cam_infos) if i not in set(train_ids.tolist())]
    print('Sparse input views:', train_ids.tolist())

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd = fetchPly(init_ply)
    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization,
                     ply_path=init_ply)
