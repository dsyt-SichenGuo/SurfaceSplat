"""Sparse-view data loader shared by the coarse and the fine SDF stages.

Expected scene layout (IDR / NeuS convention, used by both DTU and MobileBrick):

    <datadir>/image/*.png|jpg       all images of the scene (sorted order = view index)
    <datadir>/mask/*.png            object masks (only used by the normal loss)
    <datadir>/<cameras_name>        world_mat_i / scale_mat_i (cameras_sphere.npz for DTU,
                                    cameras.npz for MobileBrick)
    <normal_dir>/*.npy              (coarse stage) monocular normals, one (3, H, W) array per image,
                                    see tools/estimate_normals.py
    <gs_dir>/renders/*.png          (fine stage) novel views rendered by 3DGS,
    <gs_dir>/masks/*.png            their alpha masks,
    <gs_dir>/camera_pose_new.npy    and their camera-to-world poses in the normalized frame

The sparse input views are `select_views(n_images, num_img)`, the same rule used by the
3DGS stage and by the point sampling script.
"""
import os
from glob import glob

import cv2 as cv
import imageio
import numpy as np
import torch
import torch.nn.functional as F


def select_views(n_images, num_img):
    """Indices of the sparse input views: `num_img` views evenly spaced over the sequence."""
    return np.linspace(0, n_images - 1, num_img, dtype=int)


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(P):
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose


def load_cameras(datadir, cameras_name, n_images):
    """Normalized intrinsics / camera-to-world poses (OpenCV convention) of all images."""
    camera_dict = np.load(os.path.join(datadir, cameras_name))
    intrinsics, poses = [], []
    for i in range(n_images):
        P = camera_dict['world_mat_%d' % i].astype(np.float32) @ camera_dict['scale_mat_%d' % i].astype(np.float32)
        K, pose = load_K_Rt_from_P(P[:3, :4])
        intrinsics.append(K)
        poses.append(pose)
    scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
    return np.stack(intrinsics), np.stack(poses), scale_mat


def list_images(datadir):
    rgb_paths = sorted(glob(os.path.join(datadir, 'image', '*.png')))
    if len(rgb_paths) == 0:
        rgb_paths = sorted(glob(os.path.join(datadir, 'image', '*.jpg')))
    assert len(rgb_paths) > 0, f'no images found in {datadir}/image'
    return rgb_paths


def read_mask(path, thres=None):
    mask = (imageio.imread(path) / 255.).astype(np.float32)
    if thres is not None:
        mask = (mask > thres).astype(np.float32)
    return mask[..., :3] if mask.ndim == 3 else np.repeat(mask[..., None], 3, -1)


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:, None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far


def load_data(args, reso_level=2, num_img=5, white_bg=False, wmask=False,
              normal_dir=None, gs_dir=None, **kwargs):
    """Load the sparse input views (+ optional normals and 3DGS novel views).

    Every loaded view is a training view: the SDF stages do not use held-out views.
    """
    datadir = args.datadir
    mask_thres = args.get('mask_thres', None)
    print("[ datadir {} | num_img {} | resolution level {} | normals {} | 3DGS views {} ]".format(
        datadir, num_img, reso_level, normal_dir, gs_dir))

    rgb_paths = list_images(datadir)
    mask_paths = sorted(glob(os.path.join(datadir, 'mask', '*.png')))
    all_K, all_poses, scale_mat = load_cameras(datadir, args.cameras_name, len(rgb_paths))
    ids = select_views(len(rgb_paths), num_img)
    print('Sparse input views:', ids.tolist())

    imgs = [(imageio.imread(rgb_paths[i]) / 255.).astype(np.float32)[..., :3] for i in ids]
    poses = [all_poses[i] for i in ids]
    if len(mask_paths) > 0:
        assert len(mask_paths) == len(rgb_paths)
        masks = [read_mask(mask_paths[i], mask_thres) for i in ids]
    else:
        masks = [np.ones_like(im) for im in imgs]
    K = all_K[ids[0]].copy()

    normals = None
    if normal_dir:
        normal_paths = sorted(glob(os.path.join(normal_dir, '*.npy')))
        assert len(normal_paths) == len(rgb_paths), \
            f'expected {len(rgb_paths)} normal maps in {normal_dir}, found {len(normal_paths)}'
        normals = [np.load(normal_paths[i]).astype(np.float32) for i in ids]  # (3, H, W) each

    if gs_dir:
        gs_rgb_paths = sorted(glob(os.path.join(gs_dir, 'renders', '*.png')))
        gs_mask_paths = sorted(glob(os.path.join(gs_dir, 'masks', '*.png')))
        gs_poses = np.load(os.path.join(gs_dir, 'camera_pose_new.npy'))
        assert len(gs_rgb_paths) == len(gs_poses) > 0, f'invalid 3DGS output in {gs_dir}'
        for i, path in enumerate(gs_rgb_paths):
            img = (imageio.imread(path) / 255.).astype(np.float32)[..., :3]
            assert img.shape == imgs[0].shape, '3DGS renders must have the input image resolution'
            imgs.append(img)
            poses.append(gs_poses[i].astype(np.float32))
            masks.append(read_mask(gs_mask_paths[i]) if len(gs_mask_paths) else np.ones_like(img))
            if normals is not None:
                normals.append(np.zeros_like(normals[0]))  # no normal supervision for rendered views
        print('Added {} 3DGS views from {}'.format(len(gs_rgb_paths), gs_dir))

    imgs = np.stack(imgs, 0)
    poses = np.stack(poses, 0)
    masks = np.stack(masks, 0)
    if normals is not None:
        normals = np.stack(normals, 0)

    H, W = imgs[0].shape[:2]
    focal = K[0, 0]
    if wmask:
        bg = 1. if white_bg else 0.
        imgs = imgs * masks + bg * (1 - masks)

    if reso_level > 1:
        H, W = int(H / reso_level), int(W / reso_level)
        imgs = F.interpolate(torch.from_numpy(imgs).permute(0, 3, 1, 2), size=(H, W)).permute(0, 2, 3, 1).numpy()
        masks = F.interpolate(torch.from_numpy(masks).permute(0, 3, 1, 2), size=(H, W)).permute(0, 2, 3, 1).numpy()
        if normals is not None:
            normals = F.interpolate(torch.from_numpy(normals), size=(H, W)).numpy()
        K[:2] /= reso_level
        focal /= reso_level
    if normals is not None:
        normals = normals.transpose(0, 2, 3, 1)  # (N, H, W, 3)

    i_train = np.arange(len(imgs))
    i_test = np.array([0])  # only used for optional visual checks
    near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])
    print('Loaded {} views, H {} W {}, near {:.3f} far {:.3f}'.format(len(imgs), H, W, near, far))

    return dict(
        hwf=[H, W, focal], HW=np.array([im.shape[:2] for im in imgs]),
        Ks=K[None, :3, :3].repeat(len(poses), axis=0), near=near, far=far,
        i_train=i_train, i_val=i_test, i_test=i_test,
        poses=poses, render_poses=poses[i_test],
        images=imgs, masks=masks, normals=normals,
        irregular_shape=False, scale_mats_np=scale_mat,
    )
