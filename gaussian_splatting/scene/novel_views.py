"""Pose expansion strategies of SurfaceSplat (Sec. 3.3).

All functions take the (ordered) sparse training cameras and return a list of 4x4
camera-to-world matrices in the OpenCV convention and in the normalized scene frame,
i.e. the convention expected by the fine SDF stage.

  spline   camera pose interpolation along a spline through the input cameras (default)
  perturb  camera position perturbation around the input cameras, looking at the origin
  sphere   positions sampled on a sphere fitted to the input cameras, looking at the origin
"""
import math

import numpy as np
import scipy.interpolate
import scipy.spatial.transform
import torch
from scipy.optimize import least_squares

from utils.camera_utils import cameracenter_from_translation, translation_from_cameracenter
from utils.graphics_utils import getWorld2View, getProjectionMatrixCenterShift


class RenderCamera:
    """Minimal camera for rendering an arbitrary pose with given pinhole intrinsics."""

    def __init__(self, c2w, fx, fy, cx, cy, width, height, znear=0.01, zfar=100.0):
        self.c2w = c2w
        self.image_width, self.image_height = width, height
        self.FoVx = 2 * math.atan(width / (2 * fx))
        self.FoVy = 2 * math.atan(height / (2 * fy))
        w2c = np.linalg.inv(c2w).astype(np.float32)
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrixCenterShift(
            znear, zfar, cx, cy, fx, fy, width, height).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def interpolate_waypoints_spline(wpRs, wpts, steps=25, k=None, s=0):
    wpRs = np.array(wpRs)
    wpts = np.array(wpts)
    wpCs = cameracenter_from_translation(wpRs, wpts)

    if k is None:
        k = 2 if wpCs.shape[0] <= 3 else 3
    tck, u = scipy.interpolate.splprep(wpCs.T, k=k, s=s)
    Cs = scipy.interpolate.splev(np.linspace(0, 1, num=steps), tck)
    Cs = np.array(Cs).T

    # re-parameterize by arc length
    dists = [0] + [v for v in np.linalg.norm(Cs[1:] - Cs[:-1], axis=1)]
    dists = np.cumsum(dists)
    dists_inv = np.interp(
        np.linspace(dists[0], dists[-1], len(dists)),
        dists,
        np.linspace(0, 1, len(dists)),
    )
    Cs = scipy.interpolate.splev(dists_inv, tck)
    Cs = np.array(Cs).T

    rotations = scipy.spatial.transform.Rotation.from_matrix(wpRs)
    rot_spline = scipy.spatial.transform.RotationSpline(u, rotations)
    Rs = rot_spline(dists_inv).as_matrix()

    ts = translation_from_cameracenter(Rs, np.vstack(Cs))
    return Rs, ts


def look_at(camera_position, target, up):
    def normalize(v):
        return v / np.linalg.norm(v)
    R = np.eye(4)
    R[2, :3] = normalize(target - camera_position)
    R[0, :3] = normalize(np.cross(R[2, :3], up))
    R[1, :3] = -normalize(np.cross(R[0, :3], R[2, :3]))
    T = np.eye(4)
    T[:3, 3] = -camera_position
    return np.linalg.inv(R.dot(T))


def fit_sphere_to_points(points):
    def residuals(params):
        return np.linalg.norm(points - params[:3], axis=1) - params[3]
    center_init = np.mean(points, axis=0)
    radius_init = np.mean(np.linalg.norm(points - center_init, axis=1))
    result = least_squares(residuals, np.hstack([center_init, radius_init]))
    return result.x[:3], result.x[3]


def sample_on_sphere_surface_within_range(num_samples, original_points, rng):
    center, radius = fit_sphere_to_points(original_points)
    p = original_points - center
    theta = np.arccos(p[:, 2] / np.linalg.norm(p, axis=1))
    phi = np.arctan2(p[:, 1], p[:, 0])
    # central 80% of the angular range covered by the input cameras
    t_c, t_r = (theta.max() + theta.min()) / 2, (theta.max() - theta.min()) * 0.8
    p_c, p_r = (phi.max() + phi.min()) / 2, (phi.max() - phi.min()) * 0.8
    theta = rng.uniform(t_c - t_r / 2, t_c + t_r / 2, num_samples)
    phi = rng.uniform(p_c - p_r / 2, p_c + p_r / 2, num_samples)
    xyz = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    return radius * xyz + center


def spline_poses(train_cams, num):
    Rs, ts = interpolate_waypoints_spline(np.array([c.R for c in train_cams]),
                                          np.array([c.T for c in train_cams]), num)
    return [np.linalg.inv(getWorld2View(Rs[i], ts[i])) for i in range(len(Rs))]


def perturb_poses(train_cams, num, rng, delta=0.02):
    centers = np.array([c.camera_center.cpu().numpy() for c in train_cams])
    up = -np.mean([c.R[..., 1] for c in train_cams], axis=0)
    n_copies = max(num // len(centers), 1)
    positions = np.tile(centers, (n_copies, 1)) + rng.uniform(-delta, delta, (n_copies * len(centers), 3))
    return [look_at(p, np.zeros(3), up) for p in positions]


def sphere_poses(train_cams, num, rng):
    centers = np.array([c.camera_center.cpu().numpy() for c in train_cams])
    up = -np.mean([c.R[..., 1] for c in train_cams], axis=0)
    positions = sample_on_sphere_surface_within_range(num, centers, rng)
    return [look_at(p, np.zeros(3), up) for p in positions]


def generate_novel_cameras(train_cams, method='spline', num=10, seed=0):
    """Expanded cameras sharing the intrinsics of the first training camera."""
    rng = np.random.default_rng(seed)
    if method == 'spline':
        c2ws = spline_poses(train_cams, num)
    elif method == 'perturb':
        c2ws = perturb_poses(train_cams, num, rng)
    elif method == 'sphere':
        c2ws = sphere_poses(train_cams, num, rng)
    else:
        raise ValueError(f'unknown pose expansion method {method}')
    ref = train_cams[0]
    K = ref.intrinsics.cpu().numpy()
    cams = [RenderCamera(c2w.astype(np.float32), K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                         ref.image_width, ref.image_height) for c2w in c2ws]
    return cams, np.stack(c2ws).astype(np.float32)
