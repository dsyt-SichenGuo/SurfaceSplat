"""Mesh cleaning + depth-based point sampling (SurfaceSplat, Sec. 3.2).

1. keep the largest connected component of the coarse mesh (removes floaters);
2. render the cleaned mesh into the sparse input views to get multi-view consistent depth maps;
3. randomly sample pixels with valid depth and back-project them into colored 3D points.

The point cloud is written in the normalized scene frame (the frame of the SDF stages and of
the 3DGS stage) and is used to initialize 3DGS.

    python tools/sample_points.py --mesh <out>/scan24/coarse/meshes/scan24_mesh_normalized.ply \
        --datadir <data>/dtu_scan24 --cameras_name cameras_sphere.npz --num_img 5 \
        --out <out>/scan24/points.ply
"""
import argparse
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import imageio
import numpy as np
import open3d as o3d
import pyrender
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'voxurf'))
from lib.load_data import select_views, load_cameras, list_images  # noqa: E402


def keep_largest_component(mesh):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    largest_cluster_idx = np.asarray(cluster_n_triangles).argmax()
    mesh.remove_triangles_by_mask(triangle_clusters != largest_cluster_idx)
    mesh.remove_unreferenced_vertices()
    return mesh


class DepthRenderer:
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(self, K, c2w, mesh):
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=K[0, 2], cy=K[1, 2], fx=K[0, 0], fy=K[1, 1])
        # OpenCV -> OpenGL camera axes
        flip = np.diag([1., -1., -1., 1.])
        self.scene.add(cam, pose=c2w @ flip)
        _, depth = self.renderer.render(self.scene)
        return depth

    def delete(self):
        self.renderer.delete()


def backproject(depth, image, K, c2w, num_samples, rng):
    """Sample `num_samples` pixels with valid depth and lift them to colored world points."""
    valid = np.argwhere(depth > 0)
    if len(valid) < num_samples:
        raise ValueError(f'only {len(valid)} pixels with valid depth, fewer than {num_samples}')
    sel = valid[rng.choice(len(valid), num_samples, replace=False)]
    v, u = sel[:, 0], sel[:, 1]
    d = depth[v, u]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)], -1)
    world = (c2w @ cam.T).T[:, :3]
    return world, image[v, u, :3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mesh', required=True, help='coarse mesh in the normalized frame')
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--cameras_name', required=True, help='cameras_sphere.npz (DTU) / cameras.npz (MobileBrick)')
    parser.add_argument('--num_img', type=int, required=True, help='number of sparse input views')
    parser.add_argument('--points_per_view', type=int, default=10000)
    parser.add_argument('--out', required=True, help='output point cloud (.ply)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    mesh = keep_largest_component(o3d.io.read_triangle_mesh(args.mesh))
    clean_path = os.path.splitext(args.out)[0] + '_clean_mesh.ply'
    o3d.io.write_triangle_mesh(clean_path, mesh)
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)

    rgb_paths = list_images(args.datadir)
    all_K, all_poses, _ = load_cameras(args.datadir, args.cameras_name, len(rgb_paths))
    ids = select_views(len(rgb_paths), args.num_img)
    images = [(imageio.imread(rgb_paths[i]) / 255.).astype(np.float32) for i in ids]
    H, W = images[0].shape[:2]

    renderer = DepthRenderer(H, W)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    points, colors = [], []
    for image, i in tqdm(list(zip(images, ids)), desc='render depth + back-project'):
        depth = renderer(all_K[i][:3, :3], all_poses[i], render_mesh)
        p, c = backproject(depth, image, all_K[i][:3, :3], all_poses[i], args.points_per_view, rng)
        points.append(p)
        colors.append(c)
    renderer.delete()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(points).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors).astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_point_cloud(args.out, pcd)
    print(f'saved {len(pcd.points)} points from views {ids.tolist()} to {args.out}')


if __name__ == '__main__':
    main()
