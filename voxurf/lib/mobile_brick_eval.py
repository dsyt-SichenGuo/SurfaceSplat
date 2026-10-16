"""MobileBrick mesh evaluation (follows the official MobileBrick protocol).

The predicted mesh (in world space) is aligned to the GT mesh with ICP, culled by
the GT visibility volume, and both meshes are Poisson-disk sampled. Precision /
recall / F1 are reported at 2.5mm and 5mm.
"""
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import os


def np2pc(points):
    pts_o3d = o3d.geometry.PointCloud()
    pts_o3d.points = o3d.utility.Vector3dVector(points)
    return pts_o3d


def visibility_test(volume, min_pts, resolution, voxel_size, mesh, device):
    """Remove the vertices that fall outside the GT visibility volume."""
    points = np.asarray(mesh.vertices)
    volume = torch.from_numpy(volume).float().to(device)
    voxels = (points - min_pts) / voxel_size
    voxels = voxels / (resolution - 1) * 2 - 1
    voxels = torch.from_numpy(voxels)[..., [2, 1, 0]].float().to(device)
    mask = F.grid_sample(volume.unsqueeze(0).unsqueeze(0),            # [1,1,H,W,D]
                         voxels.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # [1,1,1,N,3]
                         mode="nearest", padding_mode="zeros", align_corners=True)
    mask = mask[0, 0, 0, 0].cpu().numpy() > 0
    mesh.remove_vertices_by_mask(mask == False)
    mesh.compute_vertex_normals()
    return mesh


def evaluate(pred_points, gt_points, threshold, verbose=False):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gt_points)
    distances, _ = nbrs.kneighbors(pred_points)
    pred_gt_dist = np.mean(distances)
    precision = np.sum(distances < threshold) / len(distances)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pred_points)
    distances, _ = nbrs.kneighbors(gt_points)
    gt_pred_dist = np.mean(distances)
    recall = np.sum(distances < threshold) / len(distances)

    F1 = 2 * precision * recall / (precision + recall)
    chamfer = pred_gt_dist + gt_pred_dist
    if verbose:
        print("precision @ {}: {:.6f}".format(threshold, precision))
        print("recall @ {}: {:.6f}".format(threshold, recall))
        print("F1: {:.6f}".format(F1))
        print("Chamfer: {:.6f}".format(chamfer))
    return dict(pred_gt=pred_gt_dist, gt_pred=gt_pred_dist, chamfer=chamfer,
                precision=precision, recall=recall, F1=F1)


def eval(mesh_path, gt_dir, n_samples=100000):
    """Evaluate a world-space mesh against <gt_dir>/mesh/gt_mesh.ply.

    Returns a dict with chamfer and precision/recall/F1 at 2.5mm and 5mm.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    visibility_mask = np.load(os.path.join(gt_dir, "visibility_mask.npy"), allow_pickle=True).item()
    resolution = visibility_mask['resolutions']
    volume = visibility_mask['mask'].reshape(resolution)
    voxel_size = visibility_mask['voxel_size']
    min_pts = visibility_mask['min_pts']
    gt_mesh = o3d.io.read_triangle_mesh(os.path.join(gt_dir, "mesh", "gt_mesh.ply"))
    gt_points = np.asarray(gt_mesh.sample_points_poisson_disk(n_samples).points)

    pred_mesh = o3d.io.read_triangle_mesh(mesh_path)

    # rigid alignment, only applied when it is reliable
    reg = o3d.pipelines.registration.registration_icp(
        np2pc(gt_mesh.vertices), np2pc(pred_mesh.vertices), 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
    if reg.fitness > 0.99:
        pred_mesh.transform(np.linalg.inv(reg.transformation))
    pred_mesh = visibility_test(volume, min_pts, resolution, voxel_size, pred_mesh, device)

    if len(np.asarray(pred_mesh.triangles)) > 0:
        pred_points = np.asarray(pred_mesh.sample_points_poisson_disk(n_samples).points)
    else:
        pred_points = np.random.permutation(np.asarray(pred_mesh.vertices))[:n_samples]

    out_25 = evaluate(pred_points, gt_points, threshold=0.0025, verbose=True)
    out_5 = evaluate(pred_points, gt_points, threshold=0.005, verbose=True)
    return {
        'chamfer': float(out_25['chamfer']),
        'precision_2.5mm': float(out_25['precision']),
        'recall_2.5mm': float(out_25['recall']),
        'F1_2.5mm': float(out_25['F1']),
        'precision_5mm': float(out_5['precision']),
        'recall_5mm': float(out_5['recall']),
        'F1_5mm': float(out_5['F1']),
    }
