import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

def depth_to_point_cloud(depth_image, camera_intrinsics, camera_pose):
    height, width = depth_image.shape
    fx, fy, cx, cy = camera_intrinsics

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    X = (x - cx) * depth_image / fx
    Y = (y - cy) * depth_image / fy
    Z = depth_image

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    _, translation, rotation = camera_pose
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    R, t = rotation_matrix, translation
    points = (R @ points.T + t.reshape(3, 1)).T

    return points


def point_cloud_to_voxel_grid(points, voxel_size, origin):
    voxel_indices = np.floor((points - origin) / voxel_size).astype(int)
    return voxel_indices


def compute_sdf(voxel_indices, point_cloud, voxel_size, origin):
    tree = cKDTree(point_cloud)
    distances, _ = tree.query(voxel_indices * voxel_size + origin)
    return distances


def sdf_1d_to_3d(sdf, voxel_indices, grid_shape):
    sdf_volume = np.zeros(grid_shape)
    for i, idx in enumerate(voxel_indices):
        sdf_volume[tuple(idx)] = sdf[i]
    return sdf_volume


def initialize_sdf_volumes(grid_shape):
    sdf_volume = np.zeros(grid_shape)
    return sdf_volume


def update_sdf_volumes(sdf_volume, sdf, voxel_indices):
    for i, idx in enumerate(voxel_indices):
        sdf_volume[tuple(idx)] += sdf[i]
    return sdf_volume


def compute_global_sdf(sdf_volume):
    global_sdf = sdf_volume
    return global_sdf
