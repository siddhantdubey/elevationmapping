import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

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


def compute_esdf(global_sdf_volume, voxel_size=0.05):
    # Compute the binary occupancy grid
    occupancy_grid = global_sdf_volume > 0

    # Calculate the Euclidean Distance Transform (EDT) for occupied and free space
    occupied_edt = distance_transform_edt(occupancy_grid, sampling=voxel_size)
    free_edt = distance_transform_edt(~occupancy_grid, sampling=voxel_size)

    # Subtract the free EDT from the occupied EDT to obtain the ESDF
    esdf = occupied_edt - free_edt

    return esdf

# def compute_tsdf(point_cloud, x_range, y_range, z_range, voxel_size, voxel_margin):
#     x, y, z = np.mgrid[x_range[0]:x_range[1]:voxel_size, y_range[0]:y_range[1]:voxel_size, z_range[0]:z_range[1]:voxel_size]
#     grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

#     tree = cKDTree(point_cloud)
#     distances, _ = tree.query(grid_points)

#     tsdf = 1.0 - np.minimum(distances / (voxel_size * voxel_margin), 1.0)
#     tsdf_volume = tsdf.reshape(x.shape)

#     return tsdf_volume

def sdf_to_elevation_map(global_sdf, voxel_size, origin, min_voxel_indices, index_offset):
    surface_points = np.argwhere(np.abs(global_sdf) < voxel_size)
    surface_points = surface_points - index_offset + min_voxel_indices
    surface_points = surface_points * voxel_size + origin

    # Project the surface points onto a 2D grid
    x_values, y_values = np.unique(surface_points[:, 0]), np.unique(surface_points[:, 1])
    xi = np.searchsorted(x_values, surface_points[:, 0])
    yi = np.searchsorted(y_values, surface_points[:, 1])

    elevation_map = np.zeros((y_values.size, x_values.size), dtype=np.float32)
    elevation_map[yi, xi] = surface_points[:, 2]

    return elevation_map, x_values, y_values
