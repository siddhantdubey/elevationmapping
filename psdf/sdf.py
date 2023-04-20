import numpy as np
import open3d as o3d
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
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    normals = np.asarray(point_cloud_o3d.normals)

    # Create a k-d tree for querying the point cloud
    tree = cKDTree(point_cloud)
    # Query the tree to find the nearest point and its index for each voxel
    distances, nearest_indices = tree.query(voxel_indices * voxel_size + origin)

    # Compute the signed distances
    vectors_to_nearest_points = (voxel_indices * voxel_size + origin) - point_cloud[nearest_indices]
    signed_distances = np.sum(vectors_to_nearest_points * normals[nearest_indices], axis=1)

    return signed_distances


def sdf_1d_to_3d(sdf, voxel_indices, grid_shape):
    sdf_volume = np.zeros(grid_shape)
    for i, idx in enumerate(voxel_indices):
        sdf_volume[tuple(idx)] = sdf[i]
    return sdf_volume


def initialize_sdf_volumes(grid_shape):
    sdf_volume = np.zeros(grid_shape)
    return sdf_volume


def compute_global_sdf(sdf_volume):
    global_sdf = sdf_volume
    return global_sdf


def compute_tsdf(sdf, truncation_limit):
    tsdf = np.where(sdf <= truncation_limit, sdf, truncation_limit)
    return tsdf


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


def find_isovalue(tsdf):
    # Compute the histogram of TSDF values
    hist, bins = np.histogram(tsdf.flatten(), bins=100, range=(-1, 1))

    # Perform binary search to find the isovalue
    lo = 0
    hi = len(bins) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if np.sum(hist[mid:]) < np.sum(hist[:mid]):
            hi = mid - 1
        else:
            lo = mid + 1

    # The isovalue is the midpoint of the bin containing the threshold
    isovalue = (bins[lo] + bins[lo + 1]) / 2

    return isovalue

def compute_esdf(tsdf, voxel_size):
    # Threshold the TSDF to obtain binary masks for occupied and free voxels
    occupied_mask = tsdf < 0
    free_mask = tsdf >= 0

    # Compute the Euclidean distance transforms for both masks
    occupied_distances = distance_transform_edt(occupied_mask, sampling=[voxel_size]*3)
    free_distances = distance_transform_edt(free_mask, sampling=[voxel_size]*3)

    # Compute the ESDF by subtracting the distance transforms
    esdf = occupied_distances - free_distances

    return esdf

def update_sdf_volumes(sdf_volume, sdf, voxel_indices):
    for i, idx in enumerate(voxel_indices):
        sdf_volume[tuple(idx)] += sdf[i]
    return sdf_volume
