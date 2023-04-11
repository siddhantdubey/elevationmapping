import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def depth_to_point_cloud(depth_image, camera_intrinsics, camera_pose):
    height, width = depth_image.shape
    fx, fy, cx, cy = camera_intrinsics

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    X = (x - cx) * depth_image / fx
    Y = (y - cy) * depth_image / fy
    Z = depth_image

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    frame_index, translation, rotation = camera_pose
    rotation_matrix = Rotation.from_quat(rotation).as_matrix() 
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    R, t = transformation_matrix[:3, :3], transformation_matrix[:3, 3]
    points = (R @ points.T + t.reshape(3, 1)).T

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)

    return point_cloud_o3d

def compute_sdf(point_cloud, grid_origin, grid_shape, voxel_size):
    sdf = np.ones(grid_shape, dtype=np.float32) * np.inf
    grid_coords = np.floor((point_cloud - grid_origin) / voxel_size).astype(np.int32)

    for point, coord in zip(point_cloud, grid_coords):
        if np.all(coord >= 0) and np.all(coord < grid_shape):
            distance = np.linalg.norm(point - (grid_origin + coord * voxel_size))
            sdf[tuple(coord)] = min(sdf[tuple(coord)], distance)

    sdf -= voxel_size / 2

    return sdf
