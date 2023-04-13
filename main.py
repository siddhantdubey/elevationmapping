import numpy as np
from psdf.data_loader import load_ground_truth_poses, load_images, preprocess_images
from psdf.sdf import (compute_sdf, update_sdf_volumes,
                      point_cloud_to_voxel_grid, depth_to_point_cloud,
                      initialize_sdf_volumes, compute_global_sdf,
                      sdf_1d_to_3d)
from psdf.visualizers import plot_sdf
from tqdm import tqdm

image_dir = '/mnt/d/Documents/Projects/Robotics/test_data/'
ground_truth_file = '/mnt/d/Documents/Projects/Robotics/test_data/livingRoom2.gt.freiburg'

poses = load_ground_truth_poses(ground_truth_file)
camera_intrinsics = np.array([520.9, 521.0, 325.1, 249.7])

sdfs = []
point_clouds = []
grid_origin = np.array([0, 0, 0])
voxel_size = 100

min_voxel_indices = float("inf")
max_voxel_indices = float("-inf")

count = 0
for frame_index, _, _ in tqdm(poses, desc='Computing SDFs...'):
    rgb_image = load_images(image_dir, frame_index, image_type='rgb')
    depth_image = load_images(image_dir, frame_index, image_type='depth')

    rgb_image, depth_image = preprocess_images(rgb_image, depth_image)
    pc = depth_to_point_cloud(depth_image, camera_intrinsics, poses[frame_index])
    voxel_grid = point_cloud_to_voxel_grid(pc, voxel_size, grid_origin)
    sdf_grid = compute_sdf(voxel_grid, pc, voxel_size, grid_origin)
    sdfs.append([sdf_grid, voxel_grid])
    min_voxel_indices = np.minimum(min_voxel_indices, voxel_grid.min(axis=0))
    max_voxel_indices = np.maximum(max_voxel_indices, voxel_grid.max(axis=0))
    point_clouds.append(pc)
    count += 1
    if count == 100:
        break

grid_shape = tuple((max_voxel_indices - min_voxel_indices + 300).astype(int))
offset = np.abs(min_voxel_indices).astype(int) + 1 # offset to make all values positive
sdf_volume = initialize_sdf_volumes(grid_shape)

for sdf, voxel_grid in tqdm(sdfs, desc='Updating SDFs...'):
    sdf_volume = update_sdf_volumes(sdf_volume, sdf, voxel_grid + offset)

plot_sdf(sdf_volume, voxel_size, file="sdf_final.png")
