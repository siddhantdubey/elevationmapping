import numpy as np
from psdf.data_loader import load_ground_truth_poses, load_images, preprocess_images
from psdf.sdf import *
from psdf.visualizers import plot_sdf, plot_elevation_map
from tqdm import tqdm


def main():
    image_dir = '/mnt/d/Documents/Projects/Robotics/test_data/'
    ground_truth_file = '/mnt/d/Documents/Projects/Robotics/test_data/livingRoom2.gt.freiburg'

    poses = load_ground_truth_poses(ground_truth_file)
    camera_intrinsics = np.array([520.9, 521.0, 325.1, 249.7])

    grid_origin = np.array([0, 0, 0])
    voxel_size = 100

    sdfs, point_clouds, min_voxel_indices, max_voxel_indices = compute_sdfs(poses, image_dir, camera_intrinsics, grid_origin, voxel_size)

    grid_shape = tuple((max_voxel_indices - min_voxel_indices + 300).astype(int))
    offset = np.abs(min_voxel_indices).astype(int) + 1
    sdf_volume = initialize_sdf_volumes(grid_shape)
    sdf_volume = update_all_sdf_volumes(sdf_volume, sdfs, offset)
    tsdf = compute_tsdf(sdf_volume, voxel_size)
    esdf = compute_esdf(tsdf, voxel_size)
    isovalue_tsdf = find_isovalue(tsdf)
    isovalue_sdf = find_isovalue(sdf_volume)
    plot_sdf(tsdf, voxel_size, file="tsdf.png", level=isovalue_tsdf)
    plot_sdf(sdf_volume, voxel_size, file="sdf_final.png", level=isovalue_sdf)
    plot_sdf(esdf, voxel_size, file="esdf.png", level=0)
    # elevation_map, x_values, y_values = esdf_to_elevation_map(esdf, voxel_size, grid_origin)
    # print(elevation_map)
    # plot_elevation_map(elevation_map, x_values, y_values, file="elevation_map.png")


def compute_sdfs(poses, image_dir, camera_intrinsics, grid_origin, voxel_size):
    sdfs = []
    point_clouds = []
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
        if count == 1:
            break

    return sdfs, point_clouds, min_voxel_indices, max_voxel_indices


def update_all_sdf_volumes(sdf_volume, sdfs, offset):
    for sdf, voxel_grid in tqdm(sdfs, desc='Updating SDFs...'):
        sdf_volume = update_sdf_volumes(sdf_volume, sdf, voxel_grid + offset)

    return sdf_volume


if __name__ == "__main__":
    main()
