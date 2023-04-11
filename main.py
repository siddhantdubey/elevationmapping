import numpy as np
import cv2

from psdf.data_loader import load_ground_truth_poses, load_images, preprocess_images
from psdf.sdf import depth_to_point_cloud, compute_sdf
from psdf.visualizers import visualize_pointcloud

image_dir = '/mnt/d/Documents/Projects/Robotics/test_data/'
ground_truth_file = '/mnt/d/Documents/Projects/Robotics/test_data/livingRoom2.gt.freiburg'

poses = load_ground_truth_poses(ground_truth_file)
camera_intrinsics = np.array([520.9, 521.0, 325.1, 249.7])


for frame_index, _, _ in poses:
    rgb_image = load_images(image_dir, frame_index, image_type='rgb')
    depth_image = load_images(image_dir, frame_index, image_type='depth')

    rgb_image, depth_image = preprocess_images(rgb_image, depth_image)
    #save rgb_image and depth_image to /mnt/d/Documents/Projects/Robotics/
    cv2.imwrite(f'/mnt/d/Documents/Projects/Robotics/rgb_{frame_index}.png', rgb_image)
    cv2.imwrite(f'/mnt/d/Documents/Projects/Robotics/depth_{frame_index}.png', depth_image)
    print(poses[frame_index])
    pc = depth_to_point_cloud(depth_image, camera_intrinsics, poses[frame_index])
    print(pc)
    visualize_pointcloud(pc, file=f'/mnt/d/Documents/Projects/Robotics/pc_{frame_index}.png')
    break

