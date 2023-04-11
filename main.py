import cv2

from psdf.data_loader import load_ground_truth_poses, load_images, preprocess_images

image_dir = '/mnt/d/Documents/Projects/Robotics/test_data/'
ground_truth_file = '/mnt/d/Documents/Projects/Robotics/test_data/livingRoom2.gt.freiburg'

poses = load_ground_truth_poses(ground_truth_file)

for frame_index, translation, quaternion in poses:
    rgb_image = load_images(image_dir, frame_index, image_type='rgb')
    depth_image = load_images(image_dir, frame_index, image_type='depth')

    rgb_image, depth_image = preprocess_images(rgb_image, depth_image)

    #save rgb_image and depth_image to /mnt/d/Documents/Projects/Robotics/
    cv2.imwrite(f'/mnt/d/Documents/Projects/Robotics/rgb_{frame_index}.png', rgb_image)
    cv2.imwrite(f'/mnt/d/Documents/Projects/Robotics/depth_{frame_index}.png', depth_image)
    break

