import os
import numpy as np
import cv2


def load_ground_truth_poses(ground_truth_file):
    poses = []
    with open(ground_truth_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            frame_index = int(data[0])
            tx, ty, tz = float(data[1]), float(data[2]), float(data[3])
            qx, qy = float(data[4]), float(data[5])
            qz, qw = float(data[6]), float(data[7])
            poses.append((frame_index, np.array([tx, ty, tz]),
                          np.array([qx, qy, qz, qw])))
    return poses


def load_images(image_dir, frame_index, image_type='rgb'):
    if image_type == 'rgb':
        file_name = f'rgb/{frame_index}.png'
    elif image_type == 'depth':
        file_name = f'depth/{frame_index}.png'
    else:
        raise ValueError('Invalid image type.')

    file_path = os.path.join(image_dir, file_name)
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


def preprocess_images(rgb_image, depth_image):
    return rgb_image, depth_image
