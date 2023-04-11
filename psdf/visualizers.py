import numpy as np
import open3d as o3d

def visualize_pointcloud(point_cloud, file="pointcloud.png"):
    vis = o3d.visualization.Visualizer()
    vis.add_geometry(point_cloud)
    image = vis.capture_screen_float_buffer(True)
    o3d.io.write_image(file, image)
    vis.destroy_window()
