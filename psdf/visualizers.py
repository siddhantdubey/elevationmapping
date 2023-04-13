import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def visualize_pointcloud(point_cloud, file="pointcloud.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(file)
    plt.close()



def plot_sdf(sdf, voxel_size, file="sdf.png"):
    verts, faces, _, _ = measure.marching_cubes(sdf, 0)

    # Scale the vertices by the voxel size to get the correct coordinates
    verts *= voxel_size

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the isosurface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis', lw=0)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(file)
    plt.close()


# ignore this for now
def visualize_elevation_map(sdf, file="map.png"):
    # 2d scatter plot
    img = plt.imshow(sdf, cmap='viridis', origin='lower')
    plt.colorbar(img, label='Height')
    plt.title('Elevation Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(file)
