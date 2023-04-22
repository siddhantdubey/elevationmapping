import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def visualize_pointcloud(point_cloud, file="pointcloud.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(file)
    plt.close()


def save_mesh_as_obj(vertices, faces, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f"Saved mesh to {filename}")


def plot_sdf(sdf, voxel_size, file="sdf.png", level=None):
    if level is None:
        level = np.min(sdf)
    verts, faces, _, _ = measure.marching_cubes(sdf, level=level)
    # Scale the vertices by the voxel size to get the correct coordinates
    verts *= voxel_size
    # flip so the back wall is not at the top
    save_mesh_as_obj(verts, faces, file.replace(".png", ".obj"))
    verts[:, [1, 2]] = verts[:, [2, 1]]
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the isosurface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis', lw=0)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(30, -45)
    plt.savefig(file)
    plt.close()


def plot_elevation_map(elevation_map, x_values, y_values, file="elevation_map.png"):
    X, Y = np.meshgrid(x_values, y_values)

    # Create a copy of the elevation_map and swap Y and Z coordinates
    elevation_map_swapped = elevation_map.copy()
    elevation_map_swapped[:, [1, 2]] = elevation_map_swapped[:, [2, 1]]

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, elevation_map_swapped, cmap='terrain')
    ax.set_title('Elevation Map')
    plt.savefig(file)

