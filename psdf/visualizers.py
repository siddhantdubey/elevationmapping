import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    save_mesh_as_obj(verts, faces, file.replace(".png", ".obj"))
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

# ignore this for now
def plot_elevation_map(elevation_map, x_values, y_values, file="map.png"):
    X, Y = np.meshgrid(x_values, y_values)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, elevation_map, cmap='terrain')
    ax.set_title('Elevation Map')
    plt.savefig(file)

def plot_esdf_mesh(esdf, level=None):
    if level is None:
        level = np.min(esdf)
    # Extract the isosurface using the Marching Cubes algorithm
    verts, faces, _, _ = measure.marching_cubes(esdf, level=level)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Add the mesh to the plot
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    # Set the plot limits
    ax.set_xlim(0, esdf.shape[0])
    ax.set_ylim(0, esdf.shape[1])
    ax.set_zlim(0, esdf.shape[2])

    # Set the axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Display the plot
    plt.savefig("esdf.png")
