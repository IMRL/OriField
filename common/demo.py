import copy
import open3d as o3d
import numpy as np

from .coordinate import Convertor
from .visulization import Visulizater


def merge_meshes(meshes):
    vertices = []
    triangles = []
    colors = []
    vertex_count = 0

    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices))
        triangles.append(np.asarray(mesh.triangles) + vertex_count)
        colors.append(np.asarray(mesh.vertex_colors))
        vertex_count += len(mesh.vertices)

    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(vertices))
    merged_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(triangles))
    merged_mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors))
    return merged_mesh


def create_main_rectangle(extents):
    # Create a single rectangle mesh
    mesh = o3d.geometry.TriangleMesh.create_box(width=extents[0], height=extents[1], depth=0.001)
    # Default color (will be overwritten)
    mesh.paint_uniform_color([0.1, 0.1, 0.1])
    return mesh


def create_mesh_by_center(positions, colors, extend):
    # Main mesh template
    template_mesh = create_main_rectangle(extend)

    # Visualizing all rectangles with different colors
    meshes = []
    for pos, color in zip(positions, colors):
        # Create a copy of the template mesh for each position
        mesh = copy.deepcopy(template_mesh)
        mesh.translate(pos, relative=False)
        # Apply color
        mesh.vertex_colors = o3d.utility.Vector3dVector([color] * len(mesh.vertices))
        # Apply translation
        meshes.append(mesh)
    return merge_meshes(meshes)


def create_voxel_grid(positions, colors, res):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(positions)
    pc.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=res)
    return voxel_grid


def rotation_matrix_from_vectors(a, b):
    """
    Calculate the rotation matrix that rotates vector a to vector b

    Args:
    - a (np.array): Source unit vector.
    - b (np.array): Destination unit vector.

    Returns:
    - np.array: Rotation matrix that aligns a with b.
    """
    # Ensure a and b are unit vectors
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # Axis of rotation (cross product of vectors)
    v = np.cross(a, b)
    c = np.dot(a, b)  # cosine of the angle

    # If vectors are the same
    if np.allclose(v, [0, 0, 0], atol=1e-7):
        return np.eye(3)

    # Compute the skew-symmetric cross-product matrix of v
    vx = np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])

    # Compute the rotation matrix using Rodrigues' formula
    I = np.eye(3)
    s = np.linalg.norm(v)  # sine of the angle
    R = I + vx + np.dot(vx, vx) * ((1 - c) / s**2)

    return R


def create_cylinder_between_points(p1, p2, radius, color):
    """
    Create a cylinder between two points.

    Args:
    - p1 (np.array): The starting point of the cylinder.
    - p2 (np.array): The ending point of the cylinder.
    - radius (float): Radius of the cylinder.
    - color (list): Color of the cylinder as [r, g, b].

    Returns:
    - o3d.geometry.TriangleMesh: A colored cylinder object.
    """
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0:
        return None

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.paint_uniform_color(color)

    # Align cylinder with line (p1, p2)
    axis = np.array([0, 0, 1])  # Z-axis, which is the default axis of the cylinder
    direction = vec / length
    R = rotation_matrix_from_vectors(axis, direction)
    cylinder.rotate(R, center=np.array([0,0,0]))

    # Translate the cylinder to the correct location in space
    cylinder.translate((p2 + p1)/2)

    return cylinder


def create_sphere_at_point(point, radius, color):
    """ Create a sphere at the given point """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.paint_uniform_color(color)
    sphere.translate(point)
    return sphere


def create_trajectory(points, egde_radius, node_radius, egde_color=[0, 1, 0], node_color=[1, 0, 0]):
    """
    Visualize a trajectory as a series of cylinders between consecutive points.

    Args:
    - points (np.array): An array of points (N, 3) defining the trajectory.
    - radius (float): Radius of the cylinders used to represent the trajectory.
    - color (list): Color of the cylinders as [r, g, b].
    """
    meshes = []
    for i in range(len(points) - 1):
        cylinder = create_cylinder_between_points(points[i], points[i + 1], egde_radius, egde_color)
        if cylinder:
            meshes.append(cylinder)
    for point in points:
        sphere = create_sphere_at_point(point, node_radius, node_color)
        meshes.append(sphere)

    return merge_meshes(meshes)


if __name__ == "__main__":
    # # Example positions and colors
    # extend = np.array([0.16, 0.16])
    # positions = np.random.rand(100, 3)*2
    # left_bottom_pixels = np.stack([
    #     positions[:, 0] / extend[0],
    #     positions[:, 1] / extend[1],
    #     np.zeros_like(positions[:, 2]),
    #     ], axis=-1).astype(np.int32)
    # left_bottom_positions = np.stack([
    #     left_bottom_pixels[:, 0] * extend[0],
    #     left_bottom_pixels[:, 1] * extend[1],
    #     left_bottom_pixels[:, 2],
    #     ], axis=-1)
    # colors = np.random.rand(100, 3)     # Corresponding colors

    # meshes = create_mesh_by_center(left_bottom_positions, colors, extend)
    # # Visualize the combined mesh
    # o3d.visualization.draw_geometries([meshes])

    # res = 0.16
    # xmin, ymin, zmin, xmax, ymax, zmax = [-32.0, -32.0, -2.5, 32.0, 32.0, 3.5]
    # lidar_range = {'Left': xmin, 'Right': xmax, 'Front': ymax, 'Back': ymin, 'Bottom': zmin, 'Top': zmax}
    # convertor = Convertor(img_h=400, img_w=400, res=res, lidar_range=lidar_range)

    # binary = np.random.rand(400, 400) > 0.5
    # tangent = (np.random.rand(400, 400, 2) - 0.5) * 2
    # matrix_points = np.transpose(np.nonzero(binary))
    # colors = Visulizater(None).tangent_vis(tangent, pltcm='hsv')[matrix_points.T[0], matrix_points.T[1], :-1]
    # left_top_positions = np.concatenate([convertor.matrix2car(matrix_points), np.full((matrix_points.shape[0], 1), 0.)], axis=-1)

    # meshes = create_mesh_by_center(left_top_positions, colors, np.array([res, res]))
    # voxels = create_voxel_grid(left_top_positions, colors, res)

    # # Visualize the combined mesh
    # o3d.visualization.draw_geometries([meshes, voxels])

    lines = create_trajectory(np.array([[0,0,0],[0,5,0],[1,0,0]]), .1, color=np.array([1,0,0]))
    # Visualize the combined mesh
    o3d.visualization.draw_geometries([lines])