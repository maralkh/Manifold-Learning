import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def project_to_torus_extended(point, center, major_radius, minor_radius,
                            axis=[0, 0, 1], active_dims=[0, 1, 2]):
    """
    Projection of multidimensional points onto a torus

    Args:
        point: n-dimensional point
        center: n-dimensional (or 3-dimensional) torus center
        major_radius: Major radius
        minor_radius: Minor radius
        axis: Torus axis (3D)
        active_dims: List of indices of active dimensions (e.g., [0,1,2] for x,y,z)

    Returns:
        Projected point with the same number of input dimensions
    """
    point = np.array(point, dtype=float)
    center = np.array(center, dtype=float)

    # If center has fewer dimensions than point, pad with zeros
    if len(center) < len(point):
        center_extended = np.zeros(len(point))
        center_extended[:len(center)] = center
        center = center_extended
    elif len(center) > len(point):
        center = center[:len(point)]

    # Extract active dimensions (first three or specified dimensions)
    if len(point) < 3:
        raise ValueError(f"Point must have at least 3 dimensions, {len(point)} dimensions provided")

    # Select active dimensions
    if max(active_dims) >= len(point):
        active_dims = list(range(min(3, len(point))))

    # Extract the 3D subspace
    point_3d = point[active_dims[:3]]
    center_3d = center[active_dims[:3]]

    # Projection in the 3D space
    proj_3d = project_to_torus_3d(point_3d, center_3d, major_radius, minor_radius, axis)

    # Construct the output point with original dimensions
    result = point.copy()
    result[active_dims[:3]] = proj_3d

    return result

def project_to_torus_3d(point, center, major_radius, minor_radius, axis=[0, 0, 1]):
    """
    Original version for 3D
    """
    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # Translate to origin
    p_translated = p - c

    # Coordinate transformation for arbitrary axis
    if not np.allclose(axis, [0, 0, 1]):
        z_axis = np.array([0, 0, 1])
        if np.allclose(axis, -z_axis):
            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            v = np.cross(axis, z_axis)
            s = np.linalg.norm(v)
            c_rot = np.dot(axis, z_axis)

            if s != 0:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_rot) / (s * s))
            else:
                rotation_matrix = np.eye(3)

        p_rotated = np.dot(rotation_matrix.T, p_translated)
    else:
        p_rotated = p_translated

    x, y, z = p_rotated

    # Distance from z-axis in xy-plane
    rho = np.sqrt(x**2 + y**2)

    if rho == 0:
        rho = major_radius
        x = major_radius
        y = 0

    # Angle in xy-plane
    phi = np.arctan2(y, x)

    # Point on the central circle
    center_circle_x = major_radius * np.cos(phi)
    center_circle_y = major_radius * np.sin(phi)
    center_circle_z = 0

    # Vector to the center of the minor circle
    dx = x - center_circle_x
    dy = y - center_circle_y
    dz = z - center_circle_z

    minor_distance = np.sqrt(dx**2 + dy**2 + dz**2)

    if minor_distance == 0:
        proj_x = center_circle_x
        proj_y = center_circle_y
        proj_z = minor_radius
    else:
        unit_minor = np.array([dx, dy, dz]) / minor_distance
        proj_x = center_circle_x + minor_radius * unit_minor[0]
        proj_y = center_circle_y + minor_radius * unit_minor[1]
        proj_z = center_circle_z + minor_radius * unit_minor[2]

    proj_rotated = np.array([proj_x, proj_y, proj_z])

    # Return to original coordinates
    if not np.allclose(axis, [0, 0, 1]):
        proj_original = np.dot(rotation_matrix, proj_rotated)
    else:
        proj_original = proj_rotated

    return c + proj_original

def project_to_torus_pca(point, center, major_radius, minor_radius, data_points=None):
    """
    Projection using PCA to find the best 3D subspace

    Args:
        point: multidimensional point
        center: multidimensional center
        major_radius, minor_radius: Torus radii
        data_points: Set of points for PCA calculation (optional)

    Returns:
        Projected point
    """
    point = np.array(point, dtype=float)
    center = np.array(center, dtype=float)

    if len(point) <= 3:
        return project_to_torus_3d(point, center, major_radius, minor_radius)

    # If no additional data, use the first dimensions
    if data_points is None:
        return project_to_torus_extended(point, center, major_radius, minor_radius)

    # PCA to find the most important dimensions
    from sklearn.decomposition import PCA

    data = np.array(data_points)
    pca = PCA(n_components=3)
    pca.fit(data)

    # Projection into PCA 3D space
    point_pca = pca.transform(point.reshape(1, -1))[0]
    center_pca = pca.transform(center.reshape(1, -1))[0]

    # Projection onto torus in PCA space
    proj_pca = project_to_torus_3d(point_pca, center_pca, major_radius, minor_radius)

    # Return to original space
    proj_original = pca.inverse_transform(proj_pca.reshape(1, -1))[0]

    return proj_original

# Various tests
def test_high_dimensional():
    """Test projection in high dimensions"""

    print("=== Multidimensional Projection Test ===\n")

    # Test 4D
    point_4d = [3, 2, 1, 4]
    center_4d = [0, 0, 0, 0]
    major_radius, minor_radius = 2, 0.5

    # Method 1: Using the first three dimensions
    proj_first3 = project_to_torus_extended(point_4d, center_4d, major_radius, minor_radius)
    print(f"4D Point: {point_4d}")
    print(f"Projection (first three dimensions): {proj_first3}")
    print(f"Distance: {np.linalg.norm(np.array(point_4d) - proj_first3):.3f}\n")

    # Method 2: Selecting custom dimensions
    proj_custom = project_to_torus_extended(point_4d, center_4d, major_radius, minor_radius,
                                          active_dims=[0, 2, 3])
    print(f"Projection (dimensions 0,2,3): {proj_custom}")
    print(f"Distance: {np.linalg.norm(np.array(point_4d) - proj_custom):.3f}\n")

    # Test 6D
    point_6d = [5, 3, 2, 1, 4, 2]
    center_6d = [0, 0, 0, 0, 0, 0]

    proj_6d = project_to_torus_extended(point_6d, center_6d, major_radius, minor_radius)
    print(f"6D Point: {point_6d}")
    print(f"6D Projection: {proj_6d}")
    print(f"Distance: {np.linalg.norm(np.array(point_6d) - proj_6d):.3f}\n")

    # Compare methods
    print("=== Comparison of Different Methods ===")
    test_point = [4, 3, 2, 5, 1]
    center = [0, 0, 0, 0, 0]

    methods = {
        "First three dimensions": project_to_torus_extended(test_point, center, 2, 0.8, active_dims=[0,1,2]),
        "Dimensions 0,1,4": project_to_torus_extended(test_point, center, 2, 0.8, active_dims=[0,1,4]),
        "Dimensions 1,2,3": project_to_torus_extended(test_point, center, 2, 0.8, active_dims=[1,2,3])
    }

    for method_name, projection in methods.items():
        distance = np.linalg.norm(np.array(test_point) - projection)
        print(f"{method_name}: Distance = {distance:.3f}")

def plot_dimension_comparison():
    """Visualize comparison of different dimensions"""

    # Test data
    np.random.seed(42)
    n_points = 20

    # 4D points
    points_4d = np.random.randn(n_points, 4) * 3
    center_4d = [0, 0, 0, 0]
    major_radius, minor_radius = 2.5, 0.8

    # Projections with different methods
    proj_xy = []  # Dimensions 0,1,2
    proj_xz = []  # Dimensions 0,2,3
    proj_yz = []  # Dimensions 1,2,3

    for point in points_4d:
        proj_xy.append(project_to_torus_extended(point, center_4d, major_radius, minor_radius,
                                               active_dims=[0,1,2]))
        proj_xz.append(project_to_torus_extended(point, center_4d, major_radius, minor_radius,
                                               active_dims=[0,2,3]))
        proj_yz.append(project_to_torus_extended(point, center_4d, major_radius, minor_radius,
                                               active_dims=[1,2,3]))

    # Calculate distances
    distances_xy = [np.linalg.norm(np.array(p1) - np.array(p2))
                   for p1, p2 in zip(points_4d, proj_xy)]
    distances_xz = [np.linalg.norm(np.array(p1) - np.array(p2))
                   for p1, p2 in zip(points_4d, proj_xz)]
    distances_yz = [np.linalg.norm(np.array(p1) - np.array(p2))
                   for p1, p2 in zip(points_4d, proj_yz)]

    # Plot comparison
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.hist(distances_xy, bins=10, alpha=0.7, label='Dimensions 0,1,2')
    plt.xlabel('Projection Distance')
    plt.ylabel('Count')
    plt.title('Distance Distribution - Dimensions 0,1,2')
    plt.grid(True, alpha=0.3)

    plt.subplot(132)
    plt.hist(distances_xz, bins=10, alpha=0.7, label='Dimensions 0,2,3', color='orange')
    plt.xlabel('Projection Distance')
    plt.ylabel('Count')
    plt.title('Distance Distribution - Dimensions 0,2,3')
    plt.grid(True, alpha=0.3)

    plt.subplot(133)
    plt.hist(distances_yz, bins=10, alpha=0.7, label='Dimensions 1,2,3', color='green')
    plt.xlabel('Projection Distance')
    plt.ylabel('Count')
    plt.title('Distance Distribution - Dimensions 1,2,3')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Comparative statistics
    print("=== Comparative Statistics ===")
    print(f"Average Distance - Dimensions 0,1,2: {np.mean(distances_xy):.3f}")
    print(f"Average Distance - Dimensions 0,2,3: {np.mean(distances_xz):.3f}")
    print(f"Average Distance - Dimensions 1,2,3: {np.mean(distances_yz):.3f}")
