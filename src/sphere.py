import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def project_to_sphere(point, center, radius):
    """
    project point to sphere
    Args:
        point: [x, y, z]
        center: [x, y, z]
        radius

    Returns:
        projected point
    """
    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)

    vec = p - c
    distance = np.linalg.norm(vec)

    if distance == 0:
        return c + np.array([radius, 0, 0])

    unit_vec = vec / distance
    projection = c + radius * unit_vec

    return projection

def project_to_ellipsoid(point, center, semi_axes):
    """
    ellisoid projection

    Args:
        point: [x, y, z]
        center: [x, y, z]
        semi_axes: [a, b, c]
    """
    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)
    a, b, c_axis = semi_axes

    vec = p - c
    x, y, z = vec

    t = 1.0
    for _ in range(10):
        denom_x = (a**2 + t)
        denom_y = (b**2 + t)
        denom_z = (c_axis**2 + t)

        fx = x**2 / denom_x**2 + y**2 / denom_y**2 + z**2 / denom_z**2 - 1

        if abs(fx) < 1e-10:
            break

        dfx = -2 * (x**2 / denom_x**3 + y**2 / denom_y**3 + z**2 / denom_z**3)

        if abs(dfx) > 1e-10:
            t = t - fx / dfx

    proj_x = (a**2 * x) / (a**2 + t)
    proj_y = (b**2 * y) / (b**2 + t)
    proj_z = (c_axis**2 * z) / (c_axis**2 + t)

    projection = c + np.array([proj_x, proj_y, proj_z])
    return projection

def project_to_cylinder(point, center, axis_direction, radius, height):
    """
    cylinder projection

    Args:
        point: [x, y, z]
        center: [x, y, z]
        axis_direction: [x, y, z]
        radius
        height
    """
    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)
    axis = np.array(axis_direction, dtype=float)
    axis = axis / np.linalg.norm(axis)

    vec = p - c

    proj_on_axis = np.dot(vec, axis)
    proj_on_axis = max(0, min(height, proj_on_axis))
    axis_point = c + proj_on_axis * axis
    perp_vec = p - axis_point
    perp_distance = np.linalg.norm(perp_vec)

    if perp_distance == 0:
        perp_vec = np.array([1, 0, 0])
        if abs(np.dot(perp_vec, axis)) > 0.9:
            perp_vec = np.array([0, 1, 0])
        perp_vec = perp_vec - np.dot(perp_vec, axis) * axis
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
    else:
        perp_vec = perp_vec / perp_distance

    projection = axis_point + radius * perp_vec

    return projection

def spherical_coordinates_from_cartesian(point, center):
    """
    cartesian to spherical
    """
    p = np.array(point) - np.array(center)
    x, y, z = p

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r > 0 else 0
    phi = np.arctan2(y, x)

    return r, theta, phi

def cartesian_from_spherical(r, theta, phi, center):
    """
    spherical to cartesian
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array(center) + np.array([x, y, z])


def plot_3d_projections():
    """
    3D plots
    """
    fig = plt.figure(figsize=(15, 5))

    # sphere
    ax1 = fig.add_subplot(131, projection='3d')

    # sphere definition
    center = [0, 0, 0]
    radius = 2
    point = [3, 3, 3]

    proj = project_to_sphere(point, center, radius)

    # sphere plot
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='blue')
    ax1.scatter(*point, color='red', s=100, label='original point')
    ax1.scatter(*proj, color='green', s=100, label='projection')
    ax1.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    ax1.set_title('projection to sphere')
    ax1.legend()

    # ellipsoid
    ax2 = fig.add_subplot(132, projection='3d')

    center = [0, 0, 0]
    semi_axes = [3, 2, 1]
    point = [4, 3, 2]

    proj = project_to_ellipsoid(point, center, semi_axes)

    #
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    a, b, c = semi_axes
    x_ellip = center[0] + a * np.outer(np.cos(u), np.sin(v))
    y_ellip = center[1] + b * np.outer(np.sin(u), np.sin(v))
    z_ellip = center[2] + c * np.outer(np.ones(np.size(u)), np.cos(v))

    ax2.plot_surface(x_ellip, y_ellip, z_ellip, alpha=0.3, color='blue')
    ax2.scatter(*point, color='red', s=100, label='original point')
    ax2.scatter(*proj, color='green', s=100, label='projection')
    ax2.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    ax2.set_title('elliptical projection')
    ax2.legend()

    # cylinder
    ax3 = fig.add_subplot(133, projection='3d')

    center = [0, 0, 0]
    axis_direction = [0, 0, 1]
    radius = 1.5
    height = 4
    point = [3, 2, 2]

    proj = project_to_cylinder(point, center, axis_direction, radius, height)

    #
    theta = np.linspace(0, 2*np.pi, 50)
    z_cyl = np.linspace(0, height, 50)
    theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
    x_cyl = center[0] + radius * np.cos(theta_mesh)
    y_cyl = center[1] + radius * np.sin(theta_mesh)
    z_cyl_mesh = center[2] + z_mesh

    ax3.plot_surface(x_cyl, y_cyl, z_cyl_mesh, alpha=0.3, color='blue')
    ax3.scatter(*point, color='red', s=100, label='original point')
    ax3.scatter(*proj, color='green', s=100, label='projection')
    ax3.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    ax3.set_title('projection to cylinder')
    ax3.legend()

    plt.tight_layout()
    plt.show()
