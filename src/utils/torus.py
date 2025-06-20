import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.remove.sphere import *

def project_to_torus(point, center, major_radius, minor_radius, axis=[0, 0, 1]):
    """
    torus projection

    Args:
        point: [x, y, z] point
        center: Torus center [x, y, z]
        major_radius: Torus major radius (R)
        minor_radius: Torus minor radius (r)
        axis: Torus axis direction (default z-axis)

    Returns:
        Projected point on the torus surface
    """
    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis

    # Translate to origin
    p_translated = p - c

    # If the torus axis is not z, we need to transform coordinates
    if not np.allclose(axis, [0, 0, 1]):
        # Create rotation matrix to align axis with z
        z_axis = np.array([0, 0, 1])
        if np.allclose(axis, -z_axis):
            # If the axis is opposite to z
            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            # Calculate rotation matrix
            v = np.cross(axis, z_axis)
            s = np.linalg.norm(v)
            c_rot = np.dot(axis, z_axis)

            if s != 0:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_rot) / (s * s))
            else:
                rotation_matrix = np.eye(3)

        # Apply inverse rotation
        p_rotated = np.dot(rotation_matrix.T, p_translated)
    else:
        p_rotated = p_translated

    x, y, z = p_rotated

    # Distance from z-axis in xy-plane
    rho = np.sqrt(x**2 + y**2)

    if rho == 0:
        # If the point is on the axis
        rho = major_radius
        x = major_radius
        y = 0

    # Angle in xy-plane
    phi = np.arctan2(y, x)

    # Point on the central circle of the torus
    center_circle_x = major_radius * np.cos(phi)
    center_circle_y = major_radius * np.sin(phi)
    center_circle_z = 0

    # Vector from the minor circle center to the original point
    dx = x - center_circle_x
    dy = y - center_circle_y
    dz = z - center_circle_z

    # Distance to the minor circle center
    minor_distance = np.sqrt(dx**2 + dy**2 + dz**2)

    if minor_distance == 0:
        # If the point is on the central circle
        proj_x = center_circle_x
        proj_y = center_circle_y
        proj_z = minor_radius
    else:
        # Normalize and multiply by minor radius
        unit_minor = np.array([dx, dy, dz]) / minor_distance

        proj_x = center_circle_x + minor_radius * unit_minor[0]
        proj_y = center_circle_y + minor_radius * unit_minor[1]
        proj_z = center_circle_z + minor_radius * unit_minor[2]

    proj_rotated = np.array([proj_x, proj_y, proj_z])

    # Apply inverse rotation to return to original coordinates
    if not np.allclose(axis, [0, 0, 1]):
        proj_original = np.dot(rotation_matrix, proj_rotated)
    else:
        proj_original = proj_rotated

    # Return to original coordinates
    projection = c + proj_original

    return projection

def torus_parametric(u, v, center, major_radius, minor_radius, axis=[0, 0, 1]):
    """
    Parametric equation of a torus

    Args:
        u, v: Spherical parameters (0 <= u <= 2π, 0 <= v <= 2π)
        center, major_radius, minor_radius, axis: Torus properties

    Returns:
        Points on the torus surface
    """
    c = np.array(center)
    axis = np.array(axis) / np.linalg.norm(axis)

    # Standard parametric equation of a torus
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)

    points = np.array([x, y, z])

    # If the torus axis is not z
    if not np.allclose(axis, [0, 0, 1]):
        z_axis = np.array([0, 0, 1])
        if np.allclose(axis, -z_axis):
            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            v_rot = np.cross(z_axis, axis)
            s = np.linalg.norm(v_rot)
            c_rot = np.dot(z_axis, axis)

            if s != 0:
                vx = np.array([[0, -v_rot[2], v_rot[1]], [v_rot[2], 0, -v_rot[0]], [-v_rot[1], v_rot[0], 0]])
                rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_rot) / (s * s))
            else:
                rotation_matrix = np.eye(3)

        # Apply rotation
        if points.ndim == 2:
            points = np.dot(rotation_matrix, points)
        else:
            points = np.dot(rotation_matrix, points.reshape(3, -1)).reshape(points.shape)

    # Translate to center
    if points.ndim == 2:
        points = points + c.reshape(-1, 1)
    else:
        points = points + c.reshape(3,1,1)
    return points

def torus_volume_inside(point, center, major_radius, minor_radius):
    """
    Check if a point is inside the torus
    """
    p = np.array(point) - np.array(center)
    x, y, z = p

    rho = np.sqrt(x**2 + y**2)
    distance_to_center_circle = np.sqrt((rho - major_radius)**2 + z**2)

    return distance_to_center_circle <= minor_radius

def project_to_disk_with_hole(point, center, outer_radius, inner_radius, normal=[0, 0, 1]):
    """
    Projection onto a disk with a hole (like a CD)

    Args:
        point: [x, y, z] point
        center: Disk center [x, y, z]
        outer_radius: Outer radius
        inner_radius: Inner radius (hole)
        normal: Normal vector of the disk plane

    Returns:
        Projected point on the disk boundary
    """
    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)
    n = np.array(normal, dtype=float)
    n = n / np.linalg.norm(n)  # Normalize

    # Translate to origin
    p_translated = p - c

    # Projection onto the disk plane
    proj_on_plane = p_translated - np.dot(p_translated, n) * n

    # Distance from the center in the plane
    distance_2d = np.linalg.norm(proj_on_plane)

    if distance_2d == 0:
        # If the point is on the axis, project to the inner edge
        return c + inner_radius * np.array([1, 0, 0])

    # Direction in the plane
    direction_2d = proj_on_plane / distance_2d

    # Determine the closest edge
    if distance_2d < inner_radius:
        # Closer to the inner edge
        projection_2d = inner_radius * direction_2d
    elif distance_2d > outer_radius:
        # Closer to the outer edge
        projection_2d = outer_radius * direction_2d
    else:
        # Inside the disk area - to the closest edge
        dist_to_inner = abs(distance_2d - inner_radius)
        dist_to_outer = abs(distance_2d - outer_radius)

        if dist_to_inner < dist_to_outer:
            projection_2d = inner_radius * direction_2d
        else:
            projection_2d = outer_radius * direction_2d

    # Return to 3D space
    projection = c + projection_2d

    return projection

def disk_with_hole_parametric(r, theta, center, outer_radius, inner_radius, normal=[0, 0, 1]):
    """
    Parametric equation of a disk with a hole

    Args:
        r: Radius (inner_radius <= r <= outer_radius)
        theta: Angle (0 <= theta <= 2π)
    """
    c = np.array(center)
    n = np.array(normal) / np.linalg.norm(normal)

    # Create two vectors orthogonal to the normal
    if abs(n[2]) < 0.9:
        v1 = np.cross(n, [0, 0, 1])
    else:
        v1 = np.cross(n, [1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(n, v1)

    # Points on the disk
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Transform to 3D coordinates
    points = c[:, np.newaxis] + x * v1[:, np.newaxis] + y * v2[:, np.newaxis]

    return points

def is_point_on_disk(point, center, outer_radius, inner_radius, normal=[0, 0, 1], tolerance=1e-6):
    """
    Check if a point is on the disk
    """
    p = np.array(point) - np.array(center)
    n = np.array(normal) / np.linalg.norm(normal)

    # Check distance from the plane
    distance_to_plane = abs(np.dot(p, n))
    if distance_to_plane > tolerance:
        return False

    # Projection onto the plane
    proj_on_plane = p - np.dot(p, n) * n
    distance_2d = np.linalg.norm(proj_on_plane)

    # Check radius
    return inner_radius <= distance_2d <= outer_radius

def plot_disk_and_torus_projection():
    """
    Plot disk, torus, and projections
    """
    fig = plt.figure(figsize=(20, 10))

    # Disk with hole (CD)
    ax1 = fig.add_subplot(231, projection='3d')

    center = [0, 0, 0]
    outer_radius = 3
    inner_radius = 1
    normal = [0, 0, 1]

    test_points = [
        [2, 2, 1],     # Above the disk
        [0.5, 0, 0.5], # Near the hole
        [4, 1, -0.5],  # Outside the disk
        [0, 2.5, 0]    # On the disk
    ]

    # Plot the disk
    theta = np.linspace(0, 2*np.pi, 100)

    # Outer edge
    outer_edge = disk_with_hole_parametric(outer_radius, theta, center, outer_radius, inner_radius, normal)
    ax1.plot(outer_edge[0], outer_edge[1], outer_edge[2], 'b-', linewidth=3, label='Outer edge')

    # Inner edge
    inner_edge = disk_with_hole_parametric(inner_radius, theta, center, outer_radius, inner_radius, normal)
    ax1.plot(inner_edge[0], inner_edge[1], inner_edge[2], 'r-', linewidth=3, label='Inner edge')

    # Some radial lines to show the surface
    for angle in np.linspace(0, 2*np.pi, 8):
        r_line = np.linspace(inner_radius, outer_radius, 20)
        line_points = disk_with_hole_parametric(r_line, angle, center, outer_radius, inner_radius, normal)
        ax1.plot(line_points[0], line_points[1], line_points[2], 'gray', alpha=0.3)

    # Project points
    colors = ['red', 'green', 'orange', 'purple']
    for i, point in enumerate(test_points):
        proj = project_to_disk_with_hole(point, center, outer_radius, inner_radius, normal)

        ax1.scatter(*point, color=colors[i], s=100, label=f'Point {i+1}')
        ax1.scatter(*proj, color=colors[i], s=100, marker='^')
        ax1.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                color=colors[i], linestyle='--', alpha=0.7)

    ax1.set_title('Projection onto Disk (CD)')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Slanted disk
    ax2 = fig.add_subplot(232, projection='3d')

    center = [0, 0, 0]
    outer_radius = 2.5
    inner_radius = 0.8
    normal = [1, 1, 0]  # Slanted
    point = [3, 2, 1]

    proj = project_to_disk_with_hole(point, center, outer_radius, inner_radius, normal)

    # Plot slanted disk
    theta = np.linspace(0, 2*np.pi, 100)
    outer_edge = disk_with_hole_parametric(outer_radius, theta, center, outer_radius, inner_radius, normal)
    inner_edge = disk_with_hole_parametric(inner_radius, theta, center, outer_radius, inner_radius, normal)

    ax2.plot(outer_edge[0], outer_edge[1], outer_edge[2], 'b-', linewidth=3)
    ax2.plot(inner_edge[0], inner_edge[1], inner_edge[2], 'r-', linewidth=3)

    # Radial lines
    for angle in np.linspace(0, 2*np.pi, 8):
        r_line = np.linspace(inner_radius, outer_radius, 20)
        line_points = disk_with_hole_parametric(r_line, angle, center, outer_radius, inner_radius, normal)
        ax2.plot(line_points[0], line_points[1], line_points[2], 'gray', alpha=0.3)

    ax2.scatter(*point, color='red', s=100, label='Original point')
    ax2.scatter(*proj, color='blue', s=100, label='Projection')
    ax2.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)

    # Display normal vector
    normal_normalized = np.array(normal) / np.linalg.norm(normal)
    ax2.quiver(0, 0, 0, normal_normalized[0]*2, normal_normalized[1]*2, normal_normalized[2]*2,
              color='black', arrow_length_ratio=0.1, linewidth=2)

    ax2.set_title('Slanted Disk')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Thick disk (Realistic CD model)
    ax3 = fig.add_subplot(233, projection='3d')

    center = [0, 0, 0]
    outer_radius = 6
    inner_radius = 0.75
    thickness = 0.12  # CD thickness (mm)
    point = [4, 3, 0.5]

    proj = project_to_disk_with_hole(point, center, outer_radius, inner_radius, [0, 0, 1])

    # Plot thick CD model
    theta = np.linspace(0, 2*np.pi, 50)

    # Top surface
    for r in [inner_radius, outer_radius]:
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        z_circle = np.full_like(theta, thickness/2)
        ax3.plot(x_circle, y_circle, z_circle, 'b-', linewidth=2)

    # Bottom surface
    for r in [inner_radius, outer_radius]:
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        z_circle = np.full_like(theta, -thickness/2)
        ax3.plot(x_circle, y_circle, z_circle, 'b-', linewidth=2)

    # Vertical edges
    for angle in np.linspace(0, 2*np.pi, 12):
        # Outer edge
        x_outer = outer_radius * np.cos(angle)
        y_outer = outer_radius * np.sin(angle)
        ax3.plot([x_outer, x_outer], [y_outer, y_outer],
                [-thickness/2, thickness/2], 'b-', linewidth=1)

        # Inner edge
        x_inner = inner_radius * np.cos(angle)
        y_inner = inner_radius * np.sin(angle)
        ax3.plot([x_inner, x_inner], [y_inner, y_inner],
                [-thickness/2, thickness/2], 'r-', linewidth=1)

    # CD tracks (concentric circles)
    for r in np.linspace(inner_radius + 0.5, outer_radius - 0.5, 8):
        x_track = r * np.cos(theta)
        y_track = r * np.sin(theta)
        z_track = np.full_like(theta, thickness/2 + 0.01)
        ax3.plot(x_track, y_track, z_track, 'gray', linewidth=0.5, alpha=0.7)

    ax3.scatter(*point, color='red', s=100, label='Original point')
    ax3.scatter(*proj, color='green', s=100, label='Projection')
    ax3.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)

    ax3.set_title('Realistic CD Model')
    ax3.legend()
    ax3.set_xlabel('X (cm)')
    ax3.set_ylabel('Y (cm)')
    ax3.set_zlabel('Z (mm)')

    # Regular torus
    ax4 = fig.add_subplot(234, projection='3d')

    center = [0, 0, 0]
    major_radius = 3
    minor_radius = 1
    points_to_project = [
        [5, 2, 1],
        [1, 1, 3]
    ]

    # Plot torus
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, 2*np.pi, 40)
    U, V = np.meshgrid(u, v)

    torus_points = torus_parametric(U, V, center, major_radius, minor_radius)
    X, Y, Z = torus_points

    ax4.plot_surface(X, Y, Z, alpha=0.4, color='lightblue')

    # Project points
    colors = ['red', 'green']
    for i, point in enumerate(points_to_project):
        proj = project_to_torus(point, center, major_radius, minor_radius)

        ax4.scatter(*point, color=colors[i], s=100, label=f'Point {i+1}')
        ax4.scatter(*proj, color=colors[i], s=100, marker='^')
        ax4.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                color=colors[i], linestyle='--', alpha=0.7)

    ax4.set_title('Projection onto Torus')
    ax4.legend()
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    # Compare disk and torus
    ax5 = fig.add_subplot(235, projection='3d')

    # Disk
    center_disk = [-2, 0, 0]
    outer_r = 2
    inner_r = 0.5
    theta = np.linspace(0, 2*np.pi, 50)

    outer_edge = disk_with_hole_parametric(outer_r, theta, center_disk, outer_r, inner_r)
    inner_edge = disk_with_hole_parametric(inner_r, theta, center_disk, outer_r, inner_r)
    ax5.plot(outer_edge[0], outer_edge[1], outer_edge[2], 'b-', linewidth=3, label='Disk')
    ax5.plot(inner_edge[0], inner_edge[1], inner_edge[2], 'b-', linewidth=3)

    # Torus
    center_torus = [2, 0, 0]
    major_r = 1.5
    minor_r = 0.5

    u_small = np.linspace(0, 2*np.pi, 30)
    v_small = np.linspace(0, 2*np.pi, 30)
    U_small, V_small = np.meshgrid(u_small, v_small)

    torus_points_small = torus_parametric(U_small, V_small, center_torus, major_r, minor_r)
    X_small, Y_small, Z_small = torus_points_small

    ax5.plot_surface(X_small, Y_small, Z_small, alpha=0.4, color='lightcoral', label='Torus')

    # Test point
    test_point = [0, 2, 1]
    proj_disk = project_to_disk_with_hole(test_point, center_disk, outer_r, inner_r)
    proj_torus = project_to_torus(test_point, center_torus, major_r, minor_r)

    ax5.scatter(*test_point, color='black', s=100, label='Test point')
    ax5.scatter(*proj_disk, color='blue', s=100, marker='^')
    ax5.scatter(*proj_torus, color='red', s=100, marker='^')
    ax5.plot([test_point[0], proj_disk[0]], [test_point[1], proj_disk[1]], [test_point[2], proj_disk[2]],
            'b--', alpha=0.7)
    ax5.plot([test_point[0], proj_torus[0]], [test_point[1], proj_torus[1]], [test_point[2], proj_torus[2]],
            'r--', alpha=0.7)

    ax5.set_title('Disk vs Torus Comparison')
    ax5.legend()
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')

    # Overview - All shapes
    ax6 = fig.add_subplot(236, projection='3d')

    # Small sphere
    sphere_center = [-3, -3, 0]
    sphere_radius = 1
    u_sphere = np.linspace(0, 2 * np.pi, 20)
    v_sphere = np.linspace(0, np.pi, 20)
    x_sphere = sphere_center[0] + sphere_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = sphere_center[1] + sphere_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    ax6.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='yellow')

    # Small disk
    disk_center = [0, -3, 0]
    outer_edge_small = disk_with_hole_parametric(1.5, theta, disk_center, 1.5, 0.3)
    inner_edge_small = disk_with_hole_parametric(0.3, theta, disk_center, 1.5, 0.3)
    ax6.plot(outer_edge_small[0], outer_edge_small[1], outer_edge_small[2], 'b-', linewidth=2)
    ax6.plot(inner_edge_small[0], inner_edge_small[1], inner_edge_small[2], 'b-', linewidth=2)

    # Small torus
    torus_center = [3, -3, 0]
    torus_points_tiny = torus_parametric(U_small, V_small, torus_center, 1, 0.3)
    X_tiny, Y_tiny, Z_tiny = torus_points_tiny
    ax6.plot_surface(X_tiny, Y_tiny, Z_tiny, alpha=0.4, color='lightgreen')

    # Small cylinder
    cyl_center = [-3, 3, 0]
    cyl_radius = 0.8
    cyl_height = 2
    theta_cyl = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(0, cyl_height, 20)
    theta_mesh, z_mesh = np.meshgrid(theta_cyl, z_cyl)
    x_cyl = cyl_center[0] + cyl_radius * np.cos(theta_mesh)
    y_cyl = cyl_center[1] + cyl_radius * np.sin(theta_mesh)
    z_cyl_mesh = cyl_center[2] + z_mesh - cyl_height/2
    ax6.plot_surface(x_cyl, y_cyl, z_cyl_mesh, alpha=0.3, color='orange')

    # Central point and projections
    central_point = [0, 0, 2]
    ax6.scatter(*central_point, color='red', s=150, label='Central point')

    # Projection to each shape
    proj_sphere = project_to_sphere(central_point, sphere_center, sphere_radius)
    proj_disk_central = project_to_disk_with_hole(central_point, disk_center, 1.5, 0.3)
    proj_torus_central = project_to_torus(central_point, torus_center, 1, 0.3)
    proj_cyl = project_to_cylinder(central_point, cyl_center, [0, 0, 1], cyl_radius, cyl_height)

    # Projection lines
    projections = [proj_sphere, proj_disk_central, proj_torus_central, proj_cyl]
    colors_proj = ['yellow', 'blue', 'green', 'orange']

    for proj, color in zip(projections, colors_proj):
        ax6.plot([central_point[0], proj[0]], [central_point[1], proj[1]], [central_point[2], proj[2]],
                color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax6.scatter(*proj, color=color, s=100, marker='^')

    ax6.set_title('All Geometric Shapes')
    ax6.legend()
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

def plot_torus_projection():
    """
    Plot torus and projections
    """
    fig = plt.figure(figsize=(18, 6))

    # Regular torus
    ax1 = fig.add_subplot(131, projection='3d')

    center = [0, 0, 0]
    major_radius = 3
    minor_radius = 1
    points_to_project = [
        [5, 2, 1],
        [1, 1, 3],
        [0, 0, 2],
        [-2, 4, -1]
    ]

    # Plot torus
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)

    torus_points = torus_parametric(U, V, center, major_radius, minor_radius)
    X, Y, Z = torus_points

    ax1.plot_surface(X, Y, Z, alpha=0.4, color='lightblue')

    # Project points
    colors = ['red', 'green', 'orange', 'purple']
    for i, point in enumerate(points_to_project):
        proj = project_to_torus(point, center, major_radius, minor_radius)

        ax1.scatter(*point, color=colors[i], s=100, label=f'Point {i+1}')
        ax1.scatter(*proj, color=colors[i], s=100, marker='^')
        ax1.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                color=colors[i], linestyle='--', alpha=0.7)

    ax1.set_title('Projection onto Regular Torus')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Stretched torus
    ax2 = fig.add_subplot(132, projection='3d')

    center = [0, 0, 0]
    major_radius = 4
    minor_radius = 0.8
    point = [3, 3, 2]

    proj = project_to_torus(point, center, major_radius, minor_radius)

    torus_points = torus_parametric(U, V, center, major_radius, minor_radius)
    X, Y, Z = torus_points

    ax2.plot_surface(X, Y, Z, alpha=0.4, color='lightgreen')
    ax2.scatter(*point, color='red', s=100, label='Original point')
    ax2.scatter(*proj, color='blue', s=100, label='Projection')
    ax2.plot([point[0], proj[0]], [point[1], point[1]], [point[2], point[2]], 'r--', alpha=0.7)

    ax2.set_title('Stretched Torus')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Slanted torus
    ax3 = fig.add_subplot(133, projection='3d')

    center = [0, 0, 0]
    major_radius = 2.5
    minor_radius = 1
    axis = [1, 1, 1]  # Slanted axis
    point = [4, 1, 3]

    proj = project_to_torus(point, center, major_radius, minor_radius, axis)

    torus_points = torus_parametric(U, V, center, major_radius, minor_radius, axis)
    X, Y, Z = torus_points

    ax3.plot_surface(X, Y, Z, alpha=0.4, color='lightcoral')
    ax3.scatter(*point, color='red', s=100, label='Original point')
    ax3.scatter(*proj, color='blue', s=100, label='Projection')
    ax3.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)

    # Plot axis
    axis_normalized = np.array(axis) / np.linalg.norm(axis)
    ax3.quiver(0, 0, 0, axis_normalized[0]*3, axis_normalized[1]*3, axis_normalized[2]*3,
              color='black', arrow_length_ratio=0.1, linewidth=2)

    ax3.set_title('Slanted Torus')
    ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()



def create_realistic_cd_model(center, outer_radius=6, inner_radius=0.75, thickness=0.12):
    """
    Realistic 3D CD model

    Args:
        center: CD center
        outer_radius: Outer radius (cm)
        inner_radius: Inner radius (cm)
        thickness: Thickness (mm)

    Returns:
        3D model data
    """
    c = np.array(center)

    # Angles for the circle
    theta = np.linspace(0, 2*np.pi, 100)

    # Various radii for data tracks
    track_radii = np.linspace(inner_radius + 0.3, outer_radius - 0.2, 20)

    model_data = {
        'center': c,
        'outer_radius': outer_radius,
        'inner_radius': inner_radius,
        'thickness': thickness,
        'theta': theta,
        'track_radii': track_radii,
        'volume': np.pi * (outer_radius**2 - inner_radius**2) * thickness,
        'surface_area': 2 * np.pi * (outer_radius**2 - inner_radius**2) +
                       2 * np.pi * (outer_radius + inner_radius) * thickness
    }

    return model_data

def project_to_thick_disk(point, center, outer_radius, inner_radius, thickness):
    """
    Projection onto a thick disk (Realistic CD model)

    Includes projection onto:
    - Top and bottom surfaces
    - Outer edge
    - Inner edge
    """
    p = np.array(point, dtype=float)
    c = np.array(center, dtype=float)

    # Translate to origin
    p_rel = p - c
    x, y, z = p_rel

    # Radial distance
    rho = np.sqrt(x**2 + y**2)

    # Determine the closest surface
    distances = {}
    projections = {}

    # Top surface
    if inner_radius <= rho <= outer_radius:
        proj_top = c + np.array([x, y, thickness/2])
        distances['top'] = abs(z - thickness/2)
        projections['top'] = proj_top

    # Bottom surface
    if inner_radius <= rho <= outer_radius:
        proj_bottom = c + np.array([x, y, -thickness/2])
        distances['bottom'] = abs(z + thickness/2)
        projections['bottom'] = proj_bottom

    # Outer edge
    if -thickness/2 <= z <= thickness/2:
        direction = np.array([x, y, 0]) / (rho if rho > 0 else 1)
        proj_outer = c + outer_radius * direction + np.array([0, 0, z])
        distances['outer'] = abs(rho - outer_radius)
        projections['outer'] = proj_outer

    # Inner edge
    if -thickness/2 <= z <= thickness/2:
        direction = np.array([x, y, 0]) / (rho if rho > 0 else 1)
        proj_inner = c + inner_radius * direction + np.array([0, 0, z])
        distances['inner'] = abs(rho - inner_radius)
        projections['inner'] = proj_inner

    # Corners (rounded edges)
    corners = []

    # Outer top corner
    corner_outer_top = c + np.array([outer_radius * x/rho if rho > 0 else outer_radius,
                                     outer_radius * y/rho if rho > 0 else 0,
                                     thickness/2])
    dist_corner_outer_top = np.linalg.norm(p - corner_outer_top)
    corners.append(('outer_top', corner_outer_top, dist_corner_outer_top))

    # Inner top corner
    corner_inner_top = c + np.array([inner_radius * x/rho if rho > 0 else inner_radius,
                                     inner_radius * y/rho if rho > 0 else 0,
                                     thickness/2])
    dist_corner_inner_top = np.linalg.norm(p - corner_inner_top)
    corners.append(('inner_top', corner_inner_top, dist_corner_inner_top))

    # Outer bottom corner
    corner_outer_bottom = c + np.array([outer_radius * x/rho if rho > 0 else outer_radius,
                                        outer_radius * y/rho if rho > 0 else 0,
                                        -thickness/2])
    dist_corner_outer_bottom = np.linalg.norm(p - corner_outer_bottom)
    corners.append(('outer_bottom', corner_outer_bottom, dist_corner_outer_bottom))

    # Inner bottom corner
    corner_inner_bottom = c + np.array([inner_radius * x/rho if rho > 0 else inner_radius,
                                        inner_radius * y/rho if rho > 0 else 0,
                                        -thickness/2])
    dist_corner_inner_bottom = np.linalg.norm(p - corner_inner_bottom)
    corners.append(('inner_bottom', corner_inner_bottom, dist_corner_inner_bottom))

    # Find the closest point
    all_candidates = []

    for name, proj in projections.items():
        dist = np.linalg.norm(p - proj)
        all_candidates.append((name, proj, dist))

    all_candidates.extend(corners)

    # Closest point
    closest = min(all_candidates, key=lambda x: x[2])

    return closest[1], closest[0], closest[2]

