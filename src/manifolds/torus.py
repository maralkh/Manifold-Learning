import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Torus:
    """
    3D Torus class for geometric operations
    """
    
    def __init__(self, center=(0, 0, 0), major_radius=2.0, minor_radius=1.0, axis=(0, 0, 1)):
        """
        Args:
            center: Center point [x, y, z]
            major_radius: Major radius (R) - distance from center to tube center
            minor_radius: Minor radius (r) - tube radius
            axis: Torus axis direction (default z-axis)
        """
        self.center = np.array(center, dtype=float)
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.axis = np.array(axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)  # Normalize
        
        if self.major_radius <= 0 or self.minor_radius <= 0:
            raise ValueError("Radii must be positive")
        if self.minor_radius >= self.major_radius:
            raise ValueError("Minor radius should be less than major radius")
    
    def _get_rotation_matrix(self):
        """
        Get rotation matrix to align torus axis with z-axis
        """
        z_axis = np.array([0, 0, 1])
        
        if np.allclose(self.axis, z_axis):
            return np.eye(3)
        elif np.allclose(self.axis, -z_axis):
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            v = np.cross(self.axis, z_axis)
            s = np.linalg.norm(v)
            c = np.dot(self.axis, z_axis)
            
            if s != 0:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
            else:
                rotation_matrix = np.eye(3)
            
            return rotation_matrix.T  # Inverse rotation
    
    def project_to_surface(self, point):
        """
        Project point to torus surface
        """
        p = np.array(point, dtype=float)
        p_translated = p - self.center
        
        # Rotate to align with z-axis if needed
        rotation_matrix = self._get_rotation_matrix()
        p_rotated = np.dot(rotation_matrix, p_translated)
        
        x, y, z = p_rotated
        
        # Distance from z-axis in xy-plane
        rho = np.sqrt(x**2 + y**2)
        
        if rho == 0:
            # Point is on the axis
            rho = self.major_radius
            x = self.major_radius
            y = 0
        
        # Angle in xy-plane
        phi = np.arctan2(y, x)
        
        # Point on the central circle of the torus
        center_circle_x = self.major_radius * np.cos(phi)
        center_circle_y = self.major_radius * np.sin(phi)
        center_circle_z = 0
        
        # Vector from minor circle center to the original point
        dx = x - center_circle_x
        dy = y - center_circle_y
        dz = z - center_circle_z
        
        # Distance to the minor circle center
        minor_distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if minor_distance == 0:
            # Point is on the central circle
            proj_x = center_circle_x
            proj_y = center_circle_y
            proj_z = self.minor_radius
        else:
            # Normalize and multiply by minor radius
            unit_minor = np.array([dx, dy, dz]) / minor_distance
            proj_x = center_circle_x + self.minor_radius * unit_minor[0]
            proj_y = center_circle_y + self.minor_radius * unit_minor[1]
            proj_z = center_circle_z + self.minor_radius * unit_minor[2]
        
        proj_rotated = np.array([proj_x, proj_y, proj_z])
        
        # Apply inverse rotation
        if not np.allclose(self.axis, [0, 0, 1]):
            proj_original = np.dot(rotation_matrix.T, proj_rotated)
        else:
            proj_original = proj_rotated
        
        # Return to original coordinates
        projection = self.center + proj_original
        return projection
    
    def distance_to_surface(self, point):
        """
        Calculate distance from point to torus surface
        """
        projected = self.project_to_surface(point)
        return np.linalg.norm(np.array(point) - projected)
    
    def is_inside(self, point):
        """
        Check if point is inside the torus
        """
        p = np.array(point) - self.center
        
        # Rotate to align with z-axis
        rotation_matrix = self._get_rotation_matrix()
        p_rotated = np.dot(rotation_matrix, p)
        
        x, y, z = p_rotated
        rho = np.sqrt(x**2 + y**2)
        distance_to_center_circle = np.sqrt((rho - self.major_radius)**2 + z**2)
        
        return distance_to_center_circle <= self.minor_radius
    
    def generate_surface_points(self, n_u=50, n_v=50):
        """
        Generate points on torus surface using parametric equations
        """
        u = np.linspace(0, 2*np.pi, n_u)
        v = np.linspace(0, 2*np.pi, n_v)
        U, V = np.meshgrid(u, v)
        
        # Standard parametric equations
        X = (self.major_radius + self.minor_radius * np.cos(V)) * np.cos(U)
        Y = (self.major_radius + self.minor_radius * np.cos(V)) * np.sin(U)
        Z = self.minor_radius * np.sin(V)
        
        points = np.array([X, Y, Z])
        
        # Apply rotation if axis is not z
        if not np.allclose(self.axis, [0, 0, 1]):
            rotation_matrix = self._get_rotation_matrix()
            points_reshaped = points.reshape(3, -1)
            points_rotated = np.dot(rotation_matrix.T, points_reshaped)
            points = points_rotated.reshape(points.shape)
        
        # Translate to center
        points[0] += self.center[0]
        points[1] += self.center[1]
        points[2] += self.center[2]
        
        return points


class DiskWithHole:
    """
    Disk with hole (like a CD) class for geometric operations
    """
    
    def __init__(self, center=(0, 0, 0), outer_radius=3.0, inner_radius=1.0, normal=(0, 0, 1)):
        """
        Args:
            center: Center point [x, y, z]
            outer_radius: Outer radius
            inner_radius: Inner radius (hole)
            normal: Normal vector of the disk plane
        """
        self.center = np.array(center, dtype=float)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize
        
        if self.outer_radius <= 0 or self.inner_radius < 0:
            raise ValueError("Radii must be non-negative and outer radius positive")
        if self.inner_radius >= self.outer_radius:
            raise ValueError("Inner radius must be less than outer radius")
    
    def _get_plane_vectors(self):
        """
        Get two orthogonal vectors in the disk plane
        """
        if abs(self.normal[2]) < 0.9:
            v1 = np.cross(self.normal, [0, 0, 1])
        else:
            v1 = np.cross(self.normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(self.normal, v1)
        return v1, v2
    
    def project_to_surface(self, point):
        """
        Project point to disk boundary (inner or outer edge)
        """
        p = np.array(point, dtype=float)
        p_translated = p - self.center
        
        # Project onto the disk plane
        proj_on_plane = p_translated - np.dot(p_translated, self.normal) * self.normal
        
        # Distance from center in the plane
        distance_2d = np.linalg.norm(proj_on_plane)
        
        if distance_2d == 0:
            # Point is on the axis, project to inner edge
            v1, v2 = self._get_plane_vectors()
            projection_2d = self.inner_radius * v1
        else:
            # Direction in the plane
            direction_2d = proj_on_plane / distance_2d
            
            # Determine closest edge
            if distance_2d < self.inner_radius:
                # Closer to inner edge
                projection_2d = self.inner_radius * direction_2d
            elif distance_2d > self.outer_radius:
                # Closer to outer edge
                projection_2d = self.outer_radius * direction_2d
            else:
                # Inside the disk area - project to closest edge
                dist_to_inner = abs(distance_2d - self.inner_radius)
                dist_to_outer = abs(distance_2d - self.outer_radius)
                
                if dist_to_inner < dist_to_outer:
                    projection_2d = self.inner_radius * direction_2d
                else:
                    projection_2d = self.outer_radius * direction_2d
        
        # Return to 3D space
        projection = self.center + projection_2d
        return projection
    
    def distance_to_surface(self, point):
        """
        Calculate distance from point to disk surface
        """
        projected = self.project_to_surface(point)
        return np.linalg.norm(np.array(point) - projected)
    
    def is_on_disk(self, point, tolerance=1e-6):
        """
        Check if point is on the disk surface
        """
        p = np.array(point) - self.center
        
        # Check distance from plane
        distance_to_plane = abs(np.dot(p, self.normal))
        if distance_to_plane > tolerance:
            return False
        
        # Project onto plane and check radius
        proj_on_plane = p - np.dot(p, self.normal) * self.normal
        distance_2d = np.linalg.norm(proj_on_plane)
        
        return self.inner_radius <= distance_2d <= self.outer_radius
    
    def generate_surface_points(self, n_radial=20, n_angular=50):
        """
        Generate points on disk surface
        """
        # Radial lines from inner to outer radius
        r_values = np.linspace(self.inner_radius, self.outer_radius, n_radial)
        theta_values = np.linspace(0, 2*np.pi, n_angular)
        
        v1, v2 = self._get_plane_vectors()
        
        points = []
        for r in r_values:
            for theta in theta_values:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                point_3d = self.center + x * v1 + y * v2
                points.append(point_3d)
        
        return np.array(points)
    
    def generate_edge_points(self, n_points=100):
        """
        Generate points on inner and outer edges
        """
        theta = np.linspace(0, 2*np.pi, n_points)
        v1, v2 = self._get_plane_vectors()
        
        # Outer edge
        outer_points = []
        for t in theta:
            x = self.outer_radius * np.cos(t)
            y = self.outer_radius * np.sin(t)
            point_3d = self.center + x * v1 + y * v2
            outer_points.append(point_3d)
        
        # Inner edge
        inner_points = []
        for t in theta:
            x = self.inner_radius * np.cos(t)
            y = self.inner_radius * np.sin(t)
            point_3d = self.center + x * v1 + y * v2
            inner_points.append(point_3d)
        
        return np.array(outer_points), np.array(inner_points)


class ThickDisk:
    """
    Thick disk (realistic CD model) with projection to all surfaces
    """
    
    def __init__(self, center=(0, 0, 0), outer_radius=6.0, inner_radius=0.75, 
                 thickness=0.12, normal=(0, 0, 1)):
        """
        Args:
            center: Center point [x, y, z]
            outer_radius: Outer radius (cm)
            inner_radius: Inner radius (cm)
            thickness: Thickness (mm or same units)
            normal: Normal vector of the disk plane
        """
        self.center = np.array(center, dtype=float)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.thickness = float(thickness)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)
        
        if self.outer_radius <= 0 or self.inner_radius < 0 or self.thickness <= 0:
            raise ValueError("All dimensions must be positive")
        if self.inner_radius >= self.outer_radius:
            raise ValueError("Inner radius must be less than outer radius")
    
    def project_to_surface(self, point):
        """
        Project point to the closest surface of thick disk
        Returns: (projection, surface_type, distance)
        """
        p = np.array(point, dtype=float)
        p_rel = p - self.center
        
        # Project to normal direction to get z-component
        z_comp = np.dot(p_rel, self.normal)
        # Project to disk plane
        proj_on_plane = p_rel - z_comp * self.normal
        rho = np.linalg.norm(proj_on_plane)
        
        # All possible projections
        candidates = []
        
        # Top surface
        if self.inner_radius <= rho <= self.outer_radius:
            proj_top = self.center + proj_on_plane + (self.thickness/2) * self.normal
            dist_top = abs(z_comp - self.thickness/2)
            candidates.append((proj_top, 'top_surface', dist_top))
        
        # Bottom surface
        if self.inner_radius <= rho <= self.outer_radius:
            proj_bottom = self.center + proj_on_plane - (self.thickness/2) * self.normal
            dist_bottom = abs(z_comp + self.thickness/2)
            candidates.append((proj_bottom, 'bottom_surface', dist_bottom))
        
        # Outer edge
        if -self.thickness/2 <= z_comp <= self.thickness/2:
            if rho > 0:
                direction = proj_on_plane / rho
                proj_outer = self.center + self.outer_radius * direction + z_comp * self.normal
                dist_outer = abs(rho - self.outer_radius)
                candidates.append((proj_outer, 'outer_edge', dist_outer))
        
        # Inner edge
        if -self.thickness/2 <= z_comp <= self.thickness/2:
            if rho > 0:
                direction = proj_on_plane / rho
                proj_inner = self.center + self.inner_radius * direction + z_comp * self.normal
                dist_inner = abs(rho - self.inner_radius)
                candidates.append((proj_inner, 'inner_edge', dist_inner))
            else:
                # Point on axis
                proj_inner = self.center + self.inner_radius * np.array([1, 0, 0]) + z_comp * self.normal
                dist_inner = self.inner_radius
                candidates.append((proj_inner, 'inner_edge', dist_inner))
        
        # Corners (edges of edges)
        corners = [
            # Outer corners
            (self.outer_radius, self.thickness/2, 'outer_top_corner'),
            (self.outer_radius, -self.thickness/2, 'outer_bottom_corner'),
            # Inner corners
            (self.inner_radius, self.thickness/2, 'inner_top_corner'),
            (self.inner_radius, -self.thickness/2, 'inner_bottom_corner')
        ]
        
        for r_corner, z_corner, corner_type in corners:
            if rho > 0:
                direction = proj_on_plane / rho
            else:
                direction = np.array([1, 0, 0])
            corner_proj = self.center + r_corner * direction + z_corner * self.normal
            corner_dist = np.linalg.norm(p - corner_proj)
            candidates.append((corner_proj, corner_type, corner_dist))
        
        # Find closest
        if not candidates:
            # Fallback: project to outer top surface
            if rho > 0:
                direction = proj_on_plane / rho
            else:
                direction = np.array([1, 0, 0])
            fallback_proj = self.center + self.outer_radius * direction + (self.thickness/2) * self.normal
            return fallback_proj, 'fallback', np.linalg.norm(p - fallback_proj)
        
        closest = min(candidates, key=lambda x: x[2])
        return closest[0], closest[1], closest[2]
    
    def get_volume(self):
        """Calculate volume of thick disk"""
        return np.pi * (self.outer_radius**2 - self.inner_radius**2) * self.thickness
    
    def get_surface_area(self):
        """Calculate total surface area"""
        # Top and bottom surfaces
        disk_area = 2 * np.pi * (self.outer_radius**2 - self.inner_radius**2)
        # Outer and inner edge areas
        edge_area = 2 * np.pi * (self.outer_radius + self.inner_radius) * self.thickness
        return disk_area + edge_area


def test_torus_and_disk():
    """
    Test torus and disk classes
    """
    print("=== Torus and Disk Test ===\n")
    
    # Test points
    test_points = [
        [3, 3, 3],
        [1, 1, 1],
        [0, 0, 0],
        [5, 2, 1],
        [-2, 4, -1]
    ]
    
    # Torus test
    print("--- Torus Test ---")
    torus = Torus(center=[0, 0, 0], major_radius=3, minor_radius=1, axis=[0, 0, 1])
    
    for i, point in enumerate(test_points):
        projected = torus.project_to_surface(point)
        distance = torus.distance_to_surface(point)
        is_inside = torus.is_inside(point)
        
        print(f"Point {i+1}: {point}")
        print(f"  Projection: {projected}")
        print(f"  Distance to surface: {distance:.3f}")
        print(f"  Inside torus: {'Yes' if is_inside else 'No'}")
        print()
    
    # Disk test
    print("--- Disk with Hole Test ---")
    disk = DiskWithHole(center=[0, 0, 0], outer_radius=3, inner_radius=1, normal=[0, 0, 1])
    
    for i, point in enumerate(test_points):
        projected = disk.project_to_surface(point)
        distance = disk.distance_to_surface(point)
        is_on_disk = disk.is_on_disk(point)
        
        print(f"Point {i+1}: {point}")
        print(f"  Projection: {projected}")
        print(f"  Distance to surface: {distance:.3f}")
        print(f"  On disk: {'Yes' if is_on_disk else 'No'}")
        print()
    
    # Thick disk test
    print("--- Thick Disk Test ---")
    thick_disk = ThickDisk(center=[0, 0, 0], outer_radius=6, inner_radius=0.75, thickness=0.12)
    
    for i, point in enumerate(test_points):
        projected, surface_type, distance = thick_disk.project_to_surface(point)
        
        print(f"Point {i+1}: {point}")
        print(f"  Projection: {projected}")
        print(f"  Surface type: {surface_type}")
        print(f"  Distance: {distance:.3f}")
        print()
    
    print(f"Thick disk volume: {thick_disk.get_volume():.3f}")
    print(f"Thick disk surface area: {thick_disk.get_surface_area():.3f}")


def plot_torus_and_disk_projections():
    """
    Visualize torus and disk projections
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Regular torus
    ax1 = fig.add_subplot(231, projection='3d')
    
    torus = Torus(center=[0, 0, 0], major_radius=3, minor_radius=1)
    test_points = [[5, 2, 1], [1, 1, 3], [0, 0, 2], [-2, 4, -1]]
    
    # Generate and plot torus surface
    torus_surface = torus.generate_surface_points(40, 40)
    X, Y, Z = torus_surface
    ax1.plot_surface(X, Y, Z, alpha=0.4, color='lightblue')
    
    # Plot projections
    colors = ['red', 'green', 'orange', 'purple']
    for i, point in enumerate(test_points):
        proj = torus.project_to_surface(point)
        ax1.scatter(*point, color=colors[i], s=100, label=f'Point {i+1}')
        ax1.scatter(*proj, color=colors[i], s=100, marker='^')
        ax1.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                color=colors[i], linestyle='--', alpha=0.7)
    
    ax1.set_title('Regular Torus Projections')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Slanted torus
    ax2 = fig.add_subplot(232, projection='3d')
    
    slanted_torus = Torus(center=[0, 0, 0], major_radius=2.5, minor_radius=1, axis=[1, 1, 1])
    point = [4, 1, 3]
    proj = slanted_torus.project_to_surface(point)
    
    # Generate and plot slanted torus
    surface = slanted_torus.generate_surface_points(30, 30)
    X, Y, Z = surface
    ax2.plot_surface(X, Y, Z, alpha=0.4, color='lightcoral')
    
    ax2.scatter(*point, color='red', s=100, label='Original point')
    ax2.scatter(*proj, color='blue', s=100, label='Projection')
    ax2.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    
    # Plot axis
    axis_end = slanted_torus.center + 3 * slanted_torus.axis
    ax2.quiver(*slanted_torus.center, *(axis_end - slanted_torus.center),
              color='black', arrow_length_ratio=0.1, linewidth=2)
    
    ax2.set_title('Slanted Torus')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Disk with hole
    ax3 = fig.add_subplot(233, projection='3d')
    
    disk = DiskWithHole(center=[0, 0, 0], outer_radius=3, inner_radius=1)
    test_points_disk = [[2, 2, 1], [0.5, 0, 0.5], [4, 1, -0.5], [0, 2.5, 0]]
    
    # Generate and plot disk edges
    outer_edge, inner_edge = disk.generate_edge_points(100)
    ax3.plot(outer_edge[:, 0], outer_edge[:, 1], outer_edge[:, 2], 'b-', linewidth=3, label='Outer edge')
    ax3.plot(inner_edge[:, 0], inner_edge[:, 1], inner_edge[:, 2], 'r-', linewidth=3, label='Inner edge')
    
    # Plot radial lines
    surface_points = disk.generate_surface_points(8, 8)
    for i in range(0, len(surface_points), 8):
        if i + 7 < len(surface_points):
            line_points = surface_points[i:i+8]
            ax3.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'gray', alpha=0.3)
    
    # Plot projections
    for i, point in enumerate(test_points_disk):
        proj = disk.project_to_surface(point)
        ax3.scatter(*point, color=colors[i], s=100, label=f'Point {i+1}')
        ax3.scatter(*proj, color=colors[i], s=100, marker='^')
        ax3.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                color=colors[i], linestyle='--', alpha=0.7)
    
    ax3.set_title('Disk with Hole Projections')
    ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Slanted disk
    ax4 = fig.add_subplot(234, projection='3d')
    
    slanted_disk = DiskWithHole(center=[0, 0, 0], outer_radius=2.5, inner_radius=0.8, normal=[1, 1, 0])
    point = [3, 2, 1]
    proj = slanted_disk.project_to_surface(point)
    
    # Plot slanted disk
    outer_edge, inner_edge = slanted_disk.generate_edge_points(100)
    ax4.plot(outer_edge[:, 0], outer_edge[:, 1], outer_edge[:, 2], 'b-', linewidth=3)
    ax4.plot(inner_edge[:, 0], inner_edge[:, 1], inner_edge[:, 2], 'r-', linewidth=3)
    
    # Plot radial lines
    surface_points = slanted_disk.generate_surface_points(8, 8)
    for i in range(0, len(surface_points), 8):
        if i + 7 < len(surface_points):
            line_points = surface_points[i:i+8]
            ax4.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'gray', alpha=0.3)
    
    ax4.scatter(*point, color='red', s=100, label='Original point')
    ax4.scatter(*proj, color='blue', s=100, label='Projection')
    ax4.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    
    # Display normal vector
    normal_end = slanted_disk.center + 2 * slanted_disk.normal
    ax4.quiver(*slanted_disk.center, *(normal_end - slanted_disk.center),
              color='black', arrow_length_ratio=0.1, linewidth=2)
    
    ax4.set_title('Slanted Disk')
    ax4.legend()
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    # Thick disk (CD model)
    ax5 = fig.add_subplot(235, projection='3d')
    
    thick_disk = ThickDisk(center=[0, 0, 0], outer_radius=3, inner_radius=0.75, thickness=0.3)
    test_points_thick = [[4, 3, 0.5], [1, 1, 0.2], [0.5, 0, -0.1]]
    
    # Plot thick disk structure
    theta = np.linspace(0, 2*np.pi, 50)
    
    # Top and bottom edges
    for z_val in [thick_disk.thickness/2, -thick_disk.thickness/2]:
        for r in [thick_disk.inner_radius, thick_disk.outer_radius]:
            x_circle = r * np.cos(theta)
            y_circle = r * np.sin(theta)
            z_circle = np.full_like(theta, z_val)
            ax5.plot(x_circle, y_circle, z_circle, 'b-', linewidth=2)
    
    # Vertical edges
    for angle in np.linspace(0, 2*np.pi, 12):
        for r in [thick_disk.outer_radius, thick_disk.inner_radius]:
            x_val = r * np.cos(angle)
            y_val = r * np.sin(angle)
            ax5.plot([x_val, x_val], [y_val, y_val],
                    [-thick_disk.thickness/2, thick_disk.thickness/2], 'b-', linewidth=1)
    
    # Plot projections
    for i, point in enumerate(test_points_thick):
        proj, surface_type, distance = thick_disk.project_to_surface(point)
        ax5.scatter(*point, color=colors[i], s=100, label=f'Point {i+1}')
        ax5.scatter(*proj, color=colors[i], s=100, marker='^')
        ax5.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                color=colors[i], linestyle='--', alpha=0.7)
    
    ax5.set_title('Thick Disk (CD Model)')
    ax5.legend()
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    
    # Comparison plot
    ax6 = fig.add_subplot(236, projection='3d')
    
    # Small versions of each shape
    small_torus = Torus(center=[-2, 0, 0], major_radius=1.5, minor_radius=0.5)
    small_disk = DiskWithHole(center=[2, 0, 0], outer_radius=1.5, inner_radius=0.3)
    
    # Generate surfaces
    torus_surf = small_torus.generate_surface_points(20, 20)
    X_t, Y_t, Z_t = torus_surf
    ax6.plot_surface(X_t, Y_t, Z_t, alpha=0.4, color='lightcoral', label='Torus')
    
    outer_edge, inner_edge = small_disk.generate_edge_points(50)
    ax6.plot(outer_edge[:, 0], outer_edge[:, 1], outer_edge[:, 2], 'b-', linewidth=3, label='Disk')
    ax6.plot(inner_edge[:, 0], inner_edge[:, 1], inner_edge[:, 2], 'b-', linewidth=3)
    
    # Test point and projections
    test_point = [0, 2, 1]
    proj_torus = small_torus.project_to_surface(test_point)
    proj_disk = small_disk.project_to_surface(test_point)
    
    ax6.scatter(*test_point, color='black', s=100, label='Test point')
    ax6.scatter(*proj_torus, color='red', s=100, marker='^')
    ax6.scatter(*proj_disk, color='blue', s=100, marker='^')
    ax6.plot([test_point[0], proj_torus[0]], [test_point[1], proj_torus[1]], [test_point[2], proj_torus[2]],
            'r--', alpha=0.7)
    ax6.plot([test_point[0], proj_disk[0]], [test_point[1], proj_disk[1]], [test_point[2], proj_disk[2]],
            'b--', alpha=0.7)
    
    ax6.set_title('Torus vs Disk Comparison')
    ax6.legend()
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()


def compare_geometric_shapes():
    """
    Compare projections across different geometric shapes
    """
    print("=== Geometric Shapes Comparison ===\n")
    
    # Create various shapes
    torus = Torus(center=[0, 0, 0], major_radius=3, minor_radius=1)
    disk = DiskWithHole(center=[0, 0, 0], outer_radius=3, inner_radius=1)
    thick_disk = ThickDisk(center=[0, 0, 0], outer_radius=3, inner_radius=1, thickness=0.2)
    
    # Test point
    test_point = [4, 2, 1.5]
    
    print(f"Test point: {test_point}")
    print()
    
    # Torus projection
    torus_proj = torus.project_to_surface(test_point)
    torus_dist = torus.distance_to_surface(test_point)
    torus_inside = torus.is_inside(test_point)
    
    print("Torus:")
    print(f"  Projection: {torus_proj}")
    print(f"  Distance: {torus_dist:.3f}")
    print(f"  Inside: {'Yes' if torus_inside else 'No'}")
    print()
    
    # Disk projection
    disk_proj = disk.project_to_surface(test_point)
    disk_dist = disk.distance_to_surface(test_point)
    disk_on_surface = disk.is_on_disk(test_point)
    
    print("Disk with Hole:")
    print(f"  Projection: {disk_proj}")
    print(f"  Distance: {disk_dist:.3f}")
    print(f"  On surface: {'Yes' if disk_on_surface else 'No'}")
    print()
    
    # Thick disk projection
    thick_proj, surface_type, thick_dist = thick_disk.project_to_surface(test_point)
    
    print("Thick Disk:")
    print(f"  Projection: {thick_proj}")
    print(f"  Surface type: {surface_type}")
    print(f"  Distance: {thick_dist:.3f}")
    print(f"  Volume: {thick_disk.get_volume():.3f}")
    print(f"  Surface area: {thick_disk.get_surface_area():.3f}")
    print()
    
    # Comparison
    distances = [torus_dist, disk_dist, thick_dist]
    shape_names = ['Torus', 'Disk', 'Thick Disk']
    closest_idx = np.argmin(distances)
    
    print(f"Closest shape: {shape_names[closest_idx]} (distance: {distances[closest_idx]:.3f})")


def create_cd_model_demo():
    """
    Create a realistic CD model demonstration
    """
    print("=== Realistic CD Model Demo ===\n")
    
    # Standard CD dimensions
    cd = ThickDisk(
        center=[0, 0, 0],
        outer_radius=6.0,    # 120mm diameter = 60mm radius
        inner_radius=0.75,   # 15mm diameter = 7.5mm radius
        thickness=0.12       # 1.2mm thickness
    )
    
    print("CD Specifications:")
    print(f"  Outer radius: {cd.outer_radius} cm")
    print(f"  Inner radius: {cd.inner_radius} cm")
    print(f"  Thickness: {cd.thickness} mm")
    print(f"  Volume: {cd.get_volume():.2f} cm³")
    print(f"  Surface area: {cd.get_surface_area():.2f} cm²")
    print()
    
    # Test various points
    test_points = [
        [0, 0, 1],          # Above center
        [3, 0, 0],          # On data area
        [7, 0, 0],          # Beyond edge
        [0.5, 0, 0],        # Near hole
        [6.5, 0, 0.1],      # Near outer edge
        [0, 3, -0.1]        # Side projection
    ]
    
    print("Projection tests:")
    for i, point in enumerate(test_points):
        proj, surface_type, distance = cd.project_to_surface(point)
        print(f"Point {i+1}: {point}")
        print(f"  → Projection: {proj}")
        print(f"  → Surface: {surface_type}")
        print(f"  → Distance: {distance:.4f}")
        print()


if __name__ == "__main__":
    test_torus_and_disk()
    print("\n" + "="*60 + "\n")
    compare_geometric_shapes()
    print("\n" + "="*60 + "\n")
    create_cd_model_demo()
    print("\n" + "="*60 + "\n")
    plot_torus_and_disk_projections()