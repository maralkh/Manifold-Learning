import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Sphere:
    """
    3D Sphere class for geometric operations
    """
    
    def __init__(self, center=(0, 0, 0), radius=1.0):
        """
        Args:
            center: Center point [x, y, z]
            radius: Sphere radius
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        
        if self.radius <= 0:
            raise ValueError("Radius must be positive")
    
    def project_to_surface(self, point):
        """
        Project point to sphere surface
        
        Args:
            point: [x, y, z] coordinates
            
        Returns:
            Projected point on sphere surface
        """
        p = np.array(point, dtype=float)
        vec = p - self.center
        distance = np.linalg.norm(vec)
        
        if distance == 0:
            return self.center + np.array([self.radius, 0, 0])
        
        unit_vec = vec / distance
        projection = self.center + self.radius * unit_vec
        return projection
    
    def distance_to_surface(self, point):
        """
        Calculate distance from point to sphere surface
        """
        p = np.array(point)
        distance_to_center = np.linalg.norm(p - self.center)
        return abs(distance_to_center - self.radius)
    
    def is_inside(self, point):
        """
        Check if point is inside the sphere
        """
        p = np.array(point)
        distance_to_center = np.linalg.norm(p - self.center)
        return distance_to_center <= self.radius
    
    def to_spherical_coordinates(self, point):
        """
        Convert cartesian point to spherical coordinates relative to sphere center
        
        Returns:
            (r, theta, phi) - radius, polar angle, azimuthal angle
        """
        p = np.array(point) - self.center
        x, y, z = p
        
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r > 0 else 0
        phi = np.arctan2(y, x)
        
        return r, theta, phi
    
    def from_spherical_coordinates(self, r, theta, phi):
        """
        Convert spherical coordinates to cartesian coordinates
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return self.center + np.array([x, y, z])
    
    def generate_surface_points(self, n_points=100):
        """
        Generate points on sphere surface
        """
        u = np.linspace(0, 2 * np.pi, int(np.sqrt(n_points)))
        v = np.linspace(0, np.pi, int(np.sqrt(n_points)))
        
        points = []
        for u_val in u:
            for v_val in v:
                x = self.center[0] + self.radius * np.sin(v_val) * np.cos(u_val)
                y = self.center[1] + self.radius * np.sin(v_val) * np.sin(u_val)
                z = self.center[2] + self.radius * np.cos(v_val)
                points.append([x, y, z])
        
        return np.array(points)


class Ellipsoid:
    """
    3D Ellipsoid class for geometric operations
    """
    
    def __init__(self, center=(0, 0, 0), semi_axes=(1, 1, 1)):
        """
        Args:
            center: Center point [x, y, z]
            semi_axes: Semi-axes lengths [a, b, c]
        """
        self.center = np.array(center, dtype=float)
        self.semi_axes = np.array(semi_axes, dtype=float)
        
        if np.any(self.semi_axes <= 0):
            raise ValueError("All semi-axes must be positive")
    
    def project_to_surface(self, point):
        """
        Project point to ellipsoid surface using Newton's method
        """
        p = np.array(point, dtype=float)
        vec = p - self.center
        x, y, z = vec
        a, b, c = self.semi_axes
        
        t = 1.0
        for _ in range(10):  # Newton's method iterations
            denom_x = (a**2 + t)
            denom_y = (b**2 + t)
            denom_z = (c**2 + t)
            
            fx = x**2 / denom_x**2 + y**2 / denom_y**2 + z**2 / denom_z**2 - 1
            
            if abs(fx) < 1e-10:
                break
            
            dfx = -2 * (x**2 / denom_x**3 + y**2 / denom_y**3 + z**2 / denom_z**3)
            
            if abs(dfx) > 1e-10:
                t = t - fx / dfx
        
        proj_x = (a**2 * x) / (a**2 + t)
        proj_y = (b**2 * y) / (b**2 + t)
        proj_z = (c**2 * z) / (c**2 + t)
        
        projection = self.center + np.array([proj_x, proj_y, proj_z])
        return projection
    
    def distance_to_surface(self, point):
        """
        Calculate distance from point to ellipsoid surface
        """
        projected = self.project_to_surface(point)
        return np.linalg.norm(np.array(point) - projected)
    
    def is_inside(self, point):
        """
        Check if point is inside the ellipsoid
        """
        p = np.array(point)
        vec = p - self.center
        x, y, z = vec
        a, b, c = self.semi_axes
        
        return (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1
    
    def generate_surface_points(self, n_points=100):
        """
        Generate points on ellipsoid surface
        """
        u = np.linspace(0, 2 * np.pi, int(np.sqrt(n_points)))
        v = np.linspace(0, np.pi, int(np.sqrt(n_points)))
        
        points = []
        a, b, c = self.semi_axes
        for u_val in u:
            for v_val in v:
                x = self.center[0] + a * np.sin(v_val) * np.cos(u_val)
                y = self.center[1] + b * np.sin(v_val) * np.sin(u_val)
                z = self.center[2] + c * np.cos(v_val)
                points.append([x, y, z])
        
        return np.array(points)


class Cylinder:
    """
    3D Cylinder class for geometric operations
    """
    
    def __init__(self, center=(0, 0, 0), axis_direction=(0, 0, 1), radius=1.0, height=2.0):
        """
        Args:
            center: Base center point [x, y, z]
            axis_direction: Cylinder axis direction [x, y, z]
            radius: Cylinder radius
            height: Cylinder height
        """
        self.center = np.array(center, dtype=float)
        self.axis_direction = np.array(axis_direction, dtype=float)
        self.axis_direction = self.axis_direction / np.linalg.norm(self.axis_direction)
        self.radius = float(radius)
        self.height = float(height)
        
        if self.radius <= 0 or self.height <= 0:
            raise ValueError("Radius and height must be positive")
    
    def project_to_surface(self, point):
        """
        Project point to cylinder surface
        """
        p = np.array(point, dtype=float)
        vec = p - self.center
        
        # Project onto axis
        proj_on_axis = np.dot(vec, self.axis_direction)
        proj_on_axis = max(0, min(self.height, proj_on_axis))
        axis_point = self.center + proj_on_axis * self.axis_direction
        
        # Perpendicular component
        perp_vec = p - axis_point
        perp_distance = np.linalg.norm(perp_vec)
        
        if perp_distance == 0:
            # Handle special case: point on axis
            perp_vec = np.array([1, 0, 0])
            if abs(np.dot(perp_vec, self.axis_direction)) > 0.9:
                perp_vec = np.array([0, 1, 0])
            perp_vec = perp_vec - np.dot(perp_vec, self.axis_direction) * self.axis_direction
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
        else:
            perp_vec = perp_vec / perp_distance
        
        projection = axis_point + self.radius * perp_vec
        return projection
    
    def distance_to_surface(self, point):
        """
        Calculate distance from point to cylinder surface
        """
        projected = self.project_to_surface(point)
        return np.linalg.norm(np.array(point) - projected)
    
    def is_inside(self, point):
        """
        Check if point is inside the cylinder
        """
        p = np.array(point)
        vec = p - self.center
        
        # Check height constraint
        proj_on_axis = np.dot(vec, self.axis_direction)
        if proj_on_axis < 0 or proj_on_axis > self.height:
            return False
        
        # Check radius constraint
        axis_point = self.center + proj_on_axis * self.axis_direction
        perp_vec = p - axis_point
        perp_distance = np.linalg.norm(perp_vec)
        
        return perp_distance <= self.radius
    
    def generate_surface_points(self, n_points=100):
        """
        Generate points on cylinder surface (including caps)
        """
        # Side surface
        theta = np.linspace(0, 2*np.pi, int(np.sqrt(n_points)))
        z_vals = np.linspace(0, self.height, int(np.sqrt(n_points)))
        
        points = []
        
        # Generate perpendicular vectors to axis
        if abs(self.axis_direction[0]) < 0.9:
            u = np.cross(self.axis_direction, [1, 0, 0])
        else:
            u = np.cross(self.axis_direction, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(self.axis_direction, u)
        
        # Side surface points
        for t in theta:
            for z in z_vals:
                radial_vec = self.radius * (np.cos(t) * u + np.sin(t) * v)
                point = self.center + z * self.axis_direction + radial_vec
                points.append(point)
        
        return np.array(points)


def test_geometric_shapes():
    """Test all geometric shape classes"""
    
    print("=== Geometric Shapes Test ===\n")
    
    # Test points
    test_points = [
        [3, 3, 3],
        [1, 1, 1],
        [0, 0, 0],
        [2, -1, 1.5]
    ]
    
    # Sphere test
    print("--- Sphere Test ---")
    sphere = Sphere(center=[0, 0, 0], radius=2)
    
    for i, point in enumerate(test_points):
        projected = sphere.project_to_surface(point)
        distance = sphere.distance_to_surface(point)
        is_inside = sphere.is_inside(point)
        
        print(f"Point {i+1}: {point}")
        print(f"  Projection: {projected}")
        print(f"  Distance to surface: {distance:.3f}")
        print(f"  Inside sphere: {'Yes' if is_inside else 'No'}")
        print()
    
    # Ellipsoid test
    print("--- Ellipsoid Test ---")
    ellipsoid = Ellipsoid(center=[0, 0, 0], semi_axes=[3, 2, 1])
    
    for i, point in enumerate(test_points):
        projected = ellipsoid.project_to_surface(point)
        distance = ellipsoid.distance_to_surface(point)
        is_inside = ellipsoid.is_inside(point)
        
        print(f"Point {i+1}: {point}")
        print(f"  Projection: {projected}")
        print(f"  Distance to surface: {distance:.3f}")
        print(f"  Inside ellipsoid: {'Yes' if is_inside else 'No'}")
        print()
    
    # Cylinder test
    print("--- Cylinder Test ---")
    cylinder = Cylinder(center=[0, 0, 0], axis_direction=[0, 0, 1], radius=1.5, height=4)
    
    for i, point in enumerate(test_points):
        projected = cylinder.project_to_surface(point)
        distance = cylinder.distance_to_surface(point)
        is_inside = cylinder.is_inside(point)
        
        print(f"Point {i+1}: {point}")
        print(f"  Projection: {projected}")
        print(f"  Distance to surface: {distance:.3f}")
        print(f"  Inside cylinder: {'Yes' if is_inside else 'No'}")
        print()


def plot_3d_projections():
    """
    Visualize 3D projections using class-based approach
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Sphere projection
    ax1 = fig.add_subplot(131, projection='3d')
    
    sphere = Sphere(center=[0, 0, 0], radius=2)
    point = [3, 3, 3]
    proj = sphere.project_to_surface(point)
    
    # Generate sphere surface
    surface_points = sphere.generate_surface_points(400)
    ax1.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
               alpha=0.1, color='blue', s=1)
    
    ax1.scatter(*point, color='red', s=100, label='Original point')
    ax1.scatter(*proj, color='green', s=100, label='Projection')
    ax1.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    ax1.set_title('Projection to Sphere')
    ax1.legend()
    
    # Ellipsoid projection
    ax2 = fig.add_subplot(132, projection='3d')
    
    ellipsoid = Ellipsoid(center=[0, 0, 0], semi_axes=[3, 2, 1])
    point = [4, 3, 2]
    proj = ellipsoid.project_to_surface(point)
    
    # Generate ellipsoid surface
    surface_points = ellipsoid.generate_surface_points(400)
    ax2.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
               alpha=0.1, color='blue', s=1)
    
    ax2.scatter(*point, color='red', s=100, label='Original point')
    ax2.scatter(*proj, color='green', s=100, label='Projection')
    ax2.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    ax2.set_title('Ellipsoid Projection')
    ax2.legend()
    
    # Cylinder projection
    ax3 = fig.add_subplot(133, projection='3d')
    
    cylinder = Cylinder(center=[0, 0, 0], axis_direction=[0, 0, 1], radius=1.5, height=4)
    point = [3, 2, 2]
    proj = cylinder.project_to_surface(point)
    
    # Generate cylinder surface
    surface_points = cylinder.generate_surface_points(400)
    ax3.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
               alpha=0.1, color='blue', s=1)
    
    ax3.scatter(*point, color='red', s=100, label='Original point')
    ax3.scatter(*proj, color='green', s=100, label='Projection')
    ax3.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', alpha=0.7)
    ax3.set_title('Projection to Cylinder')
    ax3.legend()
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_geometric_shapes()
    print("\n" + "="*60 + "\n")
    plot_3d_projections()