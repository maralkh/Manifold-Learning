import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import random


class SingleSphere:
    """
    Individual sphere for use in multi-sphere systems
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
        Project point to sphere surface and return distance
        
        Args:
            point: [x, y, z] coordinates
            
        Returns:
            (projected_point, distance_to_projection)
        """
        p = np.array(point, dtype=float)
        vec = p - self.center
        distance_to_center = np.linalg.norm(vec)
        
        if distance_to_center == 0:
            projection = self.center + np.array([self.radius, 0, 0])
        else:
            unit_vec = vec / distance_to_center
            projection = self.center + self.radius * unit_vec
        
        distance = np.linalg.norm(p - projection)
        return projection, distance
    
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


class MultiSphere:
    """
    Collection of multiple spheres with various projection methods
    """
    
    def __init__(self, spheres=None):
        """
        Args:
            spheres: List of SingleSphere objects or dict specifications
        """
        self.spheres = []
        
        if spheres is not None:
            for sphere in spheres:
                if isinstance(sphere, SingleSphere):
                    self.spheres.append(sphere)
                elif isinstance(sphere, dict):
                    # Convert dict to SingleSphere
                    center = sphere.get('center', (0, 0, 0))
                    radius = sphere.get('radius', 1.0)
                    self.spheres.append(SingleSphere(center, radius))
                else:
                    raise ValueError("Spheres must be SingleSphere objects or dicts")
    
    def add_sphere(self, center, radius):
        """
        Add a sphere to the collection
        """
        self.spheres.append(SingleSphere(center, radius))
    
    def create_predefined_configuration(self, config_type='cube_corners', radius=1.0, scale=3.0):
        """
        Create predefined sphere configurations
        
        Args:
            config_type: Type of configuration
                - 'cube_corners': 8 spheres at cube corners
                - 'random': Random positions
                - 'line': Spheres in a line
                - 'circle': Spheres in a circle
            radius: Radius for all spheres
            scale: Scale factor for positioning
        """
        self.spheres = []
        
        if config_type == 'cube_corners':
            corners = [
                (-scale, -scale, -scale), (-scale, -scale, scale),
                (-scale, scale, scale), (scale, scale, scale),
                (scale, scale, -scale), (scale, -scale, scale),
                (scale, -scale, -scale), (-scale, scale, -scale)
            ]
            for corner in corners:
                self.add_sphere(corner, radius)
                
        elif config_type == 'random':
            np.random.seed(42)  # For reproducibility
            for _ in range(8):
                center = np.random.uniform(-scale, scale, 3)
                self.add_sphere(center, radius)
                
        elif config_type == 'line':
            for i in range(8):
                center = [i * scale/4 - scale, 0, 0]
                self.add_sphere(center, radius)
                
        elif config_type == 'circle':
            for i in range(8):
                angle = i * 2 * np.pi / 8
                center = [scale * np.cos(angle), scale * np.sin(angle), 0]
                self.add_sphere(center, radius)
        
        else:
            raise ValueError("Unknown configuration type")
    
    def project_to_surface(self, point, method='closest'):
        """
        Project point onto multiple spheres using various methods
        
        Args:
            point: Point to project
            method: Projection method
                - 'closest': Closest projection
                - 'weighted': Weighted average based on inverse distance
                - 'all': Return all projections
                - 'centroid': Centroid of all projections
        
        Returns:
            Depends on method:
            - 'closest': (projection, sphere_index)
            - 'weighted': (weighted_projection, weights)
            - 'all': (projections_list, distances_list)
            - 'centroid': centroid_projection
        """
        if not self.spheres:
            raise ValueError("No spheres in collection")
        
        projections = []
        distances = []
        
        for sphere in self.spheres:
            proj, dist = sphere.project_to_surface(point)
            projections.append(proj)
            distances.append(dist)
        
        if method == 'closest':
            closest_idx = np.argmin(distances)
            return projections[closest_idx], closest_idx
            
        elif method == 'weighted':
            distances = np.array(distances)
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            weights = 1.0 / distances
            weights = weights / np.sum(weights)
            
            weighted_proj = np.zeros(3)
            for i, proj in enumerate(projections):
                weighted_proj += weights[i] * proj
            
            return weighted_proj, weights
            
        elif method == 'all':
            return projections, distances
            
        elif method == 'centroid':
            return np.mean(projections, axis=0)
            
        else:
            raise ValueError("Method must be 'closest', 'weighted', 'all', or 'centroid'")
    
    def distance_to_nearest_surface(self, point):
        """
        Calculate distance to nearest sphere surface
        """
        if not self.spheres:
            return float('inf')
        
        distances = [sphere.distance_to_surface(point) for sphere in self.spheres]
        return min(distances)
    
    def is_inside_any(self, point):
        """
        Check if point is inside any sphere
        """
        return any(sphere.is_inside(point) for sphere in self.spheres)
    
    def get_containing_spheres(self, point):
        """
        Get list of sphere indices that contain the point
        """
        containing = []
        for i, sphere in enumerate(self.spheres):
            if sphere.is_inside(point):
                containing.append(i)
        return containing
    
    def get_sphere_centers(self):
        """
        Get centers of all spheres
        """
        return np.array([sphere.center for sphere in self.spheres])
    
    def get_sphere_radii(self):
        """
        Get radii of all spheres
        """
        return np.array([sphere.radius for sphere in self.spheres])
    
    def generate_all_surface_points(self, n_points_per_sphere=100):
        """
        Generate surface points for all spheres
        """
        all_points = []
        for sphere in self.spheres:
            points = sphere.generate_surface_points(n_points_per_sphere)
            all_points.append(points)
        return all_points


def test_multi_sphere():
    """
    Test multi-sphere projection methods
    """
    print("=== Multi-Sphere Projection Test ===\n")
    
    # Create multi-sphere system
    multi_sphere = MultiSphere()
    multi_sphere.create_predefined_configuration('cube_corners', radius=0.5, scale=3.0)
    
    # Test points
    test_points = [
        [0, 0, 1],
        [1, 1, 1],
        [-2, -2, -2],
        [4, 4, 4],
        [0, 0, 0]
    ]
    
    print(f"Created {len(multi_sphere.spheres)} spheres at cube corners")
    print("Sphere centers:", multi_sphere.get_sphere_centers())
    print("Sphere radii:", multi_sphere.get_sphere_radii())
    print()
    
    for i, point in enumerate(test_points):
        print(f"Point {i+1}: {point}")
        
        # Closest projection
        closest_proj, closest_idx = multi_sphere.project_to_surface(point, 'closest')
        print(f"  Closest projection: {closest_proj} (sphere {closest_idx})")
        
        # Weighted projection
        weighted_proj, weights = multi_sphere.project_to_surface(point, 'weighted')
        print(f"  Weighted projection: {weighted_proj}")
        
        # Centroid projection
        centroid_proj = multi_sphere.project_to_surface(point, 'centroid')
        print(f"  Centroid projection: {centroid_proj}")
        
        # Distance and containment info
        distance = multi_sphere.distance_to_nearest_surface(point)
        is_inside = multi_sphere.is_inside_any(point)
        containing = multi_sphere.get_containing_spheres(point)
        
        print(f"  Distance to nearest surface: {distance:.3f}")
        print(f"  Inside any sphere: {'Yes' if is_inside else 'No'}")
        if containing:
            print(f"  Contained in spheres: {containing}")
        print()


def plot_multi_sphere_projection():
    """
    Visualize multi-sphere system and projections
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots for different methods
    methods = ['closest', 'weighted', 'centroid']
    
    for method_idx, method in enumerate(methods):
        ax = fig.add_subplot(1, 3, method_idx + 1, projection='3d')
        
        # Create multi-sphere system
        multi_sphere = MultiSphere()
        multi_sphere.create_predefined_configuration('cube_corners', radius=0.5, scale=3.0)
        
        # Test point
        point = [0, 0, 1]
        
        # Colors for spheres
        colors_spheres = ['red', 'green', 'orange', 'purple', 'brown', 'blue', 'yellow', 'pink']
        
        # Plot spheres
        for i, sphere in enumerate(multi_sphere.spheres):
            surface_points = sphere.generate_surface_points(400)
            ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
                      alpha=0.3, color=colors_spheres[i % len(colors_spheres)], s=1)
        
        # Get projection based on method
        if method == 'closest':
            proj, closest_idx = multi_sphere.project_to_surface(point, method)
            title_suffix = f" (Sphere {closest_idx})"
        else:
            proj = multi_sphere.project_to_surface(point, method)
            if method == 'weighted':
                proj = proj[0]  # Extract projection from (projection, weights) tuple
            title_suffix = ""
        
        # Plot point and projection
        ax.scatter(*point, color='red', s=100, label='Original Point')
        ax.scatter(*proj, color='blue', s=100, label=f'{method.title()} Projection')
        ax.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--', linewidth=2)
        
        ax.set_title(f'{method.title()} Projection{title_suffix}')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = 4
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.show()


def compare_projection_methods():
    """
    Compare different projection methods
    """
    print("=== Projection Methods Comparison ===\n")
    
    # Create multi-sphere system
    multi_sphere = MultiSphere()
    multi_sphere.create_predefined_configuration('cube_corners', radius=0.5, scale=3.0)
    
    # Test point
    test_point = [0, 0, 1]
    
    print(f"Test point: {test_point}")
    print(f"Number of spheres: {len(multi_sphere.spheres)}")
    print()
    
    # Compare all methods
    methods = ['closest', 'weighted', 'centroid']
    
    for method in methods:
        if method == 'closest':
            proj, idx = multi_sphere.project_to_surface(test_point, method)
            distance = np.linalg.norm(np.array(test_point) - proj)
            print(f"{method.title()} method:")
            print(f"  Projection: {proj}")
            print(f"  Selected sphere: {idx}")
            print(f"  Distance: {distance:.3f}")
            
        elif method == 'weighted':
            proj, weights = multi_sphere.project_to_surface(test_point, method)
            distance = np.linalg.norm(np.array(test_point) - proj)
            print(f"{method.title()} method:")
            print(f"  Projection: {proj}")
            print(f"  Weights: {weights}")
            print(f"  Distance: {distance:.3f}")
            
        else:  # centroid
            proj = multi_sphere.project_to_surface(test_point, method)
            distance = np.linalg.norm(np.array(test_point) - proj)
            print(f"{method.title()} method:")
            print(f"  Projection: {proj}")
            print(f"  Distance: {distance:.3f}")
        
        print()


if __name__ == "__main__":
    test_multi_sphere()
    print("\n" + "="*60 + "\n")
    compare_projection_methods()
    print("\n" + "="*60 + "\n")
    plot_multi_sphere_projection()