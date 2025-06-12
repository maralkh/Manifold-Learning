import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from extended_torus import project_to_torus_extended

class HyperTorus:
    """
    Multi-dimensional torus class (n-torus)
    
    An n-torus is the product of n circles:
    T^n = S^1 × S^1 × ... × S^1 (n times)
    """
    
    def __init__(self, radii, center=None):
        """
        Args:
            radii: List of radii for each dimension
            center: Center of the torus (default: origin)
        """
        self.radii = np.array(radii)
        self.n_dims = len(radii)
        
        if center is None:
            self.center = np.zeros(self.n_dims)
        else:
            self.center = np.array(center)
            
        if len(self.center) != self.n_dims:
            raise ValueError("Number of center dimensions must equal number of radii")
    
    def parametric_point(self, angles):
        """
        Calculate point on torus with given angles
        
        Args:
            angles: List of angles for each dimension (0 to 2π)
        
        Returns:
            Point on torus surface
        """
        angles = np.array(angles)
        if len(angles) != self.n_dims:
            raise ValueError(f"Must provide {self.n_dims} angles, {len(angles)} given")
        
        # Each dimension is a circle with corresponding radius
        point = self.radii * np.array([np.cos(angle) for angle in angles])
        return self.center + point
    
    def distance_to_surface(self, point):
        """
        Distance from point to torus surface
        
        For n-torus, distance is calculated separately for each dimension
        """
        point = np.array(point)
        if len(point) != self.n_dims:
            # If fewer dimensions, pad with zeros
            if len(point) < self.n_dims:
                padded_point = np.zeros(self.n_dims)
                padded_point[:len(point)] = point
                point = padded_point
            else:
                # If more dimensions, take only first dimensions
                point = point[:self.n_dims]
        
        relative_point = point - self.center
        
        # Distance to circle in each dimension
        distances = []
        for i, (coord, radius) in enumerate(zip(relative_point, self.radii)):
            # Distance to circle with given radius
            distance_to_circle = abs(abs(coord) - radius)
            distances.append(distance_to_circle)
        
        # Total distance (L2 norm)
        return np.linalg.norm(distances)
    
    def project_to_surface(self, point):
        """
        Project point onto multi-dimensional torus surface
        """
        point = np.array(point, dtype=float)
        
        # Dimension matching
        if len(point) < self.n_dims:
            padded_point = np.zeros(self.n_dims)
            padded_point[:len(point)] = point
            result_point = padded_point
            should_trim = True
            original_length = len(point)
        elif len(point) > self.n_dims:
            result_point = point.copy()
            should_trim = False
            original_length = len(point)
        else:
            result_point = point.copy()
            should_trim = False
            original_length = len(point)
        
        # Projection onto torus
        relative_point = result_point[:self.n_dims] - self.center
        projected = np.zeros(self.n_dims)
        
        for i, (coord, radius) in enumerate(zip(relative_point, self.radii)):
            if coord == 0:
                # If at center, project to positive point
                projected[i] = radius
            else:
                # Project onto circle
                projected[i] = radius * np.sign(coord)
        
        # Return to original coordinates
        result_point[:self.n_dims] = self.center + projected
        
        if should_trim:
            return result_point[:original_length]
        else:
            return result_point
    
    def is_inside(self, point):
        """
        Check if point is inside the torus
        """
        point = np.array(point)
        if len(point) > self.n_dims:
            point = point[:self.n_dims]
        elif len(point) < self.n_dims:
            padded_point = np.zeros(self.n_dims)
            padded_point[:len(point)] = point
            point = padded_point
        
        relative_point = point - self.center
        
        # For n-torus, point is inside if it's inside the circle in all dimensions
        for coord, radius in zip(relative_point, self.radii):
            if abs(coord) > radius:
                return False
        return True
    
    def generate_surface_points(self, n_points_per_dim=20):
        """
        Generate points on torus surface
        """
        # Generate angle grid
        angles_per_dim = [np.linspace(0, 2*np.pi, n_points_per_dim) for _ in range(self.n_dims)]
        
        # Generate all combinations
        angle_grids = np.meshgrid(*angles_per_dim, indexing='ij')
        
        points = []
        for indices in np.ndindex(*[n_points_per_dim] * self.n_dims):
            angles = [angle_grids[i][indices] for i in range(self.n_dims)]
            point = self.parametric_point(angles)
            points.append(point)
        
        return np.array(points)

def test_hyper_torus():
    """Test multi-dimensional torus"""
    
    print("=== Multi-dimensional Torus Test ===\n")
    
    # 4D torus
    radii_4d = [2, 1.5, 1, 0.8]
    center_4d = [0, 0, 0, 0]
    torus_4d = HyperTorus(radii_4d, center_4d)
    
    # Test different points
    test_points = [
        [3, 2, 1, 0.5],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [5, 3, 2, 1],
        [1, 0.5, 0.2, 0.1]  # Interior point
    ]
    
    print(f"4D Torus with radii {radii_4d}")
    print(f"Center: {center_4d}\n")
    
    for i, point in enumerate(test_points):
        projected = torus_4d.project_to_surface(point)
        distance = np.linalg.norm(np.array(point) - projected)
        is_inside = torus_4d.is_inside(point)
        surface_distance = torus_4d.distance_to_surface(point)
        
        print(f"Point {i+1}: {point}")
        print(f"  Projection: {projected}")
        print(f"  Projection distance: {distance:.3f}")
        print(f"  Distance to surface: {surface_distance:.3f}")
        print(f"  Inside torus: {'Yes' if is_inside else 'No'}")
        print()
    
    # Test with different input dimensions
    print("=== Different Dimensions Test ===")
    
    # 6D point on 4D torus
    point_6d = [2.5, 1.2, 0.8, 0.6, 3.0, 1.5]
    proj_6d = torus_4d.project_to_surface(point_6d)
    
    print(f"6D Point: {point_6d}")
    print(f"Projection on 4D Torus: {proj_6d}")
    
    # 2D point on 4D torus
    point_2d = [3, 2]
    proj_2d = torus_4d.project_to_surface(point_2d)
    
    print(f"2D Point: {point_2d}")
    print(f"Projection on 4D Torus: {proj_2d}")

def visualize_hyper_torus_projections():
    """Visualize different projections of multi-dimensional torus"""
    
    # 4D torus
    radii = [3, 2, 1.5, 1]
    torus = HyperTorus(radii)
    
    # Generate test points
    np.random.seed(42)
    test_points = np.random.randn(15, 4) * 4
    
    # Projections
    projections = [torus.project_to_surface(point) for point in test_points]
    distances = [np.linalg.norm(np.array(p1) - p2) for p1, p2 in zip(test_points, projections)]
    
    # Show 2D cross-sections
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Dimension pairs
    dim_pairs = list(combinations(range(4), 2))
    
    for idx, (dim1, dim2) in enumerate(dim_pairs):
        ax = axes[idx]
        
        # Original points
        original_x = [point[dim1] for point in test_points]
        original_y = [point[dim2] for point in test_points]
        
        # Projected points
        proj_x = [proj[dim1] for proj in projections]
        proj_y = [proj[dim2] for proj in projections]
        
        # Draw circle for this cross-section
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = radii[dim1] * np.cos(theta)
        circle_y = radii[dim2] * np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', alpha=0.5, label=f'Torus (dims {dim1},{dim2})')
        
        # Plot points
        ax.scatter(original_x, original_y, c='red', s=50, alpha=0.7, label='Original points')
        ax.scatter(proj_x, proj_y, c='blue', s=50, alpha=0.7, label='Projections')
        
        # Draw projection lines
        for i in range(len(test_points)):
            ax.plot([original_x[i], proj_x[i]], [original_y[i], proj_y[i]], 
                   'gray', alpha=0.5, linewidth=1)
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_title(f'Cross-section dims {dim1}-{dim2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # General statistics
    print("=== Projection Statistics ===")
    print(f"Average projection distance: {np.mean(distances):.3f}")
    print(f"Maximum distance: {np.max(distances):.3f}")
    print(f"Minimum distance: {np.min(distances):.3f}")

def compare_methods():
    """Compare different methods for high dimensions"""
    
    print("=== Method Comparison ===\n")
    
    # 5D test point
    test_point = [4, 3, 2, 1, 2.5]
    center = [0, 0, 0, 0, 0]
    
    # Method 1: Regular 3D torus (first three dimensions)
    proj_3d = project_to_torus_extended(test_point, center, 2.5, 1.0)
    dist_3d = np.linalg.norm(np.array(test_point) - proj_3d)
    
    # Method 2: 5D hyper-torus
    hyper_torus = HyperTorus([2.5, 2.0, 1.5, 1.2, 1.0], center)
    proj_hyper = hyper_torus.project_to_surface(test_point)
    dist_hyper = np.linalg.norm(np.array(test_point) - proj_hyper)
    
    print(f"Original point: {test_point}")
    print(f"Center: {center}\n")
    
    print("3D Torus method (first dimensions):")
    print(f"  Projection: {proj_3d}")
    print(f"  Distance: {dist_3d:.3f}\n")
    
    print("5D Hyper-torus method:")
    print(f"  Projection: {proj_hyper}")
    print(f"  Distance: {dist_hyper:.3f}\n")
    
    print(f"Hyper-torus is {'better' if dist_hyper < dist_3d else 'worse'} than 3D torus")
    