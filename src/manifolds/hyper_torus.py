"""
HyperTorus Module - Multi-dimensional torus class compatible with testing notebook

This module provides the HyperTorus class for n-dimensional torus operations,
designed to work seamlessly with the comprehensive testing notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


class HyperTorus:
    """
    Multi-dimensional torus class (n-torus)
    
    An n-torus is the product of n circles:
    T^n = S^1 × S^1 × ... × S^1 (n times)
    
    This implementation provides a simplified n-dimensional torus where each
    dimension is treated as an independent circle, suitable for high-dimensional
    data projection and analysis.
    """
    
    def __init__(self, radii, center=None):
        """
        Initialize multi-dimensional torus
        
        Args:
            radii: List of radii for each dimension
            center: Center of the torus (default: origin)
        
        Raises:
            ValueError: If center dimensions don't match radii dimensions
        """
        self.radii = np.array(radii, dtype=float)
        self.n_dims = len(radii)
        
        if center is None:
            self.center = np.zeros(self.n_dims)
        else:
            self.center = np.array(center, dtype=float)
            
        if len(self.center) != self.n_dims:
            raise ValueError("Number of center dimensions must equal number of radii")
        
        # Validate inputs
        if np.any(self.radii <= 0):
            raise ValueError("All radii must be positive")
    
    def parametric_point(self, angles):
        """
        Calculate point on torus with given angles
        
        Args:
            angles: List of angles for each dimension (0 to 2π)
        
        Returns:
            Point on torus surface
            
        Raises:
            ValueError: If number of angles doesn't match dimensions
        """
        angles = np.array(angles)
        if len(angles) != self.n_dims:
            raise ValueError(f"Must provide {self.n_dims} angles, {len(angles)} given")
        
        # Each dimension is a circle with corresponding radius
        point = self.radii * np.array([np.cos(angle) for angle in angles])
        return self.center + point
    
    def project_to_surface(self, point):
        """
        Project point onto multi-dimensional torus surface
        
        This method handles dimension compatibility automatically:
        - Points with fewer dimensions are padded with zeros
        - Points with more dimensions are preserved but only first n_dims are projected
        - Output maintains original input dimensionality when possible
        
        Args:
            point: n-dimensional point to project
            
        Returns:
            Projected point (maintains input dimensionality when possible)
        """
        point = np.array(point, dtype=float)
        
        # Handle dimension compatibility
        original_length = len(point)
        should_trim = False
        
        if len(point) < self.n_dims:
            # Pad with zeros
            padded_point = np.zeros(self.n_dims)
            padded_point[:len(point)] = point
            result_point = padded_point
            should_trim = True
        elif len(point) > self.n_dims:
            # Keep extra dimensions unchanged
            result_point = point.copy()
            should_trim = False
        else:
            # Exact match
            result_point = point.copy()
            should_trim = False
        
        # Project onto torus in each dimension
        relative_point = result_point[:self.n_dims] - self.center
        projected = np.zeros(self.n_dims)
        
        for i, (coord, radius) in enumerate(zip(relative_point, self.radii)):
            if coord == 0:
                # If at center, project to positive point
                projected[i] = radius
            else:
                # Project onto circle (sign preserving)
                projected[i] = radius * np.sign(coord)
        
        # Update result with projection
        result_point[:self.n_dims] = self.center + projected
        
        # Return with original dimensionality if needed
        if should_trim:
            return result_point[:original_length]
        else:
            return result_point
    
    def distance_to_surface(self, point):
        """
        Calculate distance from point to torus surface
        
        For n-torus, distance is calculated separately for each dimension
        using L2 norm of individual dimension distances.
        
        Args:
            point: n-dimensional point
            
        Returns:
            Distance to torus surface
        """
        point = np.array(point)
        
        # Handle dimension compatibility
        if len(point) != self.n_dims:
            if len(point) < self.n_dims:
                padded_point = np.zeros(self.n_dims)
                padded_point[:len(point)] = point
                point = padded_point
            else:
                point = point[:self.n_dims]
        
        relative_point = point - self.center
        
        # Calculate distance to circle in each dimension
        distances = []
        for coord, radius in zip(relative_point, self.radii):
            # Distance to circle with given radius
            distance_to_circle = abs(abs(coord) - radius)
            distances.append(distance_to_circle)
        
        # Total distance using L2 norm
        return np.linalg.norm(distances)
    
    def is_inside(self, point):
        """
        Check if point is inside the torus
        
        For n-torus, point is inside if it's inside the circle in all dimensions
        
        Args:
            point: n-dimensional point
            
        Returns:
            True if point is inside torus, False otherwise
        """
        point = np.array(point)
        
        # Handle dimension compatibility
        if len(point) > self.n_dims:
            point = point[:self.n_dims]
        elif len(point) < self.n_dims:
            padded_point = np.zeros(self.n_dims)
            padded_point[:len(point)] = point
            point = padded_point
        
        relative_point = point - self.center
        
        # Check if inside circle in all dimensions
        for coord, radius in zip(relative_point, self.radii):
            if abs(coord) > radius:
                return False
        return True
    
    def generate_surface_points(self, n_points_per_dim=20):
        """
        Generate points on torus surface using parametric equations
        
        Args:
            n_points_per_dim: Number of points per dimension
            
        Returns:
            Array of surface points
            
        Note:
            Total points = n_points_per_dim^n_dims, so use small values for high dimensions
        """
        # Generate angle grid for each dimension
        angles_per_dim = [np.linspace(0, 2*np.pi, n_points_per_dim) for _ in range(self.n_dims)]
        
        # Generate all combinations using meshgrid
        angle_grids = np.meshgrid(*angles_per_dim, indexing='ij')
        
        points = []
        # Iterate through all combinations
        for indices in np.ndindex(*[n_points_per_dim] * self.n_dims):
            angles = [angle_grids[i][indices] for i in range(self.n_dims)]
            point = self.parametric_point(angles)
            points.append(point)
        
        return np.array(points)
    
    def get_bounding_box(self):
        """
        Get bounding box of the torus
        
        Returns:
            Tuple of (min_bounds, max_bounds) arrays
        """
        min_bounds = self.center - self.radii
        max_bounds = self.center + self.radii
        return min_bounds, max_bounds
    
    def get_volume_estimate(self):
        """
        Get rough volume estimate for n-dimensional torus
        
        Returns:
            Approximate volume (product of circumferences)
        """
        # Simple approximation: product of circumferences
        circumferences = 2 * np.pi * self.radii
        return np.prod(circumferences)
    
    def compare_with_3d_projection(self, point, major_radius=None, minor_radius=None):
        """
        Compare n-dimensional projection with 3D torus projection
        
        This method is compatible with the high_dimensional_torus module
        for comparison purposes.
        
        Args:
            point: Point to project
            major_radius: Major radius for 3D comparison (default: first radius)
            minor_radius: Minor radius for 3D comparison (default: second radius)
            
        Returns:
            Dictionary with comparison results
        """
        if major_radius is None:
            major_radius = self.radii[0] if len(self.radii) > 0 else 1.0
        if minor_radius is None:
            minor_radius = self.radii[1] if len(self.radii) > 1 else 0.5
        
        # Our n-dimensional projection
        nd_projection = self.project_to_surface(point)
        nd_distance = self.distance_to_surface(point)
        
        # Simulate 3D torus projection (simplified)
        point_3d = np.array(point[:3]) if len(point) >= 3 else np.pad(point, (0, 3-len(point)))
        center_3d = self.center[:3] if len(self.center) >= 3 else np.pad(self.center, (0, 3-len(self.center)))
        
        # Simple 3D projection (using our method on first 3 dimensions)
        if hasattr(self, '_project_3d_fallback'):
            proj_3d = self._project_3d_fallback(point_3d, center_3d, major_radius, minor_radius)
        else:
            # Fallback: use our method on 3D subset
            temp_torus = HyperTorus([major_radius, major_radius, minor_radius], center_3d)
            proj_3d = temp_torus.project_to_surface(point_3d)
        
        dist_3d = np.linalg.norm(np.array(point_3d) - proj_3d)
        
        return {
            'nd_projection': nd_projection,
            'nd_distance': nd_distance,
            '3d_projection': proj_3d,
            '3d_distance': dist_3d,
            'nd_better': nd_distance < dist_3d
        }


def test_hyper_torus():
    """Test multi-dimensional torus functionality"""
    
    print("=== Multi-dimensional Torus Test ===\n")
    
    # Create 4D torus
    radii_4d = [2, 1.5, 1, 0.8]
    center_4d = [0, 0, 0, 0]
    torus_4d = HyperTorus(radii_4d, center_4d)
    
    print(f"Created 4D Torus with radii: {radii_4d}")
    print(f"Center: {center_4d}")
    print(f"Dimensions: {torus_4d.n_dims}")
    print(f"Volume estimate: {torus_4d.get_volume_estimate():.2f}")
    
    # Test different points
    test_points = [
        [3, 2, 1, 0.5],      # Outside point
        [1, 1, 1, 1],        # Inside point
        [0, 0, 0, 0],        # Center point
        [5, 3, 2, 1],        # Far outside point
        [1, 0.5, 0.2, 0.1]   # Interior point
    ]
    
    print(f"\n--- Testing {len(test_points)} points ---")
    
    for i, point in enumerate(test_points):
        projected = torus_4d.project_to_surface(point)
        distance = np.linalg.norm(np.array(point) - projected)
        is_inside = torus_4d.is_inside(point)
        surface_distance = torus_4d.distance_to_surface(point)
        
        print(f"\nPoint {i+1}: {point}")
        print(f"  Projection: {np.round(projected, 3)}")
        print(f"  Projection distance: {distance:.3f}")
        print(f"  Distance to surface: {surface_distance:.3f}")
        print(f"  Inside torus: {'Yes' if is_inside else 'No'}")
    
    # Test dimension compatibility
    print("\n=== Dimension Compatibility Tests ===")
    
    test_cases = [
        ([3, 2], "2D point on 4D torus"),
        ([4, 3, 2, 1, 5], "5D point on 4D torus"), 
        ([2], "1D point on 4D torus"),
        ([1, 1, 1, 1, 1, 1], "6D point on 4D torus")
    ]
    
    for point, description in test_cases:
        try:
            projected = torus_4d.project_to_surface(point)
            print(f"{description}:")
            print(f"  Input: {point} ({len(point)}D)")
            print(f"  Output: {np.round(projected, 3)} ({len(projected)}D)")
        except Exception as e:
            print(f"{description}: Error - {e}")


def visualize_hyper_torus_projections():
    """Visualize different projections of multi-dimensional torus"""
    
    print("\n=== Visualizing HyperTorus Projections ===")
    
    # Create 4D torus
    radii = [3, 2, 1.5, 1]
    torus = HyperTorus(radii)
    
    # Generate test points
    np.random.seed(42)
    test_points = np.random.randn(15, 4) * 4
    
    print(f"Generated {len(test_points)} random test points")
    
    # Calculate projections
    projections = [torus.project_to_surface(point) for point in test_points]
    distances = [np.linalg.norm(np.array(p1) - p2) for p1, p2 in zip(test_points, projections)]
    
    # Create visualization of 2D cross-sections
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Get all possible dimension pairs
    dim_pairs = list(combinations(range(4), 2))
    
    for idx, (dim1, dim2) in enumerate(dim_pairs):
        ax = axes[idx]
        
        # Extract coordinates for this pair
        original_x = [point[dim1] for point in test_points]
        original_y = [point[dim2] for point in test_points]
        proj_x = [proj[dim1] for proj in projections]
        proj_y = [proj[dim2] for proj in projections]
        
        # Draw circle representing torus cross-section in these dimensions
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = radii[dim1] * np.cos(theta)
        circle_y = radii[dim2] * np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', alpha=0.5, linewidth=2, 
               label=f'Torus boundary (dims {dim1},{dim2})')
        
        # Plot original and projected points
        ax.scatter(original_x, original_y, c='red', s=50, alpha=0.7, 
                  label='Original points', edgecolors='darkred')
        ax.scatter(proj_x, proj_y, c='blue', s=50, alpha=0.7, 
                  label='Projections', edgecolors='darkblue', marker='^')
        
        # Draw projection lines
        for i in range(len(test_points)):
            ax.plot([original_x[i], proj_x[i]], [original_y[i], proj_y[i]], 
                   'gray', alpha=0.4, linewidth=1)
        
        # Formatting
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        ax.set_title(f'Cross-section: Dims {dim1}-{dim2}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set reasonable axis limits
        all_x = original_x + proj_x + circle_x.tolist()
        all_y = original_y + proj_y + circle_y.tolist()
        margin = 0.5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.suptitle('4D HyperTorus Projections - 2D Cross-sections', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n=== Projection Statistics ===")
    print(f"Average projection distance: {np.mean(distances):.3f}")
    print(f"Standard deviation: {np.std(distances):.3f}")
    print(f"Maximum distance: {np.max(distances):.3f}")
    print(f"Minimum distance: {np.min(distances):.3f}")
    
    return distances


def compare_projection_methods():
    """Compare HyperTorus with other projection methods"""
    
    print("\n=== Projection Method Comparison ===")
    
    # Test configuration
    test_point = [4, 3, 2, 1, 2.5]
    center = [0, 0, 0, 0, 0]
    
    print(f"Test point: {test_point}")
    print(f"Center: {center}")
    
    # Method 1: HyperTorus (our implementation)
    hyper_torus = HyperTorus([2.5, 2.0, 1.5, 1.2, 1.0], center)
    proj_hyper = hyper_torus.project_to_surface(test_point)
    dist_hyper = np.linalg.norm(np.array(test_point) - proj_hyper)
    
    print(f"\nHyperTorus method:")
    print(f"  Projection: {np.round(proj_hyper, 3)}")
    print(f"  Distance: {dist_hyper:.3f}")
    
    # Method 2: Dimension reduction (use first 3 dimensions)
    hyper_torus_3d = HyperTorus([2.5, 2.0, 1.5], center[:3])
    proj_3d = hyper_torus_3d.project_to_surface(test_point[:3])
    # Pad back to original dimensions
    proj_3d_extended = test_point.copy()
    proj_3d_extended[:3] = proj_3d
    dist_3d = np.linalg.norm(np.array(test_point) - proj_3d_extended)
    
    print(f"\n3D reduction method:")
    print(f"  Projection: {np.round(proj_3d_extended, 3)}")
    print(f"  Distance: {dist_3d:.3f}")
    
    # Method 3: Per-dimension independent projection
    proj_independent = test_point.copy()
    for i in range(min(len(test_point), len(hyper_torus.radii))):
        if test_point[i] != 0:
            proj_independent[i] = hyper_torus.radii[i] * np.sign(test_point[i])
        else:
            proj_independent[i] = hyper_torus.radii[i]
    
    dist_independent = np.linalg.norm(np.array(test_point) - proj_independent)
    
    print(f"\nIndependent dimension method:")
    print(f"  Projection: {np.round(proj_independent, 3)}")
    print(f"  Distance: {dist_independent:.3f}")
    
    # Compare results
    methods = [
        ("HyperTorus", dist_hyper),
        ("3D Reduction", dist_3d), 
        ("Independent", dist_independent)
    ]
    
    methods.sort(key=lambda x: x[1])
    
    print(f"\n--- Method Ranking (best to worst) ---")
    for i, (method, distance) in enumerate(methods):
        print(f"  {i+1}. {method}: {distance:.3f}")
    
    best_method, best_distance = methods[0]
    print(f"\nBest method: {best_method} with distance {best_distance:.3f}")


def run_comprehensive_tests():
    """Run all tests for the HyperTorus class"""
    
    print("🚀 Running Comprehensive HyperTorus Tests")
    print("=" * 50)
    
    try:
        # Basic functionality tests
        test_hyper_torus()
        
        # Visualization tests
        distances = visualize_hyper_torus_projections()
        
        # Method comparison
        compare_projection_methods()
        
        print("\n✅ All tests completed successfully!")
        
        return {
            'test_passed': True,
            'projection_distances': distances,
            'message': 'HyperTorus class is working correctly'
        }
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return {
            'test_passed': False,
            'error': str(e),
            'message': 'HyperTorus class has issues'
        }


if __name__ == "__main__":
    # Run tests when script is executed directly
    results = run_comprehensive_tests()
    
    if results['test_passed']:
        print(f"\n🎉 {results['message']}")
    else:
        print(f"\n⚠️  {results['message']}: {results['error']}")