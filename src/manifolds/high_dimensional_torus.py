import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations


class HighDimensionalTorus:
    """
    Multi-dimensional torus class with various projection strategies
    """
    
    def __init__(self, center=None, major_radius=2.0, minor_radius=1.0, 
                 axis=(0, 0, 1), n_dims=3):
        """
        Args:
            center: n-dimensional center point
            major_radius: Major radius (R) - distance from center to tube center
            minor_radius: Minor radius (r) - tube radius
            axis: 3D torus axis direction
            n_dims: Number of dimensions for the space
        """
        self.n_dims = n_dims
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.axis = np.array(axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)  # Normalize
        
        if center is None:
            self.center = np.zeros(n_dims)
        else:
            self.center = np.array(center, dtype=float)
            if len(self.center) != n_dims:
                # Adjust center dimensions
                if len(self.center) < n_dims:
                    center_extended = np.zeros(n_dims)
                    center_extended[:len(self.center)] = self.center
                    self.center = center_extended
                else:
                    self.center = self.center[:n_dims]
        
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
    
    def _project_3d_subspace(self, point_3d, center_3d):
        """
        Project point in 3D subspace to torus surface
        """
        p = np.array(point_3d, dtype=float)
        c = np.array(center_3d, dtype=float)
        
        # Translate to origin
        p_translated = p - c
        
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
        proj_original = np.dot(rotation_matrix.T, proj_rotated)
        
        # Return to original coordinates
        return c + proj_original
    
    def project_to_surface(self, point, method='first_three', active_dims=None, data_points=None):
        """
        Project high-dimensional point to torus surface using various methods
        
        Args:
            point: n-dimensional point
            method: Projection method
                - 'first_three': Use first three dimensions
                - 'custom_dims': Use specified active dimensions
                - 'best_fit': Try all dimension combinations and pick best
                - 'pca': Use PCA to find best 3D subspace
                - 'random_sample': Random sampling of dimension combinations
            active_dims: List of 3 dimension indices for 'custom_dims' method
            data_points: Additional data for PCA method
        
        Returns:
            Projected point with same dimensionality as input
        """
        point = np.array(point, dtype=float)
        
        # Ensure point has correct dimensionality
        if len(point) < self.n_dims:
            padded_point = np.zeros(self.n_dims)
            padded_point[:len(point)] = point
            point = padded_point
        elif len(point) > self.n_dims:
            point = point[:self.n_dims]
        
        if len(point) < 3:
            raise ValueError(f"Point must have at least 3 dimensions, {len(point)} dimensions provided")
        
        if method == 'first_three':
            return self._project_first_three(point)
        
        elif method == 'custom_dims':
            if active_dims is None:
                active_dims = [0, 1, 2]
            return self._project_custom_dims(point, active_dims)
        
        elif method == 'best_fit':
            return self._project_best_fit(point)
        
        elif method == 'pca':
            return self._project_pca(point, data_points)
        
        elif method == 'random_sample':
            return self._project_random_sample(point)
        
        else:
            raise ValueError("Unknown projection method")
    
    def _project_first_three(self, point):
        """
        Project using first three dimensions
        """
        result = point.copy()
        point_3d = point[:3]
        center_3d = self.center[:3]
        
        proj_3d = self._project_3d_subspace(point_3d, center_3d)
        result[:3] = proj_3d
        
        return result
    
    def _project_custom_dims(self, point, active_dims):
        """
        Project using specified dimensions
        """
        if len(active_dims) != 3:
            raise ValueError("Must specify exactly 3 active dimensions")
        
        if max(active_dims) >= len(point):
            active_dims = list(range(min(3, len(point))))
        
        result = point.copy()
        
        # Extract 3D subspace
        point_3d = point[active_dims]
        center_3d = self.center[active_dims]
        
        # Project in 3D
        proj_3d = self._project_3d_subspace(point_3d, center_3d)
        
        # Put back into result
        result[active_dims] = proj_3d
        
        return result
    
    def _project_best_fit(self, point):
        """
        Try all possible 3D dimension combinations and pick the best
        """
        if len(point) <= 3:
            return self._project_first_three(point)
        
        best_projection = None
        best_distance = float('inf')
        best_dims = None
        
        # Try all combinations of 3 dimensions
        for dims in combinations(range(len(point)), 3):
            proj = self._project_custom_dims(point, list(dims))
            distance = np.linalg.norm(point - proj)
            
            if distance < best_distance:
                best_distance = distance
                best_projection = proj
                best_dims = dims
        
        return best_projection
    
    def _project_pca(self, point, data_points=None):
        """
        Use PCA to find the best 3D subspace
        """
        if len(point) <= 3:
            return self._project_first_three(point)
        
        # If no additional data, fall back to first three dimensions
        if data_points is None:
            return self._project_first_three(point)
        
        try:
            from sklearn.decomposition import PCA
            
            data = np.array(data_points)
            if data.shape[1] != len(point):
                return self._project_first_three(point)
            
            pca = PCA(n_components=3)
            pca.fit(data)
            
            # Project to PCA space
            point_pca = pca.transform(point.reshape(1, -1))[0]
            center_pca = pca.transform(self.center.reshape(1, -1))[0]
            
            # Project in PCA space
            proj_pca = self._project_3d_subspace(point_pca, center_pca)
            
            # Return to original space
            proj_original = pca.inverse_transform(proj_pca.reshape(1, -1))[0]
            
            return proj_original
            
        except ImportError:
            print("Warning: sklearn not available, using first three dimensions")
            return self._project_first_three(point)
    
    def _project_random_sample(self, point, n_samples=10):
        """
        Sample random dimension combinations and pick the best
        """
        if len(point) <= 3:
            return self._project_first_three(point)
        
        best_projection = None
        best_distance = float('inf')
        
        np.random.seed(42)  # For reproducibility
        
        # Always include the first three dimensions
        candidates = [[0, 1, 2]]
        
        # Add random samples
        for _ in range(n_samples - 1):
            dims = np.random.choice(len(point), 3, replace=False)
            candidates.append(sorted(dims))
        
        # Remove duplicates
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        # Test each candidate
        for dims in unique_candidates:
            proj = self._project_custom_dims(point, dims)
            distance = np.linalg.norm(point - proj)
            
            if distance < best_distance:
                best_distance = distance
                best_projection = proj
        
        return best_projection
    
    def distance_to_surface(self, point, method='first_three', **kwargs):
        """
        Calculate distance from point to torus surface
        """
        projected = self.project_to_surface(point, method, **kwargs)
        return np.linalg.norm(np.array(point) - projected)
    
    def compare_projection_methods(self, point, methods=None):
        """
        Compare different projection methods for a given point
        
        Returns:
            Dictionary with method names as keys and (projection, distance) as values
        """
        if methods is None:
            methods = ['first_three', 'best_fit', 'random_sample']
        
        results = {}
        
        for method in methods:
            try:
                if method == 'custom_dims':
                    # Try a few different dimension combinations
                    if len(point) >= 4:
                        dim_combinations = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
                        best_dist = float('inf')
                        best_proj = None
                        for dims in dim_combinations:
                            if max(dims) < len(point):
                                proj = self.project_to_surface(point, method, active_dims=dims)
                                dist = np.linalg.norm(np.array(point) - proj)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_proj = proj
                        results[f'{method}_best'] = (best_proj, best_dist)
                else:
                    proj = self.project_to_surface(point, method)
                    dist = np.linalg.norm(np.array(point) - proj)
                    results[method] = (proj, dist)
            except Exception as e:
                results[method] = (None, float('inf'))
        
        return results
    
    def generate_test_points(self, n_points=20, noise_level=1.0):
        """
        Generate test points for experimentation
        """
        np.random.seed(42)
        points = np.random.randn(n_points, self.n_dims) * noise_level
        
        # Add some structure - points near the torus in first 3 dimensions
        for i in range(n_points):
            # Generate point near torus surface
            u = np.random.uniform(0, 2*np.pi)
            v = np.random.uniform(0, 2*np.pi)
            
            x = (self.major_radius + self.minor_radius * np.cos(v)) * np.cos(u)
            y = (self.major_radius + self.minor_radius * np.cos(v)) * np.sin(u)
            z = self.minor_radius * np.sin(v)
            
            points[i, :3] = [x, y, z] + self.center[:3]
            # Add some noise to other dimensions
            if self.n_dims > 3:
                points[i, 3:] += np.random.randn(self.n_dims - 3) * noise_level * 0.5
        
        return points


def test_high_dimensional_torus():
    """
    Test high-dimensional torus projection methods
    """
    print("=== High-Dimensional Torus Test ===\n")
    
    # Create 5D torus
    torus = HighDimensionalTorus(
        center=[0, 0, 0, 0, 0],
        major_radius=2.5,
        minor_radius=0.8,
        n_dims=5
    )
    
    # Test points
    test_points = [
        [3, 2, 1, 4, 2],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [5, 3, 2, 1, 4],
        [-2, 4, -1, 3, -2]
    ]
    
    print(f"Created {torus.n_dims}D torus with major radius {torus.major_radius}, minor radius {torus.minor_radius}")
    print(f"Center: {torus.center}")
    print()
    
    for i, point in enumerate(test_points):
        print(f"Point {i+1}: {point}")
        
        # Compare different methods
        results = torus.compare_projection_methods(point)
        
        for method, (proj, dist) in results.items():
            if proj is not None:
                print(f"  {method}: distance = {dist:.3f}")
        
        print()
    
    # Detailed analysis for one point
    print("=== Detailed Analysis ===")
    test_point = [4, 3, 2, 5, 1]
    print(f"Test point: {test_point}")
    
    methods_detailed = ['first_three', 'best_fit', 'random_sample']
    
    for method in methods_detailed:
        proj = torus.project_to_surface(test_point, method)
        dist = torus.distance_to_surface(test_point, method)
        
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  Projection: {proj}")
        print(f"  Distance: {dist:.3f}")


def plot_dimension_comparison():
    """
    Visualize comparison of different dimension selection methods
    """
    # Create 4D torus
    torus = HighDimensionalTorus(n_dims=4, major_radius=2.5, minor_radius=0.8)
    
    # Generate test points
    test_points = torus.generate_test_points(n_points=30, noise_level=2.0)
    
    # Test different dimension combinations
    dimension_combinations = [
        [0, 1, 2],  # x, y, z
        [0, 1, 3],  # x, y, w
        [0, 2, 3],  # x, z, w
        [1, 2, 3]   # y, z, w
    ]
    
    distances = {f"dims_{dims}": [] for dims in dimension_combinations}
    distances['best_fit'] = []
    
    # Calculate distances for each method
    for point in test_points:
        # Custom dimension combinations
        for dims in dimension_combinations:
            proj = torus.project_to_surface(point, 'custom_dims', active_dims=dims)
            dist = np.linalg.norm(point - proj)
            distances[f"dims_{dims}"].append(dist)
        
        # Best fit method
        proj_best = torus.project_to_surface(point, 'best_fit')
        dist_best = np.linalg.norm(point - proj_best)
        distances['best_fit'].append(dist_best)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Distance distributions
    plt.subplot(2, 3, 1)
    for i, dims in enumerate(dimension_combinations):
        plt.hist(distances[f"dims_{dims}"], bins=10, alpha=0.6, 
                label=f'Dims {dims}')
    plt.xlabel('Projection Distance')
    plt.ylabel('Count')
    plt.title('Distance Distributions by Dimension Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot comparison
    plt.subplot(2, 3, 2)
    all_distances = [distances[f"dims_{dims}"] for dims in dimension_combinations]
    all_distances.append(distances['best_fit'])
    labels = [f'Dims {dims}' for dims in dimension_combinations] + ['Best Fit']
    
    plt.boxplot(all_distances, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('Projection Distance')
    plt.title('Distance Comparison - Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Average distances
    plt.subplot(2, 3, 3)
    avg_distances = []
    method_names = []
    
    for dims in dimension_combinations:
        avg_dist = np.mean(distances[f"dims_{dims}"])
        avg_distances.append(avg_dist)
        method_names.append(f'Dims {dims}')
    
    avg_distances.append(np.mean(distances['best_fit']))
    method_names.append('Best Fit')
    
    bars = plt.bar(range(len(avg_distances)), avg_distances)
    plt.xticks(range(len(method_names)), method_names, rotation=45)
    plt.ylabel('Average Distance')
    plt.title('Average Projection Distance')
    plt.grid(True, alpha=0.3)
    
    # Color code bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 3D visualization of one example
    ax = plt.subplot(2, 3, 4, projection='3d')
    
    # Take first test point and show projections in 3D
    test_point = test_points[0]
    
    # Project using first three dimensions
    proj_123 = torus.project_to_surface(test_point, 'custom_dims', active_dims=[0, 1, 2])
    
    # Plot original and projected points
    ax.scatter(*test_point[:3], color='red', s=100, label='Original')
    ax.scatter(*proj_123[:3], color='blue', s=100, label='Projection')
    ax.plot([test_point[0], proj_123[0]], 
           [test_point[1], proj_123[1]], 
           [test_point[2], proj_123[2]], 'r--', alpha=0.7)
    
    # Generate torus surface for visualization
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, 2*np.pi, 20)
    U, V = np.meshgrid(u, v)
    
    X = (torus.major_radius + torus.minor_radius * np.cos(V)) * np.cos(U)
    Y = (torus.major_radius + torus.minor_radius * np.cos(V)) * np.sin(U)
    Z = torus.minor_radius * np.sin(V)
    
    ax.plot_surface(X, Y, Z, alpha=0.3, color='lightblue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization (Dims 0,1,2)')
    ax.legend()
    
    # Statistics table
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    stats_text = "Statistics Summary:\n\n"
    for dims in dimension_combinations:
        dists = distances[f"dims_{dims}"]
        stats_text += f"Dims {dims}:\n"
        stats_text += f"  Mean: {np.mean(dists):.3f}\n"
        stats_text += f"  Std:  {np.std(dists):.3f}\n\n"
    
    dists_best = distances['best_fit']
    stats_text += f"Best Fit:\n"
    stats_text += f"  Mean: {np.mean(dists_best):.3f}\n"
    stats_text += f"  Std:  {np.std(dists_best):.3f}\n"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Method comparison scatter
    plt.subplot(2, 3, 6)
    
    first_three_dists = distances['dims_[0, 1, 2]']
    best_fit_dists = distances['best_fit']
    
    plt.scatter(first_three_dists, best_fit_dists, alpha=0.6)
    plt.plot([0, max(max(first_three_dists), max(best_fit_dists))], 
            [0, max(max(first_three_dists), max(best_fit_dists))], 'r--', alpha=0.5)
    plt.xlabel('First Three Dims Distance')
    plt.ylabel('Best Fit Distance')
    plt.title('Method Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("=== Projection Method Comparison Summary ===")
    for dims in dimension_combinations:
        dists = distances[f"dims_{dims}"]
        print(f"Dimensions {dims}: Mean = {np.mean(dists):.3f}, Std = {np.std(dists):.3f}")
    
    dists_best = distances['best_fit']
    print(f"Best Fit Method: Mean = {np.mean(dists_best):.3f}, Std = {np.std(dists_best):.3f}")


def demonstrate_pca_projection():
    """
    Demonstrate PCA-based projection (if sklearn is available)
    """
    print("=== PCA Projection Demonstration ===\n")
    
    try:
        from sklearn.decomposition import PCA
        
        # Create 6D torus
        torus = HighDimensionalTorus(n_dims=6, major_radius=3, minor_radius=1)
        
        # Generate structured data for PCA
        data_points = torus.generate_test_points(n_points=100, noise_level=1.5)
        
        # Test point
        test_point = [4, 3, 2, 1, 5, 2]
        
        print(f"Test point: {test_point}")
        
        # Compare PCA vs other methods
        methods = ['first_three', 'best_fit', 'pca']
        
        for method in methods:
            if method == 'pca':
                proj = torus.project_to_surface(test_point, method, data_points=data_points)
            else:
                proj = torus.project_to_surface(test_point, method)
            
            dist = np.linalg.norm(np.array(test_point) - proj)
            
            print(f"{method.replace('_', ' ').title()}:")
            print(f"  Projection: {proj}")
            print(f"  Distance: {dist:.3f}")
            print()
        
    except ImportError:
        print("sklearn not available - skipping PCA demonstration")


if __name__ == "__main__":
    test_high_dimensional_torus()
    print("\n" + "="*60 + "\n")
    plot_dimension_comparison()
    print("\n" + "="*60 + "\n")
    demonstrate_pca_projection()