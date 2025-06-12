import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist


def project_to_torus(point, center, major_radius, minor_radius, axis=None):
    """Project a point onto a torus"""
    p = np.array(point) - np.array(center)

    # If axis of rotation exists, first transform the point into local space
    if axis is not None:
        # Simplification: Assume torus is in the xy plane
        pass

    # Distance from the z-axis
    rho = np.sqrt(p[0]**2 + p[1]**2)

    if rho < 1e-10:
        # Point is on the axis
        phi = 0
        rho_proj = major_radius
    else:
        phi = np.arctan2(p[1], p[0])
        rho_proj = major_radius

    # Calculate the closest point on the minor circle
    z = p[2]
    dist_from_circle = np.sqrt((rho - major_radius)**2 + z**2)

    if dist_from_circle < 1e-10:
        theta = 0
    else:
        theta = np.arctan2(z, rho - major_radius)

    # Projected point on the torus
    x_proj = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y_proj = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z_proj = minor_radius * np.sin(theta)

    projected_point = np.array([x_proj, y_proj, z_proj]) + np.array(center)
    distance = np.linalg.norm(np.array(point) - projected_point)

    return projected_point, distance

def torus_parametric(u, v, center, major_radius, minor_radius, axis=None):
    """
    Parametric equation of a torus with rotation capability

    Parameters:
    u, v: Grid parameters
    center: Torus center [x, y, z]
    major_radius: Major radius
    minor_radius: Minor radius
    axis: Rotation axis (optional)
    """
    # Calculate standard torus points
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)

    # Combine points
    points = np.array([x, y, z])

    # Apply rotation (if axis is provided)
    if axis is not None:
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)

        # Rotation matrix (simplified)
        if not np.allclose(axis, [0, 0, 1]):
            # Calculate rotation angle
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, axis), -1, 1))

                # Apply rotation to each point
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)

                # Rodrigues' rotation formula (simplified)
                # Reshape point data for vectoruszed rotation
                points_flat = points.reshape(3, -1)
                rotated_flat = (points_flat * cos_angle +
                                np.cross(rotation_axis[:, np.newaxis], points_flat, axis=0) * sin_angle +
                                rotation_axis[:, np.newaxis] * np.dot(rotation_axis, points_flat) * (1 - cos_angle))
                points = rotated_flat.reshape(points.shape)


    # Add center with correct broadcasting
    c = np.array(center)
    # Ensure center has the same number of dimensions as points for broadcasting
    # points shape is (3, num_v, num_u)
    c_reshaped = c.reshape(3, 1, 1) # Reshape to (3, 1, 1)
    points = points + c_reshaped

    return points[0], points[1], points[2]  # Return X, Y, Z

def project_to_multiple_torus(point, torus_list, method='closest'):
    """
    Projection onto multiple torus

    Args:
        point: Point to project
        torus_list: List of torus, each a dict with keys:
                  'center', 'major_radius', 'minor_radius', 'axis'
        method: Method for selecting the torus
               - 'closest': Closest projection
               - 'weighted': Weighted average
               - 'all': All projections

    Returns:
        Projection or list of projections
    """
    projections = []
    distances = []

    for torus in torus_list:
        proj, dist = project_to_torus(
            point,
            torus['center'],
            torus['major_radius'],
            torus['minor_radius'],
            torus.get('axis', [0, 0, 1])
        )
        projections.append(proj)
        print(point, proj)
        print(np.array(point), proj)
        #distances.append(np.linalg.norm(np.array(point) - proj))
        distances.append(dist)

    if method == 'closest':
        # Closest projection
        closest_idx = np.argmin(distances)
        return projections[closest_idx], closest_idx

    elif method == 'weighted':
        # Weighted average based on inverse distance
        distances = np.array(distances)
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / distances
        weights = weights / np.sum(weights)

        weighted_proj = np.zeros(3)
        for i, proj in enumerate(projections):
            weighted_proj += weights[i] * proj[0]

        return weighted_proj, weights

    elif method == 'all':
        return projections, distances

    else:
        raise ValueError("method must be one of 'closest', 'weighted', or 'all'")

def create_infinity_shape_torus(center=[0, 0, 0], radius=3, minor_radius=0.8, separation=4):
    """
    Create two torus for an infinity shape
    """
    torus1 = {
        'center': [center[0] - separation/2, center[1], center[2]],
        'major_radius': radius,
        'minor_radius': minor_radius,
        'axis': [0, 0, 1]
    }

    torus2 = {
        'center': [center[0] + separation/2, center[1], center[2]],
        'major_radius': radius,
        'minor_radius': minor_radius,
        'axis': [0, 0, 1]
    }

    return [torus1, torus2]

def create_trefoil_torus(center=[0, 0, 0], radius=2, minor_radius=0.5):
    """
    Create three torus for a trefoil shape
    """
    torus = []
    for i in range(3):
        angle = i * 2 * np.pi / 3
        torus_center = [
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2]
        ]

        # Different axis for each torus
        axis_angle = angle + np.pi/6
        axis = [np.cos(axis_angle), np.sin(axis_angle), 0.5]

        torus = {
            'center': torus_center,
            'major_radius': radius * 0.7,
            'minor_radius': minor_radius,
            'axis': axis
        }
        torus.append(torus)

    return torus

def create_chain_torus(num_torus=4, center=[0, 0, 0], spacing=3, minor_radius=0.6):
    """
    Create a chain of torus
    """
    torus = []

    for i in range(num_torus):
        # Even torus horizontal, odd vertical
        if i % 2 == 0:
            torus_center = [center[0] + i * spacing, center[1], center[2]]
            axis = [0, 0, 1]  # Horizontal
            major_radius = 1.5
        else:
            torus_center = [center[0] + i * spacing, center[1], center[2]]
            axis = [0, 1, 0]  # Vertical
            major_radius = 1.5

        torus = {
            'center': torus_center,
            'major_radius': major_radius,
            'minor_radius': minor_radius,
            'axis': axis
        }
        torus.append(torus)

    return torus

def create_spiral_torus(num_torus=5, center=[0, 0, 0], radius=4, height_step=2, minor_radius=0.5):
    """
    Create a spiral of torus
    """
    torus = []

    for i in range(num_torus):
        angle = i * 2 * np.pi / 3  # Rotation
        height = center[2] + i * height_step

        torus_center = [
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            height
        ]

        # Axis tangent to the spiral
        axis = [-np.sin(angle), np.cos(angle), 0.3]

        torus = {
            'center': torus_center,
            'major_radius': 1.8,
            'minor_radius': minor_radius,
            'axis': axis
        }
        torus.append(torus)

    return torus

def plot_multi_torus_projection():
    """
    Plot multiple torus and projections
    """
    fig = plt.figure(figsize=(20, 12))

    # Infinity shape
    ax1 = fig.add_subplot(231, projection='3d')

    infinity_torus = create_infinity_shape_torus()
    test_points = [
        [0, 4, 1],    # Top
        [0, -4, 1],   # Bottom
        [6, 0, 1],    # Right
        [-6, 0, 1],   # Left
        [0, 0, 3]     # Middle
    ]

    # Plot torus
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, 2*np.pi, 20)
    U, V = np.meshgrid(u, v)

    colors_torus = ['lightblue', 'lightgreen']
    for i, torus in enumerate(infinity_torus):
        points = torus_parametric(U, V, torus['center'],
                                torus['major_radius'], torus['minor_radius'])
        X, Y, Z = points
        ax1.plot_surface(X, Y, Z, alpha=0.4, color=colors_torus[i])

    # Project points
    colors_points = ['red', 'green', 'orange', 'purple', 'brown']
    for i, point in enumerate(test_points):
        proj, closest_idx = project_to_multiple_torus(point, infinity_torus, 'closest')

        ax1.scatter(*point, color=colors_points[i], s=100, label=f'Point {i+1}')
        ax1.scatter(*proj, color=colors_points[i], s=100, marker='^')
        ax1.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                color=colors_points[i], linestyle='--', alpha=0.7)

    ax1.set_title('Infinity Shape (Two torus)')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Trefoil shape
    ax2 = fig.add_subplot(232, projection='3d')

    trefoil_torus = create_trefoil_torus()
    point = [0, 0, 3]

    colors_trefoil = ['lightcoral', 'lightblue', 'lightgreen']
    for i, torus in enumerate(trefoil_torus):
        points = torus_parametric(U, V, torus['center'],
                                torus['major_radius'], torus['minor_radius'],
                                torus['axis'])
        X, Y, Z = points
        ax2.plot_surface(X, Y, Z, alpha=0.4, color=colors_trefoil[i])

    proj, closest_idx = project_to_multiple_torus(point, trefoil_torus, 'closest')

    ax2.scatter(*point, color='red', s=100, label='Point')
    ax2.scatter(*proj, color='blue', s=100, label='Closest Projection')
    ax2.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--')

    # Weighted projection
    proj_weighted, weights = project_to_multiple_torus(point, trefoil_torus, 'weighted')
    ax2.scatter(*proj_weighted, color='yellow', s=100, label='Weighted Projection')
    ax2.plot([point[0], proj_weighted[0]], [point[1], proj_weighted[1]],
            [point[2], proj_weighted[2]], 'y--')

    ax2.set_title('Trefoil (Three torus)')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Chain of torus
    ax3 = fig.add_subplot(233, projection='3d')

    chain_torus = create_chain_torus(4)
    point = [6, 2, 2]

    colors_chain = ['red', 'green', 'blue', 'orange']
    for i, torus in enumerate(chain_torus):
        points = torus_parametric(U, V, torus['center'],
                                torus['major_radius'], torus['minor_radius'],
                                torus['axis'])
        X, Y, Z = points
        ax3.plot_surface(X, Y, Z, alpha=0.4, color=colors_chain[i % len(colors_chain)])

    proj, closest_idx = project_to_multiple_torus(point, chain_torus, 'closest')

    ax3.scatter(*point, color='black', s=100, label='Point')
    ax3.scatter(*proj, color='red', s=100, label=f'Projection (Torus {closest_idx+1})')
    ax3.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'k--')

    ax3.set_title('Chain of torus')
    ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Spiral of torus
    ax4 = fig.add_subplot(234, projection='3d')

    spiral_torus = create_spiral_torus(4)
    point = [2, 2, 5]

    for i, torus in enumerate(spiral_torus):
        points = torus_parametric(U, V, torus['center'],
                                torus['major_radius'], torus['minor_radius'],
                                torus['axis'])
        X, Y, Z = points
        ax4.plot_surface(X, Y, Z, alpha=0.4, color=plt.cm.rainbow(i/4))

    proj, closest_idx = project_to_multiple_torus(point, spiral_torus, 'closest')

    ax4.scatter(*point, color='black', s=100, label='Point')
    ax4.scatter(*proj, color='red', s=100, label=f'Projection (Torus {closest_idx+1})')
    ax4.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'k--')

    ax4.set_title('Spiral of torus')
    ax4.legend()
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    # Compare projection methods
    ax5 = fig.add_subplot(235, projection='3d')

    infinity_torus = create_infinity_shape_torus()
    point = [0, 0, 2]  # Middle point

    # Plot torus
    for i, torus in enumerate(infinity_torus):
        points = torus_parametric(U, V, torus['center'],
                                torus['major_radius'], torus['minor_radius'])
        X, Y, Z = points
        ax5.plot_surface(X, Y, Z, alpha=0.4, color=colors_torus[i])

    # Different projection methods
    proj_closest, _ = project_to_multiple_torus(point, infinity_torus, 'closest')
    proj_weighted, _ = project_to_multiple_torus(point, infinity_torus, 'weighted')
    all_projs, _ = project_to_multiple_torus(point, infinity_torus, 'all')

    ax5.scatter(*point, color='black', s=100, label='Original Point')
    ax5.scatter(*proj_closest, color='red', s=100, label='Closest')
    ax5.scatter(*proj_weighted, color='blue', s=100, label='Weighted')

    # Show all projections
    for i, proj in enumerate(all_projs):
        ax5.scatter(*proj, color='gray', s=50, alpha=0.7)
        ax5.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]],
                'gray', linestyle=':', alpha=0.5)

    ax5.plot([point[0], proj_closest[0]], [point[1], proj_closest[1]],
            [point[2], proj_closest[2]], 'r--', linewidth=2)
    ax5.plot([point[0], proj_weighted[0]], [point[1], proj_weighted[1]],
            [point[2], proj_weighted[2]], 'b--', linewidth=2)

    ax5.set_title('Comparison of Projection Methods')
    ax5.legend()
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')

    # Statistical chart
    ax6 = fig.add_subplot(236)

    # Test multiple points and compare methods
    test_points_stats = np.random.randn(50, 3) * 3
    distances_closest = []
    distances_weighted = []

    for point in test_points_stats:
        proj_c, _ = project_to_multiple_torus(point, infinity_torus, 'closest')
        proj_w, _ = project_to_multiple_torus(point, infinity_torus, 'weighted')

        distances_closest.append(np.linalg.norm(np.array(point) - proj_c))
        distances_weighted.append(np.linalg.norm(np.array(point) - proj_w))

    ax6.hist(distances_closest, alpha=0.7, label='Closest', bins=15, color='red')
    ax6.hist(distances_weighted, alpha=0.7, label='Weighted', bins=15, color='blue')
    ax6.set_xlabel('Projection Distance')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of Projection Distances')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analyze_multi_torus_projection():
    """
    Quantitative analysis of multi-torus projection
    """
    print("=== Multi-Torus Projection Analysis ===\n")

    # Infinity shape
    infinity_torus = create_infinity_shape_torus()

    test_points = [
        [0, 0, 0],     # Center
        [0, 4, 1],     # Top
        [4, 0, 1],     # Right
        [-3, -3, 2]    # Corner
    ]

    print("Infinity Shape (Two torus):")
    print("-" * 40)

    for i, point in enumerate(test_points):
        print(f"\nPoint {i+1}: {point}")

        # Closest method
        proj_closest, closest_idx = project_to_multiple_torus(point, infinity_torus, 'closest')
        dist_closest = np.linalg.norm(np.array(point) - proj_closest)
        print(f"  Closest: Torus {closest_idx+1}, Distance {dist_closest:.3f}")
        print(f"             Projection: [{proj_closest[0]:.2f}, {proj_closest[1]:.2f}, {proj_closest[2]:.2f}]")

        # Weighted method
        proj_weighted, weights = project_to_multiple_torus(point, infinity_torus, 'weighted')
        dist_weighted = np.linalg.norm(np.array(point) - proj_weighted)
        print(f"  Weighted: Distance {dist_weighted:.3f}")
        print(f"           Projection: [{proj_weighted[0]:.2f}, {proj_weighted[1]:.2f}, {proj_weighted[2]:.2f}]")
        print(f"           Weights: [{weights[0]:.3f}, {weights[1]:.3f}]")

        # All projections
        all_projs, all_dists = project_to_multiple_torus(point, infinity_torus, 'all')
        print(f"  All Distances: {[f'{d:.3f}' for d in all_dists]}")


import unittest
import numpy as np

# from multi_torus_projection import *

class TestMultiTorusProjection(unittest.TestCase):

    def setUp(self):
        """Default values for tests"""
        self.tolerance = 1e-6
        self.infinity_torus = create_infinity_shape_torus()
        self.trefoil_torus = create_trefoil_torus()

    def test_create_infinity_shape_torus(self):
        """Test creating infinity shape"""
        torus = create_infinity_shape_torus()

        self.assertEqual(len(torus), 2)
        self.assertEqual(torus[0]['major_radius'], 2)
        self.assertEqual(torus[1]['major_radius'], 2)

        # torus should be at different locations
        self.assertNotEqual(torus[0]['center'][0], torus[1]['center'][0])

    def test_create_trefoil_torus(self):
        """Test creating trefoil shape"""
        torus = create_trefoil_torus()

        self.assertEqual(len(torus), 3)

        # Each torus should have a different center
        centers = [torus['center'] for torus in torus]
        for i in range(3):
            for j in range(i+1, 3):
                self.assertFalse(np.allclose(centers[i], centers[j]))

    def test_project_to_multiple_torus_closest(self):
        """Test closest projection"""
        point = [0, 4, 1]  # Point near the first torus

        proj, closest_idx = project_to_multiple_torus(point, self.infinity_torus, 'closest')

        # Should be a 3-element array
        self.assertEqual(len(proj), 3)
        self.assertIsInstance(closest_idx, (int, np.integer))
        self.assertIn(closest_idx, [0, 1])

    def test_project_to_multiple_torus_weighted(self):
        """Test weighted projection"""
        point = [0, 0, 2]  # Middle point

        proj, weights = project_to_multiple_torus(point, self.infinity_torus, 'weighted')

        # Check output
        self.assertEqual(len(proj), 3)
        self.assertEqual(len(weights), 2)

        # Sum of weights should be 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)

        # Weights should be positive
        self.assertTrue(all(w >= 0 for w in weights))

    def test_project_to_multiple_torus_all(self):
        """Test all projections"""
        point = [2, 2, 1]

        projections, distances = project_to_multiple_torus(point, self.infinity_torus, 'all')

        # Should have as many projections as torus
        self.assertEqual(len(projections), 2)
        self.assertEqual(len(distances), 2)

        # Distances should be non-negative
        self.assertTrue(all(d >= 0 for d in distances))

    def test_project_to_multiple_torus_invalid_method(self):
        """Test invalid method"""
        point = [1, 1, 1]

        with self.assertRaises(ValueError):
            project_to_multiple_torus(point, self.infinity_torus, 'invalid_method')

    def test_create_chain_torus(self):
        """Test chain of torus"""
        chain = create_chain_torus(4)

        self.assertEqual(len(chain), 4)

        # Even and odd torus should have different axes
        for i in range(4):
            if i % 2 == 0:
                self.assertTrue(np.allclose(chain[i]['axis'], [0, 1, 0]))
            else:
                self.assertTrue(np.allclose(chain[i]['axis'], [1, 0, 0]))

    def test_create_spiral_torus(self):
        """Test spiral of torus"""
        spiral = create_spiral_torus(5)

        self.assertEqual(len(spiral), 5)

        # Heights of torus should be increasing
        heights = [torus['center'][2] for torus in spiral]
        for i in range(1, 5):
            self.assertGreater(heights[i], heights[i-1])

    def test_consistency_single_vs_multi(self):
        """Check consistency of single vs multi-torus projection"""
        # Single torus
        single_torus = self.infinity_torus[0]
        point = [3, 1, 1]

        # Direct projection
        direct_proj, _ = project_to_torus(
            point,
            single_torus['center'],
            single_torus['major_radius'],
            single_torus['minor_radius'],
            single_torus['axis']
        )

        # Projection via multi-torus (only one torus)
        multi_proj, _ = project_to_multiple_torus(point, [single_torus], 'closest')

        # Should be the same
        np.testing.assert_allclose(direct_proj, multi_proj, atol=self.tolerance)

    def test_weighted_projection_properties(self):
        """Test properties of weighted projection"""
        # Point exactly in the middle of two torus
        center1 = np.array(self.infinity_torus[0]['center'])
        center2 = np.array(self.infinity_torus[1]['center'])
        midpoint = (center1 + center2) / 2
        midpoint[2] = 1  # Specific height

        proj, weights = project_to_multiple_torus(midpoint, self.infinity_torus, 'weighted')

        # Weights should be approximately equal (because the point is in the middle)
        self.assertAlmostEqual(weights[0], weights[1], places=1)

    def test_edge_cases(self):
        """Test edge cases"""
        # Point on the center of one of the torus
        center_point = self.infinity_torus[0]['center']

        proj, closest_idx = project_to_multiple_torus(center_point, self.infinity_torus, 'closest')

        # The first torus should be selected
        self.assertEqual(closest_idx, 0)

        # Point very far away
        far_point = [100, 100, 100]

        proj_far, _ = project_to_multiple_torus(far_point, self.infinity_torus, 'closest')

        # Projection should not be invalid
        self.assertFalse(np.any(np.isnan(proj_far)))
        self.assertFalse(np.any(np.isinf(proj_far)))

    def test_projection_distance_minimization(self):
        """Test distance minimization"""
        point = [1, 2, 1]

        # Closest projection
        proj_closest, closest_idx = project_to_multiple_torus(point, self.infinity_torus, 'closest')

        # All projections
        all_projs, all_dists = project_to_multiple_torus(point, self.infinity_torus, 'all')

        # The closest should have the minimum distance
        min_distance = min(all_dists)
        closest_distance = all_dists[closest_idx]

        self.assertAlmostEqual(closest_distance, min_distance, places=6)

class TestMultiTorusIntegration(unittest.TestCase):
    """Integration tests"""

    def test_infinity_shape_symmetry(self):
        """Test symmetry of infinity shape"""
        torus = create_infinity_shape_torus()

        # Symmetric points
        point1 = [1, 2, 1]
        point2 = [-1, 2, 1]  # Symmetric with respect to the y-axis

        proj1, _ = project_to_multiple_torus(point1, torus, 'closest')
        proj2, _ = project_to_multiple_torus(point2, torus, 'closest')

        # Projections should be symmetric
        self.assertAlmostEqual(proj1[0], -proj2[0], places=3)
        self.assertAlmostEqual(proj1[1], proj2[1], places=3)
        self.assertAlmostEqual(proj1[2], proj2[2], places=3)

    def test_performance_with_many_torus(self):
        """Test performance with many torus"""
        # Create many torus
        many_torus = []
        for i in range(10):
            torus = {
                'center': [i*2, 0, 0],
                'major_radius': 1,
                'minor_radius': 0.3,
                'axis': [0, 0, 1]
            }
            many_torus.append(torus)

        point = [5, 3, 1]

        # Should run without error
        try:
            proj, _ = project_to_multiple_torus(point, many_torus, 'closest')
            self.assertEqual(len(proj), 3)
        except Exception as e:
            self.fail(f"Error processing many torus: {e}")

    def test_degenerate_cases(self):
        """Test degenerate cases"""
        # Torus with zero radius (point)
        degenerate_torus = {
            'center': [0, 0, 0],
            'major_radius': 0.001,  # Very small
            'minor_radius': 0.001,
            'axis': [0, 0, 1]
        }

        point = [1, 1, 1]

        try:
            proj, _ = project_to_multiple_torus(point, [degenerate_torus], 'closest')
            # Should give a reasonable result
            self.assertFalse(np.any(np.isnan(proj)))
        except:
            # If it gives an error, it's acceptable
            pass

def run_performance_benchmark():
    """Performance benchmark"""
    import time

    print("=== Performance Benchmark ===")

    # Test with infinity shape
    infinity_torus = create_infinity_shape_torus()

    # Test with many points
    points = np.random.randn(1000, 3) * 5

    start_time = time.time()
    for point in points:
        project_to_multiple_torus(point, infinity_torus, 'closest')
    closest_time = time.time() - start_time

    start_time = time.time()
    for point in points:
        project_to_multiple_torus(point, infinity_torus, 'weighted')
    weighted_time = time.time() - start_time

    print(f"Closest projection for 1000 points: {closest_time:.4f} seconds")
    print(f"Weighted projection for 1000 points: {weighted_time:.4f} seconds")

    # Test with many torus
    many_torus = []
    for i in range(10):
        torus = {
            'center': [i*3, 0, 0],
            'major_radius': 2,
            'minor_radius': 0.5,
            'axis': [0, 0, 1]
        }
        many_torus.append(torus)

    start_time = time.time()
    for point in points[:100]:  # Fewer points for a faster test
        project_to_multiple_torus(point, many_torus, 'closest')
    many_torus_time = time.time() - start_time

    print(f"Projection of 100 points onto 10 torus: {many_torus_time:.4f} seconds")


def create_infinity_shape_torus(center=[0, 0, 0], radius=3, minor_radius=0.8, separation=4):
    """
    Create two torus for an infinity shape
    """
    torus1 = {
        'center': [center[0] - separation/2, center[1], center[2]],
        'major_radius': radius,
        'minor_radius': minor_radius,
        'axis': [0, 0, 1]
    }

    torus2 = {
        'center': [center[0] + separation/2, center[1], center[2]],
        'major_radius': radius,
        'minor_radius': minor_radius,
        'axis': [0, 0, 1]
    }

    return [torus1, torus2]


def create_trefoil_torus():
    """Create three torus as a trefoil shape"""
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    radius = 4

    torus = []
    for angle in angles:
        center = [radius * np.cos(angle), radius * np.sin(angle), 0]
        axis = [np.cos(angle + np.pi/2), np.sin(angle + np.pi/2), 0]

        torus.append({
            'center': center,
            'major_radius': 1.5,
            'minor_radius': 0.5,
            'axis': axis
        })

    return torus


def create_chain_torus(n):
    """Create a chain of n torus"""
    torus = []
    spacing = 4

    for i in range(n):
        center = [i * spacing, 0, 0]
        axis = [0, 1, 0] if i % 2 == 0 else [1, 0, 0]

        torus.append({
            'center': center,
            'major_radius': 1.5,
            'minor_radius': 0.6,
            'axis': axis
        })

    return torus


def create_spiral_torus(n):
    """Create a spiral of n torus"""
    torus = []

    for i in range(n):
        t = i * 2 * np.pi / n
        radius = 3

        center = [radius * np.cos(t), radius * np.sin(t), i * 1.5]
        axis = [np.cos(t + np.pi/2), np.sin(t + np.pi/2), 0.3]

        torus.append({
            'center': center,
            'major_radius': 1.2,
            'minor_radius': 0.4,
            'axis': axis
        })

    return torus


"""
Practical example for using multi-torus projection
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def demo_infinity_projection():
    """Show projection onto infinity shape"""
    print("=== Infinity Shape Demo ===")

    # Create two torus for the infinity shape
    torus1 = {
        'center': [-2, 0, 0],
        'major_radius': 2,
        'minor_radius': 0.8,
        'axis': [0, 0, 1]
    }

    torus2 = {
        'center': [2, 0, 0],
        'major_radius': 2,
        'minor_radius': 0.8,
        'axis': [0, 0, 1]
    }

    infinity_torus = [torus1, torus2]

    # Test points
    test_points = [
        [0, 0, 2],     # Middle top
        [0, 3, 0],     # Side
        [-4, 2, 1],    # Near left torus
        [4, -2, 1],    # Near right torus
        [0, 0, 0]      # Exact center
    ]

    print("Test points and projection results:")
    print("-" * 60)

    for i, point in enumerate(test_points):
        print(f"\nPoint {i+1}: [{point[0]:4.1f}, {point[1]:4.1f}, {point[2]:4.1f}]")

        # Closest projection
        proj_closest, closest_idx = project_to_multiple_torus(point, infinity_torus, 'closest')
        dist_closest = np.linalg.norm(np.array(point) - proj_closest)

        print(f"  Closest → Torus {closest_idx+1}")
        print(f"              [{proj_closest[0]:5.2f}, {proj_closest[1]:5.2f}, {proj_closest[2]:5.2f}]")
        print(f"              Distance: {dist_closest:.3f}")

        # Weighted projection
        proj_weighted, weights = project_to_multiple_torus(point, infinity_torus, 'weighted')
        dist_weighted = np.linalg.norm(np.array(point) - proj_weighted)

        print(f"  Weighted → [{proj_weighted[0]:5.2f}, {proj_weighted[1]:5.2f}, {proj_weighted[2]:5.2f}]")
        print(f"              Distance: {dist_weighted:.3f}")
        print(f"              Weights: [{weights[0]:.2f}, {weights[1]:.2f}]")

def demo_creative_shapes():
    """Show creative shapes"""
    print("\n" + "="*60)
    print("=== Creative Shapes ===")

    # Heart shape (two torus with different angles)
    heart_torus = [
        {
            'center': [-1.5, 1, 0],
            'major_radius': 1.5,
            'minor_radius': 0.6,
            'axis': [0.3, 0, 1]  # Slight tilt
        },
        {
            'center': [1.5, 1, 0],
            'major_radius': 1.5,
            'minor_radius': 0.6,
            'axis': [-0.3, 0, 1]  # Opposite tilt
        }
    ]

    print("\nHeart Shape:")
    test_point = [0, -1, 1]
    proj, closest_idx = project_to_multiple_torus(test_point, heart_torus, 'closest')
    print(f"Point {test_point} → Projection {[round(x, 2) for x in proj]} (Torus {closest_idx+1})")

    # Ring chain
    chain_torus = []
    num_links = 4
    for i in range(num_links):
        angle = i * np.pi / 2  # 90 degree spacing

        torus = {
            'center': [3*np.cos(angle), 3*np.sin(angle), 0],
            'major_radius': 1.2,
            'minor_radius': 0.4,
            'axis': [0, 0, 1] if i % 2 == 0 else [1, 0, 0]  # Alternating horizontal/vertical
        }
        chain_torus.append(torus)

    print(f"\n{num_links}-Ring Chain:")
    test_point = [0, 0, 2]
    proj, closest_idx = project_to_multiple_torus(test_point, chain_torus, 'closest')
    print(f"Point {test_point} → Projection {[round(x, 2) for x in proj]} (Ring {closest_idx+1})")

def compare_projection_methods():
    """Compare different projection methods"""
    print("\n" + "="*60)
    print("=== Comparison of Projection Methods ===")

    # Simple infinity shape
    infinity_torus = create_infinity_shape_torus(radius=2, minor_radius=0.6, separation=3)

    # Different points for testing
    test_scenarios = [
        {"point": [0, 0, 1], "desc": "Middle point"},
        {"point": [-3, 1, 0], "desc": "Near left torus"},
        {"point": [3, -1, 0], "desc": "Near right torus"},
        {"point": [0, 4, 2], "desc": "Far point"},
        {"point": [1.5, 0, 0], "desc": "Between two torus"}
    ]

    print(f"{'Scenario':<15} {'Closest':<10} {'Weighted':<10} {'Difference':<8}")
    print("-" * 50)

    for scenario in test_scenarios:
        point = scenario["point"]
        desc = scenario["desc"]

        # Closest projection
        proj_closest, _ = project_to_multiple_torus(point, infinity_torus, 'closest')
        dist_closest = np.linalg.norm(np.array(point) - proj_closest)

        # Weighted projection
        proj_weighted, _ = project_to_multiple_torus(point, infinity_torus, 'weighted')
        dist_weighted = np.linalg.norm(np.array(point) - proj_weighted)

        diff = abs(dist_closest - dist_weighted)

        print(f"{desc:<15} {dist_closest:<10.3f} {dist_weighted:<10.3f} {diff:<8.3f}")

def interactive_demo():
    """Interactive demo (for plotting graphs)"""
    print("\n" + "="*60)
    print("=== Interactive Demo ===")

    # Create infinity shape
    infinity_torus = create_infinity_shape_torus()

    # Grid of points for testing
    x = np.linspace(-6, 6, 20)
    y = np.linspace(-4, 4, 15)
    z = [1]  # Constant height

    closest_choices = []
    distance_ratios = []

    for zi in z:
        for yi in y:
            for xi in x:
                point = [xi, yi, zi]

                # Closest projection
                proj_c, closest_idx = project_to_multiple_torus(point, infinity_torus, 'closest')
                dist_c = np.linalg.norm(np.array(point) - proj_c)

                # Weighted projection
                proj_w, _ = project_to_multiple_torus(point, infinity_torus, 'weighted')
                dist_w = np.linalg.norm(np.array(point) - proj_w)

                closest_choices.append(closest_idx)
                if dist_w > 0:
                    distance_ratios.append(dist_c / dist_w)
                else:
                    distance_ratios.append(1.0)

    # Statistics of choices
    torus1_count = closest_choices.count(0)
    torus2_count = closest_choices.count(1)
    total = len(closest_choices)

    print(f"From {total} test points:")
    print(f"  Torus 1 (Left): {torus1_count} points ({torus1_count/total*100:.1f}%)")
    print(f"  Torus 2 (Right): {torus2_count} points ({torus2_count/total*100:.1f}%)")

    # Statistics of distance ratios
    avg_ratio = np.mean(distance_ratios)
    print(f"\nAverage distance ratio (Closest/Weighted): {avg_ratio:.3f}")

    if avg_ratio < 1:
        print("→ Closest method usually has a smaller distance")
    elif avg_ratio > 1:
        print("→ Weighted method usually has a smaller distance")
    else:
        print("→ Both methods perform similarly")

def practical_application_demo():
    """Practical application example"""
    print("\n" + "="*60)
    print("=== Practical Application: Path Planning ===")

    # Assume torus are obstacles and we want to find the closest safe point
    obstacles = create_infinity_shape_torus(radius=2, minor_radius=1, separation=4)

    # Desired path (straight line)
    start_point = [-5, 0, 1]
    end_point = [5, 0, 1]

    # Divide the path into multiple points
    path_points = []
    num_segments = 20

    for i in range(num_segments + 1):
        t = i / num_segments
        point = [
            start_point[0] + t * (end_point[0] - start_point[0]),
            start_point[1] + t * (end_point[1] - start_point[1]),
            start_point[2] + t * (end_point[2] - start_point[2])
        ]
        path_points.append(point)

    print(f"Path from {start_point} to {end_point}")
    print("Path correction to avoid obstacles:")
    print("-" * 45)

    corrected_path = []
    total_deviation = 0

    for i, point in enumerate(path_points):
        # Find the closest safe point
        safe_point, closest_obstacle = project_to_multiple_torus(point, obstacles, 'closest')

        # Calculate deviation
        deviation = np.linalg.norm(np.array(point) - safe_point)
        total_deviation += deviation

        corrected_path.append(safe_point)

        if deviation > 0.1:  # Only significant deviations
            print(f"Point {i:2d}: Deviation {deviation:.2f} (Obstacle {closest_obstacle+1})")

    print(f"\nTotal deviation: {total_deviation:.2f}")
    print(f"Average deviation: {total_deviation/len(path_points):.3f}")
