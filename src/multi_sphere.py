import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist


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
    distance = np.linalg.norm(np.array(point) - projection)

    return projection, distance

def project_to_multiple_spheres(point, sphere_list, method='closest'):
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

    for sphere in sphere_list:
        proj, dist = project_to_sphere(
            point, 
            sphere['center'], 
            sphere['radius'],
            #sphere.get('axis', [0, 0, 1])
            # point,
            # torus['center'],
            # torus['major_radius'],
            # torus['minor_radius'],
            # torus.get('axis', [0, 0, 1])
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

import random

def create_multiple_spheres(center=[0, 0, 0], radius=1, n=3):
    """
    Create three torus for a trefoil shape
    """
    spheres = []
    rand_range = (-3, 3)
    centers = [(-3,-3,-3), (-3,-3,3), (-3,3,3), (3,3,3), (3,3,-3), (3,-3,3), (3,-3,-3), (-3,3,-3)]
    for i in range(n):
        angle = i *  np.pi / n
        sphere_center = [
            centers[i][0],
            centers[i][1],
            centers[i][2],
        ]

        sphere = {
            'center': sphere_center,
            'radius': radius,
        }
        spheres.append(sphere)
    return spheres
    
def cartesian_from_spherical(r, theta, phi, center):
    """
    spherical to cartesian
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array(center) + np.array([x, y, z])

def plot_multi_sphere_projection():
    """
    Plot multiple torus and projections
    """
    fig = plt.figure(figsize=(20, 12))

    ax2 = fig.add_subplot(232, projection='3d')

    spheres = create_multiple_spheres(n=8, radius=0.5)
    point = [0,0,1]

    colors_spheres = ['red', 'green', 'orange', 'purple', 'brown', 'blue', 'yellow', 'black']
    for i, sphere in enumerate(spheres):
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        X = sphere['center'][0] + sphere['radius'] * np.outer(np.cos(u), np.sin(v))
        Y = sphere['center'][1] + sphere['radius'] * np.outer(np.sin(u), np.sin(v))
        Z = sphere['center'][2] + sphere['radius'] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax2.plot_surface(X, Y, Z, alpha=0.4, color=colors_spheres[i])

    proj, closest_idx = project_to_multiple_spheres(point, spheres, 'closest')

    ax2.scatter(*point, color='red', s=100, label='Point')
    ax2.scatter(*proj, color='blue', s=100, label='Closest Projection')
    ax2.plot([point[0], proj[0]], [point[1], proj[1]], [point[2], proj[2]], 'r--')

    ax2.set_title('Multi Sphere')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.tight_layout()
    plt.show()
