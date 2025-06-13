import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.cluster import KMeans
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ManifoldFitter:
    """
    Class for fitting real-world data onto different manifolds
    """

    def __init__(self):
        self.fitted_params = {}
        self.fit_quality = {}
        self.scalers = {}

    def load_sample_datasets(self):
        """Load sample datasets"""
        datasets = {}

        # Iris dataset
        iris = load_iris()
        datasets['iris'] = {
            'data': iris.data,
            'target': iris.target,
            'feature_names': iris.feature_names,
            'description': 'Iris flower - 4 features, 3 classes'
        }

        # Digits dataset
        digits = load_digits()
        datasets['digits'] = {
            'data': digits.data,
            'target': digits.target,
            'feature_names': [f'pixel_{i}' for i in range(64)],
            'description': 'Handwritten digits - 64 features, 10 classes'
        }

        # Wine dataset
        wine = load_wine()
        datasets['wine'] = {
            'data': wine.data,
            'target': wine.target,
            'feature_names': wine.feature_names,
            'description': 'Wine - 13 features, 3 classes'
        }

        return datasets

    def preprocess_data(self, data, method='standard'):
        """Preprocess data"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return data, None

        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def estimate_manifold_type(self, data):
        """Estimate the type of the best manifold for the data"""
        scores = {}

        # Convert to 3D if necessary
        if data.shape[1] > 3:
            pca = PCA(n_components=3)
            data_3d = pca.fit_transform(data)
        else:
            data_3d = data

        # Score for each manifold
        scores['sphere'] = self._score_sphere_fit(data_3d)
        scores['torus'] = self._score_torus_fit(data_3d)
        scores['disk'] = self._score_disk_fit(data_3d)
        scores['plane'] = self._score_plane_fit(data_3d)
        scores['swiss_roll'] = self._score_swiss_roll_fit(data_3d)

        # Best manifold
        best_manifold = max(scores.keys(), key=lambda k: scores[k])

        return best_manifold, scores

    def _score_sphere_fit(self, data):
        """Score for fitting on a sphere"""
        center = np.mean(data, axis=0)
        distances = [np.linalg.norm(point - center) for point in data]

        # Uniformity of distances = better for sphere
        std_distances = np.std(distances)
        mean_distance = np.mean(distances)

        if mean_distance == 0:
            return 0

        # Higher score = more uniform distances
        score = 1 / (1 + std_distances / mean_distance)
        return score

    def _score_torus_fit(self, data):
        """Score for fitting on a torus"""
        # Check if the data has a ring shape
        center = np.mean(data, axis=0)
        centered_data = data - center

        # Radial distance in the xy plane
        radial_distances = np.sqrt(centered_data[:, 0]**2 + centered_data[:, 1]**2)

        # Distribution of distances - if it has two peaks, likely a torus
        hist, _ = np.histogram(radial_distances, bins=20)

        # Find the number of peaks
        peaks = 0
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks += 1

        # Higher score for multiple peaks
        score = min(peaks / 2.0, 1.0)
        return score

    def _score_disk_fit(self, data):
        """Score for fitting on a disk"""
        # Check if the data is approximately in a plane
        pca = PCA(n_components=3)
        pca.fit(data)

        # If the third component is very small, likely a disk
        explained_variance = pca.explained_variance_ratio_

        if len(explained_variance) >= 3:
            # Higher score if the third dimension is less important
            score = 1 - explained_variance[2]
        else:
            score = 0.5

        return score

    def _score_plane_fit(self, data):
        """Score for fitting on a plane"""
        return self._score_disk_fit(data) * 0.8  # Similar to disk but less

    def _score_swiss_roll_fit(self, data):
        """Score for fitting on a Swiss Roll"""
        # Check for nonlinearity in data
        # Use the ratio of PCA to Isomap
        try:
            pca = PCA(n_components=2)
            isomap = Isomap(n_components=2, n_neighbors=10)

            pca_result = pca.fit_transform(data)
            isomap_result = isomap.fit_transform(data)

            # If the results are very different, the data is nonlinear
            correlation = np.corrcoef(pca_result.flatten(), isomap_result.flatten())[0, 1]
            score = 1 - abs(correlation)

        except:
            score = 0.3

        return score

    def fit_to_sphere(self, data, optimize=True):
        """Fit data to a sphere"""
        # Initial estimation of center and radius
        center_init = np.mean(data, axis=0)
        distances = [np.linalg.norm(point - center_init) for point in data]
        radius_init = np.mean(distances)

        if not optimize:
            return center_init, radius_init, self._sphere_fit_error(data, center_init, radius_init)

        # Optimization
        def objective(params):
            center = params[:3]
            radius = params[3]
            return self._sphere_fit_error(data, center, radius)

        # Bounds for optimization
        bounds = [(-10, 10)] * 3 + [(0.1, 20)]
        initial_guess = np.append(center_init, radius_init)

        result = minimize(objective, initial_guess, bounds=bounds)

        center_opt = result.x[:3]
        radius_opt = result.x[3]
        error = result.fun

        return center_opt, radius_opt, error

    def _sphere_fit_error(self, data, center, radius):
        """Calculate the error of fitting on a sphere"""
        distances = [np.linalg.norm(point - center) for point in data]
        errors = [(d - radius)**2 for d in distances]
        return np.mean(errors)

    def fit_to_torus(self, data, optimize=True):
        """Fit data to a torus"""
        # Initial estimation of parameters
        center_init = np.mean(data, axis=0)

        # Estimate major radius
        centered_data = data - center_init
        radial_distances = np.sqrt(centered_data[:, 0]**2 + centered_data[:, 1]**2)
        R_init = np.mean(radial_distances)

        # Estimate minor radius
        r_init = np.std(centered_data[:, 2])

        if not optimize:
            return center_init, R_init, r_init, self._torus_fit_error(data, center_init, R_init, r_init)

        # Optimization
        def objective(params):
            center = params[:3]
            R = params[3]
            r = params[4]
            return self._torus_fit_error(data, center, R, r)

        bounds = [(-10, 10)] * 3 + [(0.5, 10), (0.1, 5)]
        initial_guess = np.append(center_init, [R_init, r_init])

        result = minimize(objective, initial_guess, bounds=bounds)

        center_opt = result.x[:3]
        R_opt = result.x[3]
        r_opt = result.x[4]
        error = result.fun

        return center_opt, R_opt, r_opt, error

    def _torus_fit_error(self, data, center, R, r):
        """Calculate the error of fitting on a torus"""
        errors = []
        for point in data:
            p = point - center
            rho = np.sqrt(p[0]**2 + p[1]**2)
            distance_to_center_circle = np.sqrt((rho - R)**2 + p[2]**2)
            error = (distance_to_center_circle - r)**2
            errors.append(error)
        return np.mean(errors)

    def fit_to_disk(self, data, optimize=True):
        """Fit data to a disk"""
        # Estimate the best plane with PCA
        pca = PCA(n_components=3)
        pca.fit(data)

        center_init = np.mean(data, axis=0)
        normal_init = pca.components_[2]  # Least component = plane normal

        # Project data onto the plane
        projected_data = self._project_to_plane(data, center_init, normal_init)
        distances_2d = [np.linalg.norm(p - center_init) for p in projected_data]

        outer_radius_init = np.max(distances_2d)
        inner_radius_init = np.min(distances_2d)

        if not optimize:
            return center_init, normal_init, outer_radius_init, inner_radius_init, \
                   self._disk_fit_error(data, center_init, normal_init, outer_radius_init, inner_radius_init)

        # Optimization
        def objective(params):
            center = params[:3]
            normal = params[3:6]
            normal = normal / np.linalg.norm(normal)  # Normalize
            outer_r = params[6]
            inner_r = params[7]
            return self._disk_fit_error(data, center, normal, outer_r, inner_r)

        bounds = [(-10, 10)] * 6 + [(0.1, 20), (0, 15)]
        initial_guess = np.concatenate([center_init, normal_init, [outer_radius_init, inner_radius_init]])

        result = minimize(objective, initial_guess, bounds=bounds)

        center_opt = result.x[:3]
        normal_opt = result.x[3:6] / np.linalg.norm(result.x[3:6])
        outer_r_opt = result.x[6]
        inner_r_opt = result.x[7]
        error = result.fun

        return center_opt, normal_opt, outer_r_opt, inner_r_opt, error

    def _project_to_plane(self, data, center, normal):
        """Project points onto a plane"""
        projected = []
        for point in data:
            v = point - center
            proj_v = v - np.dot(v, normal) * normal
            projected.append(center + proj_v)
        return np.array(projected)

    def _disk_fit_error(self, data, center, normal, outer_radius, inner_radius):
        """Calculate the error of fitting on a disk"""
        errors = []
        for point in data:
            # Distance to the plane
            plane_distance = abs(np.dot(point - center, normal))

            # Projection onto the plane
            v = point - center
            proj_v = v - np.dot(v, normal) * normal
            radial_distance = np.linalg.norm(proj_v)

            # Error = distance to the plane + radial error
            if radial_distance < inner_radius:
                radial_error = inner_radius - radial_distance
            elif radial_distance > outer_radius:
                radial_error = radial_distance - outer_radius
            else:
                radial_error = 0

            total_error = plane_distance**2 + radial_error**2
            errors.append(total_error)

        return np.mean(errors)

    def fit_data_to_manifolds(self, data, manifolds=['sphere', 'torus', 'disk']):
        """Fit data to all manifolds and compare"""
        results = {}

        for manifold in manifolds:
            try:
                if manifold == 'sphere':
                    center, radius, error = self.fit_to_sphere(data)
                    results[manifold] = {
                        'params': {'center': center, 'radius': radius},
                        'error': error,
                        'aic': self._compute_aic(error, len(data), 4),  # 4 parameters
                        'bic': self._compute_bic(error, len(data), 4)
                    }

                elif manifold == 'torus':
                    center, R, r, error = self.fit_to_torus(data)
                    results[manifold] = {
                        'params': {'center': center, 'R': R, 'r': r},
                        'error': error,
                        'aic': self._compute_aic(error, len(data), 5),  # 5 parameters
                        'bic': self._compute_bic(error, len(data), 5)
                    }

                elif manifold == 'disk':
                    center, normal, outer_r, inner_r, error = self.fit_to_disk(data)
                    results[manifold] = {
                        'params': {'center': center, 'normal': normal,
                                  'outer_radius': outer_r, 'inner_radius': inner_r},
                        'error': error,
                        'aic': self._compute_aic(error, len(data), 8),  # 8 parameters
                        'bic': self._compute_bic(error, len(data), 8)
                    }

            except Exception as e:
                results[manifold] = {'error': float('inf'), 'exception': str(e)}

        # Rank by error
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('error', float('inf')))

        return results, sorted_results

    def _compute_aic(self, mse, n, k):
        """Compute Akaike Information Criterion"""
        return n * np.log(mse) + 2 * k

    def _compute_bic(self, mse, n, k):
        """Compute Bayesian Information Criterion"""
        return n * np.log(mse) + k * np.log(n)

    def generate_synthetic_data_from_fit(self, manifold_type, params, n_samples=1000, noise=0.1):
        """Generate synthetic data from the fitted manifold"""
        if manifold_type == 'sphere':
            center = params['center']
            radius = params['radius']

            # Generate points on a sphere
            theta = np.random.uniform(0, np.pi, n_samples)
            phi = np.random.uniform(0, 2*np.pi, n_samples)

            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)

            synthetic_data = np.column_stack([x, y, z])

        elif manifold_type == 'torus':
            center = params['center']
            R = params['R']
            r = params['r']

            # Generate points on a torus
            u = np.random.uniform(0, 2*np.pi, n_samples)
            v = np.random.uniform(0, 2*np.pi, n_samples)

            x = center[0] + (R + r * np.cos(v)) * np.cos(u)
            y = center[1] + (R + r * np.cos(v)) * np.sin(u)
            z = center[2] + r * np.sin(v)

            synthetic_data = np.column_stack([x, y, z])

        elif manifold_type == 'disk':
            center = params['center']
            outer_r = params['outer_radius']
            inner_r = params['inner_radius']
            normal = params['normal']

            # Generate points on a disk
            rho = np.sqrt(np.random.uniform(inner_r**2, outer_r**2, n_samples))
            theta = np.random.uniform(0, 2*np.pi, n_samples)

            # Create two vectors orthogonal to the normal
            if abs(normal[2]) < 0.9:
                v1 = np.cross(normal, [0, 0, 1])
            else:
                v1 = np.cross(normal, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(normal, v1)

            x_local = rho * np.cos(theta)
            y_local = rho * np.sin(theta)

            synthetic_data = center + x_local[:, np.newaxis] * v1 + y_local[:, np.newaxis] * v2

        # Add noise
        if noise > 0:
            synthetic_data += np.random.normal(0, noise, synthetic_data.shape)

        return synthetic_data

    def visualize_fits(self, data, results, dataset_name="Unknown"):
        """Plot the fit results"""
        fig = plt.figure(figsize=(18, 6))

        # Original data
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.6, s=30)
        ax1.set_title(f'Original Data: {dataset_name}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Best fit
        best_manifold = min(results.keys(), key=lambda k: results[k].get('error', float('inf')))
        best_params = results[best_manifold]['params']

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.6, s=30, label='Original Data')

        # Plot the fitted manifold
        synthetic = self.generate_synthetic_data_from_fit(best_manifold, best_params, 500, 0)
        ax2.scatter(synthetic[:, 0], synthetic[:, 1], synthetic[:, 2],
                   c='red', alpha=0.3, s=10, label=f'{best_manifold} fit')

        ax2.set_title(f'Best Fit: {best_manifold}')
        ax2.legend()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Compare errors
        ax3 = fig.add_subplot(133)
        manifold_names = list(results.keys())
        errors = [results[m].get('error', float('inf')) for m in manifold_names]

        bars = ax3.bar(manifold_names, errors)
        ax3.set_title('Error Comparison')
        ax3.set_ylabel('Mean Squared Error')
        ax3.set_yscale('log')

        # Highlight the best fit
        min_idx = errors.index(min(errors))
        bars[min_idx].set_color('green')

        plt.tight_layout()
        plt.show()
