import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ManifoldEnergyIntegration:
    """
    Integration of manifold learning with energy-based methods
    """

    def __init__(self):
        self.manifold_energies = {}
        self.fitted_models = {}

    # =====================================
    # Energy Functions on Manifolds
    # =====================================

    def energy_on_sphere(self, points, center, radius, temperature=1.0):
        """Calculate energy on a sphere"""
        # Distance from the sphere surface
        distances_to_center = np.linalg.norm(points - center, axis=1)
        surface_distances = np.abs(distances_to_center - radius)

        # Energy function: less on the surface, more away from the surface
        energies = surface_distances**2 / (2 * temperature**2)

        return energies

    def energy_on_torus(self, points, center, R, r, temperature=1.0):
        """Calculate energy on a torus"""
        # Translate to center
        centered_points = points - center
        x, y, z = centered_points[:, 0], centered_points[:, 1], centered_points[:, 2]

        # Radial distance in the xy plane
        rho = np.sqrt(x**2 + y**2)

        # Distance to the central circle of the torus
        distance_to_center_circle = np.sqrt((rho - R)**2 + z**2)
        surface_distances = np.abs(distance_to_center_circle - r)

        # Energy function
        energies = surface_distances**2 / (2 * temperature**2)

        return energies

    def energy_on_disk(self, points, center, normal, outer_radius, inner_radius, temperature=1.0):
        """Calculate energy on a disk"""
        # Distance to the plane
        centered_points = points - center
        plane_distances = np.abs(np.dot(centered_points, normal))

        # Projection onto the plane
        projections = centered_points - np.outer(np.dot(centered_points, normal), normal)
        radial_distances = np.linalg.norm(projections, axis=1)

        # Distance to the disk edges
        edge_distances = np.zeros_like(radial_distances)

        # Inside the hole
        inside_hole = radial_distances < inner_radius
        edge_distances[inside_hole] = inner_radius - radial_distances[inside_hole]

        # Outside the disk
        outside_disk = radial_distances > outer_radius
        edge_distances[outside_disk] = radial_distances[outside_disk] - outer_radius

        # Energy: combined plane and edge distance
        energies = (plane_distances**2 + edge_distances**2) / (2 * temperature**2)

        return energies

    # =====================================
    # Bayesian Energy Learning
    # =====================================

    def bayesian_energy_function(self, h, centers, weights, scales):
        """Bayesian energy function with multiple centers"""
        h_tensor = torch.FloatTensor(h)
        centers_tensor = torch.FloatTensor(centers)
        weights_tensor = torch.FloatTensor(weights)
        scales_tensor = torch.FloatTensor(scales)

        # Calculate distance from all centers
        if len(h_tensor.shape) == 1:
            h_tensor = h_tensor.unsqueeze(0)

        distances = torch.norm(h_tensor.unsqueeze(1) - centers_tensor.unsqueeze(0), dim=-1)

        # Gaussian mixture energy
        exp_terms = weights_tensor * torch.exp(-distances**2 / (2 * scales_tensor**2))
        log_sum_exp = torch.logsumexp(torch.log(exp_terms + 1e-8), dim=-1)
        energy = -log_sum_exp

        return energy.numpy()

    def kl_divergence_variational(self, mu, rho, prior_std=1.0):
        """Calculate KL divergence"""
        sigma = np.log1p(np.exp(rho))

        kl = 0.5 * np.sum(
            (mu**2 + sigma**2) / prior_std**2 -
            2 * np.log(sigma / prior_std) - 1
        )

        return kl

    def monte_carlo_energy_estimation(self, mu, sigma, energy_func, n_samples=1000):
        """Estimate energy with Monte Carlo"""
        # Sample weights
        samples = np.random.normal(mu, sigma, (n_samples, len(mu)))

        # Calculate energy for each sample
        energies = [energy_func(sample) for sample in samples]

        expected_energy = np.mean(energies)
        energy_variance = np.var(energies)

        return expected_energy, energy_variance

    # =====================================
    # Combining Manifold and Energy
    # =====================================

    def fit_energy_on_manifold(self, data, manifold_type, n_energy_centers=5):
        """Fit energy model on manifold"""

        # First, fit the manifold
        if manifold_type == 'sphere':
            center, radius, manifold_error = self._fit_sphere_simple(data)
            manifold_params = {'center': center, 'radius': radius}

        elif manifold_type == 'torus':
            center, R, r, manifold_error = self._fit_torus_simple(data)
            manifold_params = {'center': center, 'R': R, 'r': r}

        elif manifold_type == 'disk':
            center, normal, outer_r, inner_r, manifold_error = self._fit_disk_simple(data)
            manifold_params = {'center': center, 'normal': normal,
                             'outer_radius': outer_r, 'inner_radius': inner_r}

        # Now place energy centers on the manifold
        energy_centers = self._place_energy_centers_on_manifold(
            data, manifold_type, manifold_params, n_energy_centers
        )

        # Optimize energy parameters
        energy_weights = np.ones(n_energy_centers) / n_energy_centers
        energy_scales = np.ones(n_energy_centers) * 0.5

        # Optimization with maximum likelihood
        optimized_params = self._optimize_energy_parameters(
            data, energy_centers, energy_weights, energy_scales
        )

        result = {
            'manifold_type': manifold_type,
            'manifold_params': manifold_params,
            'manifold_error': manifold_error,
            'energy_centers': energy_centers,
            'energy_weights': optimized_params['weights'],
            'energy_scales': optimized_params['scales'],
            'energy_likelihood': optimized_params['likelihood']
        }

        return result

    def _fit_sphere_simple(self, data):
        """Simple sphere fit"""
        center = np.mean(data, axis=0)
        distances = [np.linalg.norm(point - center) for point in data]
        radius = np.mean(distances)
        error = np.mean([(d - radius)**2 for d in distances])
        return center, radius, error

    def _fit_torus_simple(self, data):
        """Simple torus fit"""
        center = np.mean(data, axis=0)
        centered_data = data - center
        radial_distances = np.sqrt(centered_data[:, 0]**2 + centered_data[:, 1]**2)
        R = np.mean(radial_distances)
        r = np.std(centered_data[:, 2])

        # Calculate error
        errors = []
        for point in data:
            p = point - center
            rho = np.sqrt(p[0]**2 + p[1]**2)
            distance_to_center_circle = np.sqrt((rho - R)**2 + p[2]**2)
            error = (distance_to_center_circle - r)**2
            errors.append(error)

        return center, R, r, np.mean(errors)

    def _fit_disk_simple(self, data):
        """Simple disk fit"""
        # PCA to find the plane
        pca = PCA(n_components=3)
        pca.fit(data)

        center = np.mean(data, axis=0)
        normal = pca.components_[2]  # Least component

        # Projection onto the plane
        centered_data = data - center
        projections = centered_data - np.outer(np.dot(centered_data, normal), normal)
        distances_2d = [np.linalg.norm(p) for p in projections]

        outer_radius = np.max(distances_2d)
        inner_radius = np.min(distances_2d)

        # Calculate error
        plane_distances = [abs(np.dot(point - center, normal)) for point in data]
        error = np.mean([d**2 for d in plane_distances])

        return center, normal, outer_radius, inner_radius, error

    def _place_energy_centers_on_manifold(self, data, manifold_type, manifold_params, n_centers):
        """Place energy centers on the manifold"""

        if manifold_type == 'sphere':
            # Random points on the sphere
            center = manifold_params['center']
            radius = manifold_params['radius']

            # Generate points on the sphere
            theta = np.random.uniform(0, np.pi, n_centers)
            phi = np.random.uniform(0, 2*np.pi, n_centers)

            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)

            energy_centers = np.column_stack([x, y, z])

        elif manifold_type == 'torus':
            # Random points on the torus
            center = manifold_params['center']
            R = manifold_params['R']
            r = manifold_params['r']

            u = np.random.uniform(0, 2*np.pi, n_centers)
            v = np.random.uniform(0, 2*np.pi, n_centers)

            x = center[0] + (R + r * np.cos(v)) * np.cos(u)
            y = center[1] + (R + r * np.cos(v)) * np.sin(u)
            z = center[2] + r * np.sin(v)

            energy_centers = np.column_stack([x, y, z])

        elif manifold_type == 'disk':
            # Random points on the disk
            center = manifold_params['center']
            normal = manifold_params['normal']
            outer_r = manifold_params['outer_radius']
            inner_r = manifold_params['inner_radius']

            # Generate points in the annulus
            rho = np.sqrt(np.random.uniform(inner_r**2, outer_r**2, n_centers))
            theta = np.random.uniform(0, 2*np.pi, n_centers)

            # Create vectors orthogonal to the normal
            if abs(normal[2]) < 0.9:
                v1 = np.cross(normal, [0, 0, 1])
            else:
                v1 = np.cross(normal, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(normal, v1)

            x_local = rho * np.cos(theta)
            y_local = rho * np.sin(theta)

            energy_centers = center + x_local[:, np.newaxis] * v1 + y_local[:, np.newaxis] * v2

        return energy_centers

    def _optimize_energy_parameters(self, data, centers, weights_init, scales_init):
        """Optimize energy parameters"""

        def negative_log_likelihood(params):
            n_centers = len(centers)
            weights = params[:n_centers]
            scales = params[n_centers:]

            # Constrain parameters
            weights = np.abs(weights)
            weights = weights / np.sum(weights)  # Normalize
            scales = np.abs(scales) + 0.01  # Avoid zero

            # Calculate likelihood
            total_likelihood = 0
            for point in data:
                # Calculate probability density
                distances = [np.linalg.norm(point - center) for center in centers]
                exp_terms = [w * np.exp(-d**2 / (2 * s**2)) for d, w, s in zip(distances, weights, scales)]
                probability = np.sum(exp_terms) + 1e-8
                total_likelihood += np.log(probability)

            return -total_likelihood  # Negative for minimization

        # Optimization
        initial_params = np.concatenate([weights_init, scales_init])
        bounds = [(0.01, 2.0)] * len(weights_init) + [(0.01, 2.0)] * len(scales_init)

        result = minimize(negative_log_likelihood, initial_params, bounds=bounds)

        n_centers = len(centers)
        optimized_weights = result.x[:n_centers]
        optimized_scales = result.x[n_centers:]

        # Normalize weights
        optimized_weights = optimized_weights / np.sum(optimized_weights)

        return {
            'weights': optimized_weights,
            'scales': optimized_scales,
            'likelihood': -result.fun
        }

    # =====================================
    # OOD Detection on Manifolds
    # =====================================

    def ood_detection_manifold_energy(self, train_data, test_data, manifold_type='sphere'):
        """OOD detection by combining manifold and energy"""

        # Fit energy model on training data
        energy_model = self.fit_energy_on_manifold(train_data, manifold_type)

        # Calculate energy for train and test
        train_energies = self._compute_manifold_energies(train_data, energy_model)
        test_energies = self._compute_manifold_energies(test_data, energy_model)

        # Calculate uncertainty (variance of predictions)
        train_uncertainties = self._compute_prediction_uncertainties(train_data, energy_model)
        test_uncertainties = self._compute_prediction_uncertainties(test_data, energy_model)

        # Combined OOD score
        all_energies = np.concatenate([train_energies, test_energies])
        all_uncertainties = np.concatenate([train_uncertainties, test_uncertainties])

        # Normalize
        energy_norm = (all_energies - np.min(all_energies)) / (np.max(all_energies) - np.min(all_energies))
        uncertainty_norm = (all_uncertainties - np.min(all_uncertainties)) / (np.max(all_uncertainties) - np.min(all_uncertainties))

        # Combined score
        alpha = 0.7  # weight for energy
        combined_scores = alpha * energy_norm + (1 - alpha) * uncertainty_norm

        # Labels: train=0 (ID), test=1 (OOD)
        labels = np.concatenate([np.zeros(len(train_data)), np.ones(len(test_data))])

        # AUC calculation
        auc = roc_auc_score(labels, combined_scores)

        return {
            'energy_model': energy_model,
            'train_energies': train_energies,
            'test_energies': test_energies,
            'train_uncertainties': train_uncertainties,
            'test_uncertainties': test_uncertainties,
            'combined_scores': combined_scores,
            'labels': labels,
            'auc': auc
        }

    def _compute_manifold_energies(self, data, energy_model):
        """Calculate energy on the manifold"""
        manifold_type = energy_model['manifold_type']
        centers = energy_model['energy_centers']
        weights = energy_model['energy_weights']
        scales = energy_model['energy_scales']

        energies = []
        for point in data:
            energy = self.bayesian_energy_function(point, centers, weights, scales)
            energies.append(energy[0] if isinstance(energy, np.ndarray) else energy)

        return np.array(energies)

    def _compute_prediction_uncertainties(self, data, energy_model):
        """Calculate prediction uncertainty"""
        # For simplicity, use the variance of distances to energy centers
        centers = energy_model['energy_centers']

        uncertainties = []
        for point in data:
            distances = [np.linalg.norm(point - center) for center in centers]
            uncertainty = np.var(distances)  # variance of distances as uncertainty
            uncertainties.append(uncertainty)

        return np.array(uncertainties)

    # =====================================
    # Method Comparison
    # =====================================

    def compare_methods(self, data, ood_data):
        """Compare different methods"""

        print("=== Comparison of Manifold and Energy Methods ===\n")

        results = {}

        # 1. Simple Manifold
        manifolds = ['sphere', 'torus', 'disk']

        for manifold in manifolds:
            try:
                result = self.ood_detection_manifold_energy(data, ood_data, manifold)
                results[f'manifold_{manifold}'] = result['auc']
                print(f"Manifold {manifold}: AUC = {result['auc']:.4f}")
            except Exception as e:
                print(f"Error in {manifold}: {e}")

        # 2. Energy-based methods
        print(f"\n--- Energy-Based Methods ---")

        # Simple energy function
        centers = data[np.random.choice(len(data), 5, replace=False)]
        weights = np.ones(5) / 5
        scales = np.ones(5) * 0.5

        train_energies = [self.bayesian_energy_function(point, centers, weights, scales)[0]
                         for point in data]
        test_energies = [self.bayesian_energy_function(point, centers, weights, scales)[0]
                        for point in ood_data]

        all_energies = np.concatenate([train_energies, test_energies])
        labels = np.concatenate([np.zeros(len(data)), np.ones(len(ood_data))])

        auc_energy = roc_auc_score(labels, all_energies)
        results['energy_only'] = auc_energy
        print(f"Energy-only: AUC = {auc_energy:.4f}")

        # 3. Dimensionality reduction methods
        print(f"\n--- Dimensionality Reduction ---")

        # PCA reconstruction error
        pca = PCA(n_components=2)
        train_pca = pca.fit_transform(data)
        train_reconstructed = pca.inverse_transform(train_pca)
        test_pca = pca.transform(ood_data)
        test_reconstructed = pca.inverse_transform(test_pca)

        train_pca_errors = [np.linalg.norm(orig - recon) for orig, recon in zip(data, train_reconstructed)]
        test_pca_errors = [np.linalg.norm(orig - recon) for orig, recon in zip(ood_data, test_reconstructed)]

        all_pca_errors = np.concatenate([train_pca_errors, test_pca_errors])
        auc_pca = roc_auc_score(labels, all_pca_errors)
        results['pca_reconstruction'] = auc_pca
        print(f"PCA Reconstruction: AUC = {auc_pca:.4f}")

        # t-SNE (for visualization, not OOD detection)
        try:
            all_data_combined = np.vstack([data, ood_data])
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data_combined)//4))
            tsne_result = tsne.fit_transform(all_data_combined)

            # Using distance from the t-SNE center for OOD
            center_tsne = np.mean(tsne_result[:len(data)], axis=0)
            distances_tsne = [np.linalg.norm(point - center_tsne) for point in tsne_result]
            auc_tsne = roc_auc_score(labels, distances_tsne)
            results['tsne_distance'] = auc_tsne
            print(f"t-SNE Distance: AUC = {auc_tsne:.4f}")
        except:
            print(f"t-SNE: Error in calculation")

        # Best method
        best_method = max(results.keys(), key=lambda k: results[k])
        print(f"\n🏆 Best method: {best_method} with AUC = {results[best_method]:.4f}")

        return results

    # =====================================
    # Visualization
    # =====================================

    def visualize_manifold_energy(self, data, ood_data, manifold_type='sphere'):
        """Visualize the combination of manifold and energy"""

        # Fit energy model
        energy_model = self.fit_energy_on_manifold(data, manifold_type, n_energy_centers=5)

        fig = plt.figure(figsize=(20, 12))

        # 1. Original data and manifold
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.7, s=30, label='ID Data')
        ax1.scatter(ood_data[:, 0], ood_data[:, 1], ood_data[:, 2], c='red', alpha=0.7, s=30, label='OOD Data')

        # Energy centers
        centers = energy_model['energy_centers']
        ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   c='green', s=100, marker='^', label='Energy Centers')

        ax1.set_title(f'Data and Energy Centers on {manifold_type}')
        ax1.legend()

        # 2. Energy Distribution
        ax2 = fig.add_subplot(232)

        train_energies = self._compute_manifold_energies(data, energy_model)
        test_energies = self._compute_manifold_energies(ood_data, energy_model)

        ax2.hist(train_energies, bins=20, alpha=0.7, label='ID Energy', color='blue', density=True)
        ax2.hist(test_energies, bins=20, alpha=0.7, label='OOD Energy', color='red', density=True)
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('Density')
        ax2.set_title('Energy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Energy vs Distance to Manifold
        ax3 = fig.add_subplot(233)

        # Calculate distance to manifold
        if manifold_type == 'sphere':
            center = energy_model['manifold_params']['center']
            radius = energy_model['manifold_params']['radius']

            train_manifold_distances = [abs(np.linalg.norm(point - center) - radius) for point in data]
            test_manifold_distances = [abs(np.linalg.norm(point - center) - radius) for point in ood_data]
        else:
            # For simplicity, use Euclidean distance
            center = energy_model['manifold_params']['center']
            train_manifold_distances = [np.linalg.norm(point - center) for point in data]
            test_manifold_distances = [np.linalg.norm(point - center) for point in ood_data]

        ax3.scatter(train_manifold_distances, train_energies, c='blue', alpha=0.6, label='ID')
        ax3.scatter(test_manifold_distances, test_energies, c='red', alpha=0.6, label='OOD')
        ax3.set_xlabel('Distance to Manifold')
        ax3.set_ylabel('Energy')
        ax3.set_title('Energy vs Manifold Distance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. OOD Scores
        ood_result = self.ood_detection_manifold_energy(data, ood_data, manifold_type)

        ax4 = fig.add_subplot(234)

        train_scores = ood_result['combined_scores'][:len(data)]
        test_scores = ood_result['combined_scores'][len(data):]

        ax4.hist(train_scores, bins=20, alpha=0.7, label='ID Scores', color='blue', density=True)
        ax4.hist(test_scores, bins=20, alpha=0.7, label='OOD Scores', color='red', density=True)
        ax4.set_xlabel('OOD Score')
        ax4.set_ylabel('Density')
        ax4.set_title(f'OOD Scores (AUC = {ood_result["auc"]:.3f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Energy Landscape (2D projection)
        ax5 = fig.add_subplot(235)

        # PCA for projection
        all_data_combined = np.vstack([data, ood_data])
        pca = PCA(n_components=2)
        projected_data = pca.fit_transform(all_data_combined)

        # Create grid for energy landscape
        x_min, x_max = projected_data[:, 0].min() - 1, projected_data[:, 0].max() + 1
        y_min, y_max = projected_data[:, 1].min() - 1, projected_data[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

        # Transform grid to 3D (inverse PCA)
        grid_2d = np.column_stack([xx.ravel(), yy.ravel()])

        if pca.n_components_ >= 2:
            # Add the third dimension (mean)
            z_mean = np.mean(all_data_combined[:, 2]) if all_data_combined.shape[1] > 2 else 0
            grid_3d = pca.inverse_transform(grid_2d)
            if grid_3d.shape[1] < 3:
                grid_3d = np.column_stack([grid_3d, np.full(len(grid_3d), z_mean)])
        else:
            grid_3d = grid_2d

        # Calculate energy for grid
        grid_energies = self._compute_manifold_energies(grid_3d, energy_model)

        # Draw contour
        energy_surface = grid_energies.reshape(xx.shape)
        contour = ax5.contour(xx, yy, energy_surface, levels=15, alpha=0.6)
        ax5.clabel(contour, inline=True, fontsize=8)

        # Draw points
        train_proj = projected_data[:len(data)]
        test_proj = projected_data[len(data):]

        ax5.scatter(train_proj[:, 0], train_proj[:, 1], c='blue', alpha=0.7, s=30, label='ID')
        ax5.scatter(test_proj[:, 0], test_proj[:, 1], c='red', alpha=0.7, s=30, label='OOD')

        # Energy centers projected
        centers_proj = pca.transform(energy_model['energy_centers'])
        ax5.scatter(centers_proj[:, 0], centers_proj[:, 1],
                   c='green', s=100, marker='^', label='Energy Centers')

        ax5.set_xlabel('PC1')
        ax5.set_ylabel('PC2')
        ax5.set_title('Energy Landscape (2D Projection)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Compare ROC curves
        ax6 = fig.add_subplot(236)

        # ROC curve for energy-manifold
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(ood_result['labels'], ood_result['combined_scores'])
        ax6.plot(fpr, tpr, label=f'Energy-Manifold (AUC={ood_result["auc"]:.3f})', linewidth=2)

        # Compare with other methods
        # Energy-only
        all_energies = np.concatenate([train_energies, test_energies])
        labels = np.concatenate([np.zeros(len(data)), np.ones(len(ood_data))])
        fpr_energy, tpr_energy, _ = roc_curve(labels, all_energies)
        auc_energy = roc_auc_score(labels, all_energies)
        ax6.plot(fpr_energy, tpr_energy, label=f'Energy-only (AUC={auc_energy:.3f})',
                linewidth=2, linestyle='--')

        # Distance to manifold
        all_distances = np.concatenate([train_manifold_distances, test_manifold_distances])
        fpr_dist, tpr_dist, _ = roc_curve(labels, all_distances)
        auc_dist = roc_auc_score(labels, all_distances)
        ax6.plot(fpr_dist, tpr_dist, label=f'Manifold Distance (AUC={auc_dist:.3f})',
                linewidth=2, linestyle=':')

        # Reference line
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

        ax6.set_xlabel('False Positive Rate')
        ax6.set_ylabel('True Positive Rate')
        ax6.set_title('ROC Curve Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return energy_model, ood_result

# =====================================
# Practical Examples and Testing
# =====================================

def practical_examples():
    """Practical examples of manifold and energy combination"""

    print("=== Practical Examples: Manifold and Energy Combination ===\n")

    integrator = ManifoldEnergyIntegration()

    # 1. Generate sample data
    np.random.seed(42)

    # ID Data: on a sphere
    print("1️⃣ Generating Sample Data:")
    n_train = 200
    n_test = 50

    # Training data on a sphere
    theta = np.random.uniform(0, np.pi, n_train)
    phi = np.random.uniform(0, 2*np.pi, n_train)
    radius = 2.0

    train_data = np.column_stack([
        radius * np.sin(theta) * np.cos(phi) + np.random.normal(0, 0.1, n_train),
        radius * np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, n_train),
        radius * np.cos(theta) + np.random.normal(0, 0.1, n_train)
    ])

    # OOD Data: random points
    ood_data = np.random.normal(0, 3, (n_test, 3))

    print(f"Train data: {train_data.shape} (Spherical data)")
    print(f"OOD data: {ood_data.shape} (Random data)")

    # 2. Compare methods
    print(f"\n2️⃣ Comparing Methods:")
    results = integrator.compare_methods(train_data, ood_data)

    # 3. Deep analysis of the best manifold
    print(f"\n3️⃣ Deep Analysis of Different Manifolds:")

    manifolds_to_test = ['sphere', 'torus', 'disk']
    detailed_results = {}

    for manifold in manifolds_to_test:
        try:
            print(f"\n--- {manifold.upper()} ---")

            # Fit energy model
            energy_model = integrator.fit_energy_on_manifold(train_data, manifold, n_energy_centers=6)

            print(f"Manifold error: {energy_model['manifold_error']:.6f}")
            print(f"Energy likelihood: {energy_model['energy_likelihood']:.6f}")
            print(f"Energy centers: {len(energy_model['energy_centers'])}")
            print(f"Average energy weight: {np.mean(energy_model['energy_weights']):.4f}")
            print(f"Average energy scale: {np.mean(energy_model['energy_scales']):.4f}")

            # OOD detection
            ood_result = integrator.ood_detection_manifold_energy(train_data, ood_data, manifold)
            print(f"OOD AUC: {ood_result['auc']:.4f}")

            detailed_results[manifold] = {
                'manifold_error': energy_model['manifold_error'],
                'energy_likelihood': energy_model['energy_likelihood'],
                'ood_auc': ood_result['auc'],
                'energy_model': energy_model
            }

        except Exception as e:
            print(f"Error in {manifold}: {e}")

    # 4. Select the best model
    if detailed_results:
        print(f"\n4️⃣ Comparison Summary:")
        print(f"{'Manifold':<12} {'Fit Error':<12} {'Likelihood':<12} {'OOD AUC':<10}")
        print("-" * 50)

        for manifold, result in detailed_results.items():
            print(f"{manifold:<12} {result['manifold_error']:<12.6f} "
                  f"{result['energy_likelihood']:<12.2f} {result['ood_auc']:<10.4f}")

        # Select based on OOD performance
        best_manifold = max(detailed_results.keys(),
                           key=lambda k: detailed_results[k]['ood_auc'])

        print(f"\n🏆 Best manifold for OOD detection: {best_manifold}")
        print(f"   AUC = {detailed_results[best_manifold]['ood_auc']:.4f}")

        # 5. Visualization of the best model
        print(f"\n5️⃣ Visualizing the Best Model:")
        print("Drawing plots...")

        # integrator.visualize_manifold_energy(train_data, ood_data, best_manifold)

        return detailed_results, best_manifold, train_data, ood_data

    return None, None, None, None

# =====================================
# Advanced Analysis: Energy Dynamics
# =====================================

def advanced_energy_analysis():
    """Advanced analysis of energy dynamics on manifolds"""

    print("\n=== Advanced Analysis: Energy Dynamics ===\n")

    integrator = ManifoldEnergyIntegration()

    # Generate more complex data
    np.random.seed(123)

    # Multi-modal data on a torus
    n_samples = 300

    # Two different modes on the torus
    u1 = np.random.normal(np.pi/2, 0.3, n_samples//2)
    v1 = np.random.normal(0, 0.3, n_samples//2)

    u2 = np.random.normal(3*np.pi/2, 0.3, n_samples//2)
    v2 = np.random.normal(np.pi, 0.3, n_samples//2)

    u = np.concatenate([u1, u2])
    v = np.concatenate([v1, v2])

    R, r = 3.0, 1.0

    multimodal_data = np.column_stack([
        (R + r * np.cos(v)) * np.cos(u) + np.random.normal(0, 0.1, n_samples),
        (R + r * np.cos(v)) * np.sin(u) + np.random.normal(0, 0.1, n_samples),
        r * np.sin(v) + np.random.normal(0, 0.1, n_samples)
    ])

    # OOD data
    ood_samples = np.random.normal(0, 4, (100, 3))

    print(f"Multi-modal data: {multimodal_data.shape}")
    print(f"OOD data: {ood_samples.shape}")

    # 1. Compare different numbers of energy centers
    print(f"\n1️⃣ Impact of Number of Energy Centers:")

    center_counts = [3, 5, 8, 12, 15]
    auc_scores = []

    for n_centers in center_counts:
        try:
            ood_result = integrator.ood_detection_manifold_energy(
                multimodal_data, ood_samples, 'torus'
            )
            auc_scores.append(ood_result['auc'])
            print(f"  {n_centers} centers: AUC = {ood_result['auc']:.4f}")
        except:
            auc_scores.append(0)
            print(f"  {n_centers} centers: Error")

    # 2. Analyze energy landscape
    print(f"\n2️⃣ Analyzing Energy Landscape:")

    energy_model = integrator.fit_energy_on_manifold(multimodal_data, 'torus', n_energy_centers=8)

    # Calculate energy at different points
    test_points = np.random.normal(0, 3, (500, 3))
    test_energies = integrator._compute_manifold_energies(test_points, energy_model)

    print(f"Energy statistics:")
    print(f"  Mean: {np.mean(test_energies):.4f}")
    print(f"  Std:  {np.std(test_energies):.4f}")
    print(f"  Min:  {np.min(test_energies):.4f}")
    print(f"  Max:  {np.max(test_energies):.4f}")

    # 3. Analyze stability
    print(f"\n3️⃣ Analyzing Stability:")

    stability_results = []
    n_runs = 10

    for run in range(n_runs):
        np.random.seed(run)

        # Resampling
        indices = np.random.choice(len(multimodal_data), len(multimodal_data), replace=True)
        bootstrap_data = multimodal_data[indices]

        try:
            ood_result = integrator.ood_detection_manifold_energy(
                bootstrap_data, ood_samples, 'torus'
            )
            stability_results.append(ood_result['auc'])
        except:
            pass

    if stability_results:
        print(f"Stability analysis ({len(stability_results)} runs):")
        print(f"  Mean AUC: {np.mean(stability_results):.4f}")
        print(f"  Std AUC:  {np.std(stability_results):.4f}")
        print(f"  Min AUC:  {np.min(stability_results):.4f}")
        print(f"  Max AUC:  {np.max(stability_results):.4f}")

    # 4. Compare with baseline methods
    print(f"\n4️⃣ Comparing with Baseline Methods:")

    # Isolation Forest
    try:
        from sklearn.ensemble import IsolationForest

        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(multimodal_data)

        # Predict
        train_scores_iso = iso_forest.decision_function(multimodal_data)
        test_scores_iso = iso_forest.decision_function(ood_samples)

        all_scores_iso = np.concatenate([train_scores_iso, test_scores_iso])
        labels = np.concatenate([np.zeros(len(multimodal_data)), np.ones(len(ood_samples))])

        # Need to invert scores (since Isolation Forest gives negative for outlier)
        auc_iso = roc_auc_score(labels, -all_scores_iso)
        print(f"Isolation Forest: AUC = {auc_iso:.4f}")

    except ImportError:
        print(f"Isolation Forest: Library not available")

    # One-Class SVM
    try:
        from sklearn.svm import OneClassSVM

        oc_svm = OneClassSVM(gamma='scale', nu=0.1)
        oc_svm.fit(multimodal_data)

        train_scores_svm = oc_svm.decision_function(multimodal_data)
        test_scores_svm = oc_svm.decision_function(ood_samples)

        all_scores_svm = np.concatenate([train_scores_svm, test_scores_svm])
        auc_svm = roc_auc_score(labels, -all_scores_svm)  # Invert
        print(f"One-Class SVM: AUC = {auc_svm:.4f}")

    except ImportError:
        print(f"One-Class SVM: Library not available")

    return energy_model, stability_results
