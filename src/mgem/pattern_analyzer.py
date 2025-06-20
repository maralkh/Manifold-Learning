"""
Data Pattern Analyzer for Multi-Geometry Energy Models
======================================================

Analyzes data patterns to suggest optimal geometry placement
and configuration for energy-based machine learning models.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class DataPatternAnalyzer:
    """Analyzes data patterns to suggest optimal geometry placement"""
    
    def __init__(self, min_clusters=2, max_clusters=8):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.analysis_results = {}
        
    def analyze_patterns(self, X, y=None):
        """Comprehensive analysis of data patterns"""
        print("🔍 Analyzing data patterns...")
        
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = np.array(X)
        
        if y is not None:
            if torch.is_tensor(y):
                y_np = y.numpy()
            else:
                y_np = np.array(y)
        else:
            y_np = None
        
        analysis = {}
        
        # 1. Basic statistics
        analysis['statistics'] = self._compute_statistics(X_np)
        
        # 2. Clustering analysis
        analysis['clustering'] = self._analyze_clustering(X_np)
        
        # 3. Circular and elliptical pattern detection
        analysis['circular_patterns'] = self._detect_circular_patterns(X_np)
        analysis['elliptical_patterns'] = self._detect_elliptical_patterns(X_np)
        
        # 4. Gaussian mixture analysis
        analysis['gaussian_mixtures'] = self._analyze_gaussian_mixtures(X_np)
        
        # 5. Principal component analysis
        analysis['pca'] = self._analyze_pca(X_np)
        
        # 6. Target-dependent analysis if available
        if y_np is not None:
            analysis['target_dependent'] = self._analyze_target_patterns(X_np, y_np)
        
        self.analysis_results = analysis
        self._print_analysis_summary()
        
        return analysis
    
    def _compute_statistics(self, X):
        """Compute basic statistics"""
        return {
            'mean': X.mean(axis=0),
            'std': X.std(axis=0),
            'min': X.min(axis=0),
            'max': X.max(axis=0),
            'data_range': X.max(axis=0) - X.min(axis=0),
            'correlation': np.corrcoef(X.T) if X.shape[1] > 1 else np.array([[1.0]])
        }
    
    def _analyze_clustering(self, X):
        """Analyze clustering patterns"""
        clustering_results = {}
        
        # K-means clustering
        best_k = self.min_clusters
        best_inertia = float('inf')
        
        for k in range(self.min_clusters, min(self.max_clusters + 1, len(X))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_k = k
                    
                clustering_results[f'kmeans_{k}'] = {
                    'centers': kmeans.cluster_centers_,
                    'labels': labels,
                    'inertia': kmeans.inertia_
                }
            except:
                break
        
        clustering_results['best_k'] = best_k
        clustering_results['best_centers'] = clustering_results[f'kmeans_{best_k}']['centers']
        
        return clustering_results
    
    def _detect_circular_patterns(self, X):
        """Detect circular patterns in data"""
        patterns = []
        
        if X.shape[1] >= 2:
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    pattern = self._analyze_circular_pattern(X[:, [i, j]])
                    if pattern['circularity_score'] > 0.3:
                        pattern['dimensions'] = [i, j]
                        patterns.append(pattern)
        
        return patterns
    
    def _analyze_circular_pattern(self, X_2d):
        """Analyze circularity of 2D data"""
        # Center the data
        center = X_2d.mean(axis=0)
        centered_data = X_2d - center
        
        # Compute distances from center
        distances = np.linalg.norm(centered_data, axis=1)
        
        # Circularity metrics
        distance_mean = distances.mean()
        distance_std = distances.std()
        circularity = 1 - (distance_std / (distance_mean + 1e-8))
        
        # Angular distribution uniformity
        angles = np.arctan2(centered_data[:, 1], centered_data[:, 0])
        angle_uniformity = self._compute_angular_uniformity(angles)
        
        return {
            'center': center,
            'radius': distance_mean,
            'radius_std': distance_std,
            'circularity_score': circularity,
            'angular_uniformity': angle_uniformity,
            'combined_score': 0.7 * circularity + 0.3 * angle_uniformity
        }
    
    def _detect_elliptical_patterns(self, X):
        """Detect elliptical patterns in data"""
        patterns = []
        
        if X.shape[1] >= 2:
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    pattern = self._analyze_elliptical_pattern(X[:, [i, j]])
                    if pattern['ellipticity_score'] > 0.3:
                        pattern['dimensions'] = [i, j]
                        patterns.append(pattern)
        
        return patterns
    
    def _analyze_elliptical_pattern(self, X_2d):
        """Analyze ellipticity of 2D data"""
        # Center the data
        center = X_2d.mean(axis=0)
        centered_data = X_2d - center
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Ellipse parameters
        major_axis = 2 * np.sqrt(eigenvalues[0])
        minor_axis = 2 * np.sqrt(eigenvalues[1])
        eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2)
        
        # Ellipticity score
        ellipticity_score = 1 - (minor_axis / major_axis)
        
        return {
            'center': center,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'eccentricity': eccentricity,
            'ellipticity_score': ellipticity_score,
            'principal_axes': eigenvectors
        }
    
    def _analyze_gaussian_mixtures(self, X):
        """Analyze Gaussian mixture patterns"""
        mixtures = {}
        
        for n_components in range(1, min(6, len(X) // 10 + 1)):
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(X)
                
                mixtures[f'gmm_{n_components}'] = {
                    'means': gmm.means_,
                    'covariances': gmm.covariances_,
                    'weights': gmm.weights_,
                    'aic': gmm.aic(X),
                    'bic': gmm.bic(X)
                }
            except:
                break
        
        # Find best model by BIC
        if mixtures:
            best_model = min(mixtures.keys(), key=lambda k: mixtures[k]['bic'])
            mixtures['best_model'] = best_model
        
        return mixtures
    
    def _analyze_pca(self, X):
        """Analyze principal components"""
        if X.shape[1] <= 1:
            return {'n_components': 1, 'explained_variance_ratio': [1.0]}
        
        pca = PCA()
        pca.fit(X)
        
        # Find number of components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        return {
            'n_components_95': n_components_95,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumsum,
            'components': pca.components_
        }
    
    def _analyze_target_patterns(self, X, y):
        """Analyze patterns dependent on target variable"""
        patterns = {}
        
        # Separate data by target quantiles
        target_quantiles = np.quantile(y, [0.25, 0.5, 0.75])
        
        for i, threshold in enumerate(target_quantiles):
            mask = y <= threshold
            X_subset = X[mask.flatten()]
            
            if len(X_subset) > 10:
                patterns[f'quantile_{i+1}'] = {
                    'center': X_subset.mean(axis=0),
                    'std': X_subset.std(axis=0),
                    'size': len(X_subset)
                }
        
        return patterns
    
    def _compute_angular_uniformity(self, angles):
        """Compute uniformity of angular distribution"""
        # Bin the angles
        hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
        
        # Compute uniformity (inverse of variance)
        expected_count = len(angles) / 16
        uniformity = 1 - np.var(hist) / (expected_count**2 + 1e-8)
        
        return np.clip(uniformity, 0, 1)
    
    def _print_analysis_summary(self):
        """Print summary of analysis results"""
        print("   📊 Analysis Summary:")
        
        if 'clustering' in self.analysis_results:
            best_k = self.analysis_results['clustering']['best_k']
            print(f"   ✅ Best K-means clusters: {best_k}")
        
        if 'circular_patterns' in self.analysis_results:
            n_circular = len(self.analysis_results['circular_patterns'])
            print(f"   ✅ Circular patterns found: {n_circular}")
        
        if 'elliptical_patterns' in self.analysis_results:
            n_elliptical = len(self.analysis_results['elliptical_patterns'])
            print(f"   ✅ Elliptical patterns found: {n_elliptical}")
        
        if 'gaussian_mixtures' in self.analysis_results and 'best_model' in self.analysis_results['gaussian_mixtures']:
            best_gmm = self.analysis_results['gaussian_mixtures']['best_model']
            n_components = int(best_gmm.split('_')[1])
            print(f"   ✅ Best Gaussian mixture: {n_components} components")
    
    def suggest_initialization(self, output_dim=1, max_geometries=None):
        """Suggest geometry initialization based on analysis"""
        if not self.analysis_results:
            raise ValueError("Must run analyze_patterns first")
        
        suggestions = {'spheres': [], 'torus': [], 'ellipsoids': []}
        
        # Sphere suggestions from clusters
        if 'clustering' in self.analysis_results:
            centers = self.analysis_results['clustering']['best_centers']
            for center in centers:
                output_center = self._adapt_center_to_output_dim(center, output_dim)
                suggestions['spheres'].append({
                    'center': output_center,
                    'radius': 1.0
                })
        
        # Torus suggestions from circular patterns
        if 'circular_patterns' in self.analysis_results:
            for pattern in self.analysis_results['circular_patterns']:
                if pattern['combined_score'] > 0.5:
                    output_center = self._adapt_center_to_output_dim(pattern['center'], output_dim)
                    suggestions['torus'].append({
                        'center': output_center,
                        'major_radius': max(0.5, min(3.0, pattern['radius'])),
                        'minor_radius': 0.3
                    })
        
        # Ellipsoid suggestions from elliptical patterns
        if 'elliptical_patterns' in self.analysis_results:
            for pattern in self.analysis_results['elliptical_patterns']:
                if pattern['ellipticity_score'] > 0.3:
                    output_center = self._adapt_center_to_output_dim(pattern['center'], output_dim)
                    radii = [pattern['major_axis'] / 2, pattern['minor_axis'] / 2]
                    output_radii = self._adapt_radii_to_output_dim(radii, output_dim)
                    suggestions['ellipsoids'].append({
                        'center': output_center,
                        'radii': output_radii
                    })
        
        # Gaussian mixture suggestions
        if 'gaussian_mixtures' in self.analysis_results and 'best_model' in self.analysis_results['gaussian_mixtures']:
            best_model = self.analysis_results['gaussian_mixtures']['best_model']
            gmm_data = self.analysis_results['gaussian_mixtures'][best_model]
            
            for mean, cov in zip(gmm_data['means'], gmm_data['covariances']):
                output_center = self._adapt_center_to_output_dim(mean, output_dim)
                
                # Convert covariance to radii for ellipsoid
                if output_dim > 1:
                    eigenvals = np.linalg.eigvals(cov)
                    radii = np.sqrt(eigenvals)
                    output_radii = self._adapt_radii_to_output_dim(radii, output_dim)
                    suggestions['ellipsoids'].append({
                        'center': output_center,
                        'radii': output_radii
                    })
        
        # Ensure minimum geometries
        if not suggestions['spheres']:
            suggestions['spheres'].append({'center': [0.0] * output_dim, 'radius': 1.0})
        
        # Limit suggestions if specified
        if max_geometries:
            for geom_type in suggestions:
                if len(suggestions[geom_type]) > max_geometries.get(geom_type, 5):
                    suggestions[geom_type] = suggestions[geom_type][:max_geometries[geom_type]]
        
        return suggestions
    
    def _adapt_center_to_output_dim(self, center, output_dim):
        """Adapt center coordinates to output dimension"""
        center = np.array(center)
        
        if len(center) >= output_dim:
            return center[:output_dim].tolist()
        else:
            output_center = center.tolist()
            while len(output_center) < output_dim:
                output_center.append(0.0)
            return output_center
    
    def _adapt_radii_to_output_dim(self, radii, output_dim):
        """Adapt radii to output dimension"""
        radii = np.array(radii)
        
        if len(radii) >= output_dim:
            return radii[:output_dim].tolist()
        else:
            output_radii = radii.tolist()
            mean_radius = np.mean(radii)
            while len(output_radii) < output_dim:
                output_radii.append(mean_radius)
            return output_radii
