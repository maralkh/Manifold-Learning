# Manifold Type Detection from Data - Part 2

## 🎯 Core Challenge: Manifold Detection

### Problem:
Your data consists of a set of points in ℝᵈ:
```
Data = {x₁, x₂, ..., xₙ} ⊂ ℝᵈ
```

**Question**: Which **manifold** do these points lie on?

### Why is this important?
- **Correct dimensionality reduction**: Choose the appropriate method
- **Proper distance metrics**: Geodesic instead of Euclidean
- **Better prediction**: Model the true shape of data
- **Correct interpolation**: Between points on the manifold

---

## 🔍 Manifold Detection Methods

### 1. Statistical Shape Analysis

#### a) Sphericity Test:
```python
def test_sphericity(data):
    """Test if data lies on a sphere"""
    
    # Step 1: Estimate center
    center = np.mean(data, axis=0)
    
    # Step 2: Calculate distances from center
    distances = np.linalg.norm(data - center, axis=1)
    
    # Step 3: Test uniformity of distances
    # Kolmogorov-Smirnov test
    from scipy import stats
    
    # Assumption: if spherical, all distances should be equal
    mean_dist = np.mean(distances)
    normalized_distances = distances / mean_dist
    
    # Test against constant distribution
    ks_statistic, p_value = stats.kstest(
        normalized_distances, 
        lambda x: 1.0 if 0.8 < x < 1.2 else 0.0  # Uniform distribution around 1
    )
    
    # Scoring: lower = better
    sphericity_score = 1 / (1 + ks_statistic)
    
    return {
        'score': sphericity_score,
        'p_value': p_value,
        'mean_radius': mean_dist,
        'radius_std': np.std(distances),
        'is_spherical': sphericity_score > 0.7
    }
```

#### b) Toroidal Structure Test:
```python
def test_toroidal_structure(data):
    """Test if data has toroidal structure"""
    
    center = np.mean(data, axis=0)
    centered_data = data - center
    
    # Step 1: Analyze radial distance (in xy plane)
    if data.shape[1] >= 2:
        radial_distances = np.sqrt(centered_data[:, 0]**2 + centered_data[:, 1]**2)
        
        # Step 2: Analyze distance distribution
        hist, bins = np.histogram(radial_distances, bins=20, density=True)
        
        # Step 3: Detect bimodality
        peaks = find_peaks_in_histogram(hist, bins)
        
        # Step 4: Analyze vertical dimension (z)
        if data.shape[1] >= 3:
            z_distribution = centered_data[:, 2]
            z_periodicity = test_periodicity(z_distribution)
        else:
            z_periodicity = 0
        
        # Scoring
        bimodality_score = min(len(peaks) / 2.0, 1.0)  # 2 peaks = torus
        
        return {
            'score': 0.6 * bimodality_score + 0.4 * z_periodicity,
            'n_peaks': len(peaks),
            'radial_peaks': peaks,
            'z_periodicity': z_periodicity,
            'is_toroidal': bimodality_score > 0.5
        }
    
    return {'score': 0, 'is_toroidal': False}

def find_peaks_in_histogram(hist, bins):
    """Find peaks in histogram"""
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            # Peak must be significant
            if hist[i] > 0.1 * np.max(hist):
                peak_position = (bins[i] + bins[i+1]) / 2
                peaks.append((peak_position, hist[i]))
    return peaks
```

#### c) Planarity Test:
```python
def test_planarity(data):
    """Test if data lies on a plane"""
    
    # Step 1: PCA analysis
    pca = PCA(n_components=min(data.shape[1], 3))
    pca.fit(data)
    
    explained_variance = pca.explained_variance_ratio_
    
    # Step 2: If third component is very small → planar
    if len(explained_variance) >= 3:
        planarity_score = 1 - explained_variance[2]
    elif len(explained_variance) == 2:
        planarity_score = 0.8  # Probably 2D
    else:
        planarity_score = 0.9  # Definitely 1D
    
    # Step 3: Test distance to plane
    if len(explained_variance) >= 3:
        # Project data onto main plane
        data_2d = pca.transform(data)[:, :2]
        data_reconstructed_3d = pca.inverse_transform(
            np.column_stack([data_2d, np.zeros(len(data_2d))])
        )
        
        # Distance to plane
        distances_to_plane = np.linalg.norm(data - data_reconstructed_3d, axis=1)
        mean_distance = np.mean(distances_to_plane)
        
        # Refine score
        distance_factor = np.exp(-mean_distance)
        planarity_score *= distance_factor
    
    return {
        'score': planarity_score,
        'explained_variance': explained_variance,
        'principal_components': pca.components_,
        'is_planar': planarity_score > 0.8
    }
```

### 2. Topological Analysis

#### a) Intrinsic Dimension Estimation:
```python
def estimate_intrinsic_dimension(data, method='correlation'):
    """Estimate intrinsic dimension of manifold"""
    
    if method == 'correlation':
        return correlation_dimension(data)
    elif method == 'pca':
        return pca_dimension(data)
    elif method == 'mle':
        return mle_dimension(data)
    else:
        raise ValueError(f"Unknown method: {method}")

def correlation_dimension(data, eps_range=None):
    """Correlation dimension using box-counting"""
    
    if eps_range is None:
        pairwise_distances = pdist(data)
        eps_range = np.logspace(
            np.log10(np.min(pairwise_distances[pairwise_distances > 0])),
            np.log10(np.max(pairwise_distances)),
            50
        )
    
    correlations = []
    
    for eps in eps_range:
        # Count pairs of points with distance less than eps
        distance_matrix = squareform(pdist(data))
        count = np.sum(distance_matrix < eps) - len(data)  # Remove diagonal
        correlation = count / (len(data) * (len(data) - 1))
        correlations.append(correlation)
    
    # Fit line in log-log plot
    log_eps = np.log10(eps_range)
    log_corr = np.log10(np.array(correlations) + 1e-10)
    
    # Remove invalid points
    valid_indices = np.isfinite(log_corr) & (np.array(correlations) > 0)
    
    if np.sum(valid_indices) > 10:
        slope, intercept = np.polyfit(
            log_eps[valid_indices], 
            log_corr[valid_indices], 
            1
        )
        intrinsic_dim = slope
    else:
        intrinsic_dim = data.shape[1]  # fallback
    
    return max(1, min(intrinsic_dim, data.shape[1]))

def pca_dimension(data, variance_threshold=0.95):
    """PCA dimension with variance threshold"""
    
    pca = PCA()
    pca.fit(data)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    return intrinsic_dim

def mle_dimension(data, k=10):
    """Maximum Likelihood Estimation of dimension"""
    
    from sklearn.neighbors import NearestNeighbors
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Remove self (distance 0)
    distances = distances[:, 1:]
    
    # MLE estimator
    log_ratios = []
    for i in range(len(data)):
        r_k = distances[i, -1]  # Farthest neighbor
        r_1 = distances[i, 0]   # Nearest neighbor
        
        if r_1 > 0:
            log_ratios.append(np.log(r_k / r_1))
    
    if log_ratios:
        mean_log_ratio = np.mean(log_ratios)
        mle_dim = (k-1) / mean_log_ratio if mean_log_ratio > 0 else data.shape[1]
    else:
        mle_dim = data.shape[1]
    
    return max(1, min(mle_dim, data.shape[1]))
```

#### b) Curvature Analysis:
```python
def estimate_curvature_properties(data, k_neighbors=10):
    """Estimate curvature properties"""
    
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    local_curvatures = []
    
    for i in range(len(data)):
        # Local neighbors
        neighbor_indices = indices[i, 1:]  # Remove self
        local_points = data[neighbor_indices]
        center_point = data[i]
        
        # Local PCA
        centered_points = local_points - center_point
        
        if len(centered_points) >= 3:
            pca = PCA(n_components=min(3, centered_points.shape[1]))
            pca.fit(centered_points)
            
            # Estimate curvature from variance ratio
            explained_var = pca.explained_variance_ratio_
            
            if len(explained_var) >= 2:
                # Curvature ∝ variance in direction perpendicular to manifold
                curvature = explained_var[-1] / explained_var[0] if explained_var[0] > 0 else 0
                local_curvatures.append(curvature)
    
    if local_curvatures:
        return {
            'mean_curvature': np.mean(local_curvatures),
            'curvature_std': np.std(local_curvatures),
            'max_curvature': np.max(local_curvatures),
            'curvature_distribution': local_curvatures
        }
    else:
        return {'mean_curvature': 0, 'curvature_std': 0}
```

### 3. Spectral Analysis

#### a) Laplacian Analysis:
```python
def spectral_manifold_analysis(data, n_neighbors=10):
    """Spectral analysis using graph Laplacian"""
    
    from sklearn.neighbors import kneighbors_graph
    
    # Build k-NN graph
    connectivity = kneighbors_graph(
        data, n_neighbors=n_neighbors, 
        mode='distance', include_self=False
    )
    
    # Convert to symmetric matrix
    adjacency = 0.5 * (connectivity + connectivity.T)
    
    # Compute Laplacian
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    degree_matrix = np.diag(degrees)
    laplacian = degree_matrix - adjacency.toarray()
    
    # Normalization
    degree_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv
    
    # Eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)
    
    # Estimate intrinsic dimension from spectral gap
    eigenvalue_gaps = np.diff(eigenvalues)
    intrinsic_dim = np.argmax(eigenvalue_gaps) + 1
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'spectral_gap': np.max(eigenvalue_gaps),
        'intrinsic_dimension': intrinsic_dim,
        'laplacian': normalized_laplacian
    }
```

#### b) Frequency Analysis (for Hypertorus):
```python
def frequency_analysis(data):
    """Frequency analysis to detect periodic patterns"""
    
    periodicity_scores = []
    dominant_frequencies = []
    
    for feature_idx in range(data.shape[1]):
        feature = data[:, feature_idx]
        
        # Zero-mean
        feature_centered = feature - np.mean(feature)
        
        # FFT
        fft_vals = np.fft.fft(feature_centered)
        freqs = np.fft.fftfreq(len(feature))
        
        # Power spectrum (positive frequencies only)
        power_spectrum = np.abs(fft_vals[:len(fft_vals)//2])
        positive_freqs = freqs[:len(freqs)//2]
        
        if len(power_spectrum) > 1:
            # Remove DC component
            power_spectrum = power_spectrum[1:]
            positive_freqs = positive_freqs[1:]
            
            # Find main peak
            if len(power_spectrum) > 0:
                max_power_idx = np.argmax(power_spectrum)
                dominant_freq = positive_freqs[max_power_idx]
                max_power = power_spectrum[max_power_idx]
                mean_power = np.mean(power_spectrum)
                
                # Signal-to-noise ratio
                periodicity_score = max_power / (mean_power + 1e-10)
                
                periodicity_scores.append(periodicity_score)
                dominant_frequencies.append(abs(dominant_freq))
    
    if periodicity_scores:
        return {
            'mean_periodicity': np.mean(periodicity_scores),
            'max_periodicity': np.max(periodicity_scores),
            'dominant_frequencies': dominant_frequencies,
            'is_periodic': np.mean(periodicity_scores) > 5.0  # threshold
        }
    else:
        return {'mean_periodicity': 0, 'is_periodic': False}
```

---

## 🧠 Unified Detector: Combining All Tests

```python
class ManifoldDetector:
    """Comprehensive manifold type detector"""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.test_results = {}
    
    def detect_manifold_type(self, data):
        """Detect manifold type by combining all tests"""
        
        # Preprocessing
        data = self._preprocess(data)
        
        # Run all tests
        self.test_results = {
            'sphericity': test_sphericity(data),
            'toroidal': test_toroidal_structure(data),
            'planarity': test_planarity(data),
            'intrinsic_dim': estimate_intrinsic_dimension(data),
            'curvature': estimate_curvature_properties(data),
            'spectral': spectral_manifold_analysis(data),
            'periodicity': frequency_analysis(data)
        }
        
        # Compute final scores
        scores = self._compute_final_scores()
        
        # Select best
        best_manifold = max(scores.keys(), key=lambda k: scores[k])
        
        return {
            'detected_manifold': best_manifold,
            'confidence': scores[best_manifold],
            'all_scores': scores,
            'detailed_results': self.test_results,
            'is_confident': scores[best_manifold] > self.confidence_threshold
        }
    
    def _compute_final_scores(self):
        """Combine test results for final score"""
        
        scores = {}
        
        # Sphere
        sphericity_score = self.test_results['sphericity']['score']
        curvature_uniformity = 1 / (1 + self.test_results['curvature']['curvature_std'])
        intrinsic_dim_match = 1.0 if self.test_results['intrinsic_dim'] <= 3 else 0.5
        
        scores['sphere'] = 0.5 * sphericity_score + 0.3 * curvature_uniformity + 0.2 * intrinsic_dim_match
        
        # Torus
        toroidal_score = self.test_results['toroidal']['score']
        curvature_variation = min(self.test_results['curvature']['curvature_std'], 1.0)
        
        scores['torus'] = 0.6 * toroidal_score + 0.4 * curvature_variation
        
        # Plane/Disk
        planarity_score = self.test_results['planarity']['score']
        low_curvature = np.exp(-self.test_results['curvature']['mean_curvature'])
        
        scores['disk'] = 0.7 * planarity_score + 0.3 * low_curvature
        
        # Hypertorus
        periodicity_score = min(self.test_results['periodicity']['mean_periodicity'] / 10, 1.0)
        high_dim_bonus = 1.0 if self.test_results['intrinsic_dim'] > 3 else 0.3
        
        scores['hypertorus'] = 0.8 * periodicity_score + 0.2 * high_dim_bonus
        
        # Swiss roll and complex manifolds
        spectral_complexity = min(self.test_results['spectral']['spectral_gap'], 1.0)
        nonlinearity = 1 - scores['disk']  # Inverse of planarity
        
        scores['complex_manifold'] = 0.6 * spectral_complexity + 0.4 * nonlinearity
        
        return scores
    
    def _preprocess(self, data):
        """Smart preprocessing"""
        
        # Remove outliers
        data = self._remove_outliers(data)
        
        # Normalize if needed
        if np.std(np.std(data, axis=0)) > 1.0:  # If scales are different
            data = StandardScaler().fit_transform(data)
        
        return data
    
    def _remove_outliers(self, data, method='iqr'):
        """Remove outliers"""
        
        if method == 'iqr':
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Points within reasonable range in all dimensions
            mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
            
            return data[mask]
        
        return data
```

---

## 📊 Testing and Evaluation

### Practical Example:
```python
# Generate test data
def generate_test_data():
    """Generate test data for each manifold type"""
    
    np.random.seed(42)
    
    # Sphere
    theta = np.random.uniform(0, np.pi, 200)
    phi = np.random.uniform(0, 2*np.pi, 200)
    sphere_data = np.column_stack([
        2 * np.sin(theta) * np.cos(phi),
        2 * np.sin(theta) * np.sin(phi),
        2 * np.cos(theta)
    ]) + np.random.normal(0, 0.1, (200, 3))
    
    # Torus
    u = np.random.uniform(0, 2*np.pi, 200)
    v = np.random.uniform(0, 2*np.pi, 200)
    R, r = 3, 1
    torus_data = np.column_stack([
        (R + r * np.cos(v)) * np.cos(u),
        (R + r * np.cos(v)) * np.sin(u),
        r * np.sin(v)
    ]) + np.random.normal(0, 0.1, (200, 3))
    
    # Disk
    rho = np.sqrt(np.random.uniform(0, 4, 200))
    theta = np.random.uniform(0, 2*np.pi, 200)
    disk_data = np.column_stack([
        rho * np.cos(theta),
        rho * np.sin(theta),
        np.random.normal(0, 0.1, 200)  # Small noise in z
    ])
    
    return {
        'sphere': sphere_data,
        'torus': torus_data,
        'disk': disk_data
    }

# Test detector
detector = ManifoldDetector()
test_datasets = generate_test_data()

for true_type, data in test_datasets.items():
    result = detector.detect_manifold_type(data)
    
    print(f"True: {true_type}")
    print(f"Detected: {result['detected_manifold']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Correct: {result['detected_manifold'] == true_type}")
    print("-" * 40)
```

---

## 🎯 Performance Metrics

### Evaluation Metrics:
```python
def evaluate_detector_performance(detector, test_cases):
    """Comprehensive detector evaluation"""
    
    results = {
        'accuracy': 0,
        'precision': {},
        'recall': {},
        'f1_score': {},
        'confusion_matrix': {},
        'confidence_distribution': []
    }
    
    predictions = []
    true_labels = []
    confidences = []
    
    for true_type, data in test_cases.items():
        detection_result = detector.detect_manifold_type(data)
        
        predictions.append(detection_result['detected_manifold'])
        true_labels.append(true_type)
        confidences.append(detection_result['confidence'])
    
    # Calculate accuracy
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    results['accuracy'] = correct / len(predictions)
    
    # Calculate precision, recall, F1 for each class
    unique_labels = set(true_labels + predictions)
    
    for label in unique_labels:
        tp = sum(p == t == label for p, t in zip(predictions, true_labels))
        fp = sum(p == label and t != label for p, t in zip(predictions, true_labels))
        fn = sum(p != label and t == label for p, t in zip(predictions, true_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['precision'][label] = precision
        results['recall'][label] = recall
        results['f1_score'][label] = f1
    
    results['confidence_distribution'] = confidences
    
    return results
```
