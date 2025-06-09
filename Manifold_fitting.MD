# Parameter Fitting and Manifold Optimization - Part 3

## 🎯 Problem: From Detection to Precise Fitting

### Input:
- **Data**: `X = {x₁, x₂, ..., xₙ} ⊂ ℝᵈ`
- **Identified manifold type**: e.g., "sphere"

### Goal:
**Find optimal parameters** of the manifold that provide the best fit to the data.

### Challenges:
- **Local minima**: Objective function is typically non-convex
- **Noise robustness**: Data contains noise
- **Computational efficiency**: Must be fast
- **Numerical stability**: Stable across different datasets

---

## 🔧 Fitting Methods

### 1. Least Squares Fitting

#### A) Sphere:
```python
def fit_sphere_least_squares(data):
    """Algebraic sphere fitting with least squares"""
    
    # Sphere equation: (x-a)² + (y-b)² + (z-c)² = r²
    # Rewrite: x² + y² + z² - 2ax - 2by - 2cz + (a²+b²+c²-r²) = 0
    # Linear form: Ax = b
    
    n_points = len(data)
    
    # Matrix A
    A = np.column_stack([
        -2 * data,                    # -2x, -2y, -2z
        np.ones(n_points)             # constant term
    ])
    
    # Vector b
    b = -(data**2).sum(axis=1)       # -(x² + y² + z²)
    
    # Solve linear system Ax = b
    try:
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        
        center = solution[:3]         # [a, b, c]
        radius_squared = np.sum(center**2) - solution[3]
        
        if radius_squared <= 0:
            raise ValueError("Invalid radius")
        
        radius = np.sqrt(radius_squared)
        
        # Calculate residual
        distances = np.linalg.norm(data - center, axis=1)
        residual = np.mean((distances - radius)**2)
        
        return {
            'center': center,
            'radius': radius,
            'residual': residual,
            'method': 'least_squares'
        }
        
    except np.linalg.LinAlgError:
        return None

def fit_sphere_geometric(data, max_iterations=100, tolerance=1e-8):
    """Geometric sphere fitting with Gauss-Newton"""
    
    # Initial estimation
    center_init = np.mean(data, axis=0)
    distances_init = np.linalg.norm(data - center_init, axis=1)
    radius_init = np.mean(distances_init)
    
    # Initial parameters: [cx, cy, cz, r]
    params = np.append(center_init, radius_init)
    
    for iteration in range(max_iterations):
        center = params[:3]
        radius = params[3]
        
        # Calculate residuals
        distances = np.linalg.norm(data - center, axis=1)
        residuals = distances - radius
        
        # Jacobian matrix
        J = np.zeros((len(data), 4))
        
        for i, point in enumerate(data):
            if distances[i] > 1e-10:  # Avoid division by zero
                # Derivatives w.r.t center
                direction = (point - center) / distances[i]
                J[i, :3] = -direction
                
                # Derivative w.r.t radius
                J[i, 3] = -1.0
        
        # Gauss-Newton update
        try:
            JTJ = J.T @ J
            JTr = J.T @ residuals
            
            # Solve with regularization if needed
            if np.linalg.cond(JTJ) > 1e12:
                lambda_reg = 1e-6 * np.trace(JTJ) / len(JTJ)
                JTJ += lambda_reg * np.eye(len(JTJ))
            
            delta = np.linalg.solve(JTJ, JTr)
            params = params - delta
            
            # Check convergence
            if np.linalg.norm(delta) < tolerance:
                break
                
        except np.linalg.LinAlgError:
            break
    
    return {
        'center': params[:3],
        'radius': abs(params[3]),  # radius must be positive
        'residual': np.mean(residuals**2),
        'iterations': iteration + 1,
        'method': 'gauss_newton'
    }
```

#### B) Torus:
```python
def fit_torus_nonlinear(data, max_iterations=200, tolerance=1e-8):
    """Nonlinear torus fitting"""
    
    # Smart initial estimation
    center_init = np.mean(data, axis=0)
    
    # Estimate major radius from projection on xy plane
    centered_data = data - center_init
    radial_distances = np.sqrt(centered_data[:, 0]**2 + centered_data[:, 1]**2)
    R_init = np.mean(radial_distances)
    
    # Estimate minor radius from z deviation
    r_init = np.std(centered_data[:, 2])
    
    # Initial parameters: [cx, cy, cz, R, r]
    params = np.array([*center_init, R_init, r_init])
    
    def torus_residual_function(params, data):
        """Calculate residuals for torus"""
        center = params[:3]
        R = params[3]
        r = params[4]
        
        residuals = []
        
        for point in data:
            p = point - center
            
            # Distance from z-axis (radial distance)
            rho = np.sqrt(p[0]**2 + p[1]**2)
            
            # Distance to torus center circle
            distance_to_center_circle = np.sqrt((rho - R)**2 + p[2]**2)
            
            # residual = distance to torus surface
            residual = distance_to_center_circle - r
            residuals.append(residual)
        
        return np.array(residuals)
    
    def torus_jacobian(params, data):
        """Jacobian matrix for torus"""
        center = params[:3]
        R = params[3]
        r = params[4]
        
        J = np.zeros((len(data), 5))
        
        for i, point in enumerate(data):
            p = point - center
            x, y, z = p
            
            rho = np.sqrt(x**2 + y**2)
            
            if rho > 1e-10:  # Avoid division by zero
                # Coordinates in path to center circle
                distance_to_center_circle = np.sqrt((rho - R)**2 + z**2)
                
                if distance_to_center_circle > 1e-10:
                    # Derivatives w.r.t center
                    drho_dx = x / rho
                    drho_dy = y / rho
                    
                    ddist_drho = (rho - R) / distance_to_center_circle
                    ddist_dz = z / distance_to_center_circle
                    
                    J[i, 0] = -ddist_drho * drho_dx  # ∂/∂cx
                    J[i, 1] = -ddist_drho * drho_dy  # ∂/∂cy
                    J[i, 2] = -ddist_dz              # ∂/∂cz
                    
                    # Derivative w.r.t R
                    J[i, 3] = -ddist_drho            # ∂/∂R
                    
                    # Derivative w.r.t r
                    J[i, 4] = -1.0                   # ∂/∂r
    
        return J
    
    # Levenberg-Marquardt optimization
    lambda_lm = 0.01
    
    for iteration in range(max_iterations):
        residuals = torus_residual_function(params, data)
        J = torus_jacobian(params, data)
        
        # Calculate update
        JTJ = J.T @ J
        JTr = J.T @ residuals
        
        # Levenberg-Marquardt regularization
        JTJ_reg = JTJ + lambda_lm * np.diag(np.diag(JTJ))
        
        try:
            delta = np.linalg.solve(JTJ_reg, JTr)
            params_new = params - delta
            
            # Check parameter validity
            if params_new[3] > 0 and params_new[4] > 0:  # R, r positive
                residuals_new = torus_residual_function(params_new, data)
                
                # If better, update
                if np.sum(residuals_new**2) < np.sum(residuals**2):
                    params = params_new
                    lambda_lm *= 0.5  # Decrease regularization
                else:
                    lambda_lm *= 2.0  # Increase regularization
            else:
                lambda_lm *= 2.0
            
            # Check convergence
            if np.linalg.norm(delta) < tolerance:
                break
                
        except np.linalg.LinAlgError:
            lambda_lm *= 10.0
            continue
    
    return {
        'center': params[:3],
        'major_radius': abs(params[3]),
        'minor_radius': abs(params[4]),
        'residual': np.mean(residuals**2),
        'iterations': iteration + 1,
        'method': 'levenberg_marquardt'
    }
```

### 2. Maximum Likelihood Estimation

```python
def fit_manifold_mle(data, manifold_type, noise_model='gaussian'):
    """Maximum Likelihood Estimation for manifold fitting"""
    
    def negative_log_likelihood(params):
        """Calculate negative log-likelihood"""
        
        # Calculate distance of each point to manifold
        distances = []
        
        for point in data:
            if manifold_type == 'sphere':
                center, radius = params[:3], params[3]
                distance = abs(np.linalg.norm(point - center) - radius)
                
            elif manifold_type == 'torus':
                center, R, r = params[:3], params[3], params[4]
                p = point - center
                rho = np.sqrt(p[0]**2 + p[1]**2)
                distance_to_center_circle = np.sqrt((rho - R)**2 + p[2]**2)
                distance = abs(distance_to_center_circle - r)
                
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Noise model
        if noise_model == 'gaussian':
            # Assumption: Gaussian noise with constant σ
            sigma = params[-1]  # Last parameter = noise level
            
            if sigma <= 0:
                return np.inf
            
            # Gaussian log-likelihood
            log_likelihood = -0.5 * np.sum((distances / sigma)**2) - \
                           len(data) * np.log(sigma * np.sqrt(2 * np.pi))
            
        elif noise_model == 'laplace':
            # Laplace noise (robust to outliers)
            b = params[-1]  # scale parameter
            
            if b <= 0:
                return np.inf
            
            log_likelihood = -np.sum(distances / b) - len(data) * np.log(2 * b)
        
        return -log_likelihood
    
    # Initial estimation
    if manifold_type == 'sphere':
        center_init = np.mean(data, axis=0)
        radius_init = np.mean(np.linalg.norm(data - center_init, axis=1))
        sigma_init = 0.1
        
        initial_params = np.append([*center_init, radius_init], sigma_init)
        bounds = [(-np.inf, np.inf)] * 3 + [(0.1, np.inf)] + [(1e-6, np.inf)]
        
    elif manifold_type == 'torus':
        # Similar to previous...
        center_init = np.mean(data, axis=0)
        R_init = 2.0
        r_init = 1.0
        sigma_init = 0.1
        
        initial_params = np.append([*center_init, R_init, r_init], sigma_init)
        bounds = [(-np.inf, np.inf)] * 3 + [(0.1, np.inf)] * 2 + [(1e-6, np.inf)]
    
    # Optimization
    from scipy.optimize import minimize
    
    result = minimize(
        negative_log_likelihood,
        initial_params,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    if result.success:
        if manifold_type == 'sphere':
            return {
                'center': result.x[:3],
                'radius': result.x[3],
                'noise_level': result.x[4],
                'log_likelihood': -result.fun,
                'method': 'mle'
            }
        elif manifold_type == 'torus':
            return {
                'center': result.x[:3],
                'major_radius': result.x[3],
                'minor_radius': result.x[4],
                'noise_level': result.x[5],
                'log_likelihood': -result.fun,
                'method': 'mle'
            }
    
    return None
```

### 3. Robust Fitting (RANSAC)

```python
def fit_manifold_ransac(data, manifold_type, 
                       max_iterations=1000, 
                       inlier_threshold=0.1,
                       min_samples_ratio=0.6):
    """RANSAC for robust manifold fitting"""
    
    n_points = len(data)
    min_samples = max(10, int(min_samples_ratio * n_points))
    
    best_model = None
    best_inliers = []
    best_score = 0
    
    for iteration in range(max_iterations):
        # Random sampling
        sample_indices = np.random.choice(n_points, min_samples, replace=False)
        sample_data = data[sample_indices]
        
        # Fit model on sample
        if manifold_type == 'sphere':
            model = fit_sphere_least_squares(sample_data)
        elif manifold_type == 'torus':
            model = fit_torus_nonlinear(sample_data, max_iterations=50)
        
        if model is None:
            continue
        
        # Test all points
        inliers = []
        
        for i, point in enumerate(data):
            if manifold_type == 'sphere':
                distance = abs(np.linalg.norm(point - model['center']) - model['radius'])
            elif manifold_type == 'torus':
                # Calculate distance to torus
                center = model['center']
                R = model['major_radius']
                r = model['minor_radius']
                
                p = point - center
                rho = np.sqrt(p[0]**2 + p[1]**2)
                distance_to_center_circle = np.sqrt((rho - R)**2 + p[2]**2)
                distance = abs(distance_to_center_circle - r)
            
            if distance < inlier_threshold:
                inliers.append(i)
        
        # Scoring
        score = len(inliers)
        
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_model = model
            
            # If sufficient consensus reached, stop
            if score > 0.8 * n_points:
                break
    
    # Refinement with all inliers
    if best_model and len(best_inliers) > min_samples:
        inlier_data = data[best_inliers]
        
        if manifold_type == 'sphere':
            refined_model = fit_sphere_geometric(inlier_data)
        elif manifold_type == 'torus':
            refined_model = fit_torus_nonlinear(inlier_data)
        
        if refined_model:
            refined_model['inliers'] = best_inliers
            refined_model['inlier_ratio'] = len(best_inliers) / n_points
            refined_model['method'] = 'ransac'
            return refined_model
    
    return best_model
```

### 4. Global Optimization

```python
def fit_manifold_global(data, manifold_type, method='differential_evolution'):
    """Global optimization for manifold fitting"""
    
    def objective_function(params):
        """Objective function for global optimization"""
        
        # Calculate total squared error
        total_error = 0
        
        for point in data:
            if manifold_type == 'sphere':
                if len(params) < 4:
                    return np.inf
                center, radius = params[:3], params[3]
                if radius <= 0:
                    return np.inf
                distance = abs(np.linalg.norm(point - center) - radius)
                
            elif manifold_type == 'torus':
                if len(params) < 5:
                    return np.inf
                center, R, r = params[:3], params[3], params[4]
                if R <= 0 or r <= 0 or R <= r:  # validity check
                    return np.inf
                
                p = point - center
                rho = np.sqrt(p[0]**2 + p[1]**2)
                distance_to_center_circle = np.sqrt((rho - R)**2 + p[2]**2)
                distance = abs(distance_to_center_circle - r)
            
            total_error += distance**2
        
        return total_error / len(data)  # MSE
    
    # Define bounds
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_range = data_max - data_min
    
    if manifold_type == 'sphere':
        bounds = [
            (data_min[0] - data_range[0], data_max[0] + data_range[0]),  # cx
            (data_min[1] - data_range[1], data_max[1] + data_range[1]),  # cy
            (data_min[2] - data_range[2], data_max[2] + data_range[2]),  # cz
            (0.1, 2 * np.max(data_range))                               # radius
        ]
    elif manifold_type == 'torus':
        max_range = np.max(data_range)
        bounds = [
            (data_min[0] - data_range[0], data_max[0] + data_range[0]),  # cx
            (data_min[1] - data_range[1], data_max[1] + data_range[1]),  # cy
            (data_min[2] - data_range[2], data_max[2] + data_range[2]),  # cz
            (0.5, 2 * max_range),                                       # R
            (0.1, max_range)                                            # r
        ]
    
    # Global optimization
    if method == 'differential_evolution':
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=1000,
            popsize=15,
            seed=42
        )
        
    elif method == 'basin_hopping':
        from scipy.optimize import basinhopping
        
        # Initial estimation
        if manifold_type == 'sphere':
            x0 = [*np.mean(data, axis=0), np.std(np.linalg.norm(data - np.mean(data, axis=0), axis=1))]
        elif manifold_type == 'torus':
            x0 = [*np.mean(data, axis=0), 2.0, 1.0]
        
        result = basinhopping(
            objective_function,
            x0,
            niter=200,
            T=1.0,
            stepsize=0.5
        )
    
    if result.success:
        if manifold_type == 'sphere':
            return {
                'center': result.x[:3],
                'radius': result.x[3],
                'residual': result.fun,
                'method': f'global_{method}'
            }
        elif manifold_type == 'torus':
            return {
                'center': result.x[:3],
                'major_radius': result.x[3],
                'minor_radius': result.x[4],
                'residual': result.fun,
                'method': f'global_{method}'
            }
    
    return None
```

---

## 🎯 Smart Initialization

```python
def smart_initialization(data, manifold_type):
    """Smart initial parameter estimation"""
    
    if manifold_type == 'sphere':
        # Method 1: centroid + mean distance
        center = np.mean(data, axis=0)
        distances = np.linalg.norm(data - center, axis=1)
        radius = np.mean(distances)
        
        # Method 2: minimum enclosing sphere approximation
        # Find two farthest points
        max_dist = 0
        farthest_pair = None
        
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                dist = np.linalg.norm(data[i] - data[j])
                if dist > max_dist:
                    max_dist = dist
                    farthest_pair = (i, j)
        
        if farthest_pair:
            center_approx = (data[farthest_pair[0]] + data[farthest_pair[1]]) / 2
            radius_approx = max_dist / 2
            
            # Choose better option
            error1 = np.mean((np.linalg.norm(data - center, axis=1) - radius)**2)
            error2 = np.mean((np.linalg.norm(data - center_approx, axis=1) - radius_approx)**2)
            
            if error2 < error1:
                center, radius = center_approx, radius_approx
        
        return {'center': center, 'radius': radius}
    
    elif manifold_type == 'torus':
        # PCA for orientation detection
        pca = PCA(n_components=3)
        pca.fit(data)
        
        # Assumption: main torus axis = third PCA component
        main_axis = pca.components_[2]
        
        # Project data onto plane perpendicular to main axis
        center = np.mean(data, axis=0)
        centered_data = data - center
        
        # Remove component in main axis direction
        projected_data = centered_data - np.outer(
            np.dot(centered_data, main_axis), main_axis
        )
        
        # Estimate R from mean radial distance
        radial_distances = np.linalg.norm(projected_data, axis=1)
        R = np.mean(radial_distances)
        
        # Estimate r from spread in main axis direction
        axial_distances = np.abs(np.dot(centered_data, main_axis))
        r = np.std(axial_distances)
        
        return {
            'center': center,
            'major_radius': max(R, 0.5),  # minimum value
            'minor_radius': max(r, 0.1),
            'main_axis': main_axis
        }
    
    return {}
```

---

## 📊 Convergence Analysis

```python
def analyze_convergence(optimization_history):
    """Analyze convergence of optimization process"""
    
    if not optimization_history:
        return None
    
    objectives = np.array([step['objective'] for step in optimization_history])
    parameters = np.array([step['parameters'] for step in optimization_history])
    
    # Objective function analysis
    obj_improvement = np.diff(objectives)
    relative_improvement = obj_improvement / (np.abs(objectives[:-1]) + 1e-10)
    
    # Parameter stability analysis
    param_changes = np.diff(parameters, axis=0)
    param_change_norms = np.linalg.norm(param_changes, axis=1)
    
    # Detect convergence
    converged = (
        len(relative_improvement) > 10 and
        np.all(np.abs(relative_improvement[-5:]) < 1e-6) and
        np.all(param_change_norms[-5:] < 1e-6)
    )
    
    # Detect oscillation
    oscillating = (
        len(objectives) > 20 and
        np.std(objectives[-10:]) > 0.1 * np.abs(objectives[-1])
    )
    
    return {
        'converged': converged,
        'oscillating': oscillating,
        'final_objective': objectives[-1],
        'improvement_rate': np.mean(np.abs(relative_improvement[-10:])),
        'parameter_stability': np.mean(param_change_norms[-10:]),
        'total_iterations': len(objectives)
    }
```

---

## 🧪 Multi-Method Ensemble

```python
class EnsembleFitter:
    """Combine multiple fitting methods for better results"""
    
    def __init__(self, manifold_type):
        self.manifold_type = manifold_type
        self.methods = {
            'least_squares': self._fit_least_squares,
            'geometric': self._fit_geometric,
            'mle': self._fit_mle,
            'ransac': self._fit_ransac,
            'global': self._fit_global
        }
    
    def fit(self, data, use_methods='all'):
        """Fit with multiple methods and select best"""
        
        if use_methods == 'all':
            use_methods = list(self.methods.keys())
        
        results = {}
        
        for method_name in use_methods:
            try:
                result = self.methods[method_name](data)
                if result:
                    results[method_name] = result
            except Exception as e:
                print(f"Method {method_name} failed: {e}")
        
        if not results:
            return None
        
        # Select best based on different criteria
        best_method = self._select_best_method(results, data)
        
        return {
            'best_result': results[best_method],
            'all_results': results,
            'selected_method': best_method,
            'consensus_score': self._compute_consensus(results)
        }
    
    def _select_best_method(self, results, data):
        """Select best method"""
        
        scores = {}
        
        for method, result in results.items():
            score = 0
            
            # Fit quality (lower = better)
            if 'residual' in result:
                score += 1 / (1 + result['residual'])
            
            # Robust score (for RANSAC)
            if 'inlier_ratio' in result:
                score += result['inlier_ratio']
            
            # Likelihood score (for MLE)
            if 'log_likelihood' in result:
                score += result['log_likelihood'] / len(data)
            
            # Penalty for complex methods
            complexity_penalty = {
                'least_squares': 0.0,
                'geometric': 0.1,
                'mle': 0.2,
                'ransac': 0.15,
                'global': 0.3
            }
            score -= complexity_penalty.get(method, 0.0)
            
            scores[method] = score
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _compute_consensus(self, results):
        """Compute consensus between different methods"""
        
        if self.manifold_type == 'sphere':
            centers = [r['center'] for r in results.values() if 'center' in r]
            radii = [r['radius'] for r in results.values() if 'radius' in r]
            
            if centers and radii:
                center_std = np.std(centers, axis=0)
                radius_std = np.std(radii)
                
                # Normalize by data scale
                data_scale = np.mean(radii) if radii else 1.0
                
                consensus = 1.0 / (1.0 + np.mean(center_std)/data_scale + radius_std/data_scale)
                return consensus
        
        elif self.manifold_type == 'torus':
            centers = [r['center'] for r in results.values() if 'center' in r]
            major_radii = [r['major_radius'] for r in results.values() if 'major_radius' in r]
            minor_radii = [r['minor_radius'] for r in results.values() if 'minor_radius' in r]
            
            if centers and major_radii and minor_radii:
                center_std = np.std(centers, axis=0)
                major_radius_std = np.std(major_radii)
                minor_radius_std = np.std(minor_radii)
                
                data_scale = np.mean(major_radii) if major_radii else 1.0
                
                consensus = 1.0 / (1.0 + 
                                 np.mean(center_std)/data_scale + 
                                 major_radius_std/data_scale + 
                                 minor_radius_std/data_scale)
                return consensus
        
        return 0.0

class ManifoldFittingTester:
    """Comprehensive framework for testing fitting algorithms"""
    
    def __init__(self):
        self.test_results = {}
        self.benchmark_data = {}
    
    def generate_benchmark_datasets(self):
        """Generate standard datasets for testing"""
        
        np.random.seed(42)
        datasets = {}
        
        # 1. Perfect sphere
        datasets['perfect_sphere'] = self._generate_perfect_sphere(200, 2.0)
        
        # 2. Noisy sphere  
        datasets['noisy_sphere'] = self._generate_noisy_sphere(200, 2.0, noise=0.1)
        
        # 3. Sphere with outliers
        datasets['outlier_sphere'] = self._generate_outlier_sphere(200, 2.0, outlier_ratio=0.1)
        
        # 4. Perfect torus
        datasets['perfect_torus'] = self._generate_perfect_torus(300, R=3.0, r=1.0)
