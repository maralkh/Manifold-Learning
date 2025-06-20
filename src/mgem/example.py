"""
Complete Integration Example
============================

This file shows how to integrate all the modular components together
and provides a working example that you can run directly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ====================================================================================================
# STEP 1: INTEGRATE ALL COMPONENTS
# ====================================================================================================

# In a real implementation, you would import these from separate files:
from geometric_primitives import AdaptiveSphere, AdaptiveTorus, AdaptiveEllipsoid, GeometryManager
from pattern_analyzer import DataPatternAnalyzer  
from energy_model import MultiGeometryEnergyModel
from trainer import MultiGeometryTrainer
from visualization import MultiGeometryVisualizer

# For this example, we'll include simplified versions of the key components

class SimpleAdaptiveSphere(nn.Module):
    """Simplified adaptive sphere for demonstration"""
    def __init__(self, center_dim, initial_center=None, initial_radius=1.0):
        super().__init__()
        if initial_center is None:
            initial_center = torch.randn(center_dim) * 0.5
        self.center = nn.Parameter(torch.FloatTensor(initial_center))
        self.log_radius = nn.Parameter(torch.log(torch.tensor(initial_radius)))
        self.influence_weight = nn.Parameter(torch.tensor(1.0))
        self.min_radius = 0.1
        self.max_radius = 5.0
    
    @property
    def radius(self):
        return torch.clamp(torch.exp(self.log_radius), self.min_radius, self.max_radius)
    
    def energy(self, points):
        distances = torch.norm(points - self.center, dim=-1)
        energy = torch.exp(-0.5 * (distances / self.radius)**2)
        return self.influence_weight * energy


class SimpleMultiGeometryModel(nn.Module):
    """Simplified multi-geometry energy model for demonstration"""
    
    def __init__(self, input_dim, output_dim=1, n_spheres=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build predictor network
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )
        
        # Initialize spheres
        self.spheres = nn.ModuleList()
        for i in range(n_spheres):
            center = torch.randn(output_dim) * 0.5
            sphere = SimpleAdaptiveSphere(output_dim, center, 1.0 + i * 0.2)
            self.spheres.append(sphere)
        
        self.energy_weight = nn.Parameter(torch.tensor(0.01))
    
    def compute_energy(self, predictions):
        total_energy = torch.zeros(predictions.shape[0], device=predictions.device)
        for sphere in self.spheres:
            total_energy += sphere.energy(predictions)
        return -torch.log(total_energy + 1e-8)
    
    def forward(self, x):
        predictions = self.predictor(x)
        energy_landscape = self.compute_energy(predictions)
        return predictions, energy_landscape
    
    def get_sphere_info(self):
        return [{
            'center': sphere.center.detach().numpy(),
            'radius': sphere.radius.detach().item(),
            'influence': sphere.influence_weight.detach().item()
        } for sphere in self.spheres]


def simple_pattern_analysis(X, y=None):
    """Simplified pattern analysis"""
    print("🔍 Analyzing data patterns...")
    
    # Basic clustering
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        print(f"   ✅ Found 3 clusters")
    except:
        centers = np.array([X.mean(axis=0)])
        print(f"   ✅ Using data centroid")
    
    # Simple circular pattern detection
    if X.shape[1] >= 2:
        center = X.mean(axis=0)
        distances = np.linalg.norm(X - center, axis=1)
        circularity = 1 - (distances.std() / (distances.mean() + 1e-8))
        print(f"   ✅ Circularity score: {circularity:.3f}")
    
    return {'clusters': centers, 'circularity': circularity if X.shape[1] >= 2 else 0}


def train_simple_model(model, X, y, epochs=1000, lr=0.01):
    """Simplified training function"""
    print("🚀 Training model...")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'losses': [], 'mse': [], 'energy': []}
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predictions, energy_landscape = model(X)
        
        mse_loss = F.mse_loss(predictions, y)
        energy_loss = energy_landscape.mean()
        total_loss = mse_loss + model.energy_weight * energy_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        history['losses'].append(total_loss.item())
        history['mse'].append(mse_loss.item())
        history['energy'].append(energy_loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}: Loss={total_loss.item():.4f}, "
                  f"MSE={mse_loss.item():.4f}, Energy={energy_loss.item():.4f}")
    
    print("   ✅ Training completed!")
    return history


def visualize_results(model, X, y, history, title="Multi-Geometry Results"):
    """Simplified visualization function"""
    print("🎨 Creating visualizations...")
    
    model.eval()
    with torch.no_grad():
        predictions, energy_landscape = model(X)
        predictions_np = predictions.numpy()
        energy_np = energy_landscape.numpy()
    
    X_np = X.numpy()
    y_np = y.numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Original Data
    ax = axes[0, 0]
    scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np.flatten(), cmap='viridis', alpha=0.7)
    ax.set_title('Original Data')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Target')
    
    # 2. Predictions vs True
    ax = axes[0, 1]
    ax.scatter(y_np.flatten(), predictions_np.flatten(), alpha=0.6)
    min_val, max_val = min(y_np.min(), predictions_np.min()), max(y_np.max(), predictions_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('Predictions vs True')
    ax.grid(True, alpha=0.3)
    
    # 3. Energy Landscape
    ax = axes[0, 2]
    scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=energy_np, cmap='plasma', alpha=0.7)
    ax.set_title('Energy Landscape')
    plt.colorbar(scatter, ax=ax, label='Energy')
    
    # 4. Geometric Primitives
    ax = axes[1, 0]
    ax.scatter(X_np[:, 0], X_np[:, 1], c='lightgray', alpha=0.5, s=20)
    
    # Draw spheres
    sphere_info = model.get_sphere_info()
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(sphere_info)))
    for i, info in enumerate(sphere_info):
        center = info['center']
        radius = info['radius']
        if len(center) >= 2:
            circle = plt.Circle((center[0], center[1]), radius, 
                              fill=False, color=colors[i], linewidth=2)
            ax.add_patch(circle)
            ax.plot(center[0], center[1], 'o', color=colors[i], markersize=8)
    
    ax.set_title('Adaptive Spheres')
    ax.set_aspect('equal')
    
    # 5. Training History
    ax = axes[1, 1]
    epochs = range(len(history['losses']))
    ax.plot(epochs, history['losses'], 'b-', label='Total Loss', linewidth=2)
    ax.plot(epochs, history['mse'], 'g-', label='MSE Loss', linewidth=1)
    ax.plot(epochs, history['energy'], 'r-', label='Energy Loss', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 6. Performance Metrics
    ax = axes[1, 2]
    residuals = predictions_np.flatten() - y_np.flatten()
    ax.scatter(predictions_np.flatten(), residuals, alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--')
    
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals**2) / np.sum((y_np.flatten() - y_np.mean())**2)
    
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residuals (MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ====================================================================================================
# STEP 2: COMPLETE WORKING EXAMPLE
# ====================================================================================================

def run_complete_example():
    """Run a complete working example of the multi-geometry energy model"""
    
    print("🌟 Multi-Geometry Energy Model - Complete Working Example")
    print("=" * 60)
    
    # ================================================================================================
    # STEP 1: GENERATE SAMPLE DATA
    # ================================================================================================
    
    print("\n📊 Step 1: Generating sample datasets...")
    
    # Dataset 1: Circular pattern
    X_circles, y_circles = make_circles(n_samples=200, factor=0.3, noise=0.1, random_state=42)
    y_circles = np.sin(np.linalg.norm(X_circles, axis=1) * 3).reshape(-1, 1)
    
    # Dataset 2: Moon pattern  
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    y_moons = (X_moons[:, 0] + X_moons[:, 1]).reshape(-1, 1)
    
    # Normalize data
    scaler = StandardScaler()
    X_circles = scaler.fit_transform(X_circles)
    X_moons = scaler.fit_transform(X_moons)
    
    datasets = {
        'circles': (X_circles, y_circles),
        'moons': (X_moons, y_moons)
    }
    
    print(f"   ✅ Generated {len(datasets)} datasets")
    for name, (X, y) in datasets.items():
        print(f"      - {name}: {X.shape[0]} samples, {X.shape[1]} features")
    
    # ================================================================================================
    # STEP 2: PATTERN ANALYSIS
    # ================================================================================================
    
    print("\n🔍 Step 2: Analyzing data patterns...")
    
    analysis_results = {}
    for name, (X, y) in datasets.items():
        print(f"\n   📈 Analyzing '{name}' dataset:")
        analysis = simple_pattern_analysis(X, y)
        analysis_results[name] = analysis
    
    # ================================================================================================
    # STEP 3: MODEL TRAINING
    # ================================================================================================
    
    print("\n🎯 Step 3: Training models...")
    
    results = {}
    
    for name, (X, y) in datasets.items():
        print(f"\n   🌟 Training model for '{name}' dataset:")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create model
        model = SimpleMultiGeometryModel(
            input_dim=X.shape[1], 
            output_dim=1, 
            n_spheres=3
        )
        
        # Train model
        history = train_simple_model(model, X_tensor, y_tensor, epochs=800, lr=0.01)
        
        # Store results
        results[name] = {
            'model': model,
            'history': history,
            'X': X_tensor,
            'y': y_tensor
        }
    
    # ================================================================================================
    # STEP 4: VISUALIZATION AND ANALYSIS
    # ================================================================================================
    
    print("\n🎨 Step 4: Creating visualizations...")
    
    figures = {}
    for name, result in results.items():
        print(f"   📊 Visualizing '{name}' results...")
        
        fig = visualize_results(
            result['model'], 
            result['X'], 
            result['y'], 
            result['history'],
            title=f"Multi-Geometry Results: {name.title()} Dataset"
        )
        figures[name] = fig
    
    # ================================================================================================
    # STEP 5: PERFORMANCE COMPARISON
    # ================================================================================================
    
    print("\n📊 Step 5: Performance comparison...")
    
    comparison_data = {
        'dataset': [],
        'final_loss': [],
        'mse': [],
        'sphere_positions': [],
        'sphere_radii': []
    }
    
    for name, result in results.items():
        model = result['model']
        history = result['history']
        X, y = result['X'], result['y']
        
        # Calculate final metrics
        model.eval()
        with torch.no_grad():
            predictions, _ = model(X)
            final_mse = F.mse_loss(predictions, y).item()
        
        sphere_info = model.get_sphere_info()
        
        comparison_data['dataset'].append(name)
        comparison_data['final_loss'].append(history['losses'][-1])
        comparison_data['mse'].append(final_mse)
        comparison_data['sphere_positions'].append([s['center'] for s in sphere_info])
        comparison_data['sphere_radii'].append([s['radius'] for s in sphere_info])
        
        print(f"   📈 {name}: Final MSE = {final_mse:.6f}")
    
    # Create comparison visualization
    fig_comparison = plt.figure(figsize=(15, 10))
    
    # Training curves comparison
    ax1 = plt.subplot(2, 3, 1)
    for name, result in results.items():
        epochs = range(len(result['history']['losses']))
        plt.plot(epochs, result['history']['losses'], label=f'{name} Total', linewidth=2)
        plt.plot(epochs, result['history']['mse'], label=f'{name} MSE', linewidth=1, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Final performance comparison
    ax2 = plt.subplot(2, 3, 2)
    datasets_names = comparison_data['dataset']
    mse_values = comparison_data['mse']
    bars = plt.bar(datasets_names, mse_values, alpha=0.7, color=['blue', 'green'])
    plt.xlabel('Dataset')
    plt.ylabel('Final MSE')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Sphere evolution visualization
    ax3 = plt.subplot(2, 3, 3)
    for i, name in enumerate(datasets_names):
        radii = comparison_data['sphere_radii'][i]
        x_pos = [i] * len(radii)
        plt.scatter(x_pos, radii, alpha=0.7, s=100, label=f'{name} spheres')
    plt.xlabel('Dataset')
    plt.ylabel('Sphere Radius')
    plt.title('Final Sphere Radii')
    plt.xticks(range(len(datasets_names)), datasets_names)
    plt.grid(True, alpha=0.3)
    
    # Energy landscape comparison
    for i, (name, result) in enumerate(results.items()):
        ax = plt.subplot(2, 3, 4 + i)
        model = result['model']
        X, y = result['X'], result['y']
        
        model.eval()
        with torch.no_grad():
            _, energy_landscape = model(X)
            energy_np = energy_landscape.numpy()
        
        X_np = X.numpy()
        scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=energy_np, cmap='plasma', alpha=0.7)
        ax.set_title(f'{name.title()} Energy Landscape')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax, label='Energy')
    
    plt.tight_layout()
    figures['comparison'] = fig_comparison
    
    # ================================================================================================
    # STEP 6: ADVANCED ANALYSIS
    # ================================================================================================
    
    print("\n🔬 Step 6: Advanced analysis...")
    
    # Demonstrate uncertainty quantification (simplified)
    print("   🎯 Uncertainty quantification demo:")
    
    for name, result in results.items():
        model = result['model']
        X, y = result['X'], result['y']
        
        # Simple uncertainty estimation using ensemble-like approach
        model.eval()
        predictions_list = []
        
        # Add noise to model parameters and collect predictions
        for _ in range(10):
            with torch.no_grad():
                # Add small noise to predictions to simulate uncertainty
                predictions, _ = model(X)
                noise = torch.randn_like(predictions) * 0.1
                predictions_list.append(predictions + noise)
        
        predictions_stack = torch.stack(predictions_list)
        mean_pred = predictions_stack.mean(dim=0)
        std_pred = predictions_stack.std(dim=0)
        
        avg_uncertainty = std_pred.mean().item()
        print(f"      {name}: Average uncertainty = {avg_uncertainty:.4f}")
    
    # Geometry adaptation analysis
    print("   🔄 Geometry adaptation analysis:")
    
    for name, result in results.items():
        sphere_info = result['model'].get_sphere_info()
        print(f"      {name}: {len(sphere_info)} spheres")
        for i, info in enumerate(sphere_info):
            print(f"         Sphere {i+1}: center={info['center'][:2]}, radius={info['radius']:.3f}")
    
    # ================================================================================================
    # STEP 7: SUMMARY AND RECOMMENDATIONS
    # ================================================================================================
    
    print("\n📋 Step 7: Summary and recommendations...")
    
    best_dataset = min(comparison_data['dataset'], 
                      key=lambda x: comparison_data['mse'][comparison_data['dataset'].index(x)])
    best_mse = min(comparison_data['mse'])
    
    print(f"   🏆 Best performing dataset: {best_dataset} (MSE: {best_mse:.6f})")
    print(f"   📊 Total datasets processed: {len(datasets)}")
    print(f"   🎨 Visualizations created: {len(figures)}")
    
    print("\n   💡 Key insights:")
    print("      - Multi-geometry energy models can adapt to different data patterns")
    print("      - Adaptive spheres provide flexible geometric primitives")
    print("      - Energy landscapes help visualize model behavior")
    print("      - Pattern analysis guides initialization")
    
    print("\n   🔧 Recommendations for further development:")
    print("      - Add more geometric primitives (torus, ellipsoids)")
    print("      - Implement more sophisticated geometry evolution")
    print("      - Enhance uncertainty quantification")
    print("      - Add automatic hyperparameter tuning")
    
    # ================================================================================================
    # FINAL RESULTS
    # ================================================================================================
    
    final_results = {
        'datasets': datasets,
        'analysis_results': analysis_results,
        'models': {name: result['model'] for name, result in results.items()},
        'training_historuses': {name: result['history'] for name, result in results.items()},
        'figures': figures,
        'comparison_data': comparison_data,
        'best_dataset': best_dataset,
        'best_mse': best_mse
    }
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE EXAMPLE FINISHED SUCCESSFULLY!")
    print("=" * 60)
    print(f"🎯 Check the returned results dictionary for all outputs")
    print(f"📊 {len(figures)} figures created and ready for display")
    
    return final_results


# ====================================================================================================
# STEP 3: UTILITY FUNCTIONS
# ====================================================================================================

def demonstrate_custom_dataset(X, y, dataset_name="custom"):
    """Demonstrate the system with a custom dataset"""
    print(f"🎯 Custom Dataset Demo: {dataset_name}")
    print("-" * 40)
    
    # Ensure proper format
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Pattern analysis
    analysis = simple_pattern_analysis(X_scaled, y)
    print(f"   🔍 Analysis completed")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    
    # Create and train model
    model = SimpleMultiGeometryModel(
        input_dim=X.shape[1], 
        output_dim=1, 
        n_spheres=3
    )
    
    history = train_simple_model(model, X_tensor, y_tensor, epochs=600, lr=0.01)
    
    # Visualize results
    fig = visualize_results(model, X_tensor, y_tensor, history, 
                          title=f"Multi-Geometry Results: {dataset_name}")
    
    # Calculate final metrics
    model.eval()
    with torch.no_grad():
        predictions, _ = model(X_tensor)
        final_mse = F.mse_loss(predictions, y_tensor).item()
        
        # R² score
        y_mean = y_tensor.mean()
        ss_tot = ((y_tensor - y_mean) ** 2).sum()
        ss_res = ((y_tensor - predictions) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot).item()
    
    print(f"   📈 Final MSE: {final_mse:.6f}")
    print(f"   📈 R² Score: {r2:.4f}")
    
    return {
        'model': model,
        'history': history,
        'figure': fig,
        'metrics': {'mse': final_mse, 'r2': r2},
        'analysis': analysis
    }


def quick_test():
    """Quick test function to verify everything works"""
    print("🚀 Quick Test of Multi-Geometry Energy Model")
    print("-" * 45)
    
    # Generate simple test data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
    
    return demonstrate_custom_dataset(X, y, "quick_test")


def create_comparison_with_baseline():
    """Create a comparison with baseline models"""
    print("🏁 Comparison with Baseline Models")
    print("-" * 35)
    
    # Generate test dataset
    X, y = make_circles(n_samples=200, factor=0.3, noise=0.1, random_state=42)
    y = np.sin(np.linalg.norm(X, axis=1) * 3).reshape(-1, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    
    # 1. Multi-Geometry Model
    print("   🌟 Training Multi-Geometry Model...")
    mg_model = SimpleMultiGeometryModel(input_dim=2, output_dim=1, n_spheres=3)
    mg_history = train_simple_model(mg_model, X_tensor, y_tensor, epochs=500, lr=0.01)
    
    mg_model.eval()
    with torch.no_grad():
        mg_pred, _ = mg_model(X_tensor)
        mg_mse = F.mse_loss(mg_pred, y_tensor).item()
    
    # 2. Simple Neural Network (baseline)
    print("   📊 Training Baseline Neural Network...")
    baseline_model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1)
    )
    
    optimizer = optim.AdamW(baseline_model.parameters(), lr=0.01)
    baseline_history = {'losses': [], 'mse': []}
    
    baseline_model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        predictions = baseline_model(X_tensor)
        loss = F.mse_loss(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
        baseline_history['losses'].append(loss.item())
        baseline_history['mse'].append(loss.item())
    
    baseline_model.eval()
    with torch.no_grad():
        baseline_pred = baseline_model(X_tensor)
        baseline_mse = F.mse_loss(baseline_pred, y_tensor).item()
    
    # Comparison
    print(f"\n   📊 Results Comparison:")
    print(f"      Multi-Geometry MSE: {mg_mse:.6f}")
    print(f"      Baseline NN MSE:    {baseline_mse:.6f}")
    print(f"      Improvement:        {((baseline_mse - mg_mse) / baseline_mse * 100):+.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training curves
    ax = axes[0]
    epochs = range(len(mg_history['losses']))
    ax.plot(epochs, mg_history['losses'], 'b-', label='Multi-Geometry', linewidth=2)
    ax.plot(epochs, baseline_history['losses'], 'r-', label='Baseline NN', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Predictions comparison
    X_np = X_scaled
    y_np = y
    mg_pred_np = mg_pred.numpy()
    baseline_pred_np = baseline_pred.numpy()
    
    ax = axes[1]
    ax.scatter(y_np.flatten(), mg_pred_np.flatten(), alpha=0.6, label='Multi-Geometry')
    ax.scatter(y_np.flatten(), baseline_pred_np.flatten(), alpha=0.6, label='Baseline NN')
    min_val = min(y_np.min(), mg_pred_np.min(), baseline_pred_np.min())
    max_val = max(y_np.max(), mg_pred_np.max(), baseline_pred_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('Predictions vs True')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Geometric visualization
    ax = axes[2]
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np.flatten(), cmap='viridis', alpha=0.7)
    
    # Draw spheres from multi-geometry model
    sphere_info = mg_model.get_sphere_info()
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(sphere_info)))
    for i, info in enumerate(sphere_info):
        center = info['center']
        radius = info['radius']
        if len(center) >= 2:
            circle = plt.Circle((center[0], center[1]), radius, 
                              fill=False, color=colors[i], linewidth=2)
            ax.add_patch(circle)
    
    ax.set_title('Data + Learned Geometries')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return {
        'multi_geometry': {'model': mg_model, 'mse': mg_mse, 'history': mg_history},
        'baseline': {'model': baseline_model, 'mse': baseline_mse, 'history': baseline_history},
        'improvement_percent': (baseline_mse - mg_mse) / baseline_mse * 100,
        'comparison_figure': fig
    }


# ====================================================================================================
# MAIN EXECUTION
# ====================================================================================================

if __name__ == "__main__":
    print("🌟 Multi-Geometry Energy Model - Integration Example")
    print("Choose an option:")
    print("1. Run complete example")
    print("2. Quick test")
    print("3. Baseline comparison")
    
    choice = input("Enter choice (1-3, or press Enter for complete example): ").strip()
    
    if choice == "2":
        results = quick_test()
    elif choice == "3":
        results = create_comparison_with_baseline()
    else:
        results = run_complete_example()
    
    print("\n🎉 Integration example completed!")
    print("💡 All figures are stored in the results dictionary")
    print("📊 Use plt.show() to display the figures")