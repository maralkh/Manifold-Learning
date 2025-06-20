"""
Comprehensive Visualization System
==================================

Advanced visualization tools for multi-geometry energy models including
training dynamics, energy landscapes, geometry evolution, and uncertainty.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.animation import FuncAnimation
import torch
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


class MultiGeometryVisualizer:
    """Comprehensive visualization system for multi-geometry energy models"""
    
    def __init__(self, figsize=(20, 16), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.color_schemes = {
            'spheres': plt.cm.Reds,
            'torus': plt.cm.Blues,
            'ellipsoids': plt.cm.Greens,
            'energy': plt.cm.plasma,
            'predictions': plt.cm.viridis
        }
    
    def visualize_complete_analysis(self, model, trainer, X, y, title="Multi-Geometry Analysis"):
        """Create comprehensive visualization of model and training"""
        print("🎨 Creating comprehensive visualization...")
        
        # Prepare data
        X_np, y_np = self._prepare_data(X, y)
        
        # Get model predictions and components
        model_data = self._get_model_data(model, X, y)
        
        # Create main figure
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Create subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Original Data and Predictions
        self._plot_data_and_predictions(fig, gs[0, 0], X_np, y_np, model_data)
        
        # 2. Energy Landscape
        self._plot_energy_landscape(fig, gs[0, 1], X_np, model_data)
        
        # 3. Geometric Primitives
        self._plot_geometric_primitives(fig, gs[0, 2], X_np, model_data)
        
        # 4. Uncertainty Visualization
        self._plot_uncertainty(fig, gs[0, 3], X_np, y_np, model, model_data)
        
        # 5. Training History
        if trainer and trainer.training_history:
            self._plot_training_history(fig, gs[1, :2], trainer.training_history)
            
            # 6. Geometry Evolution
            self._plot_geometry_evolution(fig, gs[1, 2:], trainer.training_history)
        
        # 7. Energy Components Analysis
        self._plot_energy_components(fig, gs[2, :2], model_data)
        
        # 8. Model Performance Metrics
        self._plot_performance_metrics(fig, gs[2, 2:], X_np, y_np, model_data)
        
        plt.tight_layout()
        return fig
    
    def visualize_energy_landscape_3d(self, model, X, y, resolution=50):
        """Create 3D visualization of energy landscape"""
        print("🌄 Creating 3D energy landscape...")
        
        X_np, y_np = self._prepare_data(X, y)
        
        if X_np.shape[1] < 2:
            print("⚠️ Need at least 2D input for 3D energy landscape")
            return None
        
        # Create grid for energy landscape
        x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
        y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Create input for energy computation
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        if X_np.shape[1] > 2:
            # Add mean values for extra dimensions
            extra_dims = np.tile(X_np[:, 2:].mean(axis=0), (len(grid_points), 1))
            grid_points = np.column_stack([grid_points, extra_dims])
        
        # Get energy values
        model.eval()
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid_points)
            predictions, energy_landscape, _ = model(grid_tensor)
            energy_values = energy_landscape.numpy().reshape(resolution, resolution)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(xx, yy, energy_values, cmap='plasma', alpha=0.7)
        
        # Add data points
        model_data = self._get_model_data(model, X, y)
        scatter = ax.scatter(X_np[:, 0], X_np[:, 1], model_data['energy_landscape'], 
                           c=y_np.flatten(), cmap='viridis', s=50)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Energy')
        ax.set_title('3D Energy Landscape')
        
        plt.colorbar(surf, ax=ax, shrink=0.5, label='Energy')
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='Target Value')
        
        return fig
    
    def animate_training_progress(self, training_history, save_path=None):
        """Create animation of training progress"""
        print("🎬 Creating training animation...")
        
        if not training_history or 'train_losses' not in training_history:
            print("⚠️ No training history available for animation")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        def animate(frame):
            for ax in axes.flat:
                ax.clear()
            
            epochs = range(min(frame + 1, len(training_history['train_losses']['total_loss'])))
            
            # Loss curves
            ax = axes[0, 0]
            if epochs:
                ax.plot(epochs, training_history['train_losses']['total_loss'][:len(epochs)], 
                       'b-', label='Train Total')
                ax.plot(epochs, training_history['train_losses']['mse_loss'][:len(epochs)], 
                       'g-', label='Train MSE')
                if 'val_losses' in training_history and training_history['val_losses']:
                    ax.plot(epochs, training_history['val_losses']['total_loss'][:len(epochs)], 
                           'r--', label='Val Total')
            ax.set_title('Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Geometry counts
            ax = axes[0, 1]
            if 'geometry_counts' in training_history:
                for geom_type in ['spheres', 'torus', 'ellipsoids']:
                    if geom_type in training_history['geometry_counts']:
                        counts = training_history['geometry_counts'][geom_type][:len(epochs)]
                        if counts:
                            ax.plot(epochs, counts, 'o-', label=geom_type.title())
            ax.set_title('Geometry Evolution')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Energy components
            ax = axes[1, 0]
            if epochs:
                ax.plot(epochs, training_history['train_losses']['sphere_energy'][:len(epochs)], 
                       label='Sphere Energy')
                ax.plot(epochs, training_history['train_losses']['torus_energy'][:len(epochs)], 
                       label='Torus Energy')
                ax.plot(epochs, training_history['train_losses']['ellipsoid_energy'][:len(epochs)], 
                       label='Ellipsoid Energy')
            ax.set_title('Energy Components')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Energy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Learning rate
            ax = axes[1, 1]
            if 'learning_rates' in training_history and len(training_history['learning_rates']) > frame:
                lrs = training_history['learning_rates'][:len(epochs)]
                if lrs:
                    ax.semilogy(epochs, lrs)
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        n_frames = len(training_history['train_losses']['total_loss'])
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim
    
    def plot_uncertainty_analysis(self, model, X, y, n_samples=100):
        """Detailed uncertainty analysis visualization"""
        print("🎯 Creating uncertainty analysis...")
        
        X_np, y_np = self._prepare_data(X, y)
        
        # Get uncertainty predictions
        uncertainty_data = model.predict_with_uncertainty(
            torch.FloatTensor(X_np), n_samples=n_samples
        )
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Predictions vs True with uncertainty
        ax = axes[0, 0]
        predictions = uncertainty_data['predictions'].numpy()
        uncertainty = uncertainty_data['uncertainty'].numpy()
        
        scatter = ax.errorbar(y_np.flatten(), predictions.flatten(), 
                            yerr=uncertainty.flatten(), fmt='o', alpha=0.6, capsize=3)
        
        min_val = min(y_np.min(), predictions.min())
        max_val = max(y_np.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title('Predictions vs True (with Uncertainty)')
        ax.grid(True, alpha=0.3)
        
        # 2. Uncertainty vs Prediction Error
        ax = axes[0, 1]
        errors = np.abs(predictions.flatten() - y_np.flatten())
        ax.scatter(uncertainty.flatten(), errors, alpha=0.6)
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Uncertainty vs Error')
        ax.grid(True, alpha=0.3)
        
        # 3. Spatial uncertainty (if 2D input)
        ax = axes[0, 2]
        if X_np.shape[1] >= 2:
            scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=uncertainty.flatten(), 
                               cmap='Reds', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Uncertainty')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        ax.set_title('Spatial Uncertainty Distribution')
        
        # 4. Uncertainty histogram
        ax = axes[1, 0]
        ax.hist(uncertainty.flatten(), bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(uncertainty.mean(), color='red', linestyle='--', 
                  label=f'Mean: {uncertainty.mean():.3f}')
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Frequency')
        ax.set_title('Uncertainty Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Sample predictions
        ax = axes[1, 1]
        samples = uncertainty_data['samples'].numpy()
        for i in range(min(10, n_samples)):
            ax.plot(samples[i].flatten(), alpha=0.3, color='blue')
        ax.plot(predictions.flatten(), 'r-', linewidth=2, label='Mean Prediction')
        ax.fill_between(range(len(predictions.flatten())), 
                       (predictions - uncertainty).flatten(),
                       (predictions + uncertainty).flatten(),
                       alpha=0.3, color='red', label='Uncertainty Band')
        ax.set_xlabel('Data Point')
        ax.set_ylabel('Prediction')
        ax.set_title('Prediction Samples')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Energy landscape uncertainty
        ax = axes[1, 2]
        energy_landscape = uncertainty_data['energy_landscape'].numpy()
        if X_np.shape[1] >= 2:
            scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=energy_landscape, 
                               cmap='plasma', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Energy')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        ax.set_title('Energy Landscape')
        
        plt.tight_layout()
        return fig
    
    def _prepare_data(self, X, y):
        """Prepare data for visualization"""
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = np.array(X)
        
        if torch.is_tensor(y):
            y_np = y.numpy()
        else:
            y_np = np.array(y)
        
        return X_np, y_np
    
    def _get_model_data(self, model, X, y):
        """Get model predictions and energy components"""
        model.eval()
        with torch.no_grad():
            if torch.is_tensor(X):
                X_tensor = X
            else:
                X_tensor = torch.FloatTensor(X)
            
            predictions, energy_landscape, energy_components = model(X_tensor)
            
            return {
                'predictions': predictions.numpy(),
                'energy_landscape': energy_landscape.numpy(),
                'sphere_energy': energy_components['sphere_energy'].numpy(),
                'torus_energy': energy_components['torus_energy'].numpy(),
                'ellipsoid_energy': energy_components['ellipsoid_energy'].numpy(),
                'interaction_energy': energy_components['interaction_energy'].numpy(),
                'geometry_info': model.get_geometry_info()
            }
    
    def _plot_data_and_predictions(self, fig, gs_pos, X_np, y_np, model_data):
        """Plot original data and predictions"""
        ax = fig.add_subplot(gs_pos)
        
        # Use first two dimensions for plotting
        x_plot = X_np[:, 0]
        y_plot = X_np[:, 1] if X_np.shape[1] > 1 else np.zeros_like(x_plot)
        
        # Plot original data
        scatter1 = ax.scatter(x_plot, y_plot, c=y_np.flatten(), 
                             cmap='viridis', alpha=0.6, s=50, label='True')
        
        # Plot predictions
        scatter2 = ax.scatter(x_plot, y_plot, c=model_data['predictions'].flatten(), 
                             cmap='plasma', alpha=0.4, s=30, marker='x', label='Predicted')
        
        ax.set_title('Data and Predictions')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        
        # Add colorbars
        plt.colorbar(scatter1, ax=ax, label='True Values', shrink=0.8)
    
    def _plot_energy_landscape(self, fig, gs_pos, X_np, model_data):
        """Plot energy landscape"""
        ax = fig.add_subplot(gs_pos)
        
        x_plot = X_np[:, 0]
        y_plot = X_np[:, 1] if X_np.shape[1] > 1 else np.zeros_like(x_plot)
        
        scatter = ax.scatter(x_plot, y_plot, c=model_data['energy_landscape'], 
                           cmap='plasma', alpha=0.7, s=50)
        
        ax.set_title('Energy Landscape')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax, label='Energy', shrink=0.8)
    
    def _plot_geometric_primitives(self, fig, gs_pos, X_np, model_data):
        """Plot geometric primitives"""
        ax = fig.add_subplot(gs_pos)
        
        x_plot = X_np[:, 0]
        y_plot = X_np[:, 1] if X_np.shape[1] > 1 else np.zeros_like(x_plot)
        
        # Plot data points
        ax.scatter(x_plot, y_plot, c='lightgray', alpha=0.5, s=20)
        
        geometry_info = model_data['geometry_info']
        
        # Draw spheres
        colors_spheres = self.color_schemes['spheres'](
            np.linspace(0.3, 1, len(geometry_info['details']['spheres']))
        )
        for i, sphere_info in enumerate(geometry_info['details']['spheres']):
            center = sphere_info['center']
            radius = sphere_info['radius']
            
            if len(center) >= 2:
                circle = plt.Circle((center[0], center[1]), radius, 
                                  fill=False, color=colors_spheres[i], linewidth=2)
                ax.add_patch(circle)
                ax.plot(center[0], center[1], 'o', color=colors_spheres[i], markersize=8)
        
        # Draw torus
        colors_torus = self.color_schemes['torus'](
            np.linspace(0.3, 1, len(geometry_info['details']['torus']))
        )
        for i, torus_info in enumerate(geometry_info['details']['torus']):
            center = torus_info['center']
            major_radius = torus_info['major_radius']
            
            if len(center) >= 2:
                circle = plt.Circle((center[0], center[1]), major_radius, 
                                  fill=False, color=colors_torus[i], linewidth=2, linestyle='--')
                ax.add_patch(circle)
                ax.plot(center[0], center[1], 's', color=colors_torus[i], markersize=8)
        
        # Draw ellipsoids (as ellipses in 2D)
        colors_ellipsoids = self.color_schemes['ellipsoids'](
            np.linspace(0.3, 1, len(geometry_info['details']['ellipsoids']))
        )
        for i, ellipsoid_info in enumerate(geometry_info['details']['ellipsoids']):
            center = ellipsoid_info['center']
            radii = ellipsoid_info['radii']
            
            if len(center) >= 2 and len(radii) >= 2:
                ellipse = plt.matplotlib.patches.Ellipse(
                    (center[0], center[1]), 2*radii[0], 2*radii[1],
                    fill=False, color=colors_ellipsoids[i], linewidth=2, linestyle=':'
                )
                ax.add_patch(ellipse)
                ax.plot(center[0], center[1], '^', color=colors_ellipsoids[i], markersize=8)
        
        ax.set_title('Geometric Primitives')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_uncertainty(self, fig, gs_pos, X_np, y_np, model, model_data):
        """Plot uncertainty visualization"""
        ax = fig.add_subplot(gs_pos)
        
        # Get uncertainty data
        try:
            uncertainty_data = model.predict_with_uncertainty(torch.FloatTensor(X_np), n_samples=50)
            uncertainty = uncertainty_data['uncertainty'].numpy()
            
            x_plot = X_np[:, 0]
            y_plot = X_np[:, 1] if X_np.shape[1] > 1 else np.zeros_like(x_plot)
            
            scatter = ax.scatter(x_plot, y_plot, c=uncertainty.flatten(), 
                               cmap='Reds', alpha=0.7, s=50)
            
            ax.set_title('Prediction Uncertainty')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            plt.colorbar(scatter, ax=ax, label='Uncertainty', shrink=0.8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Uncertainty calculation failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Uncertainty (Error)')
    
    def _plot_training_history(self, fig, gs_pos, training_history):
        """Plot training history"""
        ax = fig.add_subplot(gs_pos)
        
        if 'train_losses' in training_history:
            epochs = range(len(training_history['train_losses']['total_loss']))
            
            ax.plot(epochs, training_history['train_losses']['total_loss'], 
                   'b-', label='Train Total', linewidth=2)
            ax.plot(epochs, training_history['train_losses']['mse_loss'], 
                   'g-', label='Train MSE', linewidth=1)
            
            if 'val_losses' in training_history and training_history['val_losses']:
                ax.plot(epochs, training_history['val_losses']['total_loss'], 
                       'r--', label='Val Total', linewidth=2)
                ax.plot(epochs, training_history['val_losses']['mse_loss'], 
                       'm--', label='Val MSE', linewidth=1)
        
        ax.set_title('Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_geometry_evolution(self, fig, gs_pos, training_history):
        """Plot geometry evolution"""
        ax = fig.add_subplot(gs_pos)
        
        if 'geometry_counts' in training_history:
            epochs = range(len(list(training_history['geometry_counts'].values())[0]))
            
            for geom_type in ['spheres', 'torus', 'ellipsoids']:
                if geom_type in training_history['geometry_counts']:
                    counts = training_history['geometry_counts'][geom_type]
                    ax.plot(epochs, counts, 'o-', label=geom_type.title(), linewidth=2)
        
        ax.set_title('Geometry Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_energy_components(self, fig, gs_pos, model_data):
        """Plot energy components analysis"""
        ax = fig.add_subplot(gs_pos)
        
        components = ['sphere_energy', 'torus_energy', 'ellipsoid_energy', 'interaction_energy']
        values = [model_data[comp].mean() for comp in components]
        labels = ['Spheres', 'torus', 'Ellipsoids', 'Interactions']
        
        colors = ['red', 'blue', 'green', 'orange']
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        
        ax.set_title('Energy Components')
        ax.set_ylabel('Average Energy')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_performance_metrics(self, fig, gs_pos, X_np, y_np, model_data):
        """Plot performance metrics"""
        ax = fig.add_subplot(gs_pos)
        
        predictions = model_data['predictions']
        
        # Calculate metrics
        mse = np.mean((predictions.flatten() - y_np.flatten())**2)
        mae = np.mean(np.abs(predictions.flatten() - y_np.flatten()))
        
        y_mean = y_np.mean()
        ss_tot = np.sum((y_np.flatten() - y_mean)**2)
        ss_res = np.sum((y_np.flatten() - predictions.flatten())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Create residuals plot
        residuals = predictions.flatten() - y_np.flatten()
        ax.scatter(predictions.flatten(), residuals, alpha=0.6)
        ax.axhline(y=0, color='red', linestyle='--')
        
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals (MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f})')
        ax.grid(True, alpha=0.3)
