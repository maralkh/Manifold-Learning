"""
Complete Demo and Usage Examples
================================

Comprehensive demonstration of the multi-geometry energy model system
with various datasets and use cases.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our modules (in practice, these would be separate files)
from .geometric_primitives import AdaptiveSphere, AdaptiveTorus, AdaptiveEllipsoid, GeometryManager
from .pattern_analyzer import DataPatternAnalyzer
from .energy_model import MultiGeometryEnergyModel
from .trainer import MultiGeometryTrainer
from .visualization import MultiGeometryVisualizer


class CompleteMultiGeometryDemo:
    """Complete demonstration of the multi-geometry energy model system"""
    
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.trainers = {}
        self.results = {}
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        print("🚀 Multi-Geometry Energy Model Demo System")
        print("=" * 50)
    
    def generate_demo_datasets(self):
        """Generate various datasets for demonstration"""
        print("📊 Generating demo datasets...")
        
        # 1. Circular pattern dataset
        X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.1, random_state=42)
        y_circles = np.sin(np.linalg.norm(X_circles, axis=1) * 3).reshape(-1, 1)
        
        # 2. Moon-shaped dataset
        X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
        y_moons = (X_moons[:, 0] + X_moons[:, 1]).reshape(-1, 1)
        
        # 3. Swiss roll (3D to 1D)
        X_swiss, y_swiss = make_swiss_roll(n_samples=300, noise=0.1, random_state=42)
        X_swiss = X_swiss[:, [0, 2]]  # Use only 2D projection
        y_swiss = y_swiss.reshape(-1, 1)
        
        # 4. Complex sinusoidal pattern
        X_complex = np.random.uniform(-2, 2, (300, 2))
        y_complex = (np.sin(X_complex[:, 0] * 2) * np.cos(X_complex[:, 1] * 2) + 
                    np.sin(np.linalg.norm(X_complex, axis=1))).reshape(-1, 1)
        
        # 5. Multi-scale pattern
        X_multiscale = np.random.uniform(-3, 3, (300, 2))
        y_multiscale = (np.sin(X_multiscale[:, 0]) + 
                       0.5 * np.sin(X_multiscale[:, 0] * 5) +
                       0.3 * np.cos(X_multiscale[:, 1] * 3)).reshape(-1, 1)
        
        # Normalize all datasets
        scaler = StandardScaler()
        
        datasets = {
            'circles': (scaler.fit_transform(X_circles), y_circles),
            'moons': (scaler.fit_transform(X_moons), y_moons),
            'swiss_roll': (scaler.fit_transform(X_swiss), y_swiss),
            'complex': (scaler.fit_transform(X_complex), y_complex),
            'multiscale': (scaler.fit_transform(X_multiscale), y_multiscale)
        }
        
        self.datasets = datasets
        
        print(f"   ✅ Generated {len(datasets)} datasets:")
        for name, (X, y) in datasets.items():
            print(f"      - {name}: {X.shape[0]} samples, {X.shape[1]} features")
        
        return datasets
    
    def run_pattern_analysis(self, dataset_name=None):
        """Run pattern analysis on datasets"""
        print("\n🔍 Running Pattern Analysis...")
        
        datasets_to_analyze = [dataset_name] if dataset_name else list(self.datasets.keys())
        
        analysis_results = {}
        
        for name in datasets_to_analyze:
            if name not in self.datasets:
                print(f"⚠️ Dataset '{name}' not found")
                continue
            
            print(f"\n📈 Analyzing '{name}' dataset:")
            X, y = self.datasets[name]
            
            # Initialize analyzer
            analyzer = DataPatternAnalyzer(min_clusters=2, max_clusters=6)
            
            # Run analysis
            analysis = analyzer.analyze_patterns(X, y)
            
            # Get initialization suggestions
            suggestions = analyzer.suggest_initialization(output_dim=1, max_geometries={
                'spheres': 4, 'torus': 3, 'ellipsoids': 2
            })
            
            analysis_results[name] = {
                'analysis': analysis,
                'suggestions': suggestions,
                'analyzer': analyzer
            }
            
            print(f"   💡 Suggestions: {len(suggestions['spheres'])} spheres, "
                  f"{len(suggestions['torus'])} torus, {len(suggestions['ellipsoids'])} ellipsoids")
        
        return analysis_results
    
    def train_models(self, dataset_names=None, epochs=1000):
        """Train models on specified datasets"""
        print("\n🎯 Training Multi-Geometry Models...")
        
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        training_results = {}
        
        for name in dataset_names:
            if name not in self.datasets:
                print(f"⚠️ Dataset '{name}' not found")
                continue
            
            print(f"\n🌟 Training model for '{name}' dataset:")
            X, y = self.datasets[name]
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Configure model based on dataset characteristics
            complexity_score = self._determine_complexity(name)
            
            model_config = {
                'max_spheres': 4,
                'max_torus': 3,
                'max_ellipsoids': 2,
                'complexity_score': complexity_score,
                'adaptive_geometry': True,
                'energy_weight': 0.02,
                'interaction_strength': 0.03
            }
            
            trainer_config = {
                'use_geometry_evolution': True,
                'use_adaptive_learning': True,
                'use_early_stopping': True,
                'patience': 150,
                'validation_split': 0.2
            }
            
            # Create model
            model = MultiGeometryEnergyModel(
                input_dim=X.shape[1], 
                output_dim=1, 
                config=model_config
            )
            
            # Initialize from pattern analysis if available
            if hasattr(self, 'analysis_results') and name in self.analysis_results:
                suggestions = self.analysis_results[name]['suggestions']
                model.initialize_from_analysis(suggestions)
            
            # Create trainer
            trainer = MultiGeometryTrainer(model, config=trainer_config)
            
            # Train model
            training_history = trainer.train(
                X_tensor, y_tensor, 
                epochs=epochs, 
                lr=0.01, 
                verbose=True
            )
            
            # Evaluate model
            evaluation = trainer.evaluate(X_tensor, y_tensor)
            
            # Store results
            training_results[name] = {
                'model': model,
                'trainer': trainer,
                'training_history': training_history,
                'evaluation': evaluation,
                'config': {
                    'model_config': model_config,
                    'trainer_config': trainer_config
                }
            }
            
            print(f"   ✅ Training completed for '{name}':")
            print(f"      MSE: {evaluation['mse']:.6f}")
            print(f"      R²: {evaluation['r2']:.4f}")
            
        self.results = training_results
        return training_results
    
    def create_comprehensive_visualizations(self, dataset_names=None):
        """Create comprehensive visualizations for all results"""
        print("\n🎨 Creating Comprehensive Visualizations...")
        
        if not self.results:
            print("⚠️ No training results found. Run train_models() first.")
            return {}
        
        if dataset_names is None:
            dataset_names = list(self.results.keys())
        
        visualizer = MultiGeometryVisualizer(figsize=(20, 16))
        figures = {}
        
        for name in dataset_names:
            if name not in self.results:
                print(f"⚠️ No results for dataset '{name}'")
                continue
            
            print(f"📊 Creating visualization for '{name}' dataset...")
            
            X, y = self.datasets[name]
            model = self.results[name]['model']
            trainer = self.results[name]['trainer']
            
            # Main comprehensive analysis
            fig_main = visualizer.visualize_complete_analysis(
                model, trainer, X, y, 
                title=f"Multi-Geometry Analysis: {name.title()} Dataset"
            )
            
            figures[f'{name}_main'] = fig_main
            
            # 3D energy landscape (if applicable)
            if X.shape[1] >= 2:
                try:
                    fig_3d = visualizer.visualize_energy_landscape_3d(model, X, y)
                    if fig_3d:
                        figures[f'{name}_3d'] = fig_3d
                except Exception as e:
                    print(f"⚠️ 3D visualization failed for {name}: {e}")
            
            # Uncertainty analysis
            try:
                fig_uncertainty = visualizer.plot_uncertainty_analysis(model, X, y, n_samples=50)
                figures[f'{name}_uncertainty'] = fig_uncertainty
            except Exception as e:
                print(f"⚠️ Uncertainty analysis failed for {name}: {e}")
        
        return figures
    
    def run_comparative_analysis(self):
        """Run comparative analysis across all datasets"""
        print("\n📊 Running Comparative Analysis...")
        
        if not self.results:
            print("⚠️ No training results found. Run train_models() first.")
            return
        
        # Collect metrics
        comparison_data = {
            'dataset': [],
            'mse': [],
            'mae': [],
            'r2': [],
            'final_spheres': [],
            'final_torus': [],
            'final_ellipsoids': [],
            'training_time': [],
            'final_loss': [],
            'geometry_evolutions': []
        }
        
        for name, result in self.results.items():
            evaluation = result['evaluation']
            training_history = result['training_history']
            geometry_info = result['model'].get_geometry_info()
            
            comparison_data['dataset'].append(name)
            comparison_data['mse'].append(evaluation['mse'])
            comparison_data['mae'].append(evaluation['mae'])
            comparison_data['r2'].append(evaluation['r2'])
            comparison_data['final_spheres'].append(geometry_info['counts']['spheres'])
            comparison_data['final_torus'].append(geometry_info['counts']['torus'])
            comparison_data['final_ellipsoids'].append(geometry_info['counts']['ellipsoids'])
            comparison_data['training_time'].append(training_history.get('total_training_time', 0))
            comparison_data['final_loss'].append(training_history['train_losses']['total_loss'][-1])
            comparison_data['geometry_evolutions'].append(len(training_history.get('geometry_evolution', [])))
        
        # Create comparative visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Performance metrics
        ax = axes[0, 0]
        x_pos = np.arange(len(comparison_data['dataset']))
        ax.bar(x_pos, comparison_data['r2'], alpha=0.7)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance (R² Score)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison_data['dataset'], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Geometry counts
        ax = axes[0, 1]
        width = 0.25
        x_pos = np.arange(len(comparison_data['dataset']))
        ax.bar(x_pos - width, comparison_data['final_spheres'], width, label='Spheres', alpha=0.7)
        ax.bar(x_pos, comparison_data['final_torus'], width, label='torus', alpha=0.7)
        ax.bar(x_pos + width, comparison_data['final_ellipsoids'], width, label='Ellipsoids', alpha=0.7)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Geometry Count')
        ax.set_title('Final Geometry Counts')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison_data['dataset'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training time
        ax = axes[0, 2]
        ax.bar(x_pos, comparison_data['training_time'], alpha=0.7)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Training Time')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison_data['dataset'], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # MSE vs MAE
        ax = axes[1, 0]
        ax.scatter(comparison_data['mse'], comparison_data['mae'], s=100, alpha=0.7)
        for i, dataset in enumerate(comparison_data['dataset']):
            ax.annotate(dataset, (comparison_data['mse'][i], comparison_data['mae'][i]))
        ax.set_xlabel('MSE')
        ax.set_ylabel('MAE')
        ax.set_title('MSE vs MAE')
        ax.grid(True, alpha=0.3)
        
        # Geometry evolutions
        ax = axes[1, 1]
        ax.bar(x_pos, comparison_data['geometry_evolutions'], alpha=0.7)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Number of Evolutions')
        ax.set_title('Geometry Evolution Events')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison_data['dataset'], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Performance vs Complexity
        ax = axes[1, 2]
        complexity_scores = [self._determine_complexity(name) for name in comparison_data['dataset']]
        ax.scatter(complexity_scores, comparison_data['r2'], s=100, alpha=0.7)
        for i, dataset in enumerate(comparison_data['dataset']):
            ax.annotate(dataset, (complexity_scores[i], comparison_data['r2'][i]))
        ax.set_xlabel('Complexity Score')
        ax.set_ylabel('R² Score')
        ax.set_title('Performance vs Model Complexity')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Comparative Analysis Across Datasets', fontsize=16, y=1.02)
        
        return fig, comparison_data
    
    def demonstrate_uncertainty_quantification(self, dataset_name='complex'):
        """Demonstrate uncertainty quantification capabilities"""
        print(f"\n🎯 Demonstrating Uncertainty Quantification on '{dataset_name}' dataset...")
        
        if dataset_name not in self.results:
            print(f"⚠️ No results for dataset '{dataset_name}'")
            return
        
        X, y = self.datasets[dataset_name]
        model = self.results[dataset_name]['model']
        
        # Get uncertainty predictions
        uncertainty_data = model.predict_with_uncertainty(torch.FloatTensor(X), n_samples=100)
        
        # Analysis
        predictions = uncertainty_data['predictions'].numpy()
        mean_predictions = uncertainty_data['mean_prediction'].numpy()
        uncertainty = uncertainty_data['uncertainty'].numpy()
        
        # Calculate uncertainty metrics
        prediction_error = np.abs(predictions.flatten() - y.flatten())
        uncertainty_correlation = np.corrcoef(uncertainty.flatten(), prediction_error)[0, 1]
        
        print(f"   📊 Uncertainty Analysis Results:")
        print(f"      Mean uncertainty: {uncertainty.mean():.4f}")
        print(f"      Uncertainty std: {uncertainty.std():.4f}")
        print(f"      Uncertainty-error correlation: {uncertainty_correlation:.4f}")
        
        # High uncertainty regions
        high_uncertainty_mask = uncertainty.flatten() > np.percentile(uncertainty, 75)
        high_uncertainty_error = prediction_error[high_uncertainty_mask].mean()
        low_uncertainty_error = prediction_error[~high_uncertainty_mask].mean()
        
        print(f"      High uncertainty region error: {high_uncertainty_error:.4f}")
        print(f"      Low uncertainty region error: {low_uncertainty_error:.4f}")
        print(f"      Uncertainty effectiveness ratio: {high_uncertainty_error/low_uncertainty_error:.2f}")
        
        return uncertainty_data
    
    def run_complete_demo(self, quick_mode=False):
        """Run the complete demonstration"""
        print("\n" + "="*70)
        print("🚀 RUNNING COMPLETE MULTI-GEOMETRY ENERGY MODEL DEMO")
        print("="*70)
        
        # 1. Generate datasets
        self.generate_demo_datasets()
        
        # 2. Run pattern analysis
        self.analysis_results = self.run_pattern_analysis()
        
        # 3. Train models (reduced epochs for quick mode)
        epochs = 500 if quick_mode else 1000
        training_results = self.train_models(epochs=epochs)
        
        # 4. Create visualizations
        figures = self.create_comprehensive_visualizations()
        
        # 5. Comparative analysis
        comparison_fig, comparison_data = self.run_comparative_analysis()
        
        # 6. Uncertainty demonstration
        uncertainty_data = self.demonstrate_uncertainty_quantification()
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("📊 Generated visualizations:")
        for fig_name in figures.keys():
            print(f"   - {fig_name}")
        
        print(f"\n🎯 Best performing dataset: {self._find_best_dataset(comparison_data)}")
        print(f"🔬 Most complex dataset: {self._find_most_complex_dataset()}")
        
        return {
            'datasets': self.datasets,
            'analysis_results': self.analysis_results,
            'training_results': training_results,
            'figures': figures,
            'comparison_data': comparison_data,
            'uncertainty_data': uncertainty_data
        }
    
    def _determine_complexity(self, dataset_name):
        """Determine model complexity based on dataset characteristics"""
        complexity_map = {
            'circles': 0.4,
            'moons': 0.3,
            'swiss_roll': 0.6,
            'complex': 0.8,
            'multiscale': 0.9
        }
        return complexity_map.get(dataset_name, 0.5)
    
    def _find_best_dataset(self, comparison_data):
        """Find the best performing dataset"""
        best_idx = np.argmax(comparison_data['r2'])
        return comparison_data['dataset'][best_idx]
    
    def _find_most_complex_dataset(self):
        """Find the most complex dataset"""
        complexities = {name: self._determine_complexity(name) for name in self.datasets.keys()}
        return max(complexities, key=complexities.get)


# Usage examples and quick start functions
def quick_demo():
    """Run a quick demonstration"""
    print("🚀 Quick Multi-Geometry Demo")
    demo = CompleteMultiGeometryDemo()
    return demo.run_complete_demo(quick_mode=True)


def full_demo():
    """Run the full demonstration"""
    print("🌟 Full Multi-Geometry Demo")
    demo = CompleteMultiGeometryDemo()
    return demo.run_complete_demo(quick_mode=False)


def custom_dataset_demo(X, y, dataset_name="custom"):
    """Demonstrate with custom dataset"""
    print(f"🎯 Custom Dataset Demo: {dataset_name}")
    
    demo = CompleteMultiGeometryDemo()
    demo.datasets[dataset_name] = (X, y)
    
    # Run analysis and training
    analysis = demo.run_pattern_analysis(dataset_name)
    training = demo.train_models([dataset_name])
    figures = demo.create_comprehensive_visualizations([dataset_name])
    uncertainty = demo.demonstrate_uncertainty_quantification(dataset_name)
    
    return {
        'analysis': analysis,
        'training': training,
        'figures': figures,
        'uncertainty': uncertainty
    }


if __name__ == "__main__":
    # Run the demo
    print("Starting Multi-Geometry Energy Model Demo...")
    results = quick_demo()
    print("Demo completed! Check the results dictionary for all outputs.")