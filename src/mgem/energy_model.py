"""
Multi-Geometry Energy Model
===========================

Advanced energy model with multiple adaptive geometric primitives
that evolve during training for sophisticated machine learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .geometric_primitives import GeometryManager

class MultiGeometryEnergyModel(nn.Module):
    """Advanced energy model with multiple spheres, torus, and ellipsoids"""
    
    def __init__(self, input_dim, output_dim=1, config=None):
        super().__init__()
        
        # Default configuration
        default_config = {
            'max_spheres': 5,
            'max_torus': 3,
            'max_ellipsoids': 2,
            'complexity_score': 0.5,
            'adaptive_geometry': True,
            'energy_weight': 0.01,
            'interaction_strength': 0.05,
            'evolution_frequency': 100,
            'birth_threshold': 0.1,
            'death_threshold': 0.01
        }
        
        self.config = {**default_config, **(config or {})}
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build predictor network
        self.predictor = self._build_predictor()
        
        # Initialize geometry manager
        self.geometry_manager = GeometryManager(output_dim)
        
        # Initialize with default geometries
        self._initialize_default_geometries()
        
        # Global energy parameters
        self.global_energy_weight = nn.Parameter(torch.tensor(self.config['energy_weight']))
        self.interaction_weight = nn.Parameter(torch.tensor(self.config['interaction_strength']))
        
        # Evolution tracking
        self.evolution_history = []
        self.training_step = 0
        
        print(f"🌟 Multi-Geometry Energy Model Created:")
        self._print_model_info()
    
    def _build_predictor(self):
        """Build main prediction network based on complexity"""
        complexity = self.config['complexity_score']
        
        if complexity < 0.3:
            hidden_dims = [32, 16]
            dropout_rate = 0.05
        elif complexity < 0.7:
            hidden_dims = [64, 32, 16]
            dropout_rate = 0.1
        else:
            hidden_dims = [128, 64, 32, 16]
            dropout_rate = 0.15
        
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers)
    
    def _initialize_default_geometries(self):
        """Initialize default geometric primitives"""
        complexity = self.config['complexity_score']
        
        # Number of initial geometries based on complexity
        n_spheres = max(2, int(complexity * self.config['max_spheres']))
        n_torus = max(1, int(complexity * self.config['max_torus']))
        n_ellipsoids = max(1, int(complexity * self.config['max_ellipsoids']))
        
        # Add spheres
        for i in range(n_spheres):
            center = torch.randn(self.output_dim) * 0.5
            self.geometry_manager.add_sphere(center, radius=1.0 + i * 0.2)
        
        # Add torus
        for i in range(n_torus):
            center = torch.randn(self.output_dim) * 0.3
            self.geometry_manager.add_torus(center, 
                                          major_radius=1.5 + i * 0.3, 
                                          minor_radius=0.3 + i * 0.1)
        
        # Add ellipsoids
        for i in range(n_ellipsoids):
            center = torch.randn(self.output_dim) * 0.4
            radii = torch.ones(self.output_dim) * (1.0 + i * 0.2)
            self.geometry_manager.add_ellipsoid(center, radii)
    
    def initialize_from_analysis(self, suggestions):
        """Initialize geometries based on pattern analysis suggestions"""
        print("🎯 Initializing geometries from pattern analysis...")
        
        # Clear existing geometries
        self.geometry_manager = GeometryManager(self.output_dim)
        
        # Add suggested spheres
        for sphere_config in suggestions.get('spheres', []):
            center = torch.FloatTensor(sphere_config['center'])
            radius = sphere_config['radius']
            self.geometry_manager.add_sphere(center, radius)
        
        # Add suggested torus
        for torus_config in suggestions.get('torus', []):
            center = torch.FloatTensor(torus_config['center'])
            major_radius = torus_config['major_radius']
            minor_radius = torus_config['minor_radius']
            self.geometry_manager.add_torus(center, major_radius, minor_radius)
        
        # Add suggested ellipsoids
        for ellipsoid_config in suggestions.get('ellipsoids', []):
            center = torch.FloatTensor(ellipsoid_config['center'])
            radii = torch.FloatTensor(ellipsoid_config['radii'])
            self.geometry_manager.add_ellipsoid(center, radii)
        
        # Ensure we have at least some geometries
        geom_info = self.geometry_manager.get_info()
        total_geoms = sum(geom_info['counts'].values())
        
        if total_geoms == 0:
            self._initialize_default_geometries()
        
        self._print_model_info()
    
    def compute_total_energy(self, predictions):
        """Compute total energy from all geometric primitives"""
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Initialize energy components
        energy_components = {
            'sphere_energy': torch.zeros(batch_size, device=device),
            'torus_energy': torch.zeros(batch_size, device=device),
            'ellipsoid_energy': torch.zeros(batch_size, device=device),
            'interaction_energy': torch.zeros(batch_size, device=device)
        }
        
        # Compute sphere energies
        for sphere in self.geometry_manager.geometries['spheres']:
            energy_components['sphere_energy'] += sphere.energy(predictions)
        
        # Compute torus energies
        for torus in self.geometry_manager.geometries['torus']:
            energy_components['torus_energy'] += torus.energy(predictions)
        
        # Compute ellipsoid energies
        for ellipsoid in self.geometry_manager.geometries['ellipsoids']:
            energy_components['ellipsoid_energy'] += ellipsoid.energy(predictions)
        
        # Compute interaction energies
        energy_components['interaction_energy'] = self._compute_interaction_energy(predictions)
        
        # Total energy landscape
        total_energy = (energy_components['sphere_energy'] + 
                       energy_components['torus_energy'] + 
                       energy_components['ellipsoid_energy'] + 
                       energy_components['interaction_energy'])
        
        # Convert to energy landscape (negative log)
        energy_landscape = -torch.log(total_energy + 1e-8)
        
        return energy_landscape, energy_components
    
    def _compute_interaction_energy(self, predictions):
        """Compute interaction energy between different geometry types"""
        interaction_energy = torch.zeros(predictions.shape[0], device=predictions.device)
        
        # Sphere-Torus interactions
        for sphere in self.geometry_manager.geometries['spheres']:
            for torus in self.geometry_manager.geometries['torus']:
                center_distance = torch.norm(sphere.center - torus.center)
                interaction_strength = torch.exp(-center_distance)
                sphere_influence = sphere.energy(predictions)
                torus_influence = torus.energy(predictions)
                interaction_energy += (self.interaction_weight * interaction_strength * 
                                     sphere_influence * torus_influence)
        
        # Sphere-Ellipsoid interactions
        for sphere in self.geometry_manager.geometries['spheres']:
            for ellipsoid in self.geometry_manager.geometries['ellipsoids']:
                center_distance = torch.norm(sphere.center - ellipsoid.center)
                interaction_strength = torch.exp(-center_distance)
                sphere_influence = sphere.energy(predictions)
                ellipsoid_influence = ellipsoid.energy(predictions)
                interaction_energy += (self.interaction_weight * interaction_strength * 
                                     sphere_influence * ellipsoid_influence)
        
        # Torus-Ellipsoid interactions
        for torus in self.geometry_manager.geometries['torus']:
            for ellipsoid in self.geometry_manager.geometries['ellipsoids']:
                center_distance = torch.norm(torus.center - ellipsoid.center)
                interaction_strength = torch.exp(-center_distance)
                torus_influence = torus.energy(predictions)
                ellipsoid_influence = ellipsoid.energy(predictions)
                interaction_energy += (self.interaction_weight * interaction_strength * 
                                     torus_influence * ellipsoid_influence)
        
        return interaction_energy
    
    def evolve_geometry(self, predictions, epoch):
        """Evolve geometric primitives during training"""
        if not self.config['adaptive_geometry']:
            return
        
        if epoch % self.config['evolution_frequency'] != 0:
            return
        
        with torch.no_grad():
            # Analyze performance of each geometry type
            performances = self._analyze_geometry_performance(predictions)
            
            # Remove weak geometries
            removed = self.geometry_manager.remove_weak_geometries(
                performances, self.config['death_threshold']
            )
            
            # Add new geometries if needed
            added = self._add_new_geometries(predictions, performances)
            
            # Record evolution
            if any(removed.values()) or any(added.values()):
                evolution_record = {
                    'epoch': epoch,
                    'removed': removed,
                    'added': added,
                    'final_counts': self.geometry_manager.get_info()['counts']
                }
                self.evolution_history.append(evolution_record)
                
                print(f"   🔄 Geometry evolution at epoch {epoch}:")
                for geom_type in ['spheres', 'torus', 'ellipsoids']:
                    if removed[geom_type] > 0:
                        print(f"     🗑️ Removed {removed[geom_type]} {geom_type}")
                    if added[geom_type] > 0:
                        print(f"     ➕ Added {added[geom_type]} {geom_type}")
    
    def _analyze_geometry_performance(self, predictions):
        """Analyze performance of each geometric primitive"""
        performances = {'spheres': [], 'torus': [], 'ellipsoids': []}
        
        # Sphere performances
        for sphere in self.geometry_manager.geometries['spheres']:
            performance = sphere.energy(predictions).mean().item()
            performances['spheres'].append(performance)
        
        # Torus performances
        for torus in self.geometry_manager.geometries['torus']:
            performance = torus.energy(predictions).mean().item()
            performances['torus'].append(performance)
        
        # Ellipsoid performances
        for ellipsoid in self.geometry_manager.geometries['ellipsoids']:
            performance = ellipsoid.energy(predictions).mean().item()
            performances['ellipsoids'].append(performance)
        
        return performances
    
    def _add_new_geometries(self, predictions, performances):
        """Add new geometries based on prediction variance and performance"""
        added = {'spheres': 0, 'torus': 0, 'ellipsoids': 0}
        
        pred_std = predictions.std(dim=0).mean().item()
        birth_threshold = self.config['birth_threshold']
        
        geom_counts = self.geometry_manager.get_info()['counts']
        
        # Add sphere if needed
        if (geom_counts['spheres'] < self.config['max_spheres'] and 
            pred_std > birth_threshold):
            
            new_center = predictions.mean(dim=0) + torch.randn_like(predictions.mean(dim=0)) * 0.5
            self.geometry_manager.add_sphere(new_center)
            added['spheres'] = 1
        
        # Add torus if needed
        if (geom_counts['torus'] < self.config['max_torus'] and 
            pred_std > birth_threshold * 1.2):
            
            new_center = predictions.mean(dim=0) + torch.randn_like(predictions.mean(dim=0)) * 0.3
            self.geometry_manager.add_torus(new_center)
            added['torus'] = 1
        
        # Add ellipsoid if needed
        if (geom_counts['ellipsoids'] < self.config['max_ellipsoids'] and 
            pred_std > birth_threshold * 1.5):
            
            new_center = predictions.mean(dim=0) + torch.randn_like(predictions.mean(dim=0)) * 0.4
            self.geometry_manager.add_ellipsoid(new_center)
            added['ellipsoids'] = 1
        
        return added
    
    def forward(self, x):
        """Forward pass through the model"""
        # Get base predictions
        predictions = self.predictor(x)
        
        # Compute energy landscape
        energy_landscape, energy_components = self.compute_total_energy(predictions)
        
        # Update training step
        self.training_step += 1
        
        return predictions, energy_landscape, energy_components
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """Predict with uncertainty quantification using energy landscape"""
        self.eval()
        
        with torch.no_grad():
            predictions, energy_landscape, _ = self.forward(x)
            
            # Sample from energy landscape for uncertainty
            energy_weights = torch.softmax(-energy_landscape / 0.1, dim=0)
            
            # Generate samples
            samples = []
            for _ in range(n_samples):
                # Sample indices based on energy weights
                indices = torch.multinomial(energy_weights, len(x), replacement=True)
                sample_predictions = predictions[indices]
                samples.append(sample_predictions)
            
            samples = torch.stack(samples)
            
            # Compute uncertainty metrics
            mean_pred = samples.mean(dim=0)
            std_pred = samples.std(dim=0)
            
            return {
                'predictions': predictions,
                'mean_prediction': mean_pred,
                'uncertainty': std_pred,
                'samples': samples,
                'energy_landscape': energy_landscape
            }
    
    def get_geometry_info(self):
        """Get comprehensive geometry information"""
        return self.geometry_manager.get_info()
    
    def _print_model_info(self):
        """Print model information"""
        geom_info = self.get_geometry_info()
        counts = geom_info['counts']
        
        print(f"   📊 Geometry counts: Spheres={counts['spheres']}, "
              f"torus={counts['torus']}, Ellipsoids={counts['ellipsoids']}")
        print(f"   ⚙️ Complexity: {self.config['complexity_score']:.2f}, "
              f"Adaptive: {self.config['adaptive_geometry']}")
    
    def get_training_parameters(self):
        """Get parameters specifically for training"""
        # Separate geometry parameters from predictor parameters
        predictor_params = list(self.predictor.parameters())
        geometry_params = []
        
        for geom_list in self.geometry_manager.geometries.values():
            for geom in geom_list:
                geometry_params.extend(list(geom.parameters()))
        
        global_params = [self.global_energy_weight, self.interaction_weight]
        
        return {
            'predictor': predictor_params,
            'geometry': geometry_params,
            'global': global_params,
            'all': predictor_params + geometry_params + global_params
        }
