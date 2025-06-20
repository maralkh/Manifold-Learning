"""
Core Geometric Primitives for Multi-Geometry Energy Models
==========================================================

Self-contained implementation of adaptive geometric primitives
that evolve during training for energy-based machine learning.
"""

import torch
import torch.nn as nn
import numpy as np


class AdaptiveSphere(nn.Module):
    """Adaptive sphere that evolves during training"""
    
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
        """Compute energy contribution from this sphere"""
        distances = torch.norm(points - self.center, dim=-1)
        energy = torch.exp(-0.5 * (distances / self.radius)**2)
        return self.influence_weight * energy
    
    def distance_to_surface(self, points):
        """Distance from points to sphere surface"""
        distances = torch.norm(points - self.center, dim=-1)
        return torch.abs(distances - self.radius)
    
    def get_info(self):
        """Get sphere information"""
        return {
            'center': self.center.detach().numpy(),
            'radius': self.radius.detach().item(),
            'influence': self.influence_weight.detach().item()
        }


class AdaptiveTorus(nn.Module):
    """Adaptive torus that evolves during training"""
    
    def __init__(self, center_dim, initial_center=None, initial_major_radius=2.0, initial_minor_radius=0.5):
        super().__init__()
        
        if initial_center is None:
            initial_center = torch.randn(center_dim) * 0.5
        
        self.center = nn.Parameter(torch.FloatTensor(initial_center))
        self.log_major_radius = nn.Parameter(torch.log(torch.tensor(initial_major_radius)))
        self.log_minor_radius = nn.Parameter(torch.log(torch.tensor(initial_minor_radius)))
        self.influence_weight = nn.Parameter(torch.tensor(1.0))
        
        self.min_radius = 0.1
        self.max_major_radius = 5.0
        self.max_minor_radius = 2.0
    
    @property
    def major_radius(self):
        return torch.clamp(torch.exp(self.log_major_radius), self.min_radius, self.max_major_radius)
    
    @property
    def minor_radius(self):
        return torch.clamp(torch.exp(self.log_minor_radius), 
                        torch.tensor(self.min_radius), 
                        torch.min(torch.tensor(self.max_minor_radius), 
                                self.major_radius * 0.8))
    
    def energy(self, points):
        """Compute energy contribution from this torus"""
        if self.center.shape[0] == 2:
            # 2D case: distance to circle
            distances_to_center = torch.norm(points - self.center, dim=-1)
            distances_to_torus = torch.abs(distances_to_center - self.major_radius)
            energy = torch.exp(-0.5 * (distances_to_torus / self.minor_radius)**2)
        else:
            # 3D+ case: proper torus distance
            relative_points = points - self.center
            xy_distance = torch.norm(relative_points[..., :2], dim=-1)
            z_component = relative_points[..., 2] if relative_points.shape[-1] > 2 else torch.zeros_like(xy_distance)
            
            distance_to_major_circle = torch.abs(xy_distance - self.major_radius)
            torus_distance = torch.sqrt(distance_to_major_circle**2 + z_component**2)
            energy = torch.exp(-0.5 * (torus_distance / self.minor_radius)**2)
        
        return self.influence_weight * energy
    
    def get_info(self):
        """Get torus information"""
        return {
            'center': self.center.detach().numpy(),
            'major_radius': self.major_radius.detach().item(),
            'minor_radius': self.minor_radius.detach().item(),
            'influence': self.influence_weight.detach().item()
        }


class AdaptiveEllipsoid(nn.Module):
    """Adaptive ellipsoid that evolves during training"""
    
    def __init__(self, center_dim, initial_center=None, initial_radii=None):
        super().__init__()
        
        if initial_center is None:
            initial_center = torch.randn(center_dim) * 0.5
        
        if initial_radii is None:
            initial_radii = torch.ones(center_dim) * 1.0
        
        self.center = nn.Parameter(torch.FloatTensor(initial_center))
        self.log_radii = nn.Parameter(torch.log(torch.FloatTensor(initial_radii)))
        self.influence_weight = nn.Parameter(torch.tensor(1.0))
        
        self.min_radius = 0.1
        self.max_radius = 5.0
    
    @property
    def radii(self):
        return torch.clamp(torch.exp(self.log_radii), self.min_radius, self.max_radius)
    
    def energy(self, points):
        """Compute energy contribution from this ellipsoid"""
        relative_points = points - self.center
        normalized_distances = relative_points / self.radii
        distances_squared = torch.sum(normalized_distances**2, dim=-1)
        energy = torch.exp(-0.5 * distances_squared)
        return self.influence_weight * energy
    
    def get_info(self):
        """Get ellipsoid information"""
        return {
            'center': self.center.detach().numpy(),
            'radii': self.radii.detach().numpy(),
            'influence': self.influence_weight.detach().item()
        }


class GeometryManager:
    """Manages collection of geometric primitives"""
    
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.geometries = {'spheres': [], 'torus': [], 'ellipsoids': []}
        
    def add_sphere(self, center=None, radius=1.0):
        """Add a new sphere"""
        sphere = AdaptiveSphere(self.output_dim, center, radius)
        self.geometries['spheres'].append(sphere)
        return sphere
    
    def add_torus(self, center=None, major_radius=2.0, minor_radius=0.5):
        """Add a new torus"""
        torus = AdaptiveTorus(self.output_dim, center, major_radius, minor_radius)
        self.geometries['torus'].append(torus)
        return torus
    
    def add_ellipsoid(self, center=None, radii=None):
        """Add a new ellipsoid"""
        ellipsoid = AdaptiveEllipsoid(self.output_dim, center, radii)
        self.geometries['ellipsoids'].append(ellipsoid)
        return ellipsoid
    
    def remove_weak_geometries(self, performances, threshold=0.01):
        """Remove geometries with poor performance"""
        removed = {'spheres': 0, 'torus': 0, 'ellipsoids': 0}
        
        for geom_type in self.geometries:
            if geom_type in performances and len(self.geometries[geom_type]) > 1:
                weak_indices = [i for i, perf in enumerate(performances[geom_type]) 
                              if perf < threshold]
                
                for i in reversed(weak_indices):
                    if len(self.geometries[geom_type]) > 1:
                        del self.geometries[geom_type][i]
                        removed[geom_type] += 1
        
        return removed
    
    def get_all_geometries(self):
        """Get all geometries as a flat list"""
        all_geoms = []
        for geom_list in self.geometries.values():
            all_geoms.extend(geom_list)
        return all_geoms
    
    def get_info(self):
        """Get information about all geometries"""
        info = {
            'counts': {k: len(v) for k, v in self.geometries.items()},
            'details': {}
        }
        
        for geom_type, geom_list in self.geometries.items():
            info['details'][geom_type] = [geom.get_info() for geom in geom_list]
        
        return info