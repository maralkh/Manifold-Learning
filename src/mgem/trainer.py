"""
Multi-Geometry Trainer
======================

Advanced trainer for multi-geometry energy models with adaptive learning,
geometry evolution, and comprehensive monitorusng.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict


class MultiGeometryTrainer:
    """Advanced trainer for multi-geometry energy models"""
    
    def __init__(self, model, config=None):
        self.model = model
        
        # Default training configuration
        default_config = {
            'use_geometry_evolution': True,
            'use_adaptive_learning': True,
            'use_early_stopping': True,
            'patience': 200,
            'min_delta': 1e-6,
            'max_grad_norm': 1.0,
            'warmup_epochs': 100,
            'energy_annealing': True,
            'validation_split': 0.2
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Training state
        self.training_history = {}
        self.geometry_evolution_history = []
        self.best_model_state = None
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Learning rate scheduling
        self.lr_scheduler = None
        self.optimizer = None
        
        print("🚀 Multi-Geometry Trainer Initialized")
        self._print_config()
    
    def train(self, X, y, epochs=1000, lr=0.01, batch_size=None, verbose=True):
        """Train the multi-geometry energy model"""
        if verbose:
            print("\n" + "="*60)
            print("🌟 TRAINING MULTI-GEOMETRY ENERGY MODEL")
            print("="*60)
        
        # Prepare data
        X_tensor, y_tensor, train_loader, val_loader = self._prepare_data(
            X, y, batch_size, self.config['validation_split']
        )
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(lr, epochs)
        
        # Initialize training history
        self._initialize_training_history()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step(val_metrics['total_loss'])
            
            # Record metrics
            self._record_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Geometry evolution
            if self.config['use_geometry_evolution'] and epoch > self.config['warmup_epochs']:
                self._handle_geometry_evolution(X_tensor, epoch)
            
            # Early stopping check
            if self.config['use_early_stopping']:
                if self._check_early_stopping(val_metrics['total_loss']):
                    if verbose:
                        print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # Progress reporting
            if verbose and (epoch + 1) % 100 == 0:
                self._print_progress(epoch, train_metrics, val_metrics, time.time() - start_time)
        
        # Finalize training
        total_time = time.time() - start_time
        self._finalize_training(total_time, verbose)
        
        return self.training_history
    
    def _prepare_data(self, X, y, batch_size, validation_split):
        """Prepare training and validation data"""
        # Convert to tensors
        if not torch.is_tensor(X):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X.clone()
        
        if not torch.is_tensor(y):
            y_tensor = torch.FloatTensor(y)
        else:
            y_tensor = y.clone()
        
        # Ensure proper shapes
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        # Split data if validation_split > 0
        if validation_split > 0:
            n_samples = len(X_tensor)
            n_val = int(n_samples * validation_split)
            indices = torch.randperm(n_samples)
            
            train_indices = indices[n_val:]
            val_indices = indices[:n_val]
            
            X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
            X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
            
            # Create data loaders
            if batch_size is None:
                batch_size = min(64, len(X_train) // 4)
            
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            # No validation split
            if batch_size is None:
                batch_size = min(64, len(X_tensor) // 4)
            
            train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = None
        
        return X_tensor, y_tensor, train_loader, val_loader
    
    def _setup_optimizer_and_scheduler(self, lr, epochs):
        """Setup optimizer and learning rate scheduler"""
        # Get different parameter groups
        param_groups = self.model.get_training_parameters()
        
        # Create optimizer with different learning rates for different components
        optimizer_params = [
            {'params': param_groups['predictor'], 'lr': lr},
            {'params': param_groups['geometry'], 'lr': lr * 0.5},  # Slower for geometry
            {'params': param_groups['global'], 'lr': lr * 0.1}     # Slowest for global params
        ]
        
        self.optimizer = optim.AdamW(optimizer_params, weight_decay=0.001)
        
        # Setup learning rate scheduler
        if self.config['use_adaptive_learning']:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=50, verbose=False
            )
    
    def _initialize_training_history(self):
        """Initialize training history tracking"""
        self.training_history = {
            'train_losses': defaultdict(list),
            'val_losses': defaultdict(list),
            'geometry_counts': defaultdict(list),
            'learning_rates': [],
            'geometry_evolution': [],
            'epoch_times': [],
            'config': self.config
        }
    
    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = defaultdict(float)
        n_batches = 0
        
        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, energy_landscape, energy_components = self.model(batch_X)
            
            # Compute losses
            mse_loss = F.mse_loss(predictions, batch_y)
            energy_loss = energy_landscape.mean()
            
            # Energy weight annealing
            if self.config['energy_annealing']:
                energy_weight = self.model.config['energy_weight'] * (1 + epoch / 1000)
            else:
                energy_weight = self.model.config['energy_weight']
            
            total_loss = mse_loss + energy_weight * energy_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config['max_grad_norm']
            )
            
            self.optimizer.step()
            
            # Record metrics
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['mse_loss'] += mse_loss.item()
            epoch_metrics['energy_loss'] += energy_loss.item()
            epoch_metrics['sphere_energy'] += energy_components['sphere_energy'].mean().item()
            epoch_metrics['torus_energy'] += energy_components['torus_energy'].mean().item()
            epoch_metrics['ellipsoid_energy'] += energy_components['ellipsoid_energy'].mean().item()
            epoch_metrics['interaction_energy'] += energy_components['interaction_energy'].mean().item()
            
            n_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return epoch_metrics
    
    def _validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        if val_loader is None:
            return {}
        
        self.model.eval()
        
        epoch_metrics = defaultdict(float)
        n_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Forward pass
                predictions, energy_landscape, energy_components = self.model(batch_X)
                
                # Compute losses
                mse_loss = F.mse_loss(predictions, batch_y)
                energy_loss = energy_landscape.mean()
                
                if self.config['energy_annealing']:
                    energy_weight = self.model.config['energy_weight'] * (1 + epoch / 1000)
                else:
                    energy_weight = self.model.config['energy_weight']
                
                total_loss = mse_loss + energy_weight * energy_loss
                
                # Record metrics
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['mse_loss'] += mse_loss.item()
                epoch_metrics['energy_loss'] += energy_loss.item()
                epoch_metrics['sphere_energy'] += energy_components['sphere_energy'].mean().item()
                epoch_metrics['torus_energy'] += energy_components['torus_energy'].mean().item()
                epoch_metrics['ellipsoid_energy'] += energy_components['ellipsoid_energy'].mean().item()
                epoch_metrics['interaction_energy'] += energy_components['interaction_energy'].mean().item()
                
                n_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return epoch_metrics
    
    def _record_epoch_metrics(self, epoch, train_metrics, val_metrics):
        """Record metrics for the epoch"""
        # Record training metrics
        for key, value in train_metrics.items():
            self.training_history['train_losses'][key].append(value)
        
        # Record validation metrics
        for key, value in val_metrics.items():
            self.training_history['val_losses'][key].append(value)
        
        # Record geometry counts
        geom_info = self.model.get_geometry_info()
        for geom_type, count in geom_info['counts'].items():
            self.training_history['geometry_counts'][geom_type].append(count)
        
        # Record learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['learning_rates'].append(current_lr)
    
    def _handle_geometry_evolution(self, X_tensor, epoch):
        """Handle geometry evolution during training"""
        # Get current predictions for evolution analysis
        self.model.eval()
        with torch.no_grad():
            predictions, _, _ = self.model(X_tensor)
        
        # Trigger evolution
        prev_counts = self.model.get_geometry_info()['counts'].copy()
        self.model.evolve_geometry(predictions.detach(), epoch)
        new_counts = self.model.get_geometry_info()['counts'].copy()
        
        # Record evolution if changes occurred
        if prev_counts != new_counts:
            evolution_record = {
                'epoch': epoch,
                'prev_counts': prev_counts,
                'new_counts': new_counts,
                'changes': {k: new_counts[k] - prev_counts[k] for k in new_counts}
            }
            self.training_history['geometry_evolution'].append(evolution_record)
        
        self.model.train()  # Switch back to training mode
    
    def _check_early_stopping(self, current_loss):
        """Check early stopping criteria"""
        if current_loss < self.best_loss - self.config['min_delta']:
            self.best_loss = current_loss
            self.best_model_state = self.model.state_dict().copy()
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config['patience']
    
    def _print_progress(self, epoch, train_metrics, val_metrics, elapsed_time):
        """Print training progress"""
        geom_info = self.model.get_geometry_info()['counts']
        
        print(f"\nEpoch {epoch+1:4d} | Time: {elapsed_time:.1f}s")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"MSE: {train_metrics['mse_loss']:.4f}, "
              f"Energy: {train_metrics['energy_loss']:.4f}")
        
        if val_metrics:
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"MSE: {val_metrics['mse_loss']:.4f}, "
                  f"Energy: {val_metrics['energy_loss']:.4f}")
        
        print(f"  Geometry - S:{geom_info['spheres']}, "
              f"T:{geom_info['torus']}, E:{geom_info['ellipsoids']}")
        
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.2e}")
    
    def _finalize_training(self, total_time, verbose):
        """Finalize training and restore best model if applicable"""
        # Restore best model if early stopping was used
        if self.config['use_early_stopping'] and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Add final statistics to history
        self.training_history['total_training_time'] = total_time
        self.training_history['final_geometry_info'] = self.model.get_geometry_info()
        self.training_history['evolution_history'] = self.model.evolution_history
        
        if verbose:
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETED")
            print("="*60)
            print(f"⏰ Total time: {total_time:.2f}s")
            print(f"🏆 Best loss: {self.best_loss:.6f}")
            
            final_counts = self.training_history['final_geometry_info']['counts']
            print(f"🔷 Final geometry: S:{final_counts['spheres']}, "
                  f"T:{final_counts['torus']}, E:{final_counts['ellipsoids']}")
            
            n_evolutions = len(self.training_history['geometry_evolution'])
            print(f"🔄 Geometry evolutions: {n_evolutions}")
    
    def _print_config(self):
        """Print trainer configuration"""
        print("⚙️ Trainer Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        self.model.eval()
        
        with torch.no_grad():
            if not torch.is_tensor(X_test):
                X_test = torch.FloatTensor(X_test)
            if not torch.is_tensor(y_test):
                y_test = torch.FloatTensor(y_test)
            
            if y_test.dim() == 1:
                y_test = y_test.unsqueeze(1)
            
            predictions, energy_landscape, energy_components = self.model(X_test)
            
            # Compute metrics
            mse = F.mse_loss(predictions, y_test).item()
            mae = F.l1_loss(predictions, y_test).item()
            
            # R² score
            y_mean = y_test.mean()
            ss_tot = ((y_test - y_mean) ** 2).sum()
            ss_res = ((y_test - predictions) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot).item()
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'energy_landscape_mean': energy_landscape.mean().item(),
                'predictions': predictions,
                'energy_components': energy_components
            }
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """Make predictions with uncertainty quantification"""
        return self.model.predict_with_uncertainty(X, n_samples)
