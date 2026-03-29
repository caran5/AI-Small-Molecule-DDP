"""
Training utilities for diffusion models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import time


class DiffusionTrainer:
    """Trainer for diffusion models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: torch.device = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 early_stopping_patience: int = 5,
                 num_epochs: int = 100):
        """
        Args:
            model: Diffusion model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            weight_decay: L2 regularization strength (default 1e-5)
            betas: Adam beta parameters
            early_stopping_patience: Patience for early stopping (default 5)
            num_epochs: Total number of epochs (for scheduler)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cpu')
        self.num_epochs = num_epochs
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
        # Scheduler with T_max based on num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-5
        )
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.best_epoch = 0
    
    def train_step(self, batch: Dict) -> float:
        """
        Single training step with support for variable-length molecules.
        
        Args:
            batch: Batch of data from DataLoader. Expected keys:
                   - 'features': tensor of shape (batch_size, n_atoms, features)
                   - 'n_atoms' (optional): tensor of shape (batch_size,) with valid atom counts
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Get features from batch
        features = batch['features'].to(self.device)  # (batch_size, n_atoms, features)
        
        # Get n_atoms if provided (for masking variable-length molecules)
        n_atoms = None
        if 'n_atoms' in batch:
            n_atoms = batch['n_atoms'].to(self.device)
        
        # Forward pass with optional masking
        loss = self.model.get_loss(features, n_atoms=n_atoms)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def val_step(self) -> float:
        """
        Validation step with support for variable-length molecules.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            
            # Get n_atoms if provided
            n_atoms = None
            if 'n_atoms' in batch:
                n_atoms = batch['n_atoms'].to(self.device)
            
            loss = self.model.get_loss(features, n_atoms=n_atoms)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, num_epochs: int, eval_every: int = 1, save_path: Optional[str] = None) -> Dict:
        """
        Train for multiple epochs with early stopping and regularization.
        
        Args:
            num_epochs: Number of epochs
            eval_every: Evaluate every N epochs
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        # Update scheduler if num_epochs differs from initialization
        if num_epochs != self.num_epochs:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-5
            )
            self.num_epochs = num_epochs
        
        history = {'train_loss': [], 'val_loss': [], 'epoch': [], 'early_stopped': False}
        
        for epoch in range(num_epochs):
            # Track start time at beginning of epoch
            start_time = time.time()
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Training
            for batch in self.train_loader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
            
            epoch_loss /= num_batches
            self.train_losses.append(epoch_loss)
            
            # Validation
            if (epoch + 1) % eval_every == 0:
                val_loss = self.val_step()
                self.val_losses.append(val_loss)
                
                history['epoch'].append(epoch)
                history['train_loss'].append(epoch_loss)
                history['val_loss'].append(val_loss)
                
                # Calculate elapsed time for this epoch/eval period
                elapsed = time.time() - start_time
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.2f}s")
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_epoch = epoch
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"        → Saved best model (val_loss={val_loss:.4f})")
                else:
                    self.patience_counter += 1
                    print(f"        → No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
                        print(f"Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch+1})")
                        history['early_stopped'] = True
                        break
            
            self.scheduler.step()
        
        return history
