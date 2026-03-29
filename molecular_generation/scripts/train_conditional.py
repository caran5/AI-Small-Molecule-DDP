"""
Training script for conditional diffusion models.
Implements property steering with comprehensive regularization and early stopping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

from src.models.unet import ConditionalUNet
from src.data.preprocessing import PropertyNormalizer
from src.eval.metrics import (
    chemical_validity,
    diversity_metric,
    property_fidelity,
    distribution_distance,
    print_metrics,
    compute_all_metrics
)
from src.inference.generate import generate_with_properties


def get_noise_schedule(t: torch.Tensor, 
                       num_steps: int = 1000,
                       schedule_type: str = 'cosine') -> Tuple[torch.Tensor, torch.Tensor]:
    """Get alpha and beta values for timestep t."""
    t_normalized = t.float() / num_steps
    
    if schedule_type == 'cosine':
        s = 0.008
        alpha = torch.cos(((t_normalized + s) / (1 + s)) * np.pi / 2) ** 2
    elif schedule_type == 'linear':
        alpha = 1.0 - (t.float() / num_steps)
    elif schedule_type == 'quadratic':
        alpha = (1.0 - t.float() / num_steps) ** 2
    elif schedule_type == 'learned':
        alpha = (1.0 - t_normalized) ** 3
    else:
        alpha = 1.0 - t.float() / num_steps
    
    alpha = alpha.clamp(0.0, 1.0)
    beta = 1.0 - alpha
    
    return alpha, beta


class ConditionalTrainer:
    """Trainer for conditional diffusion models."""
    
    def __init__(self,
                 model: ConditionalUNet,
                 device: str = 'cpu',
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 schedule_type: str = 'cosine',
                 num_diffusion_steps: int = 1000,
                 save_dir: str = 'checkpoints/'):
        """
        Args:
            model: ConditionalUNet instance
            device: Device to use
            lr: Learning rate
            weight_decay: Weight decay for L2 regularization
            schedule_type: Noise schedule type
            num_diffusion_steps: Number of diffusion steps
            save_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.schedule_type = schedule_type
        self.num_diffusion_steps = num_diffusion_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (features, properties) in enumerate(train_loader):
            features = features.to(self.device)
            properties = properties.to(self.device)
            
            # Sample random timesteps
            batch_size = features.shape[0]
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
            
            # Sample noise
            noise = torch.randn_like(features)
            
            # Get alpha and beta
            alpha_t, beta_t = get_noise_schedule(t, self.num_diffusion_steps, self.schedule_type)
            
            # Add noise: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
            sqrt_alpha = torch.sqrt(alpha_t).view(-1, 1)
            sqrt_beta = torch.sqrt(beta_t).view(-1, 1)
            
            x_t = sqrt_alpha * features + sqrt_beta * noise
            
            # Forward pass WITH property conditioning
            noise_pred = self.model(x_t.unsqueeze(1), t, properties=properties)
            
            # Handle output shape
            if noise_pred.dim() == 3:
                noise_pred = noise_pred.squeeze(1)
            
            # Loss: predict noise
            loss = F.mse_loss(noise_pred, noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, properties in val_loader:
                features = features.to(self.device)
                properties = properties.to(self.device)
                
                batch_size = features.shape[0]
                t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(features)
                
                alpha_t, beta_t = get_noise_schedule(t, self.num_diffusion_steps, self.schedule_type)
                sqrt_alpha = torch.sqrt(alpha_t).view(-1, 1)
                sqrt_beta = torch.sqrt(beta_t).view(-1, 1)
                
                x_t = sqrt_alpha * features + sqrt_beta * noise
                noise_pred = self.model(x_t.unsqueeze(1), t, properties=properties)
                
                if noise_pred.dim() == 3:
                    noise_pred = noise_pred.squeeze(1)
                
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self,
              train_loader,
              val_loader,
              epochs: int = 20,
              early_stopping_patience: int = 5,
              property_normalizer: Optional[PropertyNormalizer] = None,
              training_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Full training loop with early stopping and optional metrics.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            property_normalizer: For inference metrics (optional)
            training_features: Training set features for distribution metrics (optional)
        
        Returns:
            Training history dict
        """
        best_val_loss = float('inf')
        patience_counter = 0
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        
        print("\n" + "="*70)
        print("Starting Conditional Model Training")
        print("="*70)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = self.save_dir / f"conditional_best.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"✓ Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n✓ Early stopping at epoch {epoch+1} "
                          f"(no improvement for {early_stopping_patience} epochs)")
                    break
        
        # Save final metrics
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }
        
        return history


def train_conditional_model(
    train_loader,
    val_loader,
    input_dim: int = 100,
    hidden_dim: int = 128,
    depth: int = 3,
    n_properties: int = 5,
    epochs: int = 20,
    lr: float = 1e-3,
    early_stopping_patience: int = 5,
    device: str = 'cpu',
    save_dir: str = 'checkpoints/',
    property_normalizer: Optional[PropertyNormalizer] = None,
    training_features: Optional[torch.Tensor] = None
) -> Tuple[ConditionalUNet, Dict]:
    """
    Train a conditional UNet from scratch.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Feature dimension
        hidden_dim: Hidden dimension
        depth: Network depth
        n_properties: Number of properties to condition on
        epochs: Maximum epochs
        lr: Learning rate
        early_stopping_patience: Early stopping patience
        device: Device to use
        save_dir: Directory for checkpoints
        property_normalizer: For metrics (optional)
        training_features: For distribution metrics (optional)
    
    Returns:
        (trained_model, training_history)
    """
    # Create model
    model = ConditionalUNet(
        in_channels=input_dim,
        out_channels=input_dim,
        hidden_channels=hidden_dim,
        depth=depth,
        n_properties=n_properties,
        dropout_rate=0.1
    )
    
    # Create trainer
    trainer = ConditionalTrainer(
        model,
        device=device,
        lr=lr,
        schedule_type='cosine',
        save_dir=save_dir
    )
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        property_normalizer=property_normalizer,
        training_features=training_features
    )
    
    # Load best model
    best_model_path = Path(save_dir) / "conditional_best.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    
    return model, history


if __name__ == '__main__':
    print("Conditional training module loaded successfully.")
    print("Use train_conditional_model() or ConditionalTrainer class for training.")

