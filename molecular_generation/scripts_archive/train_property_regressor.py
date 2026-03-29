#!/usr/bin/env python3
"""
Train PropertyGuidanceRegressor to enable property-guided generation.

This trains a neural network to predict molecular properties from feature representations.
Once trained, this regressor enables gradient-based guidance during sampling.

Usage:
    python train_property_regressor.py --data-path data/molecules.pkl --epochs 50 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import numpy as np

from src.inference.guided_sampling import PropertyGuidanceRegressor
from src.data.preprocessing import PropertyNormalizer


def create_dummy_dataset(num_samples: int = 1000, input_dim: int = 100, n_properties: int = 5):
    """Create a dummy dataset for testing (in real use, load from preprocessed data)."""
    # Generate random features
    features = torch.randn(num_samples, input_dim)
    
    # Generate properties based on feature statistics
    # In real use, these come from RDKit computed on training molecules
    properties = torch.zeros(num_samples, n_properties)
    properties[:, 0] = torch.clamp(features[:, :10].mean(dim=1) * 2, -2, 5)  # logp
    properties[:, 1] = torch.clamp(features[:, 10:20].abs().mean(dim=1) * 100 + 150, 50, 700)  # mw
    properties[:, 2] = torch.clamp(features[:, 20:30].abs().sum(dim=1), 0, 5).long().float()  # hbd
    properties[:, 3] = torch.clamp(features[:, 30:40].abs().sum(dim=1), 0, 10).long().float()  # hba
    properties[:, 4] = torch.clamp(features[:, 40:50].abs().sum(dim=1), 0, 15).long().float()  # rotatable
    
    return features, properties


def train_regressor(
    train_features: torch.Tensor,
    train_properties: torch.Tensor,
    val_features: torch.Tensor,
    val_properties: torch.Tensor,
    input_dim: int = 100,
    n_properties: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cpu'
) -> tuple:
    """Train property guidance regressor."""
    
    # Create dataloaders
    train_dataset = TensorDataset(train_features, train_properties)
    val_dataset = TensorDataset(val_features, val_properties)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = PropertyGuidanceRegressor(input_dim, n_properties).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    
    print(f"\n{'='*70}")
    print(f"Training Property Guidance Regressor")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim}, Properties: {n_properties}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Train samples: {len(train_features)}, Val samples: {len(val_features)}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_total = 0
        num_batches = 0
        
        for features, properties in train_loader:
            features = features.to(device)
            properties = properties.to(device)
            
            optimizer.zero_grad()
            pred_properties = model(features)
            loss = criterion(pred_properties, properties)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            num_batches += 1
        
        train_loss = train_loss_total / num_batches
        
        # Validation phase
        model.eval()
        val_loss_total = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for features, properties in val_loader:
                features = features.to(device)
                properties = properties.to(device)
                
                pred_properties = model(features)
                loss = criterion(pred_properties, properties)
                
                val_loss_total += loss.item()
                num_val_batches += 1
        
        val_loss = val_loss_total / num_val_batches
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            model.load_state_dict(best_model_state)
            break
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final val loss: {val_loss:.4f}")
    print(f"{'='*70}\n")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train property guidance regressor')
    parser.add_argument('--input-dim', type=int, default=100, help='Feature dimension')
    parser.add_argument('--n-properties', type=int, default=5, help='Number of properties')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='checkpoints/property_regressor.pt',
                       help='Output path for trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create dummy dataset (in real use, load preprocessed training data)
    print("Creating dataset...")
    features, properties = create_dummy_dataset(
        num_samples=1000,
        input_dim=args.input_dim,
        n_properties=args.n_properties
    )
    
    # Split into train/val
    split_idx = int(0.8 * len(features))
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_properties, val_properties = properties[:split_idx], properties[split_idx:]
    
    # Train model
    model, history = train_regressor(
        train_features,
        train_properties,
        val_features,
        val_properties,
        input_dim=args.input_dim,
        n_properties=args.n_properties,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=str(device)
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✓ Model saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Model: PropertyGuidanceRegressor")
    print(f"Input dimension: {args.input_dim}")
    print(f"Output properties: {args.n_properties}")
    print(f"Trained for {len(history['epochs'])} epochs")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"\nNext steps:")
    print(f"1. Use this regressor with GuidedGenerator for property-guided sampling:")
    print(f"   generator = GuidedGenerator(model, regressor, normalizer, device)")
    print(f"   samples = generator.generate_guided(target_properties, num_samples=100)")
    print(f"\n2. Validate property matching with:")
    print(f"   python validate_end_to_end.py")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
