#!/usr/bin/env python3
"""
Improved training with proper regularization and better diagnostics.
Fixes: L2 regularization, dropout, noise injection, better early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import numpy as np
import json

from src.inference.guided_sampling import PropertyGuidanceRegressor
from src.data.preprocessing import PropertyNormalizer


def create_realistic_dataset(num_samples: int = 1000, input_dim: int = 100, n_properties: int = 5, noise_level: float = 0.1):
    """Create dataset with realistic noise and less perfect correlations."""
    # Generate random features
    features = torch.randn(num_samples, input_dim)
    
    # Generate properties with WEAKER correlations and added noise
    properties = torch.zeros(num_samples, n_properties)
    
    # LogP: use multiple feature ranges with noise
    logp_signal = features[:, :20].mean(dim=1) * 0.8  # Weaker correlation
    properties[:, 0] = torch.clamp(logp_signal + torch.randn(num_samples) * noise_level, -2, 5)
    
    # MW: use different features
    mw_signal = features[:, 20:40].abs().mean(dim=1) * 100 + 150
    properties[:, 1] = torch.clamp(mw_signal + torch.randn(num_samples) * 50 * noise_level, 50, 700)
    
    # HBD: use features from different ranges
    hbd_signal = features[:, 40:60].abs().sum(dim=1)
    properties[:, 2] = torch.clamp(hbd_signal + torch.randn(num_samples) * noise_level, 0, 5)
    
    # HBA: independent features
    hba_signal = features[:, 60:80].abs().sum(dim=1)
    properties[:, 3] = torch.clamp(hba_signal + torch.randn(num_samples) * noise_level, 0, 10)
    
    # Rotatable: truly independent
    rot_signal = (features[:, 80:100].abs() > 0.5).float().sum(dim=1)
    properties[:, 4] = torch.clamp(rot_signal + torch.randn(num_samples) * noise_level, 0, 15)
    
    return features, properties


class RegularizedPropertyGuidanceRegressor(nn.Module):
    """Improved regressor with dropout and better initialization."""
    
    def __init__(self, input_dim: int, n_properties: int, dropout_rate: float = 0.2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, n_properties),
        )
        
        # Better initialization
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)


def train_regressor_improved(
    train_features: torch.Tensor,
    train_properties: torch.Tensor,
    val_features: torch.Tensor,
    val_properties: torch.Tensor,
    input_dim: int = 100,
    n_properties: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,  # Stronger L2
    dropout_rate: float = 0.2,
    device: str = 'cpu'
) -> tuple:
    """Train with proper regularization."""
    
    # Create dataloaders
    train_dataset = TensorDataset(train_features, train_properties)
    val_dataset = TensorDataset(val_features, val_properties)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize regularized model
    model = RegularizedPropertyGuidanceRegressor(input_dim, n_properties, dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    history = {'train_loss': [], 'val_loss': [], 'epochs': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10  # Increased patience
    best_model_state = None
    
    print(f"\n{'='*70}")
    print(f"Training Property Guidance Regressor (IMPROVED)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim}, Properties: {n_properties}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}, Dropout: {dropout_rate}")
    print(f"Train samples: {len(train_features)}, Val samples: {len(val_features)}")
    print(f"Train/Val ratio: {len(train_features) / len(val_features):.1f}x")
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
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Calculate overfitting gap
        gap = (val_loss - train_loss) / train_loss * 100 if train_loss > 0 else 0
        gap_indicator = "🔴 HIGH" if gap > 50 else "🟡 MODERATE" if gap > 20 else "🟢 GOOD"
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping with better patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress with diagnostics
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Gap: {gap:5.1f}% {gap_indicator} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            print(f"  Patience limit reached: {patience_counter}/{patience_limit}")
            model.load_state_dict(best_model_state)
            break
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final val loss: {val_loss:.4f}")
    print(f"Final overfitting gap: {gap:.1f}%")
    print(f"{'='*70}\n")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train property guidance regressor (improved)')
    parser.add_argument('--input-dim', type=int, default=100, help='Feature dimension')
    parser.add_argument('--n-properties', type=int, default=5, help='Number of properties')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--output', type=str, default='checkpoints/property_regressor_improved.pt',
                       help='Output path for trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create realistic dataset with noise
    print("Creating realistic dataset with noise...")
    features, properties = create_realistic_dataset(
        num_samples=2000,  # Increased from 1000
        input_dim=args.input_dim,
        n_properties=args.n_properties,
        noise_level=0.15
    )
    
    # Better split: 70/15/15 (train/val/test)
    n_train = int(0.7 * len(features))
    n_val = int(0.15 * len(features))
    
    train_features = features[:n_train]
    train_properties = properties[:n_train]
    
    val_features = features[n_train:n_train+n_val]
    val_properties = properties[n_train:n_train+n_val]
    
    test_features = features[n_train+n_val:]
    test_properties = properties[n_train+n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_features)} samples (70%)")
    print(f"  Val:   {len(val_features)} samples (15%)")
    print(f"  Test:  {len(test_features)} samples (15%)")
    
    # Train model
    model, history = train_regressor_improved(
        train_features,
        train_properties,
        val_features,
        val_properties,
        input_dim=args.input_dim,
        n_properties=args.n_properties,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        device=str(device)
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    model.eval()
    with torch.no_grad():
        test_pred = model(test_features.to(device))
        test_loss = nn.MSELoss()(test_pred, test_properties.to(device))
    print(f"Test Loss: {test_loss:.4f}")
    print("="*70 + "\n")
    
    # Save model and history
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✓ Model saved to {output_path}")
    
    # Save training history
    history_path = output_path.parent / 'training_history.json'
    # Convert to native Python types for JSON serialization
    history_json = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'epochs': history['epochs'],
        'lr': [float(x) for x in history['lr']]
    }
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"✓ History saved to {history_path}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Improvements:")
    print(f"  ✓ Added dropout (20%) for regularization")
    print(f"  ✓ Increased L2 regularization (1e-4 vs 1e-5)")
    print(f"  ✓ Better weight initialization (Kaiming)")
    print(f"  ✓ ReduceLROnPlateau scheduler for better convergence")
    print(f"  ✓ Realistic dataset with noise and weaker correlations")
    print(f"  ✓ Better train/val/test split (70/15/15)")
    print(f"  ✓ Overfitting gap tracking during training")
    print(f"  ✓ Increased patience and larger dataset")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
