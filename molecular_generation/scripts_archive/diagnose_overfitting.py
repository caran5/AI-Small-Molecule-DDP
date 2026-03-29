#!/usr/bin/env python3
"""
Diagnostic script to analyze training data and identify sources of overfitting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.inference.guided_sampling import PropertyGuidanceRegressor


def analyze_dataset_bias():
    """Analyze the current training dataset for bias and class imbalance."""
    
    print("\n" + "="*70)
    print("DATASET DIAGNOSTICS")
    print("="*70)
    
    # Recreate the ORIGINAL dataset (same as train_property_regressor.py)
    num_samples = 1000
    input_dim = 100
    n_properties = 5
    
    # Generate exactly as the original script does
    features = torch.randn(num_samples, input_dim)
    
    # Original property generation (derived from features)
    properties = torch.zeros(num_samples, n_properties)
    properties[:, 0] = torch.clamp(features[:, :20].mean(dim=1) * 2 - 1, -2, 5)
    properties[:, 1] = torch.clamp(features[:, 20:40].abs().sum(dim=1) * 50 + 300, 50, 700)
    properties[:, 2] = torch.clamp(features[:, 40:60].abs().sum(dim=1), 0, 5)
    properties[:, 3] = torch.clamp(features[:, 60:80].abs().sum(dim=1), 0, 10)
    properties[:, 4] = torch.clamp((features[:, 80:100] > 0.5).float().sum(dim=1), 0, 15)
    
    # Original split
    split_idx = int(0.8 * len(features))
    train_features = features[:split_idx]
    train_properties = properties[:split_idx]
    val_features = features[split_idx:]
    val_properties = properties[split_idx:]
    
    property_names = ['LogP', 'MW', 'HBD', 'HBA', 'Rotatable']
    
    print(f"\nDataset Size: {num_samples} samples")
    print(f"  Train: {len(train_features)} samples")
    print(f"  Val:   {len(val_features)} samples")
    print(f"\nFeature Dimension: {input_dim}")
    print(f"Properties: {n_properties} ({', '.join(property_names)})")
    
    # Check for class imbalance
    print(f"\n{'='*70}")
    print("PROPERTY DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    for i, name in enumerate(property_names):
        train_mean = train_properties[:, i].mean().item()
        train_std = train_properties[:, i].std().item()
        train_min = train_properties[:, i].min().item()
        train_max = train_properties[:, i].max().item()
        
        val_mean = val_properties[:, i].mean().item()
        val_std = val_properties[:, i].std().item()
        val_min = val_properties[:, i].min().item()
        val_max = val_properties[:, i].max().item()
        
        mean_diff = abs(train_mean - val_mean) / (train_std + 1e-6) * 100
        
        print(f"\n{name}:")
        print(f"  Train: mean={train_mean:7.2f} ± {train_std:5.2f} | range=[{train_min:7.2f}, {train_max:7.2f}]")
        print(f"  Val:   mean={val_mean:7.2f} ± {val_std:5.2f} | range=[{val_min:7.2f}, {val_max:7.2f}]")
        print(f"  Mean difference: {mean_diff:.1f}% " + 
              ("⚠️  LIKELY BIASED" if mean_diff > 15 else "✓ OK"))
    
    # Check for perfect correlations (overfitting risk)
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS (Feature→Property Coupling)")
    print(f"{'='*70}")
    
    for i, name in enumerate(property_names):
        # Check correlation with specific feature ranges
        if i == 0:  # LogP uses features 0:20
            feature_range = features[:, :20]
            mean_corr = np.corrcoef(feature_range.mean(dim=1).numpy(), 
                                    properties[:, i].numpy())[0, 1]
        elif i == 1:  # MW uses features 20:40
            feature_range = features[:, 20:40]
            mean_corr = np.corrcoef(feature_range.abs().sum(dim=1).numpy(),
                                    properties[:, i].numpy())[0, 1]
        else:
            mean_corr = np.nan
        
        if not np.isnan(mean_corr):
            strength = "VERY STRONG (overfitting risk!)" if abs(mean_corr) > 0.9 else \
                       "STRONG" if abs(mean_corr) > 0.7 else "MODERATE"
            print(f"{name}: correlation = {mean_corr:.4f} ({strength})")
    
    # Effective sample count per parameter
    print(f"\n{'='*70}")
    print("REGULARIZATION ASSESSMENT")
    print(f"{'='*70}")
    
    n_params = 67333  # From PropertyGuidanceRegressor
    samples_per_param = len(train_features) / n_params
    print(f"Model parameters: {n_params:,}")
    print(f"Training samples: {len(train_features)}")
    print(f"Samples per parameter: {samples_per_param:.1f}")
    print(f"Assessment: " + 
          ("🔴 TOO HIGH (overfitting likely)" if samples_per_param < 20 else
           "🟡 MODERATE" if samples_per_param < 50 else
           "🟢 OK (sufficient regularization)"))
    
    # Synthetic data warning
    print(f"\n{'='*70}")
    print("DATA GENERATION WARNING")
    print(f"{'='*70}")
    print("⚠️  Current dataset is ENTIRELY SYNTHETIC with DETERMINISTIC properties")
    print("    Properties are mathematical functions of features:")
    print("      LogP = clamp(features[0:20].mean() * 2 - 1, -2, 5)")
    print("      MW = clamp(features[20:40].abs().sum() * 50 + 300, 50, 700)")
    print("      ... etc")
    print("\n    This means:")
    print("      ✗ Model learns perfect mappings (not realistic)")
    print("      ✗ Very high correlation with input features")
    print("      ✗ No noise or measurement error")
    print("      ✗ Overfitting is EXPECTED and UNAVOIDABLE with this data")
    
    return train_features, train_properties, val_features, val_properties


def compare_original_vs_improved():
    """Compare diagnostics of original vs improved training approach."""
    
    print(f"\n{'='*70}")
    print("IMPROVEMENTS IN NEW TRAINING SCRIPT")
    print(f"{'='*70}")
    
    improvements = [
        ("Dropout Rate", "0% (none)", "20%", "Prevents co-adaptation of neurons"),
        ("L2 Weight Decay", "1e-5 (weak)", "1e-4 (moderate)", "Stronger regularization"),
        ("Dataset Size", "1000 samples", "2000 samples", "More data = better generalization"),
        ("Train/Val/Test Split", "80/20", "70/15/15", "Separate test set for final validation"),
        ("LR Scheduler", "CosineAnnealingLR", "ReduceLROnPlateau", "Adapts to loss plateau"),
        ("Early Stopping Patience", "5 epochs", "10 epochs", "Longer convergence window"),
        ("Weight Init", "Default", "Kaiming Normal", "Better for ReLU networks"),
        ("Grad Clipping", "None", "max_norm=1.0", "Prevents gradient explosion"),
    ]
    
    print(f"\n{'Aspect':<20} {'Original':<20} {'Improved':<20} {'Benefit':<30}")
    print("-" * 90)
    for aspect, orig, imp, benefit in improvements:
        print(f"{aspect:<20} {orig:<20} {imp:<20} {benefit:<30}")
    
    print(f"\n{'='*70}")
    print("EXPECTED RESULTS")
    print(f"{'='*70}")
    print("Original training:")
    print("  - Train/Val gap grows to 4.4x (bad generalization)")
    print("  - Model learns spurious feature-property mappings")
    print("  - Not suitable for real molecular guidance")
    print("\nImproved training (expected):")
    print("  - Train/Val gap should stabilize < 1.5x")
    print("  - Better regularization prevents overfitting")
    print("  - More robust for real-world guidance")
    print("  - Separate test set reveals true generalization")


if __name__ == '__main__':
    train_f, train_p, val_f, val_p = analyze_dataset_bias()
    compare_original_vs_improved()
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. Run: python train_property_regressor_improved.py --epochs 100")
    print("2. Monitor the train/val loss curves during training")
    print("3. Compare final metrics with original training")
    print("4. Use improved model for molecular generation guidance")
    print(f"{'='*70}\n")
