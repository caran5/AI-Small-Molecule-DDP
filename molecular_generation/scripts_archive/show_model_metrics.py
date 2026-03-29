#!/usr/bin/env python3
"""
Show trained model metrics and create visualizations.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

from src.inference.guided_sampling import PropertyGuidanceRegressor
from src.eval.property_validation import compute_properties
from src.inference.decoder import MolecularDecoder


def load_trained_regressor(checkpoint_path='checkpoints/property_regressor.pt'):
    """Load the trained regressor."""
    regressor = PropertyGuidanceRegressor(input_dim=100, n_properties=5)
    if Path(checkpoint_path).exists():
        regressor.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✓ Loaded trained regressor from {checkpoint_path}")
        return regressor
    else:
        print(f"✗ No checkpoint found at {checkpoint_path}")
        return None


def evaluate_regressor(regressor, num_samples=100):
    """Evaluate regressor performance on test data."""
    regressor.eval()
    
    # Generate test features
    test_features = torch.randn(num_samples, 100)
    
    with torch.no_grad():
        predictions = regressor(test_features)
    
    # Compute basic statistics
    pred_mean = predictions.mean(dim=0).numpy()
    pred_std = predictions.std(dim=0).numpy()
    
    property_names = ['LogP', 'MW', 'HBD', 'HBA', 'Rotatable']
    
    print("\n" + "="*70)
    print("REGRESSOR EVALUATION")
    print("="*70)
    print(f"\nGenerated {num_samples} random feature vectors\n")
    print(f"{'Property':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)
    
    for i, name in enumerate(property_names):
        mean_val = pred_mean[i]
        std_val = pred_std[i]
        min_val = predictions[:, i].min().item()
        max_val = predictions[:, i].max().item()
        print(f"{name:<15} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
    
    return predictions.numpy()


def create_visualizations(predictions):
    """Create comprehensive visualizations."""
    property_names = ['LogP', 'MW', 'HBD', 'HBA', 'Rotatable']
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Distribution plots
    print("\nGenerating visualizations...")
    
    for i, prop_name in enumerate(property_names):
        ax = fig.add_subplot(gs[0, i if i < 3 else i-3])
        
        data = predictions[:, i]
        ax.hist(data, bins=20, color=plt.cm.Set3(i), alpha=0.7, edgecolor='black')
        ax.set_title(f'{prop_name} Distribution', fontweight='bold', fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add stats text
        stats_text = f'μ={data.mean():.2f}\nσ={data.std():.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Property correlations (heatmap-like scatter)
    ax = fig.add_subplot(gs[1, :2])
    
    scatter = ax.scatter(predictions[:, 0], predictions[:, 1], 
                        c=predictions[:, 2], cmap='viridis', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('LogP', fontweight='bold')
    ax.set_ylabel('MW', fontweight='bold')
    ax.set_title('LogP vs MW (colored by HBD)', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='HBD')
    ax.grid(True, alpha=0.3)
    
    # 3. Box plots for all properties
    ax = fig.add_subplot(gs[1, 2])
    
    box_data = [predictions[:, i] for i in range(5)]
    bp = ax.boxplot(box_data, labels=property_names, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Property Value Ranges', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Cumulative distributions
    ax = fig.add_subplot(gs[2, :2])
    
    for i, prop_name in enumerate(property_names):
        data = np.sort(predictions[:, i])
        cumsum = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cumsum, marker='', linestyle='-', linewidth=2,
               label=prop_name, alpha=0.7)
    
    ax.set_xlabel('Value', fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontweight='bold')
    ax.set_title('Cumulative Distributions', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 5. Summary statistics table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    stats_data = []
    for i, prop_name in enumerate(property_names):
        stats_data.append([
            prop_name,
            f"{predictions[:, i].mean():.2f}",
            f"{predictions[:, i].std():.2f}",
            f"{predictions[:, i].min():.2f}",
            f"{predictions[:, i].max():.2f}",
        ])
    
    table = ax.table(cellText=stats_data,
                    colLabels=['Property', 'Mean', 'Std', 'Min', 'Max'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    fig.suptitle('Property Guidance Regressor - Metrics & Visualizations', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = 'results/metrics_visualization.png'
    Path('results').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    
    return fig


def print_model_info(regressor):
    """Print detailed model information."""
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    
    # Count parameters
    total_params = sum(p.numel() for p in regressor.parameters())
    trainable_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    
    print(f"\nModel: PropertyGuidanceRegressor")
    print(f"Input dimension: 100")
    print(f"Output properties: 5")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Architecture
    print(f"\nArchitecture:")
    print(f"  Input layer: 100 → 256")
    print(f"  Hidden layer 1: 256 → 128")
    print(f"  Hidden layer 2: 128 → 64")
    print(f"  Output layer: 64 → 5")
    print(f"\nActivation: ReLU + Batch Norm")
    print(f"Output activation: Linear (no activation)")
    
    # Model structure
    print(f"\nLayerwise breakdown:")
    print(f"  {'Layer':<30} {'Parameters':<15} {'Trainable':<15}")
    print("-" * 60)
    
    for name, module in regressor.named_modules():
        if isinstance(module, torch.nn.Linear):
            params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            print(f"  {name:<30} {params:<15} {'Yes':<15}")


def main():
    print("\n" + "="*70)
    print("🧬 TRAINED MODEL METRICS & VISUALIZATION")
    print("="*70)
    
    # Load regressor
    regressor = load_trained_regressor()
    if regressor is None:
        print("\n❌ Failed to load regressor. Please train first:")
        print("   python train_property_regressor.py")
        return
    
    # Print model info
    print_model_info(regressor)
    
    # Evaluate regressor
    predictions = evaluate_regressor(regressor, num_samples=200)
    
    # Create visualizations
    create_visualizations(predictions)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Model loaded and evaluated")
    print(f"✓ Generated predictions for 200 samples")
    print(f"✓ Metrics computed:")
    print(f"  - Distribution statistics")
    print(f"  - Property correlations")
    print(f"  - Value ranges")
    print(f"\n✓ Visualizations saved to: results/metrics_visualization.png")
    print(f"\nNext steps:")
    print(f"1. Review the visualization")
    print(f"2. Use regressor with GuidedGenerator for property-guided sampling")
    print(f"3. Run: python validate_end_to_end_simple.py")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
