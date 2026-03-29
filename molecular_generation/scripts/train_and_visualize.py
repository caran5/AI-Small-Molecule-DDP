"""
Train the diffusion model and visualize performance metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.diffusion import DiffusionModel
from src.models.trainer import DiffusionTrainer
from src.data.loader import DataLoader, create_dummy_data, MolecularDataset


def visualize_noise_schedule():
    """Visualize the noise schedule."""
    print("Creating noise schedule visualization...")
    
    from src.models.diffusion import NoiseScheduler
    
    scheduler = NoiseScheduler(num_timesteps=1000, schedule='cosine')
    
    t = np.arange(1000)
    alphas = scheduler.alphas_cumprod.numpy()
    sqrt_alphas = scheduler.sqrt_alphas_cumprod.numpy()
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod.numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(t, alphas, linewidth=2, color='steelblue')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Alpha (cumulative)')
    axes[0].set_title('Noise Schedule: Alpha')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(t, sqrt_alphas, label='√α', linewidth=2, color='green')
    axes[1].plot(t, sqrt_one_minus, label='√(1-α)', linewidth=2, color='red')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Noise Schedule: Signal vs Noise')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Show noise progression
    x_0 = np.array([1.0, 0.0, -1.0])
    noise = np.random.randn(3)
    x_t_samples = []
    for t_idx in [0, 250, 500, 750, 999]:
        sqrt_a = sqrt_alphas[t_idx]
        sqrt_1a = sqrt_one_minus[t_idx]
        x_t = sqrt_a * x_0 + sqrt_1a * noise
        x_t_samples.append(x_t)
    
    axes[2].bar(range(len(x_t_samples)), [np.linalg.norm(x) for x in x_t_samples], 
                color=['blue', 'cyan', 'green', 'orange', 'red'])
    axes[2].set_xlabel('Timestep (0, 250, 500, 750, 999)')
    axes[2].set_ylabel('||x_t||')
    axes[2].set_title('Signal Degradation Over Time')
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('noise_schedule.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: noise_schedule.png")
    plt.close()


def train_and_visualize():
    """Train model and visualize learning curves."""
    print("\nTraining model...")
    
    # Create data
    molecules = create_dummy_data(n_samples=100)
    dataset = MolecularDataset(molecules, augment=True, augment_prob=0.5)
    
    # Split for train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = DiffusionModel(
        in_channels=5,
        num_timesteps=100,
        schedule='cosine',
        unet_channels=[64]
    )
    
    # Train
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3
    )
    
    history = trainer.train(num_epochs=10, eval_every=1)
    
    # Visualize training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    epochs = history['epoch']
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    
    axes[0].plot(epochs, train_losses, 'o-', linewidth=2, markersize=8, label='Train Loss', color='steelblue')
    axes[0].plot(epochs, val_losses, 's-', linewidth=2, markersize=8, label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Loss improvement
    improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
    axes[1].plot(epochs, improvement, 'D-', linewidth=2, markersize=8, color='green')
    axes[1].fill_between(epochs, improvement, alpha=0.3, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: training_curves.png")
    plt.close()
    
    return model, device, train_loader


def visualize_model_architecture():
    """Visualize model architecture summary."""
    print("\nCreating architecture summary...")
    
    model = DiffusionModel(in_channels=5, num_timesteps=100)
    
    # Count parameters by layer type
    layer_counts = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        layer_type = name.split('.')[1] if '.' in name else 'other'
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + param.numel()
        total_params += param.numel()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    layers = list(layer_counts.keys())
    params = list(layer_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(layers)))
    
    wedges, texts, autotexts = axes[0].pie(params, labels=layers, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    axes[0].set_title('Parameter Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    sorted_items = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
    layers_sorted = [item[0] for item in sorted_items]
    params_sorted = [item[1] for item in sorted_items]
    
    bars = axes[1].bar(layers_sorted, params_sorted, color=colors)
    axes[1].set_ylabel('Number of Parameters', fontsize=12)
    axes[1].set_title('Parameters by Layer Type', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height/1000)}K',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: model_architecture.png")
    print(f"Total parameters: {total_params:,}")
    plt.close()


def visualize_samples(model, device, train_loader):
    """Visualize generated samples."""
    print("\nGenerating samples...")
    
    # Get real samples from data
    real_batch = next(iter(train_loader))
    real_features = real_batch['features'][:4].numpy()  # (4, 128, 5)
    
    # Generate fake samples
    with torch.no_grad():
        fake_samples = model.sample(batch_size=4, device=device).cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    
    # Visualize real samples
    for i in range(4):
        # Show feature heatmap
        ax = axes[0, i]
        im = ax.imshow(real_features[i, :50, :], cmap='coolwarm', aspect='auto')
        ax.set_title(f'Real Sample {i+1}', fontweight='bold')
        ax.set_ylabel('Atom Index')
        ax.set_xlabel('Feature Dim')
        plt.colorbar(im, ax=ax)
    
    # Visualize generated samples
    for i in range(4):
        ax = axes[1, i]
        im = ax.imshow(fake_samples[i, :50, :], cmap='coolwarm', aspect='auto')
        ax.set_title(f'Generated Sample {i+1}', fontweight='bold')
        ax.set_ylabel('Atom Index')
        ax.set_xlabel('Feature Dim')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('sample_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: sample_comparison.png")
    plt.close()


def visualize_feature_statistics():
    """Visualize feature statistics."""
    print("\nAnalyzing feature statistics...")
    
    molecules = create_dummy_data(n_samples=50)
    dataset = MolecularDataset(molecules, augment=False)
    
    all_features = []
    for i in range(len(dataset)):
        batch = dataset[i]
        all_features.append(batch['features'].numpy())
    
    all_features = np.concatenate(all_features, axis=0)  # (batch, n_atoms, features)
    
    # Remove padding (features with all zeros)
    valid_features = all_features[~np.all(all_features == 0, axis=1)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    feature_names = ['Atomic Num', 'X Pos', 'Y Pos', 'Z Pos', 'Dist from COM']
    
    for i in range(5):
        ax = axes[i // 3, i % 3]
        data = valid_features[:, i]
        
        ax.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
        ax.axvline(np.median(data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(data):.3f}')
        ax.set_xlabel('Feature Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature_names[i]} Distribution', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
    
    # Statistics summary
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""
    Dataset Statistics
    
    Valid atoms: {len(valid_features):,}
    Feature mean: {valid_features.mean():.4f}
    Feature std: {valid_features.std():.4f}
    Feature min: {valid_features.min():.4f}
    Feature max: {valid_features.max():.4f}
    
    Atomic Num range: [{valid_features[:, 0].min():.2f}, {valid_features[:, 0].max():.2f}]
    Position range: [{valid_features[:, 1:4].min():.2f}, {valid_features[:, 1:4].max():.2f}]
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('feature_statistics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: feature_statistics.png")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("DIFFUSION MODEL PERFORMANCE VISUALIZATION")
    print("="*70)
    
    # Create visualizations
    visualize_noise_schedule()
    visualize_model_architecture()
    visualize_feature_statistics()
    model, device, train_loader = train_and_visualize()
    visualize_samples(model, device, train_loader)
    
    print("\n" + "="*70)
    print("✓ All visualizations saved!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. noise_schedule.png - Noise schedule progression")
    print("  2. model_architecture.png - Parameter distribution")
    print("  3. feature_statistics.png - Feature distributions")
    print("  4. training_curves.png - Training/validation loss")
    print("  5. sample_comparison.png - Real vs generated samples")
