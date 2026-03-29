"""
Test the complete diffusion model architecture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.diffusion import DiffusionModel, NoiseScheduler
from src.models.unet import SimpleUNet
from src.models.trainer import DiffusionTrainer
from src.data.loader import DataLoader


def test_noise_scheduler():
    """Test noise scheduler."""
    print("\n" + "="*70)
    print("1. Testing Noise Scheduler")
    print("="*70)
    
    scheduler = NoiseScheduler(num_timesteps=1000, schedule='linear')
    
    # Test q_sample
    x_0 = torch.randn(2, 128, 5)
    t = torch.tensor([0, 500])
    noise = torch.randn_like(x_0)
    
    x_t = scheduler.q_sample(x_0, t, noise)
    
    print(f"✓ Input shape: {x_0.shape}")
    print(f"✓ Output shape: {x_t.shape}")
    print(f"✓ Alpha at t=0: {scheduler.alphas_cumprod[0]:.4f}")
    print(f"✓ Alpha at t=500: {scheduler.alphas_cumprod[500]:.4f}")
    print(f"✓ Alpha at t=999: {scheduler.alphas_cumprod[999]:.4f}")


def test_unet():
    """Test U-Net architecture."""
    print("\n" + "="*70)
    print("2. Testing U-Net Architecture")
    print("="*70)
    
    unet = SimpleUNet(in_channels=5, out_channels=5, hidden_channels=64)
    
    # Forward pass
    x = torch.randn(2, 128, 5)  # (batch_size, n_atoms, features)
    t = torch.randn(2, 128)  # (batch_size, time_dim)
    
    out = unet(x, t)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    print(f"✓ Shapes match: {x.shape == out.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"✓ Total parameters: {num_params:,}")


def test_diffusion_model():
    """Test diffusion model."""
    print("\n" + "="*70)
    print("3. Testing Diffusion Model")
    print("="*70)
    
    model = DiffusionModel(
        in_channels=5,
        num_timesteps=100,
        schedule='cosine'
    )
    
    # Test forward pass
    x_0 = torch.randn(4, 128, 5)
    t = torch.randint(0, 100, (4,))
    
    noise_pred = model(x_0, t)
    
    print(f"✓ Input shape: {x_0.shape}")
    print(f"✓ Noise prediction shape: {noise_pred.shape}")
    print(f"✓ Shapes match: {x_0.shape == noise_pred.shape}")
    
    # Test loss
    loss = model.get_loss(x_0)
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    # Test diffuse
    x_t, noise = model.diffuse(x_0, t)
    print(f"✓ Diffused shape: {x_t.shape}")
    print(f"✓ Noise shape: {noise.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {num_params:,}")


def test_training_integration():
    """Test training integration with data loader."""
    print("\n" + "="*70)
    print("4. Testing Training Integration")
    print("="*70)
    
    # Create model
    model = DiffusionModel(
        in_channels=5,
        num_timesteps=50,
        schedule='linear',
        unet_channels=[32, 64]  # Smaller for testing
    )
    
    # Create dummy data loader
    from src.data.loader import create_dummy_data
    molecules = create_dummy_data(n_samples=32)
    
    from src.data.loader import MolecularDataset
    dataset = MolecularDataset(molecules, augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"✓ Created dataset with {len(dataset)} molecules")
    print(f"✓ Batch size: 4")
    
    # Test trainer
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    trainer = DiffusionTrainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        device=device,
        lr=1e-3
    )
    
    print(f"✓ Trainer initialized on device: {device}")
    
    # Run one training step
    batch = next(iter(loader))
    loss = trainer.train_step(batch)
    
    print(f"✓ Training step completed")
    print(f"✓ Loss: {loss:.4f}")


def test_sampling():
    """Test sampling from model."""
    print("\n" + "="*70)
    print("5. Testing Sampling")
    print("="*70)
    
    model = DiffusionModel(
        in_channels=5,
        num_timesteps=10,  # Few steps for testing
        schedule='linear'
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    # Generate samples
    samples = model.sample(batch_size=2, device=device)
    
    print(f"✓ Generated samples shape: {samples.shape}")
    print(f"✓ Sample value range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"✓ Sample mean: {samples.mean():.4f}")
    print(f"✓ Sample std: {samples.std():.4f}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DIFFUSION MODEL ARCHITECTURE TEST SUITE")
    print("="*70)
    
    try:
        test_noise_scheduler()
        test_unet()
        test_diffusion_model()
        test_training_integration()
        test_sampling()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
