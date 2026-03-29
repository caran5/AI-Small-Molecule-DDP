#!/usr/bin/env python3
"""
Batch inference testing - test conditional generation with multiple property sets.
"""

import torch
from src.models.unet import ConditionalUNet
from src.models.diffusion import NoiseScheduler


class PropertyNormalizer:
    """Normalize molecular properties to [0, 1] range."""

    def __init__(self):
        self.ranges = {
            'logp': (0.0, 5.0),
            'mw': (100, 500),
            'hbd': (0, 5),
            'hba': (0, 10),
            'rotatable': (0, 15)
        }

    def normalize(self, props: dict) -> torch.Tensor:
        """Convert property dict to normalized tensor."""
        normalized = []
        for key in ['logp', 'mw', 'hbd', 'hba', 'rotatable']:
            value = props.get(key, 0.0)
            min_val, max_val = self.ranges[key]
            norm_val = (value - min_val) / (max_val - min_val)
            norm_val = max(0.0, min(1.0, norm_val))  # Clip to [0, 1]
            normalized.append(norm_val)
        return torch.tensor(normalized, dtype=torch.float32)


def generate_with_target_properties(
    model: ConditionalUNet,
    target_properties: dict,
    num_samples: int = 3,
    num_steps: int = 30,
    max_atoms: int = 128,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate molecules with target properties."""

    normalizer = PropertyNormalizer()
    model.eval()

    # Normalize and batch properties
    norm_props = normalizer.normalize(target_properties)
    prop_batch = norm_props.unsqueeze(0).repeat(num_samples, 1).to(device)

    # Initialize with noise
    x = torch.randn(num_samples, max_atoms, 5, device=device)

    # Noise scheduler
    scheduler = NoiseScheduler(
        num_timesteps=num_steps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule='cosine'
    )

    # Reverse diffusion
    with torch.no_grad():
        for step in reversed(range(num_steps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)

            # Predict noise
            noise_pred = model(x, t, properties=prop_batch)

            # Get schedule values
            alpha_t = scheduler.alphas_cumprod[step]
            alpha_prev = scheduler.alphas_cumprod_prev[step]
            beta_t = scheduler.betas[step]

            # DDPM reverse step
            x_0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            if step > 0:
                noise = torch.randn_like(x)
                sigma_t = ((1 - alpha_prev) / (1 - alpha_t) * beta_t).sqrt()
                x = (
                    alpha_prev.sqrt() * x_0_pred / alpha_t.sqrt()
                    + (1 - alpha_prev).sqrt() * noise_pred
                    + sigma_t * noise
                )
            else:
                x = x_0_pred

    return x.cpu()


def test_generation():
    """Test conditional generation with different target properties."""

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Create conditional model
    print("Creating ConditionalUNet model...")
    model = ConditionalUNet(
        in_channels=5,
        out_channels=5,
        hidden_channels=128,
        time_dim=128,
        depth=3,
        n_properties=5,
        dropout_rate=0.1
    )
    model = model.to(device)

    # Test different molecular property targets
    test_properties = [
        {'logp': 2.0, 'mw': 300, 'hbd': 2, 'hba': 4, 'rotatable': 5},
        {'logp': 4.0, 'mw': 450, 'hbd': 1, 'hba': 2, 'rotatable': 8},
        {'logp': 1.5, 'mw': 200, 'hbd': 3, 'hba': 5, 'rotatable': 2},
    ]

    print("Testing conditional generation with target properties:\n")
    print("=" * 60)

    for i, props in enumerate(test_properties, 1):
        print(f"\n🧪 Test {i}: Target Properties")
        print(f"   LogP (lipophilicity): {props['logp']}")
        print(f"   MW (molecular weight): {props['mw']}")
        print(f"   HBD (H-bond donors): {props['hbd']}")
        print(f"   HBA (H-bond acceptors): {props['hba']}")
        print(f"   Rotatable bonds: {props['rotatable']}")

        try:
            # Generate 3 molecules with these properties
            generated = generate_with_target_properties(
                model=model,
                target_properties=props,
                num_samples=3,
                num_steps=30,
                device=device
            )

            print(f"\n   ✅ Generated {generated.shape[0]} samples")
            print(f"   Shape: {generated.shape}")
            print(f"   Value range: [{generated.min():.3f}, {generated.max():.3f}]")
            print(f"   Mean: {generated.mean():.3f}, Std: {generated.std():.3f}")

        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("\n✨ Testing complete!")
    print("\nNext steps:")
    print("  1. Use simple_inference.py for main generation")
    print("  2. Use validate_generation.py for validation")
    print("  3. See improve_model.py for improvement roadmap")


def test_unconditional_generation(num_samples=5):
    """Test unconditional generation (no property guidance)."""
    print("\n\n📊 Unconditional Generation Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = ConditionalUNet(
        in_channels=5,
        out_channels=5,
        hidden_channels=128,
        time_dim=128,
        depth=3,
        n_properties=5,
        dropout_rate=0.1
    )
    model = model.to(device)
    model.eval()

    print(f"\n✅ Testing forward pass with ConditionalUNet")
    try:
        with torch.no_grad():
            # Create dummy input
            batch_size = 2
            n_atoms = 50  # Example atom count
            x = torch.randn(batch_size, n_atoms, 5, device=device)
            t = torch.randint(0, 100, (batch_size,), device=device)
            props = torch.randn(batch_size, 5, device=device)

            # Forward pass without properties
            output_uncond = model(x, t, properties=None)
            print(f"   Unconditional output shape: {output_uncond.shape}")

            # Forward pass with properties
            output_cond = model(x, t, properties=props)
            print(f"   Conditional output shape: {output_cond.shape}")
            print(f"   Output range: [{output_cond.min():.3f}, {output_cond.max():.3f}]")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔬 Molecular Diffusion Model - Inference Testing\n")

    test_generation()
    test_unconditional_generation()
