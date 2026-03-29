#!/usr/bin/env python3
"""
Simplified inference script that works with ConditionalUNet architecture.

Shows how to generate molecules with target properties using your trained model.
"""

import torch
import numpy as np
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
            norm_val = np.clip(norm_val, 0.0, 1.0)
            normalized.append(norm_val)
        return torch.tensor(normalized, dtype=torch.float32)


def generate_conditional(
    model: ConditionalUNet,
    target_properties: dict,
    num_samples: int = 3,
    num_steps: int = 50,
    max_atoms: int = 128,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate molecular features conditioned on target properties.

    Args:
        model: ConditionalUNet model
        target_properties: Dict with keys like 'logp', 'mw', 'hbd', 'hba', 'rotatable'
        num_samples: Number of molecules to generate
        num_steps: Number of denoising steps (more = better quality but slower)
        max_atoms: Maximum number of atoms
        device: 'cpu' or 'cuda'

    Returns:
        Generated features shape (num_samples, max_atoms, 5)
    """

    # Setup
    normalizer = PropertyNormalizer()
    model.eval()

    # Normalize properties and create batch
    norm_props = normalizer.normalize(target_properties)
    prop_batch = norm_props.unsqueeze(0).repeat(num_samples, 1).to(device)

    # Initialize with random noise
    # Shape: (batch_size, max_atoms, 5) where 5 = [atomic_num, x, y, z, distance_from_com]
    x = torch.randn(num_samples, max_atoms, 5, device=device)

    # Create noise scheduler
    scheduler = NoiseScheduler(
        num_timesteps=num_steps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule='cosine'
    )

    print(f"⏳ Generating {num_samples} molecules with target properties...")
    print(f"   Properties: {target_properties}")

    # Reverse diffusion process
    with torch.no_grad():
        for step in reversed(range(num_steps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)

            # Predict noise using conditional model
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

            if (num_steps - step - 1) % max(1, num_steps // 10) == 0:
                print(f"   Step {num_steps - step}/{num_steps} ✓")

    print(f"✅ Generation complete!")
    return x.cpu()


def main():
    """Run inference tests."""

    print("=" * 70)
    print("🧬 MOLECULAR DIFFUSION MODEL - SIMPLIFIED INFERENCE")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")

    # Create model
    print("Loading ConditionalUNet model...")
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

    # Test cases with different molecular properties
    test_cases = [
        {
            'name': 'Drug-like (moderate lipophilicity)',
            'props': {'logp': 2.5, 'mw': 350, 'hbd': 2, 'hba': 4, 'rotatable': 5},
        },
        {
            'name': 'Hydrophobic (high lipophilicity)',
            'props': {'logp': 4.0, 'mw': 450, 'hbd': 1, 'hba': 2, 'rotatable': 8},
        },
        {
            'name': 'Hydrophilic (low lipophilicity)',
            'props': {'logp': 0.5, 'mw': 250, 'hbd': 4, 'hba': 6, 'rotatable': 2},
        },
    ]

    print("\n" + "=" * 70)
    print("GENERATION TESTS")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print("-" * 70)

        try:
            generated = generate_conditional(
                model=model,
                target_properties=test['props'],
                num_samples=2,  # Generate 2 molecules per test
                num_steps=30,  # Faster for testing (use 50-100 for quality)
                device=device,
            )

            print(f"\n   📊 Generated Samples Statistics:")
            print(f"      Shape: {generated.shape}")
            print(f"      Value range: [{generated.min():.4f}, {generated.max():.4f}]")
            print(f"      Mean: {generated.mean():.4f}, Std: {generated.std():.4f}")

            # Show feature breakdown
            for sample_idx in range(min(1, generated.shape[0])):
                sample = generated[sample_idx]
                # Count non-zero atoms
                non_zero_atoms = (sample[:, 0].abs() > 0.01).sum().item()
                print(f"\n      Sample {sample_idx + 1}:")
                print(f"         Non-zero atoms: {non_zero_atoms}")
                print(f"         Avg atomic number: {sample[:, 0].mean():.4f}")
                print(f"         Spatial extent (X): [{sample[:, 1].min():.4f}, {sample[:, 1].max():.4f}]")
                print(f"         Spatial extent (Y): [{sample[:, 2].min():.4f}, {sample[:, 2].max():.4f}]")
                print(f"         Spatial extent (Z): [{sample[:, 3].min():.4f}, {sample[:, 3].max():.4f}]")

        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✨ TESTING COMPLETE")
    print("=" * 70)
    print("\n📝 NEXT STEPS:")
    print("   1. Load your trained model checkpoint (if available)")
    print("   2. Decode features → molecular structures (SMILES/3D coords)")
    print("   3. Calculate properties from generated structures")
    print("   4. Validate that generated properties match targets")
    print()


if __name__ == "__main__":
    main()
