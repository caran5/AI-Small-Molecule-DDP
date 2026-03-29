#!/usr/bin/env python3
"""
Guided inference: test property-steered generation with guidance.

NOTE: GuidedGenerator requires a trained PropertyGuidanceRegressor.
This is a Phase 2 component - for now, we demonstrate the concept.
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


def generate_with_guidance(
    model: ConditionalUNet,
    target_properties: dict,
    guidance_scale: float = 1.0,
    num_samples: int = 2,
    num_steps: int = 30,
    max_atoms: int = 128,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate molecules with property guidance.

    NOTE: This simplified version doesn't use gradient-based guidance yet.
    Full guided generation requires a trained PropertyGuidanceRegressor.
    See Phase 2 in improve_model.py for complete implementation.
    """

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

    # Reverse diffusion with properties
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

    return x.cpu()


def test_guided_generation():
    """Test property-steered generation."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create conditional diffusion model
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

    print("🎯 Testing Property-Guided Generation\n")
    print("=" * 60)

    # Test different guidance strengths
    test_cases = [
        {
            'properties': {'logp': 3.5, 'mw': 350, 'hbd': 2, 'hba': 3, 'rotatable': 6},
            'guidance_scale': 0.5,
            'label': 'Weak guidance'
        },
        {
            'properties': {'logp': 3.5, 'mw': 350, 'hbd': 2, 'hba': 3, 'rotatable': 6},
            'guidance_scale': 2.0,
            'label': 'Strong guidance'
        },
        {
            'properties': {'logp': 1.0, 'mw': 200, 'hbd': 4, 'hba': 5, 'rotatable': 1},
            'guidance_scale': 1.5,
            'label': 'Hydrophilic, small molecule'
        },
    ]

    for test in test_cases:
        print(f"\n{test['label']}")
        print(f"  Target properties: {test['properties']}")
        print(f"  Guidance scale: {test['guidance_scale']}")

        try:
            samples = generate_with_guidance(
                model=model,
                target_properties=test['properties'],
                guidance_scale=test['guidance_scale'],
                num_samples=2,
                num_steps=30,
                device=device
            )

            print(f"  ✅ Generated {samples.shape[0]} samples")
            print(f"     Shape: {samples.shape}")
            print(f"     Value range: [{samples.min():.3f}, {samples.max():.3f}]")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("\n💡 About Guidance Scale:")
    print("  - 0.5: Mild property steering")
    print("  - 1.0-2.0: Moderate guidance (recommended)")
    print("  - 5.0+: Very strong steering")
    print("\n⚠️  IMPORTANT: Full gradient-based guidance requires:")
    print("  1. Trained PropertyGuidanceRegressor (Phase 2)")
    print("  2. See GuidedGenerator in guided_sampling.py")
    print("  3. See improve_model.py Phase 2 for implementation")


if __name__ == "__main__":
    print("🔬 Molecular Diffusion Model - Guided Inference\n")
    test_guided_generation()
