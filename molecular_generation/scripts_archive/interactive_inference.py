#!/usr/bin/env python3
"""
Interactive inference: test your model with custom molecules/properties.

Usage:
    python interactive_inference.py

Then enter properties interactively or use menu options.
"""

import torch
import numpy as np
from src.models.unet import ConditionalUNet
from src.models.diffusion import NoiseScheduler
from src.inference.decoder import MolecularDecoder


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
            norm_val = max(0.0, min(1.0, norm_val))
            normalized.append(norm_val)
        return torch.tensor(normalized, dtype=torch.float32)


def generate_molecules(
    model: ConditionalUNet,
    target_properties: dict,
    num_samples: int = 3,
    num_steps: int = 30,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate molecules with target properties."""

    normalizer = PropertyNormalizer()
    model.eval()

    # Normalize and batch properties
    norm_props = normalizer.normalize(target_properties)
    prop_batch = norm_props.unsqueeze(0).repeat(num_samples, 1).to(device)

    # Initialize with noise
    x = torch.randn(num_samples, 128, 5, device=device)

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


def get_user_properties():
    """Get molecular properties from user input."""
    print("\nEnter target molecular properties (or press Enter for defaults):\n")

    try:
        logp = input("LogP (lipophilicity) [2.0]: ").strip()
        logp = float(logp) if logp else 2.0

        mw = input("MW (molecular weight) [300]: ").strip()
        mw = float(mw) if mw else 300.0

        hbd = input("HBD (H-bond donors) [2]: ").strip()
        hbd = float(hbd) if hbd else 2.0

        hba = input("HBA (H-bond acceptors) [4]: ").strip()
        hba = float(hba) if hba else 4.0

        rot = input("Rotatable bonds [5]: ").strip()
        rot = float(rot) if rot else 5.0

        return {
            'logp': logp,
            'mw': mw,
            'hbd': hbd,
            'hba': hba,
            'rotatable': rot
        }

    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        print("Using defaults...")
        return {
            'logp': 2.0,
            'mw': 300,
            'hbd': 2,
            'hba': 4,
            'rotatable': 5
        }


def print_properties_summary(props):
    """Pretty print properties."""
    print("\n" + "=" * 50)
    print("🧪 TARGET PROPERTIES")
    print("=" * 50)
    print(f"  LogP (lipophilicity):     {props['logp']:.2f}")
    print(f"  MW (molecular weight):    {props['mw']:.0f}")
    print(f"  HBD (H-bond donors):      {props['hbd']:.0f}")
    print(f"  HBA (H-bond acceptors):   {props['hba']:.0f}")
    print(f"  Rotatable bonds:          {props['rotatable']:.0f}")
    print("=" * 50)


def generate_and_show_results(model, props, num_samples=3, device='cpu'):
    """Generate molecules and display decoded results."""
    print(f"\n⏳ Generating {num_samples} molecules...")

    try:
        samples = generate_molecules(
            model=model,
            target_properties=props,
            num_samples=num_samples,
            num_steps=30,
            device=device
        )

        print(f"\n✅ SUCCESS! Generated {samples.shape[0]} molecules\n")
        print("🧬 DECODED MOLECULES")
        print("=" * 70)

        valid_count = 0
        for i, sample in enumerate(samples, 1):
            # Decode feature tensor to molecular structure
            mol_dict = MolecularDecoder.features_to_molecule_dict(sample)

            print(f"\nMolecule {i}:")
            print(f"  n_atoms: {mol_dict['n_atoms']}")
            print(f"  formula: {mol_dict['formula']}")
            print(f"  valid: {mol_dict['valid']}")

            if mol_dict['valid']:
                print(f"  SMILES: {mol_dict['smiles']}")
                valid_count += 1
            else:
                print(f"  SMILES: [invalid - sanitization failed]")

            if mol_dict['coordinates'] is not None and len(mol_dict['coordinates']) > 0:
                coords = mol_dict['coordinates']
                print(f"  coord range: [{coords.min():.2f}, {coords.max():.2f}] Å")

        print("\n" + "=" * 70)
        print(f"✓ Valid structures: {valid_count}/{num_samples}")
        print("💡 Next steps:")
        print("  1. Calculate actual properties from valid SMILES")
        print("  2. Verify model learned to match target properties")
        print("  3. Train model to improve validity and property guidance")

        return samples

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "=" * 50)
    print("🧬 MOLECULAR DIFFUSION MODEL")
    print("Interactive Inference Testing")
    print("=" * 50)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")

    # Load model
    print("✓ Creating ConditionalUNet model...")
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

    print("✓ Model ready for inference")

    # Interactive loop
    while True:
        print("\n" + "=" * 50)
        print("OPTIONS")
        print("=" * 50)
        print("1. Generate with default properties")
        print("2. Generate with custom properties")
        print("3. Generate multiple times with same properties")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            props = {
                'logp': 2.0,
                'mw': 300,
                'hbd': 2,
                'hba': 4,
                'rotatable': 5
            }
            print_properties_summary(props)
            generate_and_show_results(model, props, num_samples=3, device=device)

        elif choice == "2":
            props = get_user_properties()
            print_properties_summary(props)
            generate_and_show_results(model, props, num_samples=3, device=device)

        elif choice == "3":
            props = get_user_properties()
            print_properties_summary(props)
            num_times = input("\nHow many times? [3]: ").strip()
            num_times = int(num_times) if num_times else 3
            for i in range(num_times):
                print(f"\n--- Iteration {i+1}/{num_times} ---")
                generate_and_show_results(model, props, num_samples=1, device=device)

        elif choice == "4":
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
