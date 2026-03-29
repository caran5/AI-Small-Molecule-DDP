#!/usr/bin/env python3
"""
Validation script: Analyze generated molecules and compare with target properties.
"""

import torch
import numpy as np
from src.models.unet import ConditionalUNet
from src.models.diffusion import NoiseScheduler
from src.inference.decoder import MolecularDecoder
from simple_inference import PropertyNormalizer, generate_conditional


def estimate_properties_from_features(features: np.ndarray) -> dict:
    """
    Estimate molecular properties from generated features.

    This is a simple placeholder - real property calculation would use
    the decoded molecular structure and RDKit or similar.
    """
    # Extract atoms and coordinates
    atomic_nums, coords = MolecularDecoder.features_to_atoms(
        torch.tensor(features, dtype=torch.float32)
    )

    n_atoms = len(atomic_nums)

    # Very rough property estimates based on features
    # In practice, use actual molecular property calculators
    estimates = {
        'n_atoms': n_atoms,
        'atomic_nums': atomic_nums,
        'avg_coord_spread': float(np.std(coords)) if len(coords) > 0 else 0.0,
    }

    return estimates


def compare_properties(target: dict, estimated: dict) -> dict:
    """Compare target vs estimated properties."""
    return {
        'target': target,
        'estimated': estimated,
        'n_atoms_match': estimated.get('n_atoms', 0) > 0,
        'has_valid_atoms': len(estimated.get('atomic_nums', [])) > 0,
    }


def main():
    """Run validation pipeline."""

    print("=" * 70)
    print("🔍 VALIDATION: Generated vs Target Properties")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")

    # Setup model
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

    # Test case
    target_properties = {
        'logp': 2.5,
        'mw': 350,
        'hbd': 2,
        'hba': 4,
        'rotatable': 5
    }

    print("Target Properties:")
    for key, val in target_properties.items():
        print(f"  {key}: {val}")

    # Generate
    print("\n⏳ Generating molecules...")
    generated = generate_conditional(
        model=model,
        target_properties=target_properties,
        num_samples=2,
        num_steps=30,
        device=device
    )

    # Decode and validate
    print("\n" + "=" * 70)
    print("DECODED MOLECULES")
    print("=" * 70)

    for sample_idx, features in enumerate(generated):
        print(f"\n[Sample {sample_idx + 1}]")
        print("-" * 70)

        # Decode features
        mol_dict = MolecularDecoder.features_to_molecule_dict(features)

        print(f"  Valid: {mol_dict['valid']}")
        print(f"  Number of atoms: {mol_dict['n_atoms']}")
        print(f"  Molecular formula: {mol_dict['formula']}")
        print(f"  Atomic numbers: {mol_dict['atoms'][:min(10, len(mol_dict['atoms']))]}")

        if mol_dict['valid']:
            coords = mol_dict['coordinates']
            print(f"  Coordinate range:")
            print(f"    X: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
            print(f"    Y: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
            print(f"    Z: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")

            # Estimate properties
            estimated = estimate_properties_from_features(features.numpy())
            comparison = compare_properties(target_properties, estimated)

            print(f"\n  Property Analysis:")
            print(f"    Atoms detected: {comparison['has_valid_atoms']}")
            print(f"    Avg coordinate spread: {estimated['avg_coord_spread']:.4f}")

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print("""
Generated molecules successfully created! However, to fully validate property matching:

1. **Infer Molecular Structure**
   - Use coordinates to build graph (connectivity inference)
   - Determine bond orders from distances

2. **Calculate Properties**
   - Use RDKit: Descriptors.MolWt(), Descriptors.MolLogP(), etc.
   - Validate against target properties

3. **Improve Model**
   - If properties don't match: train with property loss
   - Use GuidedGenerator with stronger guidance_scale
   - Implement property regressor for guidance

See simple_inference.py and test_guided_inference.py for more examples.
""")


if __name__ == "__main__":
    main()
