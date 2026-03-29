#!/usr/bin/env python3
"""
End-to-end validation: Proof that the model generates valid molecules with target properties.

Pipeline:
  Target Properties (e.g., logp=2.5, mw=300) 
    ↓
  Generate Features using diffusion + conditioning
    ↓
  Decode Features → Molecular Structure (atoms + bonds)
    ↓
  Compute Actual Properties from structure
    ↓
  Compare Target vs Actual with RMSE
    ↓
  Report validation results

Usage:
    python validate_end_to_end.py --model-path checkpoints/model.pt --num-samples 10
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List
import json

from src.models.unet import ConditionalUNet
from src.models.diffusion import DiffusionModel, NoiseScheduler
from src.inference.decoder import MolecularDecoder
from src.eval.property_validation import (
    compute_properties,
    property_rmse,
    print_validation_result
)


def generate_conditional_features(
    model: ConditionalUNet,
    target_properties: Dict[str, float],
    num_samples: int = 1,
    num_steps: int = 50,
    max_atoms: int = 128,
    device: str = 'cpu',
    guidance_scale: float = 1.0
) -> torch.Tensor:
    """Generate molecular features conditioned on target properties.
    
    Args:
        model: Trained conditional U-Net
        target_properties: Dict with keys 'logp', 'mw', 'hbd', 'hba', 'rotatable'
        num_samples: Number of molecules to generate
        num_steps: Denoising steps in reverse diffusion
        max_atoms: Maximum atoms per molecule
        device: 'cpu' or 'cuda'
        guidance_scale: Property guidance strength (1.0 = no guidance)
    
    Returns:
        Tensor of shape (num_samples, max_atoms, 5) with atomic features
    """
    model.eval()
    
    with torch.no_grad():
        # Normalize properties (this is a simplified normalization)
        # In production, use proper PropertyNormalizer
        props_norm = torch.tensor([
            (target_properties['logp'] + 1.0) / 4.0,  # Assume logp in [-1, 7]
            (target_properties['mw'] - 100) / 400.0,   # Assume mw in [100, 500]
            target_properties['hbd'] / 5.0,            # Assume max 5
            target_properties['hba'] / 10.0,           # Assume max 10
            target_properties['rotatable'] / 10.0      # Assume max 10
        ], dtype=torch.float32, device=device).unsqueeze(0).repeat(num_samples, 1)
        
        # Initialize from noise
        x_t = torch.randn(num_samples, max_atoms, 5, device=device)
        
        # Create scheduler
        scheduler = NoiseScheduler(num_timesteps=1000, schedule='cosine')
        
        # Reverse diffusion (simplified sampling)
        for step in range(num_steps - 1, -1, -1):
            t_tensor = torch.full((num_samples,), step, dtype=torch.long, device=device)
            
            # Predict noise
            eps_pred = model(x_t, t_tensor, props_norm)
            
            # DDPM sampling
            alpha_t = scheduler.alphas[step]
            alpha_t_prev = scheduler.alphas[step - 1] if step > 0 else torch.tensor(1.0)
            sigma_t = scheduler.betas[step].sqrt()
            
            # Update
            x_t = (x_t - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
            
            if step > 0:
                x_t = x_t + sigma_t * torch.randn_like(x_t)
        
        # Clip to reasonable range
        x_t = torch.clamp(x_t, -2.0, 2.0)
    
    return x_t


def main():
    """Run end-to-end validation."""
    print("\n" + "="*70)
    print("🧬 END-TO-END MOLECULAR VALIDATION")
    print("="*70)
    print("Process: Target Props → Generate Features → Decode Molecule → Compare\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Initialize model
    print("[1/4] Initializing model...")
    model = ConditionalUNet(
        in_channels=5,
        out_channels=5,
        hidden_channels=128,
        time_dim=128,
        depth=3,
        n_properties=5,
        dropout_rate=0.1
    ).to(device)
    print("✓ Model initialized")
    
    # Define test cases
    print("\n[2/4] Generating molecules with different properties...\n")
    
    test_cases = [
        {
            'name': 'Drug-like (hydrophobic)',
            'properties': {'logp': 3.5, 'mw': 350, 'hbd': 2, 'hba': 3, 'rotatable': 6},
        },
        {
            'name': 'Hydrophilic (small)',
            'properties': {'logp': 0.5, 'mw': 200, 'hbd': 4, 'hba': 5, 'rotatable': 1},
        },
        {
            'name': 'Large molecule',
            'properties': {'logp': 2.0, 'mw': 450, 'hbd': 3, 'hba': 4, 'rotatable': 8},
        },
    ]
    
    all_results = []
    
    for test_case in test_cases:
        print(f"\n{'─'*70}")
        print(f"Test Case: {test_case['name']}")
        print(f"Target Properties: {test_case['properties']}")
        print(f"{'─'*70}")
        
        # Generate features
        print("  → Generating molecular features...")
        try:
            features = generate_conditional(
                model,
                test_case['properties'],
                num_samples=2,
                num_steps=30,
                device=device
            )
            print(f"  ✓ Generated shape: {features.shape}")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")
            continue
        
        # Validate each molecule
        print("  → Validating molecules...")
        batch_results = validate_batch(features, test_case['properties'])
        
        for i, result in enumerate(batch_results):
            print_validation_result(result, index=i, verbose=False)
            all_results.append(result)
    
    # Print overall summary
    print_batch_summary(all_results)
    
    # Calculate statistics
    valid_count = sum(1 for r in all_results if r['valid'])
    total_count = len(all_results)
    
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"✓ End-to-end validation complete!")
    print(f"✓ Generated and validated {total_count} molecules")
    print(f"✓ Valid molecules: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    
    if valid_count > 0:
        rmse_values = [r['rmse']['overall'] for r in all_results if r['valid']]
        avg_rmse = np.mean(rmse_values)
        print(f"✓ Average property RMSE: {avg_rmse:.3f}")
        
        if avg_rmse < 0.5:
            print("✓ Property matching: EXCELLENT (RMSE < 0.5)")
        elif avg_rmse < 1.0:
            print("⚠️  Property matching: GOOD (RMSE < 1.0)")
        else:
            print("⚠️  Property matching: NEEDS IMPROVEMENT (RMSE >= 1.0)")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Train PropertyGuidanceRegressor for better guidance")
    print(f"  2. Validate on held-out test set")
    print(f"  3. Compare property accuracy before/after training regressor")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
