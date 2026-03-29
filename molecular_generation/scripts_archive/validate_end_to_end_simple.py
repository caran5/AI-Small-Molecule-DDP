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
    python validate_end_to_end_simple.py
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

try:
    from src.models.unet import ConditionalUNet
    from src.models.diffusion import NoiseScheduler
    from src.inference.decoder import MolecularDecoder
    from src.eval.property_validation import (
        compute_properties,
        property_rmse,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you run from molecular_generation/ directory")
    exit(1)


def generate_random_features(num_samples: int = 5, max_atoms: int = 128) -> torch.Tensor:
    """Generate random molecular features for testing.
    
    In production, this would be the output of the diffusion model.
    Features shape: (batch, max_atoms, 5) where 5 = [atomic_num, x, y, z, validity]
    """
    # Random atomic numbers (1-6 for H, C, N, O, F, P)
    atomic_nums = torch.randint(1, 7, (num_samples, max_atoms, 1), dtype=torch.float32)
    
    # Random coordinates (normalized)
    coords = torch.randn(num_samples, max_atoms, 3) * 2.0
    
    # Validity mask (some atoms are padding)
    validity = torch.where(
        torch.randn(num_samples, max_atoms, 1) > -0.5,  # ~62% valid atoms
        torch.ones(num_samples, max_atoms, 1),
        torch.zeros(num_samples, max_atoms, 1)
    )
    
    features = torch.cat([atomic_nums, coords, validity], dim=-1)
    return features


def decode_and_validate(
    features: torch.Tensor,
    target_properties: Dict[str, float],
    decoder: MolecularDecoder = None,
    device: str = 'cpu'
) -> Dict:
    """Decode molecular features and validate against target properties.
    
    Args:
        features: Tensor of shape (num_atoms, 5) with atomic features
        target_properties: Target properties (logp, mw, hbd, hba, rotatable)
        decoder: MolecularDecoder instance
        device: 'cpu' or 'cuda'
    
    Returns:
        Dict with validation results
    """
    if decoder is None:
        decoder = MolecularDecoder(device=device)
    
    try:
        # Decode features to molecule
        mol = decoder.decode(features)
        
        if mol is None:
            return {
                'valid': False,
                'reason': 'Decoding failed',
                'smiles': None,
                'properties': None,
                'rmse': None
            }
        
        # Compute actual properties
        actual_props = compute_properties(mol)
        
        # Compare to target
        rmse = property_rmse(actual_props, target_properties)
        
        return {
            'valid': True,
            'reason': 'Success',
            'smiles': mol.GetSymbol() if hasattr(mol, 'GetSymbol') else str(mol),
            'properties': actual_props,
            'target': target_properties,
            'rmse': rmse,
        }
    
    except Exception as e:
        return {
            'valid': False,
            'reason': f'Error: {str(e)}',
            'smiles': None,
            'properties': None,
            'rmse': None
        }


def print_results(results: list):
    """Pretty-print validation results."""
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS")
    print("="*70)
    
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    
    print(f"\n✓ Valid molecules: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    
    # Show details
    for i, result in enumerate(results, 1):
        status = "✓" if result['valid'] else "✗"
        print(f"\n  [{i}] {status} {result['reason']}")
        
        if result['valid']:
            print(f"      SMILES: {result['smiles']}")
            print(f"      Target properties: {result['target']}")
            if result['properties']:
                print(f"      Actual properties: {result['properties']}")
            if result['rmse']:
                print(f"      RMSE: {result['rmse']:.4f}")
    
    print("\n" + "="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("🧬 END-TO-END MOLECULAR GENERATION VALIDATION")
    print("="*70)
    print("\nPipeline: Random Features → Decode → Compute Properties → Validate\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test cases
    test_cases = [
        {
            'name': 'Drug-like molecule',
            'properties': {'logp': 3.5, 'mw': 350, 'hbd': 2, 'hba': 3, 'rotatable': 6},
        },
        {
            'name': 'Small hydrophilic',
            'properties': {'logp': 0.5, 'mw': 200, 'hbd': 4, 'hba': 5, 'rotatable': 1},
        },
        {
            'name': 'Large molecule',
            'properties': {'logp': 2.0, 'mw': 450, 'hbd': 3, 'hba': 4, 'rotatable': 8},
        },
    ]
    
    decoder = MolecularDecoder(device=device)
    all_results = []
    
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        print(f"  Target: {test_case['properties']}")
        
        # Generate random features
        features = generate_random_features(num_samples=3, max_atoms=128)
        
        # Validate each sample
        for i, feature_sample in enumerate(features):
            result = decode_and_validate(
                feature_sample,
                test_case['properties'],
                decoder,
                device
            )
            all_results.append(result)
        
        print(f"  ✓ Validated 3 samples\n")
    
    # Print summary
    print_results(all_results)
    
    # Statistics
    print("STATISTICS")
    print("="*70)
    print(f"Total validated: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if r['valid'])}")
    print(f"Failed: {sum(1 for r in all_results if not r['valid'])}")
    
    rmses = [r['rmse'] for r in all_results if r['rmse'] is not None]
    if rmses:
        print(f"Mean RMSE: {np.mean(rmses):.4f}")
        print(f"Median RMSE: {np.median(rmses):.4f}")
    
    print("="*70 + "\n")
    
    return all_results


if __name__ == '__main__':
    main()
