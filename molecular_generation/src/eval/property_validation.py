"""
End-to-end property validation: Features → Molecules → Properties → Comparison
Validates that generated molecules match target properties.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

from src.inference.decoder import MolecularDecoder


def compute_properties(mol) -> Optional[Dict]:
    """
    Compute drug-like properties for a molecule.
    
    Args:
        mol: RDKit Mol object or SMILES string
    
    Returns:
        Dict with computed properties or None if invalid
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    
    if mol is None:
        return None
    
    try:
        return {
            'logp': float(Crippen.MolLogP(mol)),
            'mw': float(Descriptors.MolWt(mol)),
            'hbd': float(Lipinski.NumHDonors(mol)),
            'hba': float(Lipinski.NumHAcceptors(mol)),
            'rotatable': float(Descriptors.NumRotatableBonds(mol)),
            'tpsa': float(Descriptors.TPSA(mol)),
        }
    except Exception as e:
        return None


def property_rmse(actual: Dict, target: Dict) -> Dict:
    """
    Compute RMSE for each property and overall.
    
    Args:
        actual: Dict of computed properties
        target: Dict of target properties
    
    Returns:
        Dict with per-property errors and overall RMSE
    """
    if actual is None:
        return {'overall': float('inf'), 'valid': False}
    
    errors = {}
    squared_errors = []
    
    for key in target.keys():
        if key not in actual:
            errors[key] = float('inf')
            squared_errors.append(float('inf'))
        else:
            error = (actual[key] - target[key]) ** 2
            errors[key] = np.sqrt(error)
            squared_errors.append(error)
    
    # Overall RMSE (only over common properties)
    if len(squared_errors) > 0 and not all(np.isinf(e) for e in squared_errors):
        overall_mse = np.mean([e for e in squared_errors if not np.isinf(e)])
        overall_rmse = np.sqrt(overall_mse)
    else:
        overall_rmse = float('inf')
    
    return {
        'per_property': errors,
        'overall': overall_rmse,
        'valid': overall_rmse != float('inf')
    }


def validate_generated_molecule(
    features: torch.Tensor,
    target_properties: Dict,
    tolerance: float = 0.4,
    return_mol: bool = False
) -> Dict:
    """
    End-to-end validation pipeline: features → molecule → properties → comparison
    
    Args:
        features: Generated features, shape (n_atoms, 5)
        target_properties: Target property values
        tolerance: Tolerance for bond distance threshold
        return_mol: If True, return RDKit Mol object
    
    Returns:
        Dict with validation results including actual properties and RMSE
    """
    result = {
        'target': target_properties,
        'actual': None,
        'rmse': {},
        'mol': None,
        'smiles': None,
        'valid': False,
        'error': None
    }
    
    try:
        # Step 1: Decode features to atoms and coordinates
        atomic_nums, coords = MolecularDecoder.features_to_atoms(features)
        
        if len(atomic_nums) == 0:
            result['error'] = 'No atoms extracted'
            return result
        
        # Step 2: Infer bonds from coordinates
        bonds = MolecularDecoder.infer_bonds_from_coords(atomic_nums, coords, tolerance)
        
        # Step 3: Build RDKit molecule
        mol, smiles = MolecularDecoder.build_rdkit_mol(atomic_nums, coords, bonds)
        
        if mol is None:
            result['error'] = 'Failed to build valid molecule'
            result['smiles'] = smiles
            return result
        
        result['smiles'] = smiles
        result['mol'] = mol if return_mol else None
        
        # Step 4: Compute actual properties
        actual_props = compute_properties(mol)
        
        if actual_props is None:
            result['error'] = 'Failed to compute properties'
            return result
        
        result['actual'] = actual_props
        
        # Step 5: Compare to targets
        comparison = property_rmse(actual_props, target_properties)
        result['rmse'] = comparison
        result['valid'] = comparison['valid']
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def validate_batch(
    features_batch: torch.Tensor,
    target_properties: Dict,
    tolerance: float = 0.4
) -> List[Dict]:
    """
    Validate a batch of molecules.
    
    Args:
        features_batch: Batch of features, shape (batch_size, n_atoms, 5)
        target_properties: Target properties (same for all in batch)
        tolerance: Bond distance tolerance
    
    Returns:
        List of validation results
    """
    results = []
    for i in range(features_batch.shape[0]):
        result = validate_generated_molecule(
            features_batch[i],
            target_properties,
            tolerance
        )
        results.append(result)
    return results


def print_validation_result(result: Dict, index: int = 0, verbose: bool = True):
    """
    Pretty-print validation result.
    
    Args:
        result: Result dict from validate_generated_molecule
        index: Molecule index for labeling
        verbose: If True, print detailed info
    """
    if not verbose:
        if result['valid']:
            print(f"  [Mol {index}] ✓ VALID (RMSE: {result['rmse']['overall']:.3f})")
        else:
            print(f"  [Mol {index}] ✗ INVALID ({result['error']})")
        return
    
    print(f"\n{'='*70}")
    print(f"Molecule {index}")
    print(f"{'='*70}")
    
    if result['error']:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"✓ SMILES: {result['smiles']}")
    
    print(f"\n{'Property':<15} {'Target':<12} {'Actual':<12} {'Error':<10} {'Status':<8}")
    print(f"{'-'*70}")
    
    if result['actual']:
        for prop_name in result['target'].keys():
            target_val = result['target'][prop_name]
            actual_val = result['actual'].get(prop_name, None)
            error_val = result['rmse']['per_property'].get(prop_name, float('inf'))
            
            if actual_val is not None and not np.isinf(error_val):
                status = "✓" if error_val < 0.5 else "⚠️"
                print(f"{prop_name:<15} {target_val:<12.2f} {actual_val:<12.2f} {error_val:<10.3f} {status:<8}")
            else:
                print(f"{prop_name:<15} {target_val:<12.2f} {'N/A':<12} {'N/A':<10} ❌")
    
    overall_rmse = result['rmse']['overall']
    if overall_rmse != float('inf'):
        overall_status = "✓ PASS" if overall_rmse < 0.5 else "⚠️ WARN"
        print(f"\n{'Overall RMSE':<15} {overall_rmse:<12.3f} {overall_status}")
    else:
        print(f"\n{'Overall RMSE':<15} {'N/A':<12} {'❌ FAIL'}")


def print_batch_summary(results: List[Dict]):
    """Print summary statistics for batch of molecules."""
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    
    rmse_values = [r['rmse']['overall'] for r in results if r['valid']]
    
    print(f"\n{'='*70}")
    print(f"BATCH SUMMARY ({total_count} molecules)")
    print(f"{'='*70}")
    print(f"Valid molecules: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    
    if len(rmse_values) > 0:
        print(f"Property RMSE:   {np.mean(rmse_values):.3f} ± {np.std(rmse_values):.3f}")
        print(f"  Min: {np.min(rmse_values):.3f}")
        print(f"  Max: {np.max(rmse_values):.3f}")
    else:
        print(f"Property RMSE:   N/A (no valid molecules)")
    
    # Per-property statistics
    all_props = {}
    for r in results:
        if r['valid'] and r['rmse']['per_property']:
            for prop_name, error in r['rmse']['per_property'].items():
                if prop_name not in all_props:
                    all_props[prop_name] = []
                if not np.isinf(error):
                    all_props[prop_name].append(error)
    
    if len(all_props) > 0:
        print(f"\nPer-property errors:")
        for prop_name in sorted(all_props.keys()):
            errors = all_props[prop_name]
            print(f"  {prop_name:<15} {np.mean(errors):.3f} ± {np.std(errors):.3f}")
