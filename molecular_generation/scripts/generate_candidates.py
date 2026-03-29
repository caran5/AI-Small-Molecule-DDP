"""
End-to-end pipeline for drug candidate generation.
Combines guided sampling, ensemble, filtering, and property validation.

Supports three generation modes:
  1. Standard ensemble generation (Phase 1)
  2. Guided sampling with property steering (Phase 2)
  3. Ensemble + guided generation (Phase 2)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

from src.inference.ensemble import EnsembleModel
from src.inference.guided_sampling import GuidedGenerator, PropertyGuidanceRegressor, TrainableGuidance
from src.inference.decoder import MolecularDecoder
from src.filtering.energy_filter import ConformationFilter
from src.eval.metrics import (
    chemical_validity,
    property_fidelity,
    print_metrics,
    compute_all_metrics
)
from src.data.preprocessing import PropertyNormalizer


def compute_druglike_properties(mol) -> Dict:
    """Compute drug-like properties for a molecule."""
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
            'violations': int(Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol) > 12)
        }
    except:
        return None


def decode_to_smiles(features: torch.Tensor,
                     decoder=None) -> List[str]:
    """
    Decode feature tensors to SMILES strings.

    Converts generated 128×5 feature tensors back to chemical structures
    by inferring bonds from atomic coordinates and building RDKit molecules.

    Args:
        features: Tensor of shape (n_samples, 128, 5)
        decoder: Ignored (uses MolecularDecoder directly)

    Returns:
        List of SMILES strings (may include None for failed sanitization)
    """
    smiles_list = []

    for feat in features:
        # Decode feature tensor to molecule dict
        mol_dict = MolecularDecoder.features_to_molecule_dict(feat)

        # Extract SMILES if molecule is valid
        if mol_dict['valid'] and mol_dict['smiles'] is not None:
            smiles_list.append(mol_dict['smiles'])
        else:
            # Sanitization failed - record as None
            smiles_list.append(None)

    return smiles_list


def generate_drug_candidates(
    ensemble: EnsembleModel,
    target_properties: Dict,
    num_candidates: int = 100,
    confidence_threshold: float = 1.0,
    property_normalizer=None,
    decoder=None
) -> Dict:
    """
    Full pipeline: ensemble → uncertainty filter → property check → SMILES
    
    Args:
        ensemble: EnsembleModel instance
        target_properties: Target properties dict
        num_candidates: Number of candidates to generate
        confidence_threshold: Max ensemble std for filtering
        property_normalizer: Property normalizer
        decoder: Molecule decoder (optional)
    
    Returns:
        Dict with SMILES, properties, confidence, and fidelity
    """
    
    print("\n" + "="*70)
    print("Drug Candidate Generation Pipeline")
    print("="*70)
    
    # Step 1: Generate from ensemble
    print(f"\n[1/5] Generating {num_candidates} samples from ensemble...")
    results = ensemble.generate(
        target_properties,
        num_samples=num_candidates,
        num_steps=100,
        property_normalizer=property_normalizer
    )
    
    # Step 2: Filter by confidence
    print(f"\n[2/5] Filtering by ensemble uncertainty (threshold={confidence_threshold})...")
    samples, confidence, mask = ensemble.filter_by_confidence(
        results,
        threshold=confidence_threshold
    )
    
    kept_count = mask.sum().item()
    removed_count = (~mask).sum().item()
    print(f"  Kept {kept_count} samples, removed {removed_count}")
    
    # Step 3: Decode to SMILES
    print(f"\n[3/5] Decoding {len(samples)} samples to SMILES...")
    molecules = decode_to_smiles(samples, decoder)
    
    # Filter invalid SMILES
    valid_smiles = []
    valid_mols = []
    valid_confidence = []
    
    for smiles, conf in zip(molecules, confidence):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            valid_mols.append(mol)
            valid_confidence.append(conf)
    
    print(f"  Valid SMILES: {len(valid_smiles)}/{len(molecules)}")
    
    if len(valid_smiles) == 0:
        print("✗ No valid molecules generated!")
        return {
            'smiles': [],
            'properties': [],
            'confidence': [],
            'fidelity': {'overall_mse': float('inf')}
        }
    
    # Step 4: Compute chemical properties
    print(f"\n[4/5] Computing chemical properties for {len(valid_smiles)} molecules...")
    properties_list = []
    for mol in valid_mols:
        props = compute_druglike_properties(mol)
        if props is not None:
            properties_list.append(props)
    
    print(f"  Successfully computed properties for {len(properties_list)} molecules")
    
    # Step 5: Compute fidelity
    print(f"\n[5/5] Computing property fidelity...")
    fidelity = property_fidelity(valid_smiles, target_properties)
    print(f"  Fidelity MSE: {fidelity['overall_mse']:.4f}")
    
    results_dict = {
        'smiles': valid_smiles,
        'molecules': valid_mols,
        'properties': properties_list,
        'confidence': valid_confidence,
        'fidelity': fidelity,
        'target_properties': target_properties
    }
    
    return results_dict


def rank_candidates(candidates: Dict,
                   sort_by: str = 'fidelity') -> List[Dict]:
    """
    Rank drug candidates.
    
    Args:
        candidates: Output from generate_drug_candidates()
        sort_by: 'fidelity', 'confidence', or 'property_fit'
    
    Returns:
        Ranked list of candidate dicts
    """
    ranked = []
    
    for i, (smiles, props, conf) in enumerate(
        zip(candidates['smiles'], candidates['properties'], candidates['confidence'])
    ):
        # Compute fidelity for this candidate
        fidelity_error = 0.0
        for key, target_val in candidates['target_properties'].items():
            if key in props:
                fidelity_error += (props[key] - target_val) ** 2
        
        ranked.append({
            'rank': i + 1,
            'smiles': smiles,
            'properties': props,
            'ensemble_confidence': float(conf.max().item()),
            'fidelity_error': float(fidelity_error),
            'score': float(1.0 / (1.0 + fidelity_error))
        })
    
    # Sort by score (higher is better)
    ranked.sort(key=lambda x: x['score'], reverse=True)
    
    # Update ranks
    for i, cand in enumerate(ranked):
        cand['rank'] = i + 1
    
    return ranked


def print_candidates(ranked_candidates: List[Dict], top_n: int = 10) -> None:
    """Pretty-print top drug candidates."""
    print("\n" + "="*70)
    print(f"Top {min(top_n, len(ranked_candidates))} Drug Candidates")
    print("="*70 + "\n")
    
    for cand in ranked_candidates[:top_n]:
        print(f"Rank {cand['rank']}: Score={cand['score']:.3f}")
        print(f"  SMILES: {cand['smiles']}")
        print(f"  Properties:")
        props = cand['properties']
        print(f"    LogP: {props['logp']:.2f}")
        print(f"    MW: {props['mw']:.1f}")
        print(f"    HBD: {props['hbd']}")
        print(f"    HBA: {props['hba']}")
        print(f"    Rotatable bonds: {props['rotatable']}")
        print(f"  Ensemble confidence: {cand['ensemble_confidence']:.3f}")
        print(f"  Fidelity error: {cand['fidelity_error']:.4f}")
        print()


def main_pipeline(
    ensemble_checkpoint_paths: List[str],
    target_properties: Dict,
    property_normalizer,
    num_candidates: int = 200,
    confidence_threshold: float = 0.8,
    device: str = 'cpu'
) -> Dict:
    """
    Complete drug discovery pipeline.
    
    Args:
        ensemble_checkpoint_paths: Paths to ensemble models
        target_properties: Target properties for generation
        property_normalizer: Fitted property normalizer
        num_candidates: Number of candidates to generate
        confidence_threshold: Confidence threshold for filtering
        device: Device to use
    
    Returns:
        Dict with results and rankings
    """
    
    # Load ensemble
    print("Loading ensemble...")
    ensemble = EnsembleModel(ensemble_checkpoint_paths, device=device)
    
    # Generate candidates
    candidates = generate_drug_candidates(
        ensemble,
        target_properties,
        num_candidates=num_candidates,
        confidence_threshold=confidence_threshold,
        property_normalizer=property_normalizer
    )
    
    # Rank candidates
    ranked = rank_candidates(candidates, sort_by='fidelity')
    
    # Print results
    print_candidates(ranked, top_n=10)
    
    return {
        'candidates': candidates,
        'ranked': ranked
    }


def generate_guided_candidates(
    model,
    property_regressor: PropertyGuidanceRegressor,
    normalizer: PropertyNormalizer,
    target_properties: Dict[str, float],
    num_samples: int = 100,
    guidance_scale: float = 1.0,
    num_steps: int = 50,
    confidence_threshold: float = 0.7,
    decoder=None,
    device='cuda'
) -> Dict:
    """
    Generate drug candidates using guided sampling (Phase 2).
    
    Uses gradient-based guidance during diffusion to steer generation
    toward target property values. More efficient than ensemble for
    single-property optimization.
    
    Args:
        model: ConditionalUNet diffusion model
        property_regressor: PropertyGuidanceRegressor for guidance signals
        normalizer: PropertyNormalizer for property scaling
        target_properties: Target property values
        num_samples: Number of molecules to generate
        guidance_scale: Guidance strength (0=no guidance, 5-10=strong)
        num_steps: Diffusion steps (more=better quality, slower)
        confidence_threshold: Min property uncertainty (not for guided)
        decoder: SMILES decoder
        device: torch device
        
    Returns:
        Dict with generated SMILES, properties, and fidelity metrics
    """
    print(f"\n{'='*70}")
    print("PHASE 2: GUIDED SAMPLING (Property Steering)")
    print(f"{'='*70}")
    print(f"\nTarget Properties: {target_properties}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Diffusion Steps: {num_steps}")
    
    device_obj = torch.device(device)
    
    # Initialize guided generator
    print(f"\n[1/4] Initializing guided generator...")
    generator = GuidedGenerator(
        model,
        property_regressor,
        normalizer,
        device_obj,
        guidance_scale=guidance_scale
    )
    
    # Generate with guidance
    print(f"\n[2/4] Generating {num_samples} samples with guidance...")
    samples = generator.generate_guided(
        target_properties,
        num_samples=num_samples,
        num_steps=num_steps,
        noise_schedule='cosine'
    )
    
    # Decode to SMILES
    print(f"\n[3/4] Decoding {len(samples)} samples to SMILES...")
    molecules = decode_to_smiles(samples, decoder)
    
    # Filter valid SMILES
    valid_smiles = []
    valid_mols = []
    
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            valid_mols.append(mol)
    
    print(f"  Valid SMILES: {len(valid_smiles)}/{len(molecules)}")
    
    if len(valid_smiles) == 0:
        print("✗ No valid molecules generated!")
        return {
            'smiles': [],
            'properties': [],
            'fidelity': {'overall_mse': float('inf')}
        }
    
    # Compute properties
    print(f"\n[4/4] Computing properties...")
    properties_list = []
    for mol in valid_mols:
        props = compute_druglike_properties(mol)
        if props is not None:
            properties_list.append(props)
    
    # Compute fidelity
    fidelity = property_fidelity(valid_smiles, target_properties)
    print(f"  Fidelity MSE: {fidelity['overall_mse']:.4f}")
    
    return {
        'smiles': valid_smiles,
        'properties': properties_list,
        'fidelity': fidelity,
        'target': target_properties
    }


def generate_with_energy_filtering(
    ensemble_or_generator,
    target_properties: Dict[str, float],
    energy_threshold: float = 100.0,
    use_guided: bool = False,
    num_samples: int = 100,
    decoder=None,
    verbose: bool = True,
    device='cuda'
) -> Dict:
    """
    Generate candidates with energy-based filtering (Phase 2).
    
    Combines generation (ensemble or guided) with 3D conformation
    validation using MMFF94 force field. Removes strained/implausible
    molecules before returning final candidates.
    
    Energy Filtering:
        - Generates 3D coordinates via distance geometry
        - Optimizes with MMFF94 force field
        - Removes molecules with high energy (strain indicator)
        - Typical filtering removes 20-40% of generated molecules
    
    Args:
        ensemble_or_generator: EnsembleModel or GuidedGenerator
        target_properties: Target properties
        energy_threshold: Max energy for molecules (kcal/mol)
        use_guided: Use guided sampling or ensemble
        num_samples: Number to generate before filtering
        decoder: SMILES decoder
        verbose: Print progress
        device: torch device
        
    Returns:
        Dict with filtered SMILES, properties, energies, and fidelity
    """
    print(f"\n{'='*70}")
    print("PHASE 2: ENERGY-BASED FILTERING")
    print(f"{'='*70}")
    print(f"\nEnergy Threshold: {energy_threshold} kcal/mol")
    print(f"Generation Mode: {'Guided' if use_guided else 'Ensemble'}")
    
    # Step 1: Generate candidates
    print(f"\n[1/4] Generating {num_samples} candidates...")
    if use_guided:
        gen_result = generate_guided_candidates(
            ensemble_or_generator.model,
            ensemble_or_generator.property_regressor,
            ensemble_or_generator.normalizer,
            target_properties,
            num_samples=num_samples,
            decoder=decoder,
            device=device
        )
        generated_smiles = gen_result['smiles']
    else:
        gen_result = generate_drug_candidates(
            ensemble_or_generator,
            target_properties,
            num_samples=num_samples,
            decoder=decoder
        )
        generated_smiles = gen_result['smiles']
    
    print(f"  Generated: {len(generated_smiles)} valid molecules")
    
    if len(generated_smiles) == 0:
        print("✗ No molecules generated!")
        return {
            'original': generated_smiles,
            'filtered': [],
            'filter_results': None
        }
    
    # Step 2: Energy filtering
    print(f"\n[2/4] Running energy-based filtering...")
    filter_obj = ConformationFilter(energy_threshold=energy_threshold)
    filtered_smiles, filter_results = filter_obj.filter_smiles(
        generated_smiles,
        verbose=verbose
    )
    
    print(f"  {len(filtered_smiles)}/{len(generated_smiles)} molecules pass energy filter")
    print(f"  Filter statistics:")
    for key, value in filter_results.summary.items():
        print(f"    {key}: {value}")
    
    # Step 3: Get filtered with energies
    print(f"\n[3/4] Ranking filtered candidates by energy...")
    filtered_with_e = filter_obj.get_filtered_with_energies(generated_smiles)
    
    # Step 4: Compute final properties
    print(f"\n[4/4] Computing final properties...")
    properties_list = []
    for smiles, energy, strain in filtered_with_e:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            props = compute_druglike_properties(mol)
            if props is not None:
                props['mmff94_energy'] = energy
                props['strain_indicator'] = strain
                properties_list.append(props)
    
    print(f"  Final candidates: {len(properties_list)}")
    
    # Compute fidelity on filtered set
    filtered_smiles_final = [s for s, _, _ in filtered_with_e]
    fidelity = property_fidelity(filtered_smiles_final, target_properties)
    
    return {
        'original': generated_smiles,
        'filtered': filtered_smiles_final,
        'properties': properties_list,
        'fidelity': fidelity,
        'filter_results': filter_results,
        'target': target_properties
    }


if __name__ == '__main__':
    print("""
    Drug Candidate Generation Pipeline
    
    Usage:
        1. Load ensemble:
           ensemble = EnsembleModel(checkpoint_paths)
        
        2. Generate candidates:
           results = generate_drug_candidates(
               ensemble,
               target_properties={'logp': 3.5, 'mw': 400, ...}
           )
        
        3. Rank candidates:
           ranked = rank_candidates(results)
        
        4. Print results:
           print_candidates(ranked, top_n=10)
    """)

