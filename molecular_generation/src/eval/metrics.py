"""
Evaluation metrics for molecular diffusion models.
Includes validity, diversity, property fidelity, and distribution metrics.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def chemical_validity(molecules, return_details=False) -> Dict:
    """
    Check what fraction of molecules are valid.
    
    Args:
        molecules: List of molecule objects (RDKit Mol) or SMILES strings
        return_details: If True, return per-molecule validation flags
    
    Returns:
        Dict with 'validity', 'valid_count', 'total_count', and optional 'details'
    """
    valid_count = 0
    details = []
    
    for i, mol in enumerate(molecules):
        # Convert SMILES to Mol if needed
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is not None:
            valid_count += 1
            flag = 'valid'
        else:
            flag = 'invalid_smiles'
        
        if return_details:
            details.append({'index': i, 'flag': flag})
    
    validity_rate = valid_count / len(molecules) if len(molecules) > 0 else 0.0
    
    return {
        'validity': float(validity_rate),
        'valid_count': valid_count,
        'total_count': len(molecules),
        'details': details if return_details else None
    }


def diversity_metric(molecules: np.ndarray, metric: str = 'cosine') -> float:
    """
    Compute pairwise diversity between molecules in feature space.
    Higher = more diverse.
    
    Args:
        molecules: Feature array [n_samples, n_features] or list of features
        metric: 'cosine', 'euclidean', or 'jaccard'
    
    Returns:
        Mean pairwise distance (diversity score)
    """
    from scipy.spatial.distance import pdist
    
    if isinstance(molecules, list):
        molecules = np.array(molecules)
    
    if molecules.shape[0] < 2:
        return 0.0
    
    # Compute pairwise distances
    distances = pdist(molecules, metric=metric)
    
    return float(np.mean(distances))


def property_fidelity(generated_molecules: List[str], 
                     target_properties: Dict) -> Dict:
    """
    Compute MSE between target and actual properties.
    Lower = better fidelity.
    
    Args:
        generated_molecules: List of SMILES strings or RDKit Mol objects
        target_properties: Dict like {'logp': 3.5, 'mw': 400, ...}
    
    Returns:
        Dict with 'overall_mse', 'per_property' MSE, and 'valid_molecules' count
    """
    actual_properties = []
    valid_count = 0
    
    for smiles in generated_molecules:
        # Convert to Mol if needed
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = smiles
        
        if mol is None:
            continue
        
        valid_count += 1
        
        try:
            props = {
                'logp': float(Crippen.MolLogP(mol)),
                'mw': float(Descriptors.MolWt(mol)),
                'hbd': float(Lipinski.NumHDonors(mol)),
                'hba': float(Lipinski.NumHAcceptors(mol)),
                'rotatable': float(Descriptors.NumRotatableBonds(mol)),
            }
            actual_properties.append(props)
        except Exception as e:
            print(f"Error computing properties: {e}")
            continue
    
    if len(actual_properties) == 0:
        return {
            'overall_mse': float('inf'),
            'per_property': {},
            'valid_molecules': 0
        }
    
    # Compute MSE per property
    errors_per_property = {}
    total_error = 0
    property_count = 0
    
    for key in target_properties:
        if key not in actual_properties[0]:
            continue
        
        actual_values = np.array([p[key] for p in actual_properties])
        target_value = target_properties[key]
        
        mse = float(np.mean((actual_values - target_value) ** 2))
        errors_per_property[key] = mse
        total_error += mse
        property_count += 1
    
    overall_mse = total_error / property_count if property_count > 0 else float('inf')
    
    return {
        'overall_mse': overall_mse,
        'per_property': errors_per_property,
        'valid_molecules': valid_count
    }


def distribution_distance(generated_features: np.ndarray,
                        training_features: np.ndarray,
                        metric: str = 'mmd') -> float:
    """
    Compute distance between generated and training distributions.
    Lower = model stays close to training data.
    Higher = model generalizes/creates novel examples.
    
    Args:
        generated_features: [num_generated, feature_dim]
        training_features: [num_train, feature_dim]
        metric: 'mmd' or 'sliced_wasserstein'
    
    Returns:
        Distance metric (float)
    """
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.stats import ks_2samp
    
    if isinstance(generated_features, torch.Tensor):
        generated_features = generated_features.cpu().numpy()
    if isinstance(training_features, torch.Tensor):
        training_features = training_features.cpu().numpy()
    
    if metric == 'mmd':
        sigma = 1.0
        
        # RBF kernel for MMD computation
        kxx = rbf_kernel(generated_features, generated_features, sigma=sigma).mean()
        kyy = rbf_kernel(training_features, training_features, sigma=sigma).mean()
        kxy = rbf_kernel(generated_features, training_features, sigma=sigma).mean()
        
        mmd = max(0.0, kxx + kyy - 2 * kxy)
        return float(np.sqrt(mmd))
    
    elif metric == 'sliced_wasserstein':
        # Simplified: 1D projections of Wasserstein distance
        distances = []
        
        for dim in range(min(generated_features.shape[1], training_features.shape[1])):
            stat, _ = ks_2samp(generated_features[:, dim], training_features[:, dim])
            distances.append(float(stat))
        
        return float(np.mean(distances))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def novel_statistics(generated_features: np.ndarray,
                    training_features: np.ndarray,
                    novelty_threshold: float = 0.5) -> Dict:
    """
    Compute fraction of novel (out-of-distribution) samples.
    
    Args:
        generated_features: [num_generated, feature_dim]
        training_features: [num_train, feature_dim]
        novelty_threshold: Distance threshold for considering sample novel
    
    Returns:
        Dict with novelty fraction and statistics
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(training_features) == 0:
        return {'novelty': 0.0, 'mean_distance': 0.0, 'novel_count': 0}
    
    # Find nearest neighbor in training set for each generated sample
    knn = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn.fit(training_features)
    
    distances, _ = knn.kneighbors(generated_features)
    distances = distances.flatten()
    
    # Count samples beyond novelty threshold
    novel_count = np.sum(distances > novelty_threshold)
    novelty_fraction = novel_count / len(distances)
    
    return {
        'novelty': float(novelty_fraction),
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'novel_count': int(novel_count),
        'total_count': len(distances)
    }


def compute_all_metrics(generated_smiles: List[str],
                       generated_features: np.ndarray,
                       training_features: np.ndarray,
                       target_properties: Dict,
                       property_normalizer=None) -> Dict:
    """
    Compute all evaluation metrics at once.
    
    Args:
        generated_smiles: List of generated SMILES strings
        generated_features: Generated molecule features
        training_features: Training set features for distribution comparison
        target_properties: Target properties for fidelity evaluation
        property_normalizer: PropertyNormalizer for denormalization (optional)
    
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # Validity
    validity_result = chemical_validity(generated_smiles)
    metrics['validity'] = validity_result['validity']
    metrics['valid_count'] = validity_result['valid_count']
    
    # Diversity
    metrics['diversity'] = diversity_metric(generated_features)
    
    # Property fidelity
    fidelity_result = property_fidelity(generated_smiles, target_properties)
    metrics['fidelity_mse'] = fidelity_result['overall_mse']
    metrics['fidelity_per_property'] = fidelity_result['per_property']
    
    # Distribution distance
    metrics['mmd_distance'] = distribution_distance(
        generated_features,
        training_features,
        metric='mmd'
    )
    
    # Novelty
    novelty_result = novel_statistics(generated_features, training_features)
    metrics['novelty'] = novelty_result['novelty']
    metrics['mean_nn_distance'] = novelty_result['mean_distance']
    
    return metrics


def print_metrics(metrics: Dict, epoch: Optional[int] = None) -> None:
    """
    Pretty-print evaluation metrics.
    
    Args:
        metrics: Dict from compute_all_metrics()
        epoch: Optional epoch number for context
    """
    prefix = f"Epoch {epoch}" if epoch is not None else "Metrics"
    
    print(f"\n{prefix}:")
    print(f"  Validity: {metrics['validity']:.1%} ({metrics['valid_count']} valid)")
    print(f"  Diversity: {metrics['diversity']:.3f}")
    print(f"  Property Fidelity MSE: {metrics['fidelity_mse']:.4f}")
    print(f"  MMD Distance: {metrics['mmd_distance']:.4f}")
    print(f"  Novelty: {metrics['novelty']:.1%}")
    print(f"  Mean NN Distance: {metrics['mean_nn_distance']:.4f}")

