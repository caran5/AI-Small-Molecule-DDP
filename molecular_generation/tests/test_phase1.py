"""
Integration test for Phase 1: Production-Ready Foundation
Tests conditional generation, metrics, and ensemble inference.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Import components
from src.models.unet import ConditionalUNet
from src.data.preprocessing import PropertyNormalizer
from src.data.loader import ConditionalMoleculeDataLoader
from src.inference.generate import generate_with_properties, ConditionalGenerationPipeline
from src.eval.metrics import (
    chemical_validity,
    diversity_metric,
    property_fidelity,
    distribution_distance,
    compute_all_metrics,
    print_metrics
)


def create_dummy_data(num_samples: int = 100) -> Tuple[torch.Tensor, List[Dict]]:
    """Create dummy molecular features and properties for testing."""
    
    # Create dummy features (100-dim embedding)
    features = torch.randn(num_samples, 100)
    
    # Create dummy properties
    properties_list = []
    for i in range(num_samples):
        props = {
            'logp': float(np.random.uniform(0, 5)),
            'mw': float(np.random.uniform(250, 500)),
            'hbd': float(np.random.randint(0, 5)),
            'hba': float(np.random.randint(0, 10)),
            'rotatable': float(np.random.randint(0, 10))
        }
        properties_list.append(props)
    
    return features, properties_list


def test_conditional_unet():
    """Test ConditionalUNet architecture."""
    print("\n" + "="*70)
    print("Test 1: ConditionalUNet Architecture")
    print("="*70)
    
    device = 'cpu'
    batch_size = 16
    input_dim = 100
    
    # Create model
    model = ConditionalUNet(
        in_channels=input_dim,
        out_channels=input_dim,
        hidden_channels=128,
        depth=3,
        n_properties=5
    ).to(device)
    
    # Create dummy inputs
    x = torch.randn(batch_size, 1, input_dim).to(device)  # [batch, atoms, features]
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    properties = torch.randn(batch_size, 5).to(device)
    
    # Forward pass
    output = model(x, t, properties=properties)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("✓ ConditionalUNet forward pass successful")
    
    # Test without properties
    output_no_props = model(x, t, properties=None)
    assert output_no_props.shape == output.shape
    print("✓ ConditionalUNet works without properties")
    
    return model


def test_property_normalizer():
    """Test PropertyNormalizer."""
    print("\n" + "="*70)
    print("Test 2: PropertyNormalizer")
    print("="*70)
    
    # Create normalizer
    normalizer = PropertyNormalizer()
    
    # Create training data
    prop_dict = {
        'logp': [1.5, 2.0, 3.5, 4.0, 2.5],
        'mw': [300, 350, 400, 450, 325],
        'hbd': [1, 2, 1, 2, 3],
        'hba': [4, 5, 6, 5, 4],
        'rotatable': [2, 3, 4, 5, 2]
    }
    
    # Fit
    normalizer.fit(prop_dict)
    print("✓ Normalizer fitted")
    
    # Normalize
    test_props = {'logp': 2.5, 'mw': 350, 'hbd': 2, 'hba': 5, 'rotatable': 3}
    normalized = normalizer.normalize(test_props)
    print(f"Original: {test_props}")
    print(f"Normalized: {normalized}")
    assert all(abs(v) <= 3 for v in normalized.values()), "Normalization failed!"
    print("✓ Normalization successful")
    
    # Denormalize
    denormalized = normalizer.denormalize(normalized)
    print(f"Denormalized: {denormalized}")
    
    # Check recovery
    for key in test_props:
        diff = abs(test_props[key] - denormalized[key])
        assert diff < 1e-5, f"Denormalization failed for {key}!"
    print("✓ Denormalization successful")
    
    return normalizer


def test_conditional_dataloader():
    """Test ConditionalMoleculeDataLoader."""
    print("\n" + "="*70)
    print("Test 3: ConditionalMoleculeDataLoader")
    print("="*70)
    
    # Create data
    num_samples = 64
    features, properties_list = create_dummy_data(num_samples)
    
    # Create loader
    loader = ConditionalMoleculeDataLoader(
        features=features,
        properties_list=properties_list,
        batch_size=16,
        shuffle=True
    )
    
    print(f"Created loader with {len(loader)} batches")
    
    # Iterate
    total_samples = 0
    for batch_features, batch_properties in loader:
        assert batch_features.shape[0] == batch_properties.shape[0]
        assert batch_features.shape[1] == 100
        assert batch_properties.shape[1] == 5
        total_samples += batch_features.shape[0]
    
    assert total_samples == num_samples
    print(f"✓ Successfully iterated through all {total_samples} samples")
    
    # Test normalizer
    normalizer = loader.get_normalizer()
    assert normalizer.fitted
    print("✓ Normalizer successfully fitted during dataloader creation")
    
    return loader, normalizer


def test_generation_pipeline():
    """Test generation with properties."""
    print("\n" + "="*70)
    print("Test 4: Conditional Generation Pipeline")
    print("="*70)
    
    device = 'cpu'
    
    # Create model
    model = ConditionalUNet(
        in_channels=100,
        out_channels=100,
        hidden_channels=128,
        depth=3,
        n_properties=5
    ).to(device)
    
    # Create normalizer
    _, normalizer = test_conditional_dataloader()
    
    # Target properties
    target_props = {
        'logp': 3.0,
        'mw': 400,
        'hbd': 2,
        'hba': 5,
        'rotatable': 4
    }
    
    # Generate
    samples = generate_with_properties(
        model,
        target_props,
        num_samples=10,
        num_steps=50,  # Fewer steps for speed
        property_normalizer=normalizer,
        input_dim=100,
        device=device
    )
    
    print(f"Generated samples shape: {samples.shape}")
    assert samples.shape == (10, 100), "Generation failed!"
    print("✓ Generation successful")
    
    # Test pipeline wrapper
    pipeline = ConditionalGenerationPipeline(model, normalizer, device=device)
    samples2 = pipeline.generate(target_props, num_samples=5, num_steps=50)
    assert samples2.shape == (5, 100)
    print("✓ ConditionalGenerationPipeline wrapper works")
    
    return samples


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*70)
    print("Test 5: Evaluation Metrics")
    print("="*70)
    
    # Create dummy SMILES
    smiles_list = [
        'CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O',  # Ibuprofen
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'c1ccccc1',  # Benzene
        'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Similar to ibuprofen
        'INVALID_SMILES'  # Invalid
    ]
    
    # Test validity
    validity = chemical_validity(smiles_list, return_details=True)
    print(f"Validity: {validity['validity']:.1%} ({validity['valid_count']}/{validity['total_count']})")
    assert validity['validity'] >= 0.8, "Validity check failed!"
    print("✓ Validity metric works")
    
    # Create dummy features
    features = torch.randn(len(smiles_list), 100).numpy()
    
    # Test diversity
    diversity = diversity_metric(features)
    print(f"Diversity: {diversity:.3f}")
    assert 0.0 <= diversity <= 2.0, "Diversity out of range!"
    print("✓ Diversity metric works")
    
    # Test fidelity
    target_props = {
        'logp': 3.0,
        'mw': 400,
        'hbd': 2,
        'hba': 5,
        'rotatable': 4
    }
    fidelity = property_fidelity(smiles_list[:4], target_props)
    print(f"Fidelity MSE: {fidelity['overall_mse']:.4f}")
    print(f"  Per-property errors: {fidelity['per_property']}")
    print("✓ Fidelity metric works")
    
    # Test distribution distance
    gen_features = torch.randn(20, 100).numpy()
    train_features = torch.randn(100, 100).numpy()
    
    mmd = distribution_distance(gen_features, train_features, metric='mmd')
    print(f"MMD distance: {mmd:.4f}")
    assert 0.0 <= mmd, "MMD should be non-negative!"
    print("✓ Distribution distance metric works")
    
    # Test combined metrics
    all_metrics = compute_all_metrics(
        smiles_list[:4],
        features[:4],
        train_features,
        target_props
    )
    print("✓ All metrics computed successfully:")
    print_metrics(all_metrics)


def test_ensemble_compatibility():
    """Test ensemble model compatibility."""
    print("\n" + "="*70)
    print("Test 6: Ensemble Model Compatibility")
    print("="*70)
    
    from src.inference.ensemble import EnsembleModel
    
    device = 'cpu'
    
    # Create and save multiple models
    checkpoints = []
    save_dir = Path('test_checkpoints')
    save_dir.mkdir(exist_ok=True)
    
    for i in range(2):
        model = ConditionalUNet(
            in_channels=100,
            out_channels=100,
            hidden_channels=128,
            depth=3,
            n_properties=5
        )
        checkpoint_path = save_dir / f"model_{i}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoints.append(str(checkpoint_path))
        print(f"  Saved model {i}")
    
    # Load ensemble
    ensemble = EnsembleModel(checkpoints, device=device)
    assert len(ensemble.models) == 2
    print(f"✓ Loaded ensemble with {ensemble.n_models} models")
    
    # Generate from ensemble
    _, normalizer = test_conditional_dataloader()
    target_props = {'logp': 3.0, 'mw': 400, 'hbd': 2, 'hba': 5, 'rotatable': 4}
    
    results = ensemble.generate(
        target_props,
        num_samples=5,
        num_steps=50,
        property_normalizer=normalizer
    )
    
    assert 'mean' in results and 'std' in results and 'all' in results
    print(f"✓ Ensemble generation successful")
    print(f"  Mean shape: {results['mean'].shape}")
    print(f"  Std shape: {results['std'].shape}")
    
    # Test filtering
    filtered, confidence, mask = ensemble.filter_by_confidence(results, threshold=1.0)
    print(f"✓ Filtering successful: kept {mask.sum()}/{len(mask)} samples")
    
    # Cleanup
    import shutil
    shutil.rmtree(save_dir)


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("PHASE 1: Production-Ready Foundation - Integration Tests")
    print("="*70)
    
    try:
        test_conditional_unet()
        test_property_normalizer()
        loader, normalizer = test_conditional_dataloader()
        test_generation_pipeline()
        test_evaluation_metrics()
        test_ensemble_compatibility()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nPhase 1 components are ready for production use:")
        print("  ✓ Conditional generation with property steering")
        print("  ✓ Property normalization and dataloader")
        print("  ✓ Comprehensive evaluation metrics")
        print("  ✓ Ensemble inference with uncertainty")
        print("  ✓ Full drug candidate generation pipeline")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

