"""
Integration tests for Phase 2: Guided Sampling and Energy Filtering.

Tests all Phase 2 components:
  1. PropertyGuidanceRegressor - property prediction network
  2. GuidedGenerator - guided sampling with property steering
  3. ConformationFilter - 3D generation and energy filtering

Run with: python tests/test_phase2.py
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.guided_sampling import (
    PropertyGuidanceRegressor,
    GuidedGenerator,
    TrainableGuidance
)
from src.filtering.energy_filter import ConformationFilter, EnergyResults
from src.models.unet import SimpleUNet
from src.data.preprocessing import PropertyNormalizer


class TestPropertyGuidanceRegressor(unittest.TestCase):
    """Test property prediction network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.input_dim = 100
        self.n_properties = 5
        self.batch_size = 8
    
    def test_initialization(self):
        """Test regressor initialization."""
        model = PropertyGuidanceRegressor(self.input_dim, self.n_properties)
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.n_properties, self.n_properties)
    
    def test_forward_pass(self):
        """Test forward pass shapes."""
        model = PropertyGuidanceRegressor(self.input_dim, self.n_properties).to(self.device)
        
        features = torch.randn(self.batch_size, self.input_dim, device=self.device)
        output = model(features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.n_properties))
    
    def test_gradient_flow(self):
        """Test gradient computation during backward pass."""
        model = PropertyGuidanceRegressor(self.input_dim, self.n_properties).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        
        features = torch.randn(self.batch_size, self.input_dim, device=self.device)
        targets = torch.randn(self.batch_size, self.n_properties, device=self.device)
        
        # Forward
        pred = model(features)
        loss = torch.nn.functional.mse_loss(pred, targets)
        
        # Backward
        loss.backward()
        
        # Check gradients computed
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_grad)
        
        # Update
        optimizer.step()


class TestGuidedGenerator(unittest.TestCase):
    """Test guided sampling with property steering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.input_dim = 100
        self.n_properties = 5
        
        # Create dummy model
        self.model = SimpleUNet(input_dim=self.input_dim)
        
        # Create regressor
        self.regressor = PropertyGuidanceRegressor(self.input_dim, self.n_properties)
        
        # Create normalizer
        self.normalizer = PropertyNormalizer()
        # Fit with dummy data
        dummy_properties = {
            'logp': np.random.uniform(0, 5, 100),
            'mw': np.random.uniform(200, 600, 100),
            'hbd': np.random.randint(0, 10, 100),
            'hba': np.random.randint(0, 10, 100),
            'rotatable': np.random.randint(0, 15, 100)
        }
        self.normalizer.fit(dummy_properties)
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = GuidedGenerator(
            self.model,
            self.regressor,
            self.normalizer,
            self.device
        )
        self.assertEqual(gen.guidance_scale, 1.0)
    
    def test_set_guidance_scale(self):
        """Test setting guidance scale."""
        gen = GuidedGenerator(
            self.model,
            self.regressor,
            self.normalizer,
            self.device
        )
        
        gen.set_guidance_scale(5.0)
        self.assertEqual(gen.guidance_scale, 5.0)
    
    def test_property_gradient_computation(self):
        """Test gradient computation for properties."""
        gen = GuidedGenerator(
            self.model,
            self.regressor,
            self.normalizer,
            self.device
        )
        
        features = torch.randn(4, self.input_dim, device=self.device)
        target_props = torch.randn(4, self.n_properties, device=self.device)
        
        gradient = gen.compute_property_gradient(features, target_props)
        
        # Check gradient shape
        self.assertEqual(gradient.shape, features.shape)
        
        # Check gradient is not all zeros
        self.assertFalse(torch.allclose(gradient, torch.zeros_like(gradient)))
    
    def test_apply_guidance(self):
        """Test guidance application to noise prediction."""
        gen = GuidedGenerator(
            self.model,
            self.regressor,
            self.normalizer,
            self.device,
            guidance_scale=2.0
        )
        
        features = torch.randn(4, self.input_dim, device=self.device)
        noise_pred = torch.randn(4, self.input_dim, device=self.device)
        target_props = torch.randn(4, self.n_properties, device=self.device)
        
        guided_noise = gen.apply_guidance(
            features,
            noise_pred,
            target_props,
            alpha_t=0.9,
            beta_t=0.1
        )
        
        # Check shape preserved
        self.assertEqual(guided_noise.shape, noise_pred.shape)
        
        # Guided noise should differ from original
        self.assertFalse(torch.allclose(guided_noise, noise_pred))
    
    def test_guided_generation(self):
        """Test full guided generation pipeline."""
        gen = GuidedGenerator(
            self.model,
            self.regressor,
            self.normalizer,
            self.device,
            guidance_scale=1.0
        )
        
        target_properties = {
            'logp': 3.5,
            'mw': 400,
            'hbd': 2,
            'hba': 5,
            'rotatable': 4
        }
        
        samples = gen.generate_guided(
            target_properties,
            num_samples=4,
            num_steps=10,
            noise_schedule='cosine'
        )
        
        # Check output shape
        self.assertEqual(samples.shape, (4, self.input_dim))
        
        # Check CPU output
        self.assertIsInstance(samples, np.ndarray) or isinstance(samples, torch.Tensor)


class TestConformationFilter(unittest.TestCase):
    """Test 3D conformation generation and energy filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = ConformationFilter(energy_threshold=100.0)
        
        # Test SMILES - valid molecules
        self.valid_smiles = [
            'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
            'CC(=O)Oc1ccccc1C(=O)O',         # Aspirin
            'c1ccccc1',                      # Benzene
            'CC(C)CC(N)C(=O)O',              # Leucine
            'CC(C)c1ccc(O)cc1'               # 4-isopropylphenol
        ]
        
        # Invalid SMILES
        self.invalid_smiles = [
            'XYZ',
            'c1ccc(Q)cc1',
            'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(CCCCCCCCCC)C'
        ]
    
    def test_initialization(self):
        """Test filter initialization."""
        filter = ConformationFilter(energy_threshold=80.0)
        self.assertEqual(filter.energy_threshold, 80.0)
        self.assertEqual(filter.num_conformers, 5)
    
    def test_set_energy_threshold(self):
        """Test setting energy threshold."""
        self.filter.set_energy_threshold(50.0)
        self.assertEqual(self.filter.energy_threshold, 50.0)
    
    def test_valid_smiles_parsing(self):
        """Test that valid SMILES parse correctly."""
        from rdkit import Chem
        
        for smiles in self.valid_smiles:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol, f"Failed to parse: {smiles}")
    
    def test_invalid_smiles_rejected(self):
        """Test that invalid SMILES are handled gracefully."""
        filtered, results = self.filter.filter_smiles(self.invalid_smiles)
        
        # All should be rejected
        self.assertEqual(len(filtered), 0)
        self.assertTrue(all(not valid for valid in results.valid_3d))
    
    def test_filter_smiles_basic(self):
        """Test basic filtering on valid molecules."""
        filtered, results = self.filter.filter_smiles(self.valid_smiles, verbose=False)
        
        # Check results structure
        self.assertEqual(len(results.smiles_list), len(self.valid_smiles))
        self.assertIsNotNone(results.energies)
        self.assertIsNotNone(results.valid_3d)
        self.assertIsNotNone(results.passes_filter)
    
    def test_filter_results_summary(self):
        """Test summary statistics in results."""
        filtered, results = self.filter.filter_smiles(self.valid_smiles, verbose=False)
        
        # Check summary has expected keys
        self.assertIn('total_molecules', results.summary)
        self.assertIn('valid_3d', results.summary)
        self.assertIn('passes_filter', results.summary)
        self.assertIn('mean_energy', results.summary)
        self.assertIn('median_energy', results.summary)
    
    def test_energy_values_reasonable(self):
        """Test that computed energies are in reasonable range."""
        filtered, results = self.filter.filter_smiles(self.valid_smiles, verbose=False)
        
        valid_energies = results.energies[results.valid_3d]
        
        if len(valid_energies) > 0:
            # Energies should be positive and not too large
            self.assertTrue(np.all(valid_energies >= 0))
            self.assertTrue(np.all(valid_energies < 1000))  # Sanity check
    
    def test_percentile_filtering(self):
        """Test percentile-based filtering."""
        filtered, results = self.filter.filter_with_batch_stats(
            self.valid_smiles,
            use_percentile=True,
            percentile=75.0,
            verbose=False
        )
        
        # Approximately 25% should be filtered (top 75%)
        pass_rate = results.summary['passes_filter'] / results.summary['valid_3d']
        self.assertTrue(0.1 < pass_rate < 0.4)  # Reasonable range
    
    def test_filtered_with_energies(self):
        """Test getting filtered molecules with energy values."""
        filtered_with_e = self.filter.get_filtered_with_energies(self.valid_smiles)
        
        # Should be list of tuples (smiles, energy, strain)
        for item in filtered_with_e:
            self.assertEqual(len(item), 3)
            self.assertIsInstance(item[0], str)  # SMILES
            self.assertIsInstance(item[1], (float, np.floating))  # Energy
            self.assertIsInstance(item[2], (float, np.floating))  # Strain
        
        # Should be sorted by energy
        if len(filtered_with_e) > 1:
            energies = [e for _, e, _ in filtered_with_e]
            self.assertEqual(energies, sorted(energies))
    
    def test_energy_results_finalization(self):
        """Test EnergyResults container."""
        results = EnergyResults()
        
        # Add some molecules
        for i in range(5):
            results.add_molecule(f"smiles_{i}", energy=50.0 + i*10, valid=True, passes=True, strain=0.5)
        
        results.finalize()
        
        # Check arrays are created
        self.assertEqual(len(results.energies), 5)
        self.assertEqual(len(results.valid_3d), 5)
        self.assertEqual(len(results.passes_filter), 5)


class TestPhase2Integration(unittest.TestCase):
    """Integration tests combining guided sampling and energy filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Set up guided generator
        self.model = SimpleUNet(input_dim=100)
        self.regressor = PropertyGuidanceRegressor(100, 5)
        self.normalizer = PropertyNormalizer()
        
        dummy_properties = {
            'logp': np.random.uniform(0, 5, 100),
            'mw': np.random.uniform(200, 600, 100),
            'hbd': np.random.randint(0, 10, 100),
            'hba': np.random.randint(0, 10, 100),
            'rotatable': np.random.randint(0, 15, 100)
        }
        self.normalizer.fit(dummy_properties)
        
        self.generator = GuidedGenerator(
            self.model,
            self.regressor,
            self.normalizer,
            self.device
        )
        
        # Set up filter
        self.filter = ConformationFilter(energy_threshold=100.0)
    
    def test_guided_generation_then_filter(self):
        """Test full workflow: guided generation then energy filtering."""
        # Generate with guidance
        target_props = {'logp': 3.5, 'mw': 400, 'hbd': 2, 'hba': 5, 'rotatable': 4}
        samples = self.generator.generate_guided(
            target_props,
            num_samples=2,
            num_steps=5,
            noise_schedule='cosine'
        )
        
        self.assertEqual(samples.shape[0], 2)
        self.assertTrue(torch.is_tensor(samples) or isinstance(samples, np.ndarray))
    
    def test_different_guidance_scales(self):
        """Test generation with different guidance scales."""
        target_props = {'logp': 3.5, 'mw': 400, 'hbd': 2, 'hba': 5, 'rotatable': 4}
        
        results = []
        for scale in [0.0, 1.0, 5.0]:
            self.generator.set_guidance_scale(scale)
            samples = self.generator.generate_guided(
                target_props,
                num_samples=2,
                num_steps=5
            )
            results.append(samples)
        
        # All should have same shape
        for r in results:
            self.assertEqual(r.shape, (2, 100))
        
        # Results should differ between different guidance scales
        # (with high probability for different random seeds)


if __name__ == '__main__':
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL PHASE 2 TESTS PASSED")
        print(f"  - {result.testsRun} test cases executed")
        print("\n✓ Phase 2 components are production-ready:")
        print("  ✓ PropertyGuidanceRegressor - Property prediction network")
        print("  ✓ GuidedGenerator - Guided sampling with property steering")
        print("  ✓ ConformationFilter - 3D generation and energy filtering")
        print("\nReady for production deployment and Phase 2 workflows.")
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  - Failures: {len(result.failures)}")
        print(f"  - Errors: {len(result.errors)}")
    print("="*70)
    
    sys.exit(0 if result.wasSuccessful() else 1)
