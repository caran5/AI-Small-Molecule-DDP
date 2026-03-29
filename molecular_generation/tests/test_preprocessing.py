"""
Unit tests and validation for preprocessing module.
"""

import numpy as np
import torch
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import MolecularPreprocessor, DataAugmentation


class TestMolecularPreprocessor(unittest.TestCase):
    """Test cases for MolecularPreprocessor."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.preprocessor = MolecularPreprocessor(normalize=True, max_atoms=128)
        
        # Create sample molecule (e.g., water: O + 2H)
        self.atoms = np.array([8, 1, 1])  # O, H, H
        self.positions = np.array([
            [0.0, 0.0, 0.0],   # O at origin
            [0.96, 0.0, 0.0],  # H at 0.96 Å
            [-0.24, 0.93, 0.0] # H at angle
        ])
    
    def test_normalize_features(self):
        """Test feature normalization."""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized = self.preprocessor.normalize_features(features)
        
        # Check mean ≈ 0 and std ≈ 1
        assert np.abs(np.mean(normalized, axis=0)).max() < 1e-6, "Mean not zero"
        assert np.abs(np.std(normalized, axis=0) - 1.0).max() < 1e-6, "Std not one"
        
        print("✓ Normalization test passed")
    
    def test_denormalize_features(self):
        """Test denormalization reverses normalization."""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized = self.preprocessor.normalize_features(features)
        denormalized = self.preprocessor.denormalize_features(normalized)
        
        # Check we recover original features
        assert np.allclose(denormalized, features), "Denormalization failed"
        
        print("✓ Denormalization test passed")
    
    def test_pad_molecule(self):
        """Test molecule padding."""
        atoms, features = self.preprocessor.pad_molecule(self.atoms, np.zeros((3, 5)))
        
        # Check shape
        assert len(atoms) == self.preprocessor.max_atoms, "Atoms not padded to max_atoms"
        assert features.shape == (self.preprocessor.max_atoms, 5), "Features shape mismatch"
        
        # Check padding with zeros
        assert np.all(atoms[3:] == 0), "Padding not zero"
        assert np.all(features[3:] == 0), "Feature padding not zero"
        
        print("✓ Padding test passed")
    
    def test_pad_molecule_truncate(self):
        """Test molecule truncation when too large."""
        large_atoms = np.ones(200)
        large_features = np.ones((200, 5))
        
        atoms, features = self.preprocessor.pad_molecule(large_atoms, large_features)
        
        assert len(atoms) == self.preprocessor.max_atoms, "Truncation failed"
        assert features.shape[0] == self.preprocessor.max_atoms, "Features truncation failed"
        
        print("✓ Truncation test passed")
    
    def test_create_adjacency_matrix(self):
        """Test adjacency matrix creation."""
        adj = self.preprocessor.create_adjacency_matrix(self.positions, distance_threshold=1.8)
        
        # Check shape
        assert adj.shape == (3, 3), "Adjacency matrix shape mismatch"
        
        # Check properties
        assert np.all(adj >= 0) and np.all(adj <= 1), "Adjacency values out of range"
        assert np.all(np.diag(adj) == 0), "Self-loops not removed"
        assert np.allclose(adj, adj.T), "Adjacency not symmetric"
        
        print("✓ Adjacency matrix test passed")
    
    def test_extract_features(self):
        """Test feature extraction."""
        features = self.preprocessor.extract_features(self.atoms, self.positions)
        
        # Check shape
        assert features.shape == (3, 5), "Feature shape mismatch"
        
        # Check feature ranges (should be roughly normalized)
        assert np.all(features[:, 0] >= 0) and np.all(features[:, 0] <= 1), "Atomic number not normalized"
        assert np.all(np.abs(features[:, 1:4]) <= 2), "Positions out of reasonable range"
        
        print("✓ Feature extraction test passed")
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        result = self.preprocessor.preprocess(self.atoms, self.positions)
        
        # Check all required keys
        required_keys = ['atoms', 'features', 'adjacency', 'n_atoms']
        assert all(k in result for k in required_keys), "Missing keys in preprocessing output"
        
        # Check tensors
        assert isinstance(result['atoms'], torch.Tensor), "Atoms not a tensor"
        assert isinstance(result['features'], torch.Tensor), "Features not a tensor"
        assert isinstance(result['adjacency'], torch.Tensor), "Adjacency not a tensor"
        
        # Check shapes
        assert result['atoms'].shape[0] == self.preprocessor.max_atoms, "Atoms tensor shape mismatch"
        assert result['features'].shape == (self.preprocessor.max_atoms, 5), "Features tensor shape mismatch"
        
        # Check n_atoms
        assert result['n_atoms'] == 3, "n_atoms incorrect"
        
        print("✓ Full pipeline test passed")
    
    def test_preprocessing_consistency(self):
        """Test that preprocessing is deterministic."""
        result1 = self.preprocessor.preprocess(self.atoms, self.positions)
        result2 = self.preprocessor.preprocess(self.atoms, self.positions)
        
        assert torch.allclose(result1['features'], result2['features']), "Features not deterministic"
        
        print("✓ Consistency test passed")


class TestDataAugmentation(unittest.TestCase):
    """Test cases for DataAugmentation."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
    
    def test_random_rotation(self):
        """Test that rotation preserves distances."""
        rotated = DataAugmentation.random_rotation(self.positions)
        
        # Compute pairwise distances
        distances_orig = np.linalg.norm(self.positions[:, np.newaxis] - self.positions[np.newaxis, :], axis=2)
        distances_rot = np.linalg.norm(rotated[:, np.newaxis] - rotated[np.newaxis, :], axis=2)
        
        assert np.allclose(distances_orig, distances_rot, atol=1e-6), "Rotation changed distances"
        
        print("✓ Rotation test passed")
    
    def test_random_noise(self):
        """Test noise addition."""
        noisy = DataAugmentation.random_noise(self.positions, noise_level=0.1)
        
        # Check that something changed
        assert not np.allclose(noisy, self.positions), "Noise not added"
        
        # Check rough magnitude
        diff = np.abs(noisy - self.positions)
        assert np.mean(diff) < 0.2, "Noise magnitude too large"
        
        print("✓ Noise test passed")
    
    def test_random_scale(self):
        """Test scaling."""
        scaled = DataAugmentation.random_scale(self.positions, scale_range=(0.9, 1.1))
        
        # Compute distances
        distances_orig = np.linalg.norm(self.positions[:, np.newaxis] - self.positions[np.newaxis, :], axis=2)
        distances_scaled = np.linalg.norm(scaled[:, np.newaxis] - scaled[np.newaxis, :], axis=2)
        
        # Distances should scale by same factor
        scale_factor = distances_scaled[0, 1] / distances_orig[0, 1]
        assert 0.85 < scale_factor < 1.15, "Scale factor out of range"
        
        print("✓ Scaling test passed")


class TestDataValidation(unittest.TestCase):
    """Validation tests for data quality."""
    
    def setUp(self):
        self.preprocessor = MolecularPreprocessor(max_atoms=128)
    
    def test_empty_molecule(self):
        """Test handling of edge case: empty molecule."""
        atoms = np.array([])
        positions = np.array([]).reshape(0, 3)
        
        result = self.preprocessor.preprocess(atoms, positions)
        
        assert result['n_atoms'] == 0, "Empty molecule not handled"
        assert result['atoms'].shape[0] == self.preprocessor.max_atoms, "Padding not applied"
        
        print("✓ Empty molecule test passed")
    
    def test_single_atom_molecule(self):
        """Test handling of single atom."""
        atoms = np.array([6])  # Carbon
        positions = np.array([[0.0, 0.0, 0.0]])
        
        result = self.preprocessor.preprocess(atoms, positions)
        
        assert result['n_atoms'] == 1, "Single atom not handled"
        assert result['atoms'][0] > 0, "Atom not preserved"
        
        print("✓ Single atom test passed")
    
    def test_large_molecule(self):
        """Test handling of very large molecule."""
        n_atoms = 500
        atoms = np.random.randint(1, 9, n_atoms)  # C, N, O, etc.
        positions = np.random.randn(n_atoms, 3)
        
        result = self.preprocessor.preprocess(atoms, positions)
        
        assert result['n_atoms'] == n_atoms, "Large molecule not handled"
        assert result['atoms'].shape[0] == self.preprocessor.max_atoms, "Not truncated to max"
        
        print("✓ Large molecule test passed")


if __name__ == '__main__':
    print("Running Preprocessing Validation Tests...\n")
    
    # Run test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print(f"{'='*60}")
