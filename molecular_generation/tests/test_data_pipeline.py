"""
Integration test showing the complete data loading pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import DataLoader, create_dummy_data
import pickle


def test_data_pipeline():
    """Test the complete data loading and preprocessing pipeline."""
    
    print("="*60)
    print("Data Loading Pipeline Integration Test")
    print("="*60)
    
    # Step 1: Create dummy data
    print("\n1. Creating dummy molecular data...")
    molecules = create_dummy_data(n_samples=100)
    print(f"   ✓ Generated {len(molecules)} molecules")
    print(f"   Sample molecule: {molecules[0]}")
    
    # Step 2: Save data to file
    print("\n2. Saving data to disk...")
    data_path = '/tmp/test_molecules.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(molecules, f)
    print(f"   ✓ Saved to {data_path}")
    
    # Step 3: Initialize data loader
    print("\n3. Initializing DataLoader...")
    loader = DataLoader(
        data_path=data_path,
        batch_size=16,
        max_atoms=128,
        normalize=True,
        augment=True,
        augment_prob=0.5
    )
    print("   ✓ DataLoader created")
    
    # Step 4: Setup (load and split data)
    print("\n4. Loading and splitting data...")
    loader.setup(train_split=0.7, val_split=0.15, test_split=0.15)
    
    # Step 5: Get loaders and iterate
    print("\n5. Iterating through batches...")
    train_loader = loader.get_train_loader()
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:
            print(f"\n   Batch {batch_idx + 1}:")
            print(f"   - Keys in batch: {batch.keys()}")
            print(f"   - Atoms shape: {batch['atoms'].shape}")
            print(f"   - Features shape: {batch['features'].shape}")
            print(f"   - Adjacency shape: {batch['adjacency'].shape}")
            print(f"   - N_atoms: {batch['n_atoms']}")
            
            # Print sample values
            print(f"\n   Sample features (first 3 atoms, 5 features):")
            print(f"   {batch['features'][0, :3, :].numpy()}")
        
        if batch_idx >= 2:  # Show first 3 batches
            break
    
    print(f"\n   ✓ Successfully iterated through {batch_idx + 1} batches")
    
    # Step 6: Test validation and test loaders
    print("\n6. Testing validation and test loaders...")
    val_loader = loader.get_val_loader()
    test_loader = loader.get_test_loader()
    
    val_batches = len(val_loader)
    test_batches = len(test_loader)
    
    print(f"   - Validation batches: {val_batches}")
    print(f"   - Test batches: {test_batches}")
    print(f"   ✓ All loaders working")
    
    # Step 7: Save preprocessor
    print("\n7. Saving preprocessor state...")
    preprocessor_path = '/tmp/preprocessor.pkl'
    loader.save_preprocessor(preprocessor_path)
    print(f"   ✓ Preprocessor saved")
    
    # Step 8: Load preprocessor for inference
    print("\n8. Loading preprocessor for inference...")
    new_loader = DataLoader(batch_size=16)
    new_loader.load_preprocessor(preprocessor_path)
    print(f"   ✓ Preprocessor loaded")
    print(f"   - Feature normalization stats: {new_loader.preprocessor.feature_stats}")
    
    print("\n" + "="*60)
    print("✓ All integration tests passed!")
    print("="*60)


if __name__ == '__main__':
    test_data_pipeline()
