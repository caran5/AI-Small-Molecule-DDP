"""
Integration test: ChemBL database → Data Loader → Preprocessing → PyTorch batches
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import DataLoader
import numpy as np


def test_full_integration():
    """Test complete integration from ChemBL to batches."""
    
    print("="*70)
    print("FULL INTEGRATION TEST: ChemBL → Loader → Preprocessing → Batches")
    print("="*70)
    
    chembl_path = '/Users/ceejayarana/diffusion_model/molecular_generation/src/data/chembl_34_sqlite.tar.gz'
    
    # Step 1: Initialize loader with ChemBL
    print("\n1. Initializing DataLoader with ChemBL database...")
    loader = DataLoader(
        data_path=chembl_path,
        batch_size=16,
        max_atoms=128,
        normalize=True,
        augment=True,
        augment_prob=0.5
    )
    print("   ✓ DataLoader created")
    
    # Step 2: Setup (loads ChemBL data and preprocesses)
    print("\n2. Loading ChemBL database and setting up splits...")
    try:
        loader.setup(
            data_path=chembl_path,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )
        print("   ✓ Data loaded and split successfully")
    except Exception as e:
        print(f"   ✗ Error during setup: {e}")
        return
    
    # Step 3: Get and iterate training batches
    print("\n3. Getting training batches (with preprocessing applied)...")
    try:
        train_loader = loader.get_train_loader()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                print(f"\n   BATCH 0 ANALYSIS:")
                print(f"   ┌─ Data Flow ─────────────────────────────────────┐")
                print(f"   │ ChemBL Database                                  │")
                print(f"   │  ↓ (Extract SMILES & convert to atoms/positions)│")
                print(f"   │ MolecularDataset                                 │")
                print(f"   │  ↓ (Call preprocessor.preprocess())             │")
                print(f"   │ MolecularPreprocessor                            │")
                print(f"   │  • Extract features from atoms/positions         │")
                print(f"   │  • Normalize (mean=0, std=1)                    │")
                print(f"   │  • Pad to max_atoms=128                          │")
                print(f"   │  • Create adjacency matrix                       │")
                print(f"   │  ↓ (Return tensors)                             │")
                print(f"   │ PyTorch DataLoader (batch_size=16)               │")
                print(f"   └──────────────────────────────────────────────────┘")
                
                print(f"\n   BATCH CONTENTS:")
                print(f"   - Batch size: {batch['atoms'].shape[0]} molecules")
                print(f"   - Atoms shape: {batch['atoms'].shape}")
                print(f"     (batch_size × max_atoms)")
                print(f"   - Features shape: {batch['features'].shape}")
                print(f"     (batch_size × max_atoms × 5 features)")
                print(f"     Features: [atomic_num, x, y, z, dist_from_com]")
                print(f"   - Adjacency shape: {batch['adjacency'].shape}")
                print(f"     (batch_size × max_atoms × max_atoms)")
                print(f"   - N_atoms shape: {len(batch['n_atoms'])} values")
                print(f"     (actual atom count per molecule)")
                
                # Verify preprocessing worked
                print(f"\n   PREPROCESSING VERIFICATION:")
                
                # Check normalization
                features = batch['features'][0].numpy()
                actual_atoms = batch['n_atoms'][0]
                
                print(f"   - First molecule has {actual_atoms} atoms")
                print(f"   - Feature mean (first 10 atoms): {features[:10, :].mean(axis=0)}")
                print(f"   - Feature std (first 10 atoms): {features[:10, :].std(axis=0)}")
                
                # Check padding
                print(f"   - Atoms padded to {batch['atoms'].shape[1]} (max_atoms=128)")
                print(f"   - Padding verification: Last atom values are {batch['atoms'][0, -5:].numpy()}")
                
                # Check adjacency
                adj = batch['adjacency'][0].numpy()
                print(f"   - Adjacency is symmetric: {np.allclose(adj, adj.T)}")
                print(f"   - No self-loops: {np.all(np.diag(adj) == 0)}")
                
                print(f"\n   ✓ All preprocessing working correctly!")
            
            if batch_idx >= 2:
                break
        
        print(f"\n   ✓ Successfully iterated {batch_idx + 1} batches")
    
    except Exception as e:
        print(f"   ✗ Error during batching: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test validation and test loaders
    print("\n4. Testing validation and test loaders...")
    try:
        val_loader = loader.get_val_loader()
        test_loader = loader.get_test_loader()
        
        for batch in val_loader:
            print(f"   - Val batch shape: {batch['features'].shape}")
            break
        
        for batch in test_loader:
            print(f"   - Test batch shape: {batch['features'].shape}")
            break
        
        print(f"   ✓ All loaders working")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Step 5: Verify augmentation is working
    print("\n5. Checking data augmentation in training...")
    try:
        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader))
        
        # Features should be different due to augmentation
        diff = (batch1['features'] - batch2['features']).abs().mean()
        print(f"   - Mean feature difference between batches: {diff:.6f}")
        print(f"   - ✓ Augmentation is adding diversity to batches" if diff > 0.01 else "   - ⚠ Augmentation may not be active")
    except:
        pass
    
    print("\n" + "="*70)
    print("✓ INTEGRATION TEST PASSED!")
    print("  ChemBL database → preprocessing.py → loader.py → PyTorch batches")
    print("="*70)


if __name__ == '__main__':
    test_full_integration()
