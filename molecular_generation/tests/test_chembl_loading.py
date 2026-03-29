"""
Test loading molecules from ChemBL SQLite database.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import DataLoader


def test_chembl_loading():
    """Test loading molecules from ChemBL tar.gz archive."""
    
    print("="*60)
    print("ChemBL Database Loading Test")
    print("="*60)
    
    chembl_path = '/Users/ceejayarana/diffusion_model/molecular_generation/src/data/chembl_34_sqlite.tar.gz'
    
    if not os.path.exists(chembl_path):
        print(f"✗ ChemBL database not found at {chembl_path}")
        return
    
    # Initialize loader
    print("\n1. Initializing DataLoader...")
    loader = DataLoader(batch_size=32, max_atoms=128)
    print("   ✓ DataLoader created")
    
    # Load ChemBL data
    print("\n2. Loading molecules from ChemBL database...")
    try:
        molecules = loader.load_data(chembl_path)
        print(f"   ✓ Loaded {len(molecules)} molecules")
        
        # Show sample molecule
        if molecules:
            sample = molecules[0]
            print(f"\n   Sample molecule:")
            print(f"   - Mol ID: {sample.get('mol_id', 'N/A')}")
            print(f"   - SMILES: {sample.get('smiles', 'N/A')}")
            print(f"   - Atoms: {sample['atoms']}")
            print(f"   - N atoms: {len(sample['atoms'])}")
            print(f"   - Positions shape: {np.array(sample['positions']).shape}")
    
    except Exception as e:
        print(f"   ✗ Error loading ChemBL database:")
        print(f"   {str(e)}")
        return
    
    # Setup splits
    print("\n3. Setting up train/val/test splits...")
    try:
        loader.setup(data_path=chembl_path, 
                    train_split=0.7, 
                    val_split=0.15, 
                    test_split=0.15)
        
        # Get first batch
        print("\n4. Iterating through training batches...")
        train_loader = loader.get_train_loader()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                print(f"\n   First batch:")
                print(f"   - Batch size: {batch['atoms'].shape[0]}")
                print(f"   - Atoms shape: {batch['atoms'].shape}")
                print(f"   - Features shape: {batch['features'].shape}")
                print(f"   - Adjacency shape: {batch['adjacency'].shape}")
                print(f"\n   ✓ Data successfully preprocessed and batched")
            
            if batch_idx >= 2:
                break
        
        print(f"\n   ✓ Processed {batch_idx + 1} batches successfully")
    
    except Exception as e:
        print(f"   ✗ Error during setup/batching:")
        print(f"   {str(e)}")
        return
    
    print("\n" + "="*60)
    print("✓ ChemBL loading test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    import numpy as np
    test_chembl_loading()
