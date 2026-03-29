#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from data.loader import DataLoader as MolDataLoader
from rdkit import Chem

loader = MolDataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=10)

for i, mol_data in enumerate(molecules):
    print(f"\nMolecule {i}:")
    print(f"  Keys: {list(mol_data.keys())}")
    print(f"  SMILES: {mol_data.get('smiles', 'NONE')}")
    
    smiles = mol_data.get('smiles', '')
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            print(f"  ✅ Valid RDKit mol")
        else:
            print(f"  ❌ Invalid SMILES")
    else:
        print(f"  ❌ No SMILES")
