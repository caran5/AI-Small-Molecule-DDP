#!/usr/bin/env python3
"""
PHASE 3 PHASE 1: Debug where molecules are lost
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

print("="*80)
print("PHASE 3 PHASE 1: Debug Molecule Extraction")
print("="*80)

# Use the EXACT loader that phase2_fix_noncircular.py uses
from data.loader import DataLoader as MolDataLoader

loader = MolDataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)

print(f"\n✅ Loader returned: {type(molecules)}, length: {len(molecules)}")

if len(molecules) == 0:
    print("ERROR: molecules list is empty!")
    sys.exit(1)

# Check first few molecules
print("\nFirst 5 molecules:")
for idx in range(min(5, len(molecules))):
    mol_data = molecules[idx]
    print(f"  Mol {idx}: type={type(mol_data)}")
    print(f"    Keys: {mol_data.keys() if isinstance(mol_data, dict) else 'NOT A DICT'}")
    if isinstance(mol_data, dict):
        smiles = mol_data.get('smiles')
        print(f"    SMILES: {smiles}")
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            print(f"    Valid mol: {mol is not None}, Atoms: {mol.GetNumAtoms() if mol else 'N/A'}")

print("\n" + "="*80)
print("Testing feature extraction on first valid molecule:")
print("="*80)

for mol_data in molecules:
    try:
        smiles = mol_data.get('smiles')
        print(f"\nMol SMILES: {smiles}")
        if not smiles:
            print("  → Skipped: No SMILES")
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        print(f"  → RDKit parse: {mol is not None}")
        if not mol or mol.GetNumAtoms() == 0:
            print("  → Skipped: Invalid mol or 0 atoms")
            continue
        
        print(f"  → Valid! Atoms: {mol.GetNumAtoms()}")
        
        # Try extracting original features
        print("  Extracting original 9 features:")
        feat_orig = [
            float(mol.GetNumAtoms()),
            float(mol.GetNumHeavyAtoms()),
            float(len(Chem.GetSSSR(mol))),
            float(Descriptors.NumAromaticRings(mol)),
            float(Descriptors.NumHeteroatoms(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.TPSA(mol)) if Descriptors.TPSA(mol) else 0.0,
        ]
        print(f"    ✓ Original features: {feat_orig}")
        
        # Try extracting RDKit features
        print("  Extracting RDKit descriptors:")
        feat_rdkit = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(Descriptors.RingCount(mol)),
            float(Descriptors.NumAromaticRings(mol)),
            float(Descriptors.BertzCT(mol)),
            float(Descriptors.Chi0(mol)) if Descriptors.Chi0(mol) else 0.0,
            float(Descriptors.HallKierAlpha(mol)),
            float(Descriptors.Kappa1(mol)),
            float(Descriptors.Kappa2(mol)),
            float(Descriptors.Kappa3(mol)) if len(mol.GetAtoms()) >= 3 else 0.0,
            float(Crippen.MolLogP(mol)),
            float(Descriptors.Asphericity(mol)),
            float(Descriptors.Eccentricity(mol)),
        ]
        print(f"    ✓ RDKit descriptors: {feat_rdkit}")
        
        logp = float(Descriptors.MolLogP(mol))
        print(f"    ✓ LogP: {logp}")
        
        print("\n✅ SUCCESS - First molecule extracted!")
        break
        
    except Exception as e:
        print(f"  ✗ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

EOF
