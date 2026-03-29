#!/usr/bin/env python3
"""Debug: extract feature from single molecule"""
import sys
sys.path.insert(0, 'src')
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import numpy as np
import sqlite3
import tarfile
import tempfile
import os

db_path = 'src/data/chembl_34_sqlite.tar.gz'

with tempfile.TemporaryDirectory() as tmpdir:
    with tarfile.open(db_path, 'r:gz') as tar:
        tar.extractall(tmpdir)
    
    db_file = None
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            if f.endswith('.db'):
                db_file = os.path.join(root, f)
                break
        if db_file:
            break
    
    # Load one molecule
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT canonical_smiles FROM compound_structures LIMIT 1")
    (smiles,) = cursor.fetchone()
    conn.close()
    
    print(f"Testing SMILES: {smiles}")
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol or mol.GetNumAtoms() == 0:
            print(f"❌ Invalid molecule")
        
        print("Extracting features...")
        original = [
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
        print(f"Original 9 features: {original}")
        
        # Try one RDKit descriptor
        print(f"MolWt: {Descriptors.MolWt(mol)}")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
