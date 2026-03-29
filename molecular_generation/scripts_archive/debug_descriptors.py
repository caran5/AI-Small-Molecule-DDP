#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import sqlite3
import os
import tarfile
import tempfile

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
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT canonical_smiles FROM compound_structures LIMIT 3")
    smiles_list = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    for idx, smiles in enumerate(smiles_list):
        print(f"\nMol {idx}: {smiles[:50]}")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                print("  Invalid SMILES")
                continue
            print(f"  Atoms: {mol.GetNumAtoms()}")
            print(f"  MolWt: {Descriptors.MolWt(mol):.2f}")
            print(f"  FractionCSP3: {Descriptors.FractionCSP3(mol):.2f}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
