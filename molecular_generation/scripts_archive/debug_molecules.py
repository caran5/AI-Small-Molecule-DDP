#!/usr/bin/env python3
"""Debug: check molecules"""
import sys
sys.path.insert(0, 'src')
from rdkit import Chem
from rdkit.Chem import Descriptors
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
    
    # Load molecules from database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT canonical_smiles FROM compound_structures LIMIT 10")
    molecules_raw = cursor.fetchall()
    conn.close()
    
    print("Sample molecules:")
    for idx, (smiles_tuple,) in enumerate(molecules_raw):
        if smiles_tuple:
            mol = Chem.MolFromSmiles(smiles_tuple)
            print(f"  {idx}: {smiles_tuple[:50]:50s} -> {'VALID' if mol else 'INVALID'}")
        else:
            print(f"  {idx}: None")
