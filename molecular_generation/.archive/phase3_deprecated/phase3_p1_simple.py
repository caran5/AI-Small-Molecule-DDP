#!/usr/bin/env python3
"""Phase 3 Phase 1: Simple version"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json
import sqlite3
import os
import tarfile
import tempfile

print("="*80)
print("PHASE 3 PHASE 1: RDKit Descriptors Test")
print("="*80)

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
    cursor.execute("SELECT canonical_smiles FROM compound_structures LIMIT 500")
    all_smiles = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"\n✅ Loaded {len(all_smiles)} SMILES")
    
    X_orig_list = []
    X_enh_list = []
    y_list = []
    valid = 0
    
    print("Processing molecules...")
    for idx, smiles in enumerate(all_smiles):
        try:
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if not mol or mol.GetNumAtoms() == 0:
                continue
            
            # Original 9
            orig = np.array([
                float(mol.GetNumAtoms()),
                float(mol.GetNumHeavyAtoms()),
                float(len(Chem.GetSSSR(mol))),
                float(Descriptors.NumAromaticRings(mol)),
                float(Descriptors.NumHeteroatoms(mol)),
                float(Descriptors.NumHDonors(mol)),
                float(Descriptors.NumHAcceptors(mol)),
                float(Descriptors.NumRotatableBonds(mol)),
                float(Descriptors.TPSA(mol)) if Descriptors.TPSA(mol) else 0.0,
            ])
            
            # 13 RDKit descriptors
            rdkit = np.array([
                Descriptors.MolWt(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol) if Descriptors.Chi0(mol) else 0.0,
                Descriptors.HallKierAlpha(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol) if len(mol.GetAtoms()) >= 3 else 0.0,
                Crippen.MolLogP(mol),
                Descriptors.Asphericity(mol),
                Descriptors.Eccentricity(mol),
            ])
            
            enh = np.concatenate([orig, rdkit])
            logp = Descriptors.MolLogP(mol)
            
            X_orig_list.append(orig)
            X_enh_list.append(enh)
            y_list.append(logp)
            valid += 1
            
            if valid % 100 == 0:
                print(f"  ... {valid}")
        except:
            pass
    
    print(f"✅ Got {valid} valid molecules")
    
    X_orig = np.array(X_orig_list)
    X_enh = np.array(X_enh_list)
    y = np.array(y_list)
    
    print(f"   Original: {X_orig.shape}")
    print(f"   Enhanced: {X_enh.shape}")
    
    # Split
    n = len(X_orig)
    train_idx = np.arange(int(0.7*n))
    test_idx = np.arange(int(0.85*n), n)
    
    X_orig_train, y_train = X_orig[train_idx], y[train_idx]
    X_orig_test, y_test = X_orig[test_idx], y[test_idx]
    X_enh_train = X_enh[train_idx]
    X_enh_test = X_enh[test_idx]
    
    # Normalize
    for X_train, X_test in [(X_orig_train, X_orig_test), (X_enh_train, X_enh_test)]:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std==0] = 1.0
        X_train[:] = (X_train - mean) / std
        X_test[:] = (X_test - mean) / std
    
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std
    
    # Train
    print(f"\n✅ Training...")
    lr_orig = LinearRegression()
    lr_orig.fit(X_orig_train, y_train_n)
    pred_orig = lr_orig.predict(X_orig_test)
    success_orig = (np.abs(pred_orig - y_test_n) < 0.2).mean()
    
    lr_enh = LinearRegression()
    lr_enh.fit(X_enh_train, y_train_n)
    pred_enh = lr_enh.predict(X_enh_test)
    success_enh = (np.abs(pred_enh - y_test_n) < 0.2).mean()
    
    print(f"\nResults:")
    print(f"  Original (9D): {success_orig*100:.1f}%")
    print(f"  Enhanced ({X_enh.shape[1]}D): {success_enh*100:.1f}%")
    print(f"  Improvement: {(success_enh-success_orig)*100:+.1f}%")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'molecules': len(X_orig),
        'original_dims': X_orig.shape[1],
        'enhanced_dims': X_enh.shape[1],
        'original_success': float(success_orig),
        'enhanced_success': float(success_enh),
        'improvement': float(success_enh - success_orig),
    }
    
    with open('phase3_phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved")
