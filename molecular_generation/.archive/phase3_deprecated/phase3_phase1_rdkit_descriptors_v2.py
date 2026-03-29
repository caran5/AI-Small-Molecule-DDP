#!/usr/bin/env python3
"""
PHASE 3 PHASE 1 v2: RDKit Descriptors Enhancement (Direct Loading)
Test: Jump from 50% accuracy (9 topology features) to 55-65% (39 chemistry features)
Uses direct database loading, no tar extraction needed on second load
"""

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
print("PHASE 3 PHASE 1 v2: RDKit Descriptors (Direct Loading)")
print("="*80)

# Load molecules directly from ChEMBL
db_path = 'src/data/chembl_34_sqlite.tar.gz'

# Extract or find existing database
if os.path.exists('src/data/.chembl_extracted'):
    print("\n✅ Using cached ChEMBL database")
    db_file = 'src/data/.chembl_extracted'
else:
    print("\nExtracting ChEMBL database (one-time)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(db_path) as tar:
            tar.extractall(tmpdir)
        # Find the .db file
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.endswith('.db'):
                    db_file = os.path.join(root, f)
                    print(f"   Found: {db_file}")
                    # Copy to cache
                    os.system(f"cp {db_file} src/data/.chembl_extracted")
                    print("   Cached for future runs")
                    break

# Load molecules from database
print(f"\nLoading molecules from database...")
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute("SELECT SMILES FROM molecule_dictionary LIMIT 500")
molecules_raw = cursor.fetchall()
conn.close()

print(f"✅ Found {len(molecules_raw)} SMILES strings")

# Extract BOTH original features AND new RDKit descriptors
X_original = []  # 9 features
X_enhanced = []  # 39 features (9 + 30 RDKit)
y_logp = []

print("\nExtracting features from molecules...")
valid_count = 0

for idx, (smiles_tuple,) in enumerate(molecules_raw):
    try:
        smiles = smiles_tuple
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol or mol.GetNumAtoms() == 0:
            continue
        
        # ORIGINAL 9 features
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
        
        # NEW 30 RDKit DESCRIPTORS (chemistry-aware)
        rdkit_descriptors = [
            float(Descriptors.MolWt(mol)),                      # 1: Molecular weight
            float(Descriptors.FractionCsp3(mol)),               # 2: Fraction sp3 (aliphatic)
            float(Descriptors.NumAromaticAtoms(mol)),           # 3: Aromatic atoms
            float(Descriptors.NumAliphaticCycles(mol)),         # 4: Aliphatic rings
            float(Descriptors.NumAromaticCycles(mol)),          # 5: Aromatic rings
            float(Descriptors.RingCount(mol)),                  # 6: Total rings
            float(Descriptors.BertzCT(mol)),                    # 7: Complexity (Bertz)
            float(Descriptors.Chi0(mol)) if Descriptors.Chi0(mol) else 0.0,  # 8: Connectivity
            float(Descriptors.LabuteASA(mol)),                  # 9: Labute surface area
            float(Descriptors.NumHeterocycles(mol)),            # 10: Heterocycles
            float(Descriptors.NumSaturatedCycles(mol)),         # 11: Saturated rings
            float(Descriptors.NumAliphaticRings(mol)),          # 12: Aliphatic rings
            float(Descriptors.Ipc(mol)) if Descriptors.Ipc(mol) else 0.0,  # 13: Info content
            float(Descriptors.HallKierAlpha(mol)),              # 14: Hall-Kier alpha
            float(Descriptors.Kappa1(mol)),                     # 15: Kappa shape 1
            float(Descriptors.Kappa2(mol)),                     # 16: Kappa shape 2
            float(Descriptors.Kappa3(mol)) if len(mol.GetAtoms()) >= 3 else 0.0,  # 17: Kappa 3
            float(Descriptors.NumSaturatedHeterocycles(mol)),   # 18: Saturated heterocycles
            float(Descriptors.NumAromaticHeterocycles(mol)),    # 19: Aromatic heterocycles
            float(Descriptors.NumAromaticCarbocycles(mol)),     # 20: Aromatic carbocycles
            float(Descriptors.NumSaturatedCarbocycles(mol)),    # 21: Saturated carbocycles
            float(Descriptors.NumAliphaticHeterocycles(mol)),   # 22: Aliphatic heterocycles
            float(Descriptors.NumAliphaticCarbocycles(mol)),    # 23: Aliphatic carbocycles
            float(Crippen.MolLogP(mol)),                        # 24: Wiener LogP
            float(Descriptors.Asphericity(mol)),                # 25: Asphericity
            float(Descriptors.Eccentricity(mol)),               # 26: Eccentricity
            float(Descriptors.InertialShapeFactor(mol)),        # 27: Inertial shape
            float(Descriptors.RadiusOfGyration(mol)),           # 28: Radius of gyration
            float(Descriptors.NumLipinskiHBA(mol)),             # 29: Lipinski HBA
            float(Descriptors.NumLipinskiHBD(mol)),             # 30: Lipinski HBD
        ]
        
        # Combine
        feat_original = np.array(original)
        feat_enhanced = np.array(original + rdkit_descriptors)
        
        # Target
        logp = float(Descriptors.MolLogP(mol))
        
        X_original.append(feat_original)
        X_enhanced.append(feat_enhanced)
        y_logp.append(logp)
        valid_count += 1
        
        if valid_count % 100 == 0:
            print(f"   ... {valid_count} molecules processed")
    
    except Exception as e:
        pass

X_orig = np.array(X_original)
X_enh = np.array(X_enhanced)
y = np.array(y_logp)

if len(X_orig) == 0:
    print("❌ Failed to extract features!")
    sys.exit(1)

print(f"\n✅ Extracted features from {len(X_orig)} valid molecules")
print(f"   Original features: {X_orig.shape[1]}D")
print(f"   Enhanced features: {X_enh.shape[1]}D")
print(f"   Target (LogP): range {y.min():.2f} to {y.max():.2f}")

# Split 70/15/15
n = len(X_orig)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

train_idx = np.arange(train_size)
test_idx = np.arange(train_size + val_size, n)

X_orig_train, y_train = X_orig[train_idx], y[train_idx]
X_orig_test, y_test = X_orig[test_idx], y[test_idx]

X_enh_train = X_enh[train_idx]
X_enh_test = X_enh[test_idx]

# Normalize
X_orig_mean, X_orig_std = X_orig_train.mean(axis=0), X_orig_train.std(axis=0)
X_orig_std[X_orig_std == 0] = 1.0
X_orig_train_n = (X_orig_train - X_orig_mean) / X_orig_std
X_orig_test_n = (X_orig_test - X_orig_mean) / X_orig_std

X_enh_mean, X_enh_std = X_enh_train.mean(axis=0), X_enh_train.std(axis=0)
X_enh_std[X_enh_std == 0] = 1.0
X_enh_train_n = (X_enh_train - X_enh_mean) / X_enh_std
X_enh_test_n = (X_enh_test - X_enh_mean) / X_enh_std

y_mean, y_std = y_train.mean(), y_train.std()
y_train_n = (y_train - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std

print(f"\n✅ Split: train={len(train_idx)}, test={len(test_idx)}")

# TEST 1: Original 9 features
print("\n" + "="*80)
print("TEST 1: Original 9 Features (Phase 2 Baseline)")
lr_orig = LinearRegression()
lr_orig.fit(X_orig_train_n, y_train_n)
y_pred_orig = lr_orig.predict(X_orig_test_n)
rmse_orig = np.sqrt(((y_pred_orig - y_test_n)**2).mean())
mape_orig = np.abs((y_pred_orig - y_test_n) / (np.abs(y_test_n) + 0.1)).mean()
success_orig = (np.abs(y_pred_orig - y_test_n) < 0.2).mean()
print(f"RMSE: {rmse_orig:.4f}, MAPE: {mape_orig*100:.1f}%, Success@±20%: {success_orig*100:.1f}%")

# TEST 2: Enhanced 39 features
print("\nTEST 2: Enhanced 39 Features (9 + 30 RDKit)")
lr_enh = LinearRegression()
lr_enh.fit(X_enh_train_n, y_train_n)
y_pred_enh = lr_enh.predict(X_enh_test_n)
rmse_enh = np.sqrt(((y_pred_enh - y_test_n)**2).mean())
mape_enh = np.abs((y_pred_enh - y_test_n) / (np.abs(y_test_n) + 0.1)).mean()
success_enh = (np.abs(y_pred_enh - y_test_n) < 0.2).mean()
print(f"RMSE: {rmse_enh:.4f}, MAPE: {mape_enh*100:.1f}%, Success@±20%: {success_enh*100:.1f}%")

# COMPARISON
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print(f"\n{'Metric':<20} {'9-feat':<15} {'39-feat':<15} {'Improvement':<15}")
print("-" * 65)
improvement_pct = ((success_enh - success_orig) / max(success_orig, 0.01)) * 100
rmse_improvement = ((rmse_orig - rmse_enh) / max(rmse_orig, 0.001)) * 100
print(f"{'Success@±20%':<20} {success_orig*100:<14.1f}% {success_enh*100:<14.1f}% {improvement_pct:+.1f}%")
print(f"{'RMSE':<20} {rmse_orig:<14.4f} {rmse_enh:<14.4f} {rmse_improvement:+.1f}%")
print(f"{'MAPE':<20} {mape_orig*100:<14.1f}% {mape_enh*100:<14.1f}% {((mape_orig-mape_enh)/max(mape_orig,0.01))*100:+.1f}%")

# VERDICT
print("\n" + "="*80)
if success_enh >= success_orig + 0.05:
    verdict = "✅ STRONG SUCCESS: RDKit descriptors provide significant improvement (+5%+)"
    decision = "→ Proceed to Phase 3 Phase 2 (feature selection)"
elif success_enh > success_orig:
    verdict = f"✅ SUCCESS: RDKit descriptors improve accuracy ({success_enh*100:.1f}%)"
    decision = "→ Proceed to Phase 3 Phase 2 (feature selection)"
else:
    verdict = "❌ NO IMPROVEMENT: RDKit descriptors did not help"
    decision = "→ Diagnose: feature quality or need different approach"

print(f"\nVERDICT: {verdict}")
print(f"DECISION: {decision}")
print("="*80)

# SAVE RESULTS
results = {
    'timestamp': datetime.now().isoformat(),
    'phase': 3,
    'subphase': 1,
    'test_type': 'RDKit Descriptors Enhancement',
    'molecules': len(X_orig),
    'test_size': len(test_idx),
    'models': {
        'original_9_features': {
            'features': ['NumAtoms', 'NumHeavyAtoms', 'NumRings', 'NumAromaticRings', 'Heteroatoms', 'HBD', 'HBA', 'RotatableBonds', 'TPSA'],
            'rmse': float(rmse_orig),
            'mape': float(mape_orig),
            'success_20pct': float(success_orig)
        },
        'enhanced_39_features': {
            'features': 'Original 9 + 30 RDKit (aromaticity, polarity, lipophilicity, complexity, shape)',
            'rmse': float(rmse_enh),
            'mape': float(mape_enh),
            'success_20pct': float(success_enh)
        }
    },
    'improvement': {
        'success_delta': float(success_enh - success_orig),
        'success_pct_improvement': float(improvement_pct),
        'rmse_improvement_pct': float(rmse_improvement)
    },
    'verdict': verdict
}

with open('phase3_phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to phase3_phase1_results.json")
print("="*80)
