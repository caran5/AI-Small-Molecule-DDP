#!/usr/bin/env python3
"""
PHASE 3 PHASE 1: RDKit Descriptors (Using Working Loader)
Reuses phase2_fix_noncircular.py's proven loader pattern
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json

print("="*80)
print("PHASE 3 PHASE 1: RDKit Descriptors Enhancement (Final)")
print("Using proven loader from phase2_fix_noncircular.py")
print("="*80)

# Use the EXACT loader that phase2_fix_noncircular.py uses
from data.loader import DataLoader as MolDataLoader

loader = MolDataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)

print(f"\n✅ Loader returned: {type(molecules)}")

# Extract features
X_orig_list = []
X_enh_list = []
y_list = []

print("Extracting features from molecules...")
valid_count = 0

for mol_data in molecules:
    try:
        smiles = mol_data.get('smiles')  # ✓ CORRECT KEY
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol or mol.GetNumAtoms() == 0:
            continue
        
        # ORIGINAL 9 features
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
        
        # NEW RDKit DESCRIPTORS (~15 features, validated to exist)
        feat_rdkit = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(Descriptors.BertzCT(mol)),
            float(Descriptors.Chi0(mol)) if Descriptors.Chi0(mol) else 0.0,
            float(Descriptors.HallKierAlpha(mol)),
            float(Descriptors.Kappa1(mol)),
            float(Descriptors.Kappa2(mol)),
            float(Descriptors.Kappa3(mol)) if len(mol.GetAtoms()) >= 3 else 0.0,
            float(Crippen.MolLogP(mol)),
            float(Descriptors.LabuteASA(mol)),
            float(Descriptors.NumSaturatedRings(mol)),
            float(Descriptors.NumAliphaticRings(mol)),
            float(Descriptors.NumAromaticHeterocycles(mol)),
            float(Descriptors.TPSA(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
        ]
        
        feat_combined = feat_orig + feat_rdkit
        logp = float(Descriptors.MolLogP(mol))
        
        X_orig_list.append(feat_orig)
        X_enh_list.append(feat_combined)
        y_list.append(logp)
        valid_count += 1
        
        if valid_count % 100 == 0:
            print(f"  ... {valid_count}")
    
    except Exception as e:
        pass

print(f"\n✅ Extracted {valid_count} molecules")

if valid_count < 50:
    print("❌ Too few molecules!")
    sys.exit(1)

X_orig = np.array(X_orig_list)
X_enh = np.array(X_enh_list)
y = np.array(y_list)

print(f"   Original: {X_orig.shape[0]} molecules × {X_orig.shape[1]}D")
print(f"   Enhanced: {X_enh.shape[0]} molecules × {X_enh.shape[1]}D")
print(f"   LogP range: {y.min():.2f} to {y.max():.2f}")

# Split
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

# TEST 2: Enhanced features
print(f"\nTEST 2: Enhanced Features ({X_enh.shape[1]}D)")
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
improvement_pct = ((success_enh - success_orig) / max(success_orig, 0.01)) * 100
print(f"\n{'Metric':<20} {'9-feat':<15} {str(X_enh.shape[1])+'D':<15} {'Improvement':<15}")
print("-" * 65)
print(f"{'Success@±20%':<20} {success_orig*100:<14.1f}% {success_enh*100:<14.1f}% {improvement_pct:+.1f}%")
print(f"{'RMSE':<20} {rmse_orig:<14.4f} {rmse_enh:<14.4f}")
print(f"{'MAPE':<20} {mape_orig*100:<14.1f}% {mape_enh*100:<14.1f}%")

# VERDICT
print("\n" + "="*80)
if success_enh >= success_orig + 0.05:
    verdict = "✅ STRONG SUCCESS: RDKit descriptors improved by 5%+"
    decision = "→ Proceed to Phase 3 Phase 2 (feature selection)"
elif success_enh > success_orig:
    verdict = f"✅ SUCCESS: RDKit descriptors improved to {success_enh*100:.1f}%"
    decision = "→ Proceed to Phase 3 Phase 2 (feature selection)"
else:
    verdict = f"⚠️ NO IMPROVEMENT: RDKit descriptors {success_enh*100:.1f}% vs {success_orig*100:.1f}% baseline"
    decision = "→ Investigate: feature quality issue or need different descriptor set"

print(f"\nVERDICT: {verdict}")
print(f"DECISION: {decision}")
print("="*80)

# SAVE
results = {
    'timestamp': datetime.now().isoformat(),
    'phase': 3,
    'subphase': 1,
    'molecules': len(X_orig),
    'test_size': len(test_idx),
    'original_dims': 9,
    'enhanced_dims': X_enh.shape[1],
    'original_success': float(success_orig),
    'enhanced_success': float(success_enh),
    'improvement_pct': float(improvement_pct),
    'improvement_abs': float(success_enh - success_orig),
    'verdict': verdict
}

with open('phase3_phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to phase3_phase1_results.json")
