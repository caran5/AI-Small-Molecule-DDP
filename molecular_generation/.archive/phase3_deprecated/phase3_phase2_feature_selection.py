#!/usr/bin/env python3
"""
PHASE 3 PHASE 2: Feature Selection via Correlation
Takes top 15-20 RDKit descriptors by LogP correlation
Target: 70-75% accuracy with reduced features
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 3 PHASE 2: Feature Selection via Correlation")
print("="*80)

# Reload molecules to get feature names
from data.loader import DataLoader as MolDataLoader

loader = MolDataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)

X_orig_list = []
X_enh_list = []
y_list = []

feature_names_orig = [
    'NumAtoms', 'NumHeavyAtoms', 'NumRings', 'NumAromaticRings',
    'NumHeteroatoms', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'TPSA'
]

feature_names_rdkit = [
    'MolWt', 'FractionCSP3', 'BertzCT', 'Chi0', 'HallKierAlpha',
    'Kappa1', 'Kappa2', 'Kappa3', 'MolLogP', 'LabuteASA',
    'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles',
    'TPSA_2', 'NumRotatableBonds_2'
]

print("Extracting features from 500 molecules...")
valid_count = 0

for mol_data in molecules:
    try:
        smiles = mol_data.get('smiles')
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
        
        # NEW RDKit DESCRIPTORS (15 features)
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
            float(Descriptors.TPSA(mol)) if Descriptors.TPSA(mol) else 0.0,
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

X_orig = np.array(X_orig_list)
X_enh = np.array(X_enh_list)
y = np.array(y_list)

print(f"\n✅ Extracted {valid_count} molecules")
print(f"   Enhanced: {X_enh.shape[0]} molecules × {X_enh.shape[1]}D")

# ============================================================================
# FEATURE IMPORTANCE: Calculate correlation with LogP
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

all_feature_names = feature_names_orig + feature_names_rdkit
correlations = []

for idx, feat_name in enumerate(all_feature_names):
    corr = np.abs(np.corrcoef(X_enh[:, idx], y)[0, 1])
    correlations.append((feat_name, corr))
    print(f"{feat_name:30s} | Correlation: {corr:.4f}")

# Sort by importance
correlations_sorted = sorted(correlations, key=lambda x: x[1], reverse=True)

print("\n" + "-"*80)
print("TOP 15 FEATURES (by correlation with LogP)")
print("-"*80)

top_k = 15
top_features = correlations_sorted[:top_k]
selected_indices = [all_feature_names.index(name) for name, _ in top_features]

for idx, (name, corr) in enumerate(top_features, 1):
    print(f"{idx:2d}. {name:30s} | Correlation: {corr:.4f}")

X_selected = X_enh[:, selected_indices]

# ============================================================================
# COMPARISON: 9D vs 24D vs 15D selected
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

def evaluate_model(X, y_true, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.15, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    mape = np.mean(np.abs((y_pred - y_test) / np.abs(y_test))) * 100 if np.all(y_test != 0) else 0
    within_20_percent = np.sum(np.abs(y_pred - y_test) <= 0.2 * np.abs(y_test)) / len(y_test) * 100
    
    print(f"\n{name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  Success@±20%: {within_20_percent:.1f}%")
    print(f"  Train/Test: {len(X_train)}/{len(X_test)}")
    
    return {
        'name': name,
        'dimensions': X.shape[1],
        'rmse': float(rmse),
        'mape': float(mape),
        'success_pct': float(within_20_percent)
    }

results = []
results.append(evaluate_model(X_orig, y, "Model 1: Original 9 Features"))
results.append(evaluate_model(X_enh, y, "Model 2: All 24 Features"))
results.append(evaluate_model(X_selected, y, "Model 3: Selected 15 Features"))

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("PHASE 3 PHASE 2 VERDICT")
print("="*80)

best_model = max(results, key=lambda x: x['success_pct'])
print(f"\n🏆 Best Model: {best_model['name']}")
print(f"   Dimensions: {best_model['dimensions']}D")
print(f"   Success@±20%: {best_model['success_pct']:.1f}%")

# Check if selected features maintain high accuracy
if results[2]['success_pct'] >= results[1]['success_pct'] * 0.95:
    print(f"\n✅ SUCCESS: 15D features maintain {results[2]['success_pct']:.1f}% accuracy")
    print("   (within 5% of 24D model)")
    print("   → Ready for Phase 3 Phase 3 (ensemble voting)")
else:
    print(f"\n⚠️  Note: 15D features have {results[2]['success_pct']:.1f}% accuracy")
    print(f"   (vs 24D at {results[1]['success_pct']:.1f}%)")

output = {
    'phase': '3.2',
    'title': 'Feature Selection via Correlation',
    'total_features_available': X_enh.shape[1],
    'selected_features': top_k,
    'selected_feature_names': [name for name, _ in top_features],
    'selected_feature_indices': selected_indices if not isinstance(selected_indices, np.ndarray) else selected_indices.tolist(),
    'all_correlations': {name: float(corr) for name, corr in correlations_sorted},
    'model_comparison': results
}

with open('phase3_phase2_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to phase3_phase2_results.json")
