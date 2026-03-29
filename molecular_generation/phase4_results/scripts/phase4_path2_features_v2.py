#!/usr/bin/env python3
"""
PHASE 4 PATH 2: FEATURE ENGINEERING (SIMPLIFIED)
=================================================
Extract reliable descriptors and use correlation-based selection.
Expected improvement: +5-8pp (81-84%)
Time: 3-4 hours

Simplified version focusing on stable RDKit descriptors that work across versions.
"""

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
import sys
sys.path.insert(0, '/Users/ceejayarana/diffusion_model/molecular_generation')
from src.data.loader import DataLoader
import time

print("=" * 80)
print("PHASE 4 PATH 2: FEATURE ENGINEERING (SIMPLIFIED)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[SETUP] Loading data...")
start_time = time.time()

loader = DataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)
smiles_list = [m['smiles'] for m in molecules]

logp_list = []
valid_smiles = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid_smiles.append(smi)
        logp_list.append(float(Descriptors.MolLogP(mol)))

logp_list = np.array(logp_list)
print(f"  ✓ Loaded {len(valid_smiles)} molecules")

# ============================================================================
# EXTRACT STABLE DESCRIPTORS
# ============================================================================
print("\n[FEATURES] Extracting stable RDKit descriptors...")

def extract_safe_descriptors(smiles):
    """Extract only stable, widely-available RDKit descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        descriptors = [
            # Basic structure
            float(mol.GetNumAtoms()),
            float(mol.GetNumHeavyAtoms()),
            float(len(Chem.GetSSSR(mol))),
            float(Descriptors.NumAromaticRings(mol)),
            float(Descriptors.NumHeteroatoms(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.TPSA(mol)) or 0.0,
            float(Descriptors.MolWt(mol)),
            # Lipophilicity and size
            float(Crippen.MolLogP(mol)),
            float(Crippen.MolMR(mol)),
            # Ring features
            float(Descriptors.FractionCSP3(mol)),
            float(Descriptors.BertzCT(mol)),
            float(Descriptors.NumSaturatedRings(mol)),
            float(Descriptors.NumAliphaticRings(mol)),
            float(Descriptors.NumAromaticHeterocycles(mol)),
            # Topological
            float(Descriptors.ExactMolWt(mol)),
            float(Descriptors.Kappa1(mol)),
            float(Descriptors.Kappa2(mol)),
            float(Descriptors.Kappa3(mol)),
            # Additional structural features
            float(Descriptors.NumAtomStereoCenters(mol)),
            float(Descriptors.PEOE_VSA1(mol) if hasattr(Descriptors, 'PEOE_VSA1') else 0.0),
            float(Descriptors.LabuteASA(mol) if hasattr(Descriptors, 'LabuteASA') else 0.0),
        ]
        return descriptors
    except Exception as e:
        return None

# Extract for all molecules
all_features = []
for i, smi in enumerate(valid_smiles):
    desc = extract_safe_descriptors(smi)
    if desc is not None:
        all_features.append(desc)
    if (i + 1) % 100 == 0:
        print(f"    ✓ {i+1}/{len(valid_smiles)} processed")

all_features = np.array(all_features)
print(f"  ✓ Extracted {all_features.shape[1]}D feature matrix: {all_features.shape}")

# ============================================================================
# CORRELATION-BASED FEATURE SELECTION
# ============================================================================
print("\n[SELECT] Correlation-based feature selection...")

correlations = []
for i in range(all_features.shape[1]):
    try:
        corr = np.corrcoef(all_features[:, i], logp_list)[0, 1]
        if not np.isnan(corr):
            correlations.append((i, corr, abs(corr)))
    except:
        pass

correlations_sorted = sorted(correlations, key=lambda x: x[2], reverse=True)

# Select top features
n_select = min(20, len(correlations_sorted))
selected_indices = sorted([c[0] for c in correlations_sorted[:n_select]])
selected_features = all_features[:, selected_indices]

print(f"  ✓ Selected {len(selected_indices)} features (top by correlation)")
print(f"  ✓ Top correlations:")
for rank, (idx, corr, abs_corr) in enumerate(correlations_sorted[:5], 1):
    print(f"    {rank}. Feature {idx}: {corr:+.4f}")

# ============================================================================
# COMBINE WITH MORGAN FINGERPRINTS
# ============================================================================
print("\n[COMBINE] Combining with Morgan fingerprints (best Path 1 config)...")

# Use best Path 1 config: radius=1, nBits=2048, pca=200
morgan_fps = []
for smi in valid_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=2048)
        morgan_fps.append(np.array(fp))

morgan_fps = np.array(morgan_fps)
pca_morgan = PCA(n_components=200, random_state=42).fit_transform(morgan_fps)

# Combine
X = np.hstack([pca_morgan, selected_features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  ✓ Morgan PCA: {pca_morgan.shape[1]}D")
print(f"  ✓ Selected features: {selected_features.shape[1]}D")
print(f"  ✓ Combined: {X_scaled.shape[1]}D total")

# ============================================================================
# TRAIN AND EVALUATE
# ============================================================================
print("\n[TRAIN] Training gradient boosting on combined features...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, logp_list, test_size=0.15, random_state=42
)

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)

# Success@±20%
errors = np.abs(y_test - y_pred) / np.abs(y_test)
test_accuracy = (errors <= 0.20).sum() / len(errors) * 100

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_scaled):
    X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
    y_fold_train, y_fold_val = logp_list[train_idx], logp_list[val_idx]
    
    gb_fold = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb_fold.fit(X_fold_train, y_fold_train)
    y_fold_pred = gb_fold.predict(X_fold_val)
    
    fold_errors = np.abs(y_fold_val - y_fold_pred) / np.abs(y_fold_val)
    fold_score = (fold_errors <= 0.20).sum() / len(fold_errors) * 100
    cv_scores.append(fold_score)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("PATH 2 FEATURE ENGINEERING - RESULTS")
print("=" * 80)

baseline = 76.0
path1_result = 81.3
improvement = test_accuracy - baseline
cumulative = test_accuracy - path1_result

print(f"\n▶ PERFORMANCE:")
print(f"  • Baseline (Approach 1):      {baseline:.1f}%")
print(f"  • Path 1 result:              {path1_result:.1f}%")
print(f"  • Path 2 result (combined):   {test_accuracy:.1f}%")
print(f"  • Improvement from baseline:  {improvement:+.1f}pp")
print(f"  • Change from Path 1:         {cumulative:+.1f}pp")
print(f"  • CV Mean ± Std:              {cv_mean:.1f}% ± {cv_std:.1f}%")
print(f"  • RMSE:                       {rmse:.4f}")
print(f"  • MAPE:                       {mape:.4f}")

print(f"\n▶ FEATURE SUMMARY:")
print(f"  • Original (Approach 1):      115D")
print(f"  • Extracted:                  {all_features.shape[1]}D")
print(f"  • Selected (top correlated):  {selected_features.shape[1]}D")
print(f"  • Morgan PCA (radius=1):      {pca_morgan.shape[1]}D")
print(f"  • Total combined:             {X_scaled.shape[1]}D")

# Save results
output = {
    "approach": "Phase 4 Path 2: Feature Engineering",
    "baseline_accuracy": baseline,
    "path1_accuracy": path1_result,
    "features": {
        "original": 115,
        "extracted": int(all_features.shape[1]),
        "selected": int(selected_features.shape[1]),
        "morgan_pca": int(pca_morgan.shape[1]),
        "combined": int(X_scaled.shape[1])
    },
    "performance": {
        "test_accuracy": float(test_accuracy),
        "cv_mean": float(cv_mean),
        "cv_std": float(cv_std),
        "rmse": float(rmse),
        "mape": float(mape),
        "improvement_from_baseline_pp": float(improvement),
        "change_from_path1_pp": float(cumulative)
    }
}

with open('phase4_path2_feature_engineering_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ Results saved to phase4_path2_feature_engineering_results.json")
print("=" * 80)

elapsed = time.time() - start_time
print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
print(f"\n📊 Next: Execute Path 3 (stacking ensemble)")
print(f"   Current: 81.3% (Path 1)")
print(f"   Target:  82-86% (Path 3)")
