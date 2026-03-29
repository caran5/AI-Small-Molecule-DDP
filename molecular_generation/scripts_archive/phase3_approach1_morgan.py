#!/usr/bin/env python3
"""
PHASE 3 APPROACH 1: MORGAN FINGERPRINTS + DIMENSIONALITY REDUCTION
==================================================================
Strategy: Extract circular fingerprints (2048 bits), reduce via PCA to 100D,
combine with 15D RDKit descriptors to create 115D feature space.

What it captures: Atom connectivity, ring systems, functional groups, 
heteroatom patterns, molecular topology

Expected improvement: +6-10 percentage points (69% → 75-80%)
Timeline: 1-2 hours
"""

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
import sys
sys.path.insert(0, '/Users/ceejayarana/diffusion_model/molecular_generation')
from src.data.loader import DataLoader

print("=" * 80)
print("PHASE 3 APPROACH 1: MORGAN FINGERPRINTS + PCA REDUCTION")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading ChemBL molecules...")
loader = DataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)
print(f"  ✓ Loaded {len(molecules)} molecules")

# Extract SMILES and LogP
smiles_list = [m['smiles'] for m in molecules]
logp_list = [float(Descriptors.MolLogP(Chem.MolFromSmiles(m['smiles']))) 
             for m in molecules if Chem.MolFromSmiles(m['smiles']) is not None]

# Filter: only keep molecules with valid SMILES
valid_indices = []
valid_smiles = []
valid_logp = []
for i, smi in enumerate(smiles_list):
    if i < len(logp_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_indices.append(i)
            valid_smiles.append(smi)
            valid_logp.append(logp_list[i])

print(f"  ✓ Valid SMILES: {len(valid_smiles)}")

# ============================================================================
# EXTRACT MORGAN FINGERPRINTS (2048 bits, radius=2)
# ============================================================================
print("\n[2/6] Extracting Morgan fingerprints...")
morgan_fps = []
for smi in valid_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        morgan_fps.append(np.array(fp))

morgan_fps = np.array(morgan_fps)
print(f"  ✓ Morgan fingerprints shape: {morgan_fps.shape}")
print(f"  ✓ Sparsity: {(morgan_fps == 0).sum() / morgan_fps.size * 100:.1f}%")

# ============================================================================
# PCA REDUCTION: 2048D → 100D
# ============================================================================
print("\n[3/6] Applying PCA reduction (2048D → 100D)...")
pca = PCA(n_components=100, random_state=42)
morgan_pca = pca.fit_transform(morgan_fps)
explained_variance = pca.explained_variance_ratio_.sum()
print(f"  ✓ PCA reduced to {morgan_pca.shape}")
print(f"  ✓ Explained variance: {explained_variance * 100:.2f}%")
print(f"  ✓ Component range: [{morgan_pca.min():.4f}, {morgan_pca.max():.4f}]")

# ============================================================================
# EXTRACT RDKIT DESCRIPTORS (SAFE 15D - NO MolLogP)
# ============================================================================
print("\n[4/6] Extracting 15D RDKit descriptors (no MolLogP)...")

def extract_rdkit_descriptors(smiles):
    """Extract safe RDKit descriptors without target leakage"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = [
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
        float(Descriptors.FractionCSP3(mol)),
        float(Descriptors.BertzCT(mol)),
        float(Descriptors.NumSaturatedRings(mol)),
        float(Descriptors.NumAliphaticRings(mol)),
        float(Descriptors.NumAromaticHeterocycles(mol))
    ]
    return descriptors

rdkit_features = []
valid_smiles_filtered = []
valid_logp_filtered = []

for smi, logp in zip(valid_smiles, valid_logp):
    desc = extract_rdkit_descriptors(smi)
    if desc is not None:
        rdkit_features.append(desc)
        valid_smiles_filtered.append(smi)
        valid_logp_filtered.append(logp)

rdkit_features = np.array(rdkit_features)
morgan_pca_filtered = morgan_pca[:len(valid_smiles_filtered)]
valid_logp_filtered = np.array(valid_logp_filtered)

print(f"  ✓ RDKit features shape: {rdkit_features.shape}")
print(f"  ✓ Final dataset size: {len(valid_logp_filtered)}")

# ============================================================================
# COMBINE: 100D Morgan + 15D RDKit = 115D features
# ============================================================================
print("\n[5/6] Combining features (100D Morgan + 15D RDKit = 115D)...")
combined_features = np.hstack([morgan_pca_filtered, rdkit_features])
print(f"  ✓ Combined feature shape: {combined_features.shape}")
print(f"  ✓ Feature range: [{combined_features.min():.4f}, {combined_features.max():.4f}]")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_features)

# Train/test split (85/15)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, valid_logp_filtered, test_size=0.15, random_state=42
)

print(f"\n  Train set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# ============================================================================
# TRAIN ENSEMBLE (LR + RF + GB with SAME weights as baseline)
# ============================================================================
print("\n[6/6] Training ensemble models (LR:10% + RF:10% + GB:80%)...")

# Train individual models
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

gb = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

# Ensemble with optimal weights
weights = {'lr': 0.10, 'rf': 0.10, 'gb': 0.80}
ensemble_pred = (
    weights['lr'] * lr_pred + 
    weights['rf'] * rf_pred + 
    weights['gb'] * gb_pred
)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred)

# Calculate Success@±20%
def calculate_success_at_threshold(y_true, y_pred, threshold=0.20):
    """Calculate % of predictions within threshold of target"""
    errors = np.abs(y_true - y_pred) / np.abs(y_true)
    return (errors <= threshold).sum() / len(errors) * 100

ensemble_success = calculate_success_at_threshold(y_test, ensemble_pred, threshold=0.20)
lr_success = calculate_success_at_threshold(y_test, lr_pred, threshold=0.20)
rf_success = calculate_success_at_threshold(y_test, rf_pred, threshold=0.20)
gb_success = calculate_success_at_threshold(y_test, gb_pred, threshold=0.20)

# ============================================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================================
print("\n[CV] Running 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
    y_fold_train, y_fold_val = valid_logp_filtered[train_idx], valid_logp_filtered[val_idx]
    
    # Train ensemble on fold
    lr_fold = LinearRegression()
    lr_fold.fit(X_fold_train, y_fold_train)
    lr_fold_pred = lr_fold.predict(X_fold_val)
    
    rf_fold = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_fold.fit(X_fold_train, y_fold_train)
    rf_fold_pred = rf_fold.predict(X_fold_val)
    
    gb_fold = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
    gb_fold.fit(X_fold_train, y_fold_train)
    gb_fold_pred = gb_fold.predict(X_fold_val)
    
    ensemble_fold_pred = (
        weights['lr'] * lr_fold_pred + 
        weights['rf'] * rf_fold_pred + 
        weights['gb'] * gb_fold_pred
    )
    fold_score = calculate_success_at_threshold(y_fold_val, ensemble_fold_pred, threshold=0.20)
    cv_scores.append(fold_score)
    print(f"  Fold {fold}: {fold_score:.1f}%")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3 APPROACH 1: MORGAN FINGERPRINTS - FINAL RESULTS")
print("=" * 80)

results = {
    "approach": "Morgan Fingerprints + PCA",
    "features": {
        "morgan_fingerprints": {
            "bits": 2048,
            "radius": 2,
            "pca_components": 100,
            "explained_variance": float(explained_variance * 100)
        },
        "rdkit_descriptors": 15,
        "total_features": 115,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0])
    },
    "individual_model_performance": {
        "linear_regression": {
            "rmse": float(lr_rmse),
            "success_at_20percent": float(lr_success)
        },
        "random_forest": {
            "rmse": float(rf_rmse),
            "success_at_20percent": float(rf_success)
        },
        "gradient_boosting": {
            "rmse": float(gb_rmse),
            "success_at_20percent": float(gb_success)
        }
    },
    "ensemble_performance": {
        "weights": weights,
        "rmse": float(ensemble_rmse),
        "mape": float(ensemble_mape),
        "success_at_20percent": float(ensemble_success),
        "predictions_sample": {
            "actual": y_test[:5].tolist(),
            "predicted": ensemble_pred[:5].tolist(),
            "errors": (ensemble_pred[:5] - y_test[:5]).tolist()
        }
    },
    "cross_validation": {
        "folds": 5,
        "mean_accuracy": float(cv_mean),
        "std_accuracy": float(cv_std),
        "fold_scores": [float(s) for s in cv_scores]
    },
    "comparison_to_baseline": {
        "baseline_accuracy": 69.3,
        "approach1_accuracy": float(ensemble_success),
        "improvement_percentage_points": float(ensemble_success - 69.3),
        "improvement_relative_percent": float((ensemble_success - 69.3) / 69.3 * 100)
    }
}

# Print formatted results
print("\n▶ TEST SET PERFORMANCE (115D Features)")
print(f"  • Linear Regression:        {lr_success:6.1f}% Success@±20%  (RMSE: {lr_rmse:.4f})")
print(f"  • Random Forest:            {rf_success:6.1f}% Success@±20%  (RMSE: {rf_rmse:.4f})")
print(f"  • Gradient Boosting:        {gb_success:6.1f}% Success@±20%  (RMSE: {gb_rmse:.4f})")
print(f"\n  ⭐ ENSEMBLE (LR:10% RF:10% GB:80%)")
print(f"     Success@±20%: {ensemble_success:.1f}%")
print(f"     RMSE:         {ensemble_rmse:.4f}")
print(f"     MAPE:         {ensemble_mape:.2f}%")

print(f"\n▶ 5-FOLD CROSS-VALIDATION")
print(f"  • Mean:     {cv_mean:.1f}%")
print(f"  • Std Dev:  ±{cv_std:.1f}%")
print(f"  • Range:    [{min(cv_scores):.1f}%, {max(cv_scores):.1f}%]")

print(f"\n▶ COMPARISON TO BASELINE (69.3%)")
print(f"  • Baseline (9D):            69.3%")
print(f"  • Approach 1 (115D):        {ensemble_success:.1f}%")
print(f"  • Improvement:              {ensemble_success - 69.3:+.1f} percentage points")
print(f"  • Relative improvement:     {(ensemble_success - 69.3) / 69.3 * 100:+.1f}%")

print("\n" + "=" * 80)

# Save results
with open('phase3_approach1_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved to phase3_approach1_results.json")
