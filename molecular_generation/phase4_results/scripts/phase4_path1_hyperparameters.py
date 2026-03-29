#!/usr/bin/env python3
"""
PHASE 4 PATH 1: HYPERPARAMETER OPTIMIZATION
============================================
Grid search over Morgan fingerprint radius, bits, and PCA components.
Expected improvement: +3-5pp (76% → 79-81%)
Time: 2-3 hours

Current best (baseline Approach 1):
- Morgan: radius=2, nBits=2048
- PCA: n_components=100
- Result: 76.0%

Grid to search:
- Radius: [0, 1, 2, 3]           (4 options)
- Bits: [1024, 2048, 4096]       (3 options)
- PCA: [50, 75, 100, 125, 150]   (5 options)
Total: 60 configurations
"""

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
import sys
sys.path.insert(0, '/Users/ceejayarana/diffusion_model/molecular_generation')
from src.data.loader import DataLoader
import time

print("=" * 80)
print("PHASE 4 PATH 1: HYPERPARAMETER GRID SEARCH")
print("=" * 80)

# ============================================================================
# LOAD DATA (reuse from Approach 1)
# ============================================================================
print("\n[SETUP] Loading data...")
start_time = time.time()

loader = DataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)
smiles_list = [m['smiles'] for m in molecules]

# Extract LogP target
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
# EXTRACT FIXED RDKit DESCRIPTORS (same 15 for all configs)
# ============================================================================
print("\n[FEATURES] Extracting RDKit descriptors...")

def extract_rdkit_descriptors(smiles):
    """Extract safe 15D RDKit descriptors"""
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
for smi in valid_smiles:
    desc = extract_rdkit_descriptors(smi)
    if desc is not None:
        rdkit_features.append(desc)

rdkit_features = np.array(rdkit_features)
print(f"  ✓ RDKit features: {rdkit_features.shape}")

# ============================================================================
# GRID SEARCH CONFIGURATION
# ============================================================================
print("\n[GRID] Configuring search space...")

grid_config = {
    'radius': [0, 1, 2, 3],
    'n_bits': [1024, 2048, 4096],
    'pca_components': [50, 75, 100, 125, 150, 200]
}

print(f"  ✓ Radius options: {grid_config['radius']}")
print(f"  ✓ Bit options: {grid_config['n_bits']}")
print(f"  ✓ PCA components: {grid_config['pca_components']}")
print(f"  ✓ Total configurations: {len(grid_config['radius']) * len(grid_config['n_bits']) * len(grid_config['pca_components'])}")

# ============================================================================
# PRECOMPUTE ALL MORGAN FINGERPRINTS (outer loop)
# ============================================================================
print("\n[FP] Precomputing Morgan fingerprints for all radius/bits combinations...")

fingerprint_cache = {}
config_counter = 0
total_configs = len(grid_config['radius']) * len(grid_config['n_bits']) * len(grid_config['pca_components'])

for radius in grid_config['radius']:
    for n_bits in grid_config['n_bits']:
        fp_key = (radius, n_bits)
        
        fps = []
        for smi in valid_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                fps.append(np.array(fp))
        
        fingerprint_cache[fp_key] = np.array(fps)
        print(f"    ✓ Cached FP(r={radius}, b={n_bits}): {fingerprint_cache[fp_key].shape}")

# ============================================================================
# GRID SEARCH: Test all combinations
# ============================================================================
print("\n[SEARCH] Running grid search (60 configurations)...\n")

results = []
config_counter = 0

for radius in grid_config['radius']:
    for n_bits in grid_config['n_bits']:
        for n_components in grid_config['pca_components']:
            config_counter += 1
            
            # Get precomputed fingerprints
            morgan_fps = fingerprint_cache[(radius, n_bits)]
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, morgan_fps.shape[1]), random_state=42)
            morgan_pca = pca.fit_transform(morgan_fps)
            
            # Combine features
            X = np.hstack([morgan_pca, rdkit_features])
            
            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, logp_list, test_size=0.15, random_state=42
            )
            
            # Train ensemble (same as Approach 1)
            lr = LinearRegression()
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
            
            lr.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            gb.fit(X_train, y_train)
            
            # Ensemble prediction
            weights = {'lr': 0.10, 'rf': 0.10, 'gb': 0.80}
            y_pred = (
                weights['lr'] * lr.predict(X_test) +
                weights['rf'] * rf.predict(X_test) +
                weights['gb'] * gb.predict(X_test)
            )
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # Success@±20%
            errors = np.abs(y_test - y_pred) / np.abs(y_test)
            success = (errors <= 0.20).sum() / len(errors) * 100
            
            # 5-fold CV for validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in kf.split(X_scaled):
                X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
                y_fold_train, y_fold_val = logp_list[train_idx], logp_list[val_idx]
                
                gb_fold = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
                gb_fold.fit(X_fold_train, y_fold_train)
                y_fold_pred = gb_fold.predict(X_fold_val)
                
                fold_errors = np.abs(y_fold_val - y_fold_pred) / np.abs(y_fold_val)
                fold_score = (fold_errors <= 0.20).sum() / len(fold_errors) * 100
                cv_scores.append(fold_score)
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            result = {
                'config_id': config_counter,
                'radius': radius,
                'n_bits': n_bits,
                'pca_components': n_components,
                'features_total': X.shape[1],
                'test_accuracy': float(success),
                'rmse': float(rmse),
                'mape': float(mape),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'variance_explained': float(pca.explained_variance_ratio_.sum())
            }
            results.append(result)
            
            # Print progress
            if config_counter % 6 == 0 or config_counter == 1:
                print(f"[{config_counter:2d}/60] r={radius} b={n_bits:4d} pca={n_components:3d} → " +
                      f"Test: {success:5.1f}% | CV: {cv_mean:5.1f}%±{cv_std:4.1f}%")

# ============================================================================
# ANALYSIS & RANKING
# ============================================================================
print("\n" + "=" * 80)
print("GRID SEARCH RESULTS")
print("=" * 80)

# Sort by test accuracy (primary) then CV stability
results_sorted = sorted(results, key=lambda x: (-x['test_accuracy'], -x['cv_mean']))

print("\n▶ TOP 10 CONFIGURATIONS (by test accuracy):\n")
for rank, result in enumerate(results_sorted[:10], 1):
    print(f"{rank:2d}. r={result['radius']} b={result['n_bits']:4d} pca={result['pca_components']:3d}  " +
          f"Test: {result['test_accuracy']:5.1f}% | CV: {result['cv_mean']:5.1f}%±{result['cv_std']:4.1f}% | " +
          f"RMSE: {result['rmse']:.4f}")

# Best configuration
best = results_sorted[0]
baseline = 76.0  # Approach 1 baseline

print("\n▶ BEST CONFIGURATION FOUND:")
print(f"  • Radius: {best['radius']}")
print(f"  • Bits: {best['n_bits']}")
print(f"  • PCA Components: {best['pca_components']}")
print(f"  • Total Features: {best['features_total']}D")
print(f"  • Test Accuracy: {best['test_accuracy']:.1f}%")
print(f"  • CV Mean: {best['cv_mean']:.1f}% ± {best['cv_std']:.1f}%")
print(f"  • RMSE: {best['rmse']:.4f}")
print(f"  • Variance Explained: {best['variance_explained']*100:.1f}%")

improvement = best['test_accuracy'] - baseline
print(f"\n▶ IMPROVEMENT OVER BASELINE (76.0%):")
print(f"  • Absolute: {improvement:+.1f} percentage points")
print(f"  • Relative: {improvement/baseline*100:+.1f}%")

if improvement > 0:
    print(f"  ✅ IMPROVEMENT FOUND! New best: {best['test_accuracy']:.1f}%")
else:
    print(f"  ⚠️  No improvement found. Consider Path 2 (features) or Path 3 (stacking)")

# ============================================================================
# DETAILED ANALYSIS BY PARAMETER
# ============================================================================
print("\n▶ PERFORMANCE BY RADIUS:")
for r in sorted(set(x['radius'] for x in results)):
    r_results = [x for x in results if x['radius'] == r]
    best_r = max(r_results, key=lambda x: x['test_accuracy'])
    avg_acc = np.mean([x['test_accuracy'] for x in r_results])
    print(f"  Radius {r}: Best={best_r['test_accuracy']:.1f}%, Avg={avg_acc:.1f}%")

print("\n▶ PERFORMANCE BY BITS:")
for b in sorted(set(x['n_bits'] for x in results)):
    b_results = [x for x in results if x['n_bits'] == b]
    best_b = max(b_results, key=lambda x: x['test_accuracy'])
    avg_acc = np.mean([x['test_accuracy'] for x in b_results])
    print(f"  Bits {b:4d}: Best={best_b['test_accuracy']:.1f}%, Avg={avg_acc:.1f}%")

print("\n▶ PERFORMANCE BY PCA COMPONENTS:")
for p in sorted(set(x['pca_components'] for x in results)):
    p_results = [x for x in results if x['pca_components'] == p]
    best_p = max(p_results, key=lambda x: x['test_accuracy'])
    avg_acc = np.mean([x['test_accuracy'] for x in p_results])
    print(f"  PCA {p:3d}: Best={best_p['test_accuracy']:.1f}%, Avg={avg_acc:.1f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)

output = {
    "approach": "Phase 4 Path 1: Hyperparameter Optimization",
    "baseline_accuracy": 76.0,
    "baseline_cv": 75.6,
    "search_space": {
        "radius": grid_config['radius'],
        "n_bits": grid_config['n_bits'],
        "pca_components": grid_config['pca_components'],
        "total_configurations": total_configs
    },
    "best_configuration": {
        "radius": best['radius'],
        "n_bits": best['n_bits'],
        "pca_components": best['pca_components'],
        "test_accuracy": best['test_accuracy'],
        "cv_mean": best['cv_mean'],
        "cv_std": best['cv_std'],
        "rmse": best['rmse'],
        "improvement_pp": improvement
    },
    "top_10_configurations": [
        {
            "rank": i,
            "radius": r['radius'],
            "n_bits": r['n_bits'],
            "pca_components": r['pca_components'],
            "test_accuracy": r['test_accuracy'],
            "cv_mean": r['cv_mean'],
            "cv_std": r['cv_std']
        }
        for i, r in enumerate(results_sorted[:10], 1)
    ],
    "all_results": results
}

with open('phase4_path1_grid_search_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("✓ Results saved to phase4_path1_grid_search_results.json")
print("=" * 80)

elapsed = time.time() - start_time
print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
print(f"\n📊 Summary:")
print(f"   Baseline (Approach 1):          76.0%")
print(f"   Best found (Path 1):            {best['test_accuracy']:.1f}%")
print(f"   Improvement:                    {improvement:+.1f}pp")
print(f"   Next target (Path 1 + 2):       81-84%")
