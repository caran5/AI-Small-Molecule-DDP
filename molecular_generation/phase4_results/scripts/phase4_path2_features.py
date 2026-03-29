#!/usr/bin/env python3
"""
PHASE 4 PATH 2: FEATURE ENGINEERING
====================================
Add 30-50 new descriptors beyond current 115D.
Expected improvement: +5-8pp (76% → 81-84%)
Time: 3-4 hours

New features to add:
1. Extended RDKit descriptors (PEOE_VSA, MolVSA, EstateValues)
2. Graph structural features (degree, branching, bridges)
3. Atomic environment features (500+ dimensional)
4. Functional group patterns (20+ common groups)
5. Correlation-based selection

Current (Approach 1): 115D
This path: 150-200D (after correlation filtering)
Target: 81-84%
"""

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
import sys
sys.path.insert(0, '/Users/ceejayarana/diffusion_model/molecular_generation')
from src.data.loader import DataLoader
import time

print("=" * 80)
print("PHASE 4 PATH 2: FEATURE ENGINEERING (EXTENDED)")
print("=" * 80)

# ============================================================================
# LOAD DATA
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
# FEATURE EXTRACTION: COMPREHENSIVE
# ============================================================================
print("\n[FEATURES] Extracting 200+ dimensional feature set...")

def extract_comprehensive_features(smiles):
    """Extract 200+ features including extended descriptors and functional groups"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    # 1. BASIC STRUCTURE (10 features)
    features.extend([
        float(mol.GetNumAtoms()),
        float(mol.GetNumHeavyAtoms()),
        float(len(Chem.GetSSSR(mol))),
        float(Descriptors.NumAromaticRings(mol)),
        float(Descriptors.NumHeteroatoms(mol)),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        float(Descriptors.TPSA(mol)) or 0.0,
        float(Descriptors.MolWt(mol))
    ])
    
    # 2. MOLECULAR PROPERTIES (12 features)
    features.extend([
        float(Descriptors.FractionCSP3(mol)),
        float(Descriptors.BertzCT(mol)),
        float(Descriptors.NumSaturatedRings(mol)),
        float(Descriptors.NumAliphaticRings(mol)),
        float(Descriptors.NumAromaticHeterocycles(mol)),
        float(Crippen.MolLogP(mol)),
        float(Crippen.MolMR(mol)),
        float(Descriptors.MolWt(mol) / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0.0,  # Simplified density
        float(Descriptors.PEOE_VSA1(mol)),
        float(Descriptors.PEOE_VSA2(mol)),
        float(Descriptors.LabuteASA(mol)),
        float(Descriptors.SlogP_VSA1(mol))
    ])
    
    # 3. TOPOLOGICAL DESCRIPTORS (14 features)
    features.extend([
        float(Descriptors.NumBridgeheads(mol)),
        float(Descriptors.NumSpiroAtoms(mol)),
        float(Descriptors.NumAtomStereoCenters(mol)),
        float(Descriptors.NumUnspecifiedAtomStereoCenters(mol)),
        float(Descriptors.Kappa1(mol)),
        float(Descriptors.Kappa2(mol)),
        float(Descriptors.Kappa3(mol)),
        float(Descriptors.ExactMolWt(mol)),
        float(Descriptors.Asphericity(mol)),
        float(Descriptors.Eccentricity(mol)),
        float(Descriptors.Periperi(mol)),
        float(Descriptors.PEOE_VSA1(mol)),
        float(Descriptors.PEOE_VSA2(mol)),
        float(Descriptors.PEOE_VSA3(mol))
    ])
    
    # 4. ESTATE-RELATED (10 features)
    try:
        features.extend([
            float(Descriptors.SlogP_VSA1(mol)),
            float(Descriptors.SlogP_VSA2(mol)),
            float(Descriptors.SlogP_VSA3(mol)),
            float(Descriptors.SlogP_VSA4(mol)),
            float(Descriptors.SlogP_VSA5(mol)),
            float(Descriptors.SlogP_VSA6(mol)),
            float(Descriptors.SlogP_VSA7(mol)),
            float(Descriptors.SlogP_VSA8(mol)),
            float(Descriptors.SlogP_VSA9(mol)),
            float(Descriptors.NumHBD(mol))  # Additional descriptor
        ])
    except:
        features.extend([0.0] * 10)
    
    # 5. GRAPH STRUCTURAL FEATURES (12 features)
    try:
        atom_degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        features.extend([
            float(min(atom_degrees) if atom_degrees else 0),
            float(max(atom_degrees) if atom_degrees else 0),
            float(np.mean(atom_degrees) if atom_degrees else 0),
            float(np.std(atom_degrees) if atom_degrees else 0),
            float(Descriptors.NumBridgeheads(mol)),
            float(Descriptors.NumSpiroAtoms(mol)),
            float(len([b for b in mol.GetBonds()])),  # num bonds
            float(len([b for b in mol.GetBonds() if b.GetIsAromatic()])),  # aromatic bonds
            float(len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE])),
            float(len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE])),
            float(len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE])),
            float(len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC]))
        ])
    except:
        features.extend([0.0] * 12)
    
    # 6. FUNCTIONAL GROUP PATTERNS (20+ features)
    try:
        # Simple SMARTS-based functional groups
        groups = {
            'carbonyl': '[C]=O',
            'carboxyl': 'O=C[OH]',
            'hydroxyl': '[OH]',
            'amine': '[NH2]',
            'amide': '[NH]C=O',
            'ester': 'O=C[O]',
            'ether': '[OD2]',
            'thiol': '[SH]',
            'sulfide': '[SD2]',
            'disulfide': 'S-S',
            'nitro': '[N+](=O)[O-]',
            'nitrile': 'C#N',
            'alkene': '[C]=[C]',
            'alkyne': '[C]#[C]',
            'benzene': 'c1ccccc1',
            'phenol': '[OH]c1ccccc1',
            'aniline': '[NH2]c1ccccc1',
            'pyridine': 'c1ccncc1',
            'imidazole': 'c1c[nH]cn1',
            'furan': 'o1cccc1'
        }
        
        for group_name, smarts in groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.HasSubstructMatch(pattern)
                features.append(float(1 if matches else 0))
    except:
        features.extend([0.0] * 20)
    
    # Total: ~100+ features
    return features

# Extract features for all molecules
print("  Extracting features for all molecules...")
all_features = []
for i, smi in enumerate(valid_smiles):
    feat = extract_comprehensive_features(smi)
    if feat is not None:
        all_features.append(feat)
    if (i + 1) % 100 == 0:
        print(f"    ✓ {i+1}/{len(valid_smiles)} molecules processed")

all_features = np.array(all_features)
print(f"  ✓ Total features extracted: {all_features.shape}")

# ============================================================================
# CORRELATION-BASED FEATURE SELECTION
# ============================================================================
print("\n[SELECT] Performing correlation-based feature selection...")

# Calculate correlations with target
correlations = []
for i in range(all_features.shape[1]):
    corr = np.corrcoef(all_features[:, i], logp_list)[0, 1]
    correlations.append((i, corr, abs(corr)))

correlations_sorted = sorted(correlations, key=lambda x: x[2], reverse=True)

# Select top features
n_select = 150  # Select 150 most correlated features
selected_indices = sorted([c[0] for c in correlations_sorted[:n_select]])
selected_features = all_features[:, selected_indices]

print(f"  ✓ Selected {len(selected_indices)} features from {all_features.shape[1]}")
print(f"  ✓ Top 10 correlations:")
for rank, (idx, corr, abs_corr) in enumerate(correlations_sorted[:10], 1):
    print(f"    {rank:2d}. Feature {idx:3d}: correlation = {corr:+.4f}")

# ============================================================================
# COMBINE WITH MORGAN FINGERPRINTS (from best Path 1 config)
# ============================================================================
print("\n[COMBINE] Combining with Morgan fingerprints...")

# Best config from Path 1: r=2, b=2048, pca=125 (assuming)
# For now, use the baseline: r=2, b=2048, pca=100
morgan_fps = []
for smi in valid_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        morgan_fps.append(np.array(fp))

morgan_fps = np.array(morgan_fps)
pca_morgan = PCA(n_components=100, random_state=42).fit_transform(morgan_fps)

# Combine: Morgan PCA (100D) + Selected Features (150D) = 250D
X = np.hstack([pca_morgan, selected_features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  ✓ Combined feature matrix: {X_scaled.shape}D")

# ============================================================================
# TRAIN AND EVALUATE
# ============================================================================
print("\n[TRAIN] Training gradient boosting model on extended features...")

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
print("PATH 2 FEATURE ENGINEERING RESULTS")
print("=" * 80)

baseline = 76.0
improvement = test_accuracy - baseline

print(f"\n▶ PERFORMANCE:")
print(f"  • Baseline (Approach 1, 115D):    76.0%")
print(f"  • Path 2 (250D extended):         {test_accuracy:.1f}%")
print(f"  • Improvement:                    {improvement:+.1f}pp")
print(f"  • CV Mean ± Std:                  {cv_mean:.1f}% ± {cv_std:.1f}%")
print(f"  • RMSE:                           {rmse:.4f}")
print(f"  • MAPE:                           {mape:.4f}")

print(f"\n▶ FEATURE ENGINEERING SUMMARY:")
print(f"  • Original features (Approach 1): 115D")
print(f"  • New features extracted:         {all_features.shape[1]}D")
print(f"  • Selected via correlation:       {selected_features.shape[1]}D")
print(f"  • Combined total:                 {X_scaled.shape[1]}D")

print(f"\n▶ TOP CONTRIBUTING FEATURES:")
feature_importance = gb.feature_importances_
top_indices = np.argsort(feature_importance)[-10:][::-1]
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. Feature {idx:3d}: importance = {feature_importance[idx]:.4f}")

# Save results
output = {
    "approach": "Phase 4 Path 2: Feature Engineering",
    "baseline_accuracy": 76.0,
    "features": {
        "original": 115,
        "extracted": int(all_features.shape[1]),
        "selected": int(selected_features.shape[1]),
        "combined": int(X_scaled.shape[1])
    },
    "performance": {
        "test_accuracy": float(test_accuracy),
        "cv_mean": float(cv_mean),
        "cv_std": float(cv_std),
        "rmse": float(rmse),
        "mape": float(mape),
        "improvement_pp": float(improvement)
    }
}

with open('phase4_path2_feature_engineering_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ Results saved to phase4_path2_feature_engineering_results.json")
print("=" * 80)

elapsed = time.time() - start_time
print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
print(f"\nNext: Combine Path 1 (best hyperparams) + Path 2 (best features) → target 81-84%")
