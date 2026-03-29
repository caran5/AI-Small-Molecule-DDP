#!/usr/bin/env python3
"""Phase 2 FIX: Non-Circular Features + Honest Baseline Comparison"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import json

print("="*80)
print("PHASE 2 FIX: Non-Circular Features (Real Problem)")
print("="*80)

# Load molecules
from data.loader import DataLoader as MolDataLoader

loader = MolDataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)

# Extract STRUCTURAL features ONLY (no properties we're predicting)
X_list, y_logp = [], []
for mol_data in molecules:
    try:
        smiles = mol_data.get('smiles')
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        
        # STRUCTURAL features ONLY - NO LogP, NO properties
        feat = [
            float(mol.GetNumAtoms()),                    # Total atoms
            float(mol.GetNumHeavyAtoms()),              # Non-hydrogen atoms
            float(len(Chem.GetSSSR(mol))),              # Number of rings
            float(Descriptors.NumAromaticRings(mol)),   # Aromatic rings
            float(Descriptors.NumHeteroatoms(mol)),     # Heteroatoms
            float(Descriptors.NumHDonors(mol)),         # H-bond donors
            float(Descriptors.NumHAcceptors(mol)),      # H-bond acceptors
            float(Descriptors.NumRotatableBonds(mol)),  # Rotatable bonds
            float(Descriptors.TPSA(mol)) if Descriptors.TPSA(mol) else 0.0,  # Polar surface
            float(Descriptors.MolWt(mol)),              # Molecular weight
        ]
        
        # TARGET: LogP (lipophilicity) - what we want to predict
        logp = float(Descriptors.MolLogP(mol))
        
        # Pad to 50D
        feat = np.array(feat + [0.0] * (50 - len(feat)))[:50]
        X_list.append(feat)
        y_logp.append(logp)
    except:
        pass

X = np.array(X_list)
y = np.array(y_logp)

if len(X) == 0:
    print("❌ No valid molecules!")
    sys.exit(1)

print(f"\n✅ Extracted {len(X)} molecules")
print(f"   Features: {X.shape[1]} (structural only, no LogP)")
print(f"   Target (LogP): range {y.min():.2f} to {y.max():.2f}")

# Split
n = len(X)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Normalize
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_std[X_std==0] = 1.0
X_train_n = (X_train - X_mean) / X_std
X_val_n = (X_val - X_mean) / X_std
X_test_n = (X_test - X_mean) / X_std

y_mean, y_std = y_train.mean(), y_train.std()
y_train_n = (y_train - y_mean) / y_std
y_val_n = (y_val - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std

print(f"\n✅ Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"   Normalized features: mean=0, std=1")

# BASELINES
print("\n" + "="*80)
print("BASELINE 1: Linear Regression")
lr = LinearRegression()
lr.fit(X_train_n, y_train_n)
y_pred_lr = lr.predict(X_test_n)
rmse_lr = np.sqrt(((y_pred_lr - y_test_n)**2).mean())
mape_lr = np.abs((y_pred_lr - y_test_n) / (np.abs(y_test_n) + 0.1)).mean()
success_20_lr = (np.abs(y_pred_lr - y_test_n) < 0.2).mean()
print(f"RMSE: {rmse_lr:.4f}, MAPE: {mape_lr*100:.1f}%, Success@±20%: {success_20_lr*100:.1f}%")

print("\nBASELINE 2: Random Forest (50 trees, max_depth=10)")
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
rf.fit(X_train_n, y_train_n)
y_pred_rf = rf.predict(X_test_n)
rmse_rf = np.sqrt(((y_pred_rf - y_test_n)**2).mean())
mape_rf = np.abs((y_pred_rf - y_test_n) / (np.abs(y_test_n) + 0.1)).mean()
success_20_rf = (np.abs(y_pred_rf - y_test_n) < 0.2).mean()
print(f"RMSE: {rmse_rf:.4f}, MAPE: {mape_rf*100:.1f}%, Success@±20%: {success_20_rf*100:.1f}%")

# PROPOSED MODEL
print("\nPROPOSED: MLPDeep Neural Network (REGULARIZED - 18K params, dropout 0.7)")
print("  Testing: Can stronger regularization beat Linear Regression (50.7%)?")

class MLPDeep(nn.Module):
    """
    REGULARIZED VERSION (March 27, 2026)
    Test: Does reducing parameters and increasing dropout beat Linear (50.7%)?
    
    Original (overfitting):   294K params, dropout 0.2
    Regularized (this):       18K params, dropout 0.7
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # REDUCED: 256 → 128
            nn.Linear(50, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.7),
            # REDUCED: 128 → 64
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.7),
            # REDUCED: 64 → 32
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.7),
            # OUTPUT
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")
model = MLPDeep().to(device)

X_train_t = torch.tensor(X_train_n, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_n.reshape(-1,1), dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_n.reshape(-1,1), dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.MSELoss()

best_val = float('inf')
patience_cnt = 0
for epoch in range(300):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        y_p = model(X_b)
        loss = criterion(y_p, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_b)
    
    train_loss /= len(X_train)
    
    model.eval()
    with torch.no_grad():
        y_val_p = model(X_val_t)
        val_loss = criterion(y_val_p, y_val_t).item()
    
    if val_loss < best_val:
        best_val = val_loss
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= 25:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

model.eval()
with torch.no_grad():
    y_pred_nn = model(X_test_t).cpu().numpy().flatten()

rmse_nn = np.sqrt(((y_pred_nn - y_test_n)**2).mean())
mape_nn = np.abs((y_pred_nn - y_test_n) / (np.abs(y_test_n) + 0.1)).mean()
success_20_nn = (np.abs(y_pred_nn - y_test_n) < 0.2).mean()
print(f"RMSE: {rmse_nn:.4f}, MAPE: {mape_nn*100:.1f}%, Success@±20%: {success_20_nn*100:.1f}%")

# SAVE REGRESSOR FOR PHASE 2b
torch.save(model.state_dict(), 'phase2_mlpdeep_regressor_regularized.pt')
print("\n✅ Regularized regressor saved to phase2_mlpdeep_regressor_regularized.pt")

# RESULTS
print("\n" + "="*80)
print("\nHONEST COMPARISON (Non-Circular Features):\n")
print(f"{'Model':<30} {'RMSE':<10} {'MAPE':<10} {'Success@±20%':<15}")
print(f"-" * 65)
print(f"{'Linear Regression':<30} {rmse_lr:.4f}      {mape_lr*100:5.1f}%    {success_20_lr*100:5.1f}%")
print(f"{'Random Forest (50, d=10)':<30} {rmse_rf:.4f}      {mape_rf*100:5.1f}%    {success_20_rf*100:5.1f}%")
print(f"{'MLPDeep (proposed)':<30} {rmse_nn:.4f}      {mape_nn*100:5.1f}%    {success_20_nn*100:5.1f}%  ← YOURS")
print(f"-" * 65)

# Improvements
imp_lr = ((rmse_rf - rmse_nn) / max(rmse_rf, 0.001)) * 100
imp_rf = ((rmse_rf - rmse_nn) / max(rmse_rf, 0.001)) * 100

print(f"\nMLPDeep vs Linear:      {(rmse_lr - rmse_nn) / max(rmse_lr, 0.001) * 100:+.1f}%")
print(f"MLPDeep vs RandomForest: {(rmse_rf - rmse_nn) / max(rmse_rf, 0.001) * 100:+.1f}%")

# Verdict - CRITICAL TEST FOR REGULARIZATION
if success_20_nn >= success_20_lr:
    verdict = "✅ PASS - MLPDeep beats Linear Regression! Regularization worked."
    decision = "→ Keep improved MLPDeep model for Phase 3"
elif success_20_nn >= 0.45:
    verdict = "⚠️  CLOSE - MLPDeep within 5% of Linear. Borderline."
    decision = "→ Use Linear (safer, simpler)"
else:
    verdict = "❌ FAIL - MLPDeep still worse than Linear. Problem is not just overfitting."
    decision = "→ Switch to Linear Regression for Phase 3"

print(f"\nVERDICT: {verdict}")
print(f"DECISION: {decision}")
print("="*80)

results = {
    'timestamp': datetime.now().isoformat(),
    'phase': 2,
    'test_type': 'Regularization Test - Reduced Params (18K) + Dropout (0.7)',
    'test_purpose': 'Does regularization fix overfitting? Can we beat Linear (50.7%)?',
    'molecules': len(X),
    'test_size': len(X_test),
    'features': 'Structural only (10 features: NumAtoms, NumHeavyAtoms, Rings, AromaticRings, Heteroatoms, HBD, HBA, RotatableBonds, TPSA, MolWt)',
    'target': 'LogP (NOT included in features)',
    'success_metric': '±20% error (±0.2 normalized units)',
    'models': {
        'linear_regression': {
            'type': 'baseline',
            'rmse': float(rmse_lr),
            'mape': float(mape_lr),
            'success_20pct': float(success_20_lr)
        },
        'random_forest': {
            'type': 'baseline',
            'rmse': float(rmse_rf),
            'mape': float(mape_rf),
            'success_20pct': float(success_20_rf)
        },
        'mlpdeep': {
            'type': 'proposed',
            'rmse': float(rmse_nn),
            'mape': float(mape_nn),
            'success_20pct': float(success_20_nn)
        }
    },
    'winner': 'MLPDeep' if success_20_nn >= max(success_20_lr, success_20_rf) else ('Random Forest' if success_20_rf >= success_20_lr else 'Linear'),
    'verdict': verdict
}

with open('phase2_honest_noncircular.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to phase2_honest_noncircular.json\n")
