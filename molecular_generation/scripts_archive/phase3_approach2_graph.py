#!/usr/bin/env python3
"""
PHASE 3 APPROACH 2: GRAPH CONVOLUTIONAL NETWORK (GCN)
=====================================================
Strategy: Extract molecule graph structure (adjacency matrix + node features),
train 2-layer GCN (64D → 32D) with global pooling, add MLP head for LogP prediction.

What it captures: Graph topology, learned aggregation of neighbor information,
ring systems, heteroatom positions, connectivity patterns

Expected improvement: +10-15 percentage points (69% → 80-85%)
Timeline: 2-3 hours
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem
import sys
sys.path.insert(0, '/Users/ceejayarana/diffusion_model/molecular_generation')
from src.data.loader import DataLoader

print("=" * 80)
print("PHASE 3 APPROACH 2: GRAPH CONVOLUTIONAL NETWORK (GCN)")
print("=" * 80)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ============================================================================
# LOAD DATA & EXTRACT MOLECULAR GRAPHS
# ============================================================================
print("\n[1/6] Loading ChemBL molecules...")
loader = DataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)
print(f"  ✓ Loaded {len(molecules)} molecules")

smiles_list = [m['smiles'] for m in molecules]
logp_list = []
graph_data = []

print("\n[2/6] Extracting molecular graphs...")

def extract_graph_features(smiles, atom_features_list=None):
    """
    Extract graph structure and node features.
    
    Returns:
        adjacency (np.array): NxN adjacency matrix
        node_features (np.array): Nx7 node feature matrix
        num_atoms (int): number of atoms
        logp (float): target LogP
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None
    
    # Adjacency matrix
    adjacency = np.zeros((num_atoms, num_atoms))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    
    # Node features: [atomic_num, degree, hybridization, aromatic, h_count, charge, valence]
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum() / 118.0,  # Normalize to 0-1
            atom.GetTotalDegree() / 4.0,  # Max degree typically 4
            int(atom.GetHybridization()),  # SP, SP2, SP3, etc.
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs() / 4.0,
            (atom.GetFormalCharge() + 2) / 4.0,  # Shift charge to 0-1 range
            atom.GetTotalValence() / 4.0
        ]
        node_features.append(features)
    
    node_features = np.array(node_features, dtype=np.float32)
    logp = float(Descriptors.MolLogP(mol))
    
    return {
        'adjacency': adjacency.astype(np.float32),
        'node_features': node_features,
        'num_atoms': num_atoms,
        'logp': logp
    }

# Extract graphs
for i, smi in enumerate(smiles_list):
    result = extract_graph_features(smi)
    if result is not None:
        graph_data.append(result)
        logp_list.append(result['logp'])
    if (i + 1) % 100 == 0:
        print(f"  ✓ Processed {i + 1}/{len(smiles_list)} molecules")

print(f"  ✓ Valid graphs: {len(graph_data)}")
print(f"  ✓ Atom count range: {min([g['num_atoms'] for g in graph_data])} - {max([g['num_atoms'] for g in graph_data])}")

# ============================================================================
# GRAPH CONVOLUTIONAL NETWORK MODEL
# ============================================================================
print("\n[3/6] Building GCN model...")

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, X, A):
        """
        X: node features (batch, num_atoms, in_features)
        A: adjacency matrix (batch, num_atoms, num_atoms)
        """
        # Graph convolution: A @ X @ W
        aggregated = torch.bmm(A, X)  # (batch, num_atoms, in_features)
        output = self.linear(aggregated)  # (batch, num_atoms, out_features)
        return self.relu(output)


class GCNModel(nn.Module):
    def __init__(self, node_feature_dim=7, hidden_dim=64, output_dim=32):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNLayer(node_feature_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, X, A):
        """
        X: node features (batch, num_atoms, node_feature_dim)
        A: adjacency matrices (batch, num_atoms, num_atoms)
        """
        # First GCN layer
        h1 = self.gcn1(X, A)  # (batch, num_atoms, 64)
        
        # Second GCN layer
        h2 = self.gcn2(h1, A)  # (batch, num_atoms, 32)
        
        # Global pooling (mean over atoms)
        h_pool = h2.mean(dim=1)  # (batch, 32)
        
        # MLP head
        output = self.mlp(h_pool)  # (batch, 1)
        return output


# ============================================================================
# COLLATE FUNCTION FOR VARIABLE-SIZE GRAPHS
# ============================================================================
def pad_and_batch(batch_graphs):
    """
    Pad graphs to same size for batching.
    Graphs have variable number of atoms.
    """
    max_atoms = max([g['node_features'].shape[0] for g in batch_graphs])
    
    batch_X = []
    batch_A = []
    batch_y = []
    
    for graph in batch_graphs:
        num_atoms = graph['node_features'].shape[0]
        
        # Pad node features
        X_padded = np.zeros((max_atoms, graph['node_features'].shape[1]), dtype=np.float32)
        X_padded[:num_atoms] = graph['node_features']
        batch_X.append(X_padded)
        
        # Pad adjacency
        A_padded = np.zeros((max_atoms, max_atoms), dtype=np.float32)
        A_padded[:num_atoms, :num_atoms] = graph['adjacency']
        # Add self-loops for padding
        for i in range(num_atoms, max_atoms):
            A_padded[i, i] = 1.0
        batch_A.append(A_padded)
        
        batch_y.append(graph['logp'])
    
    batch_X = torch.from_numpy(np.array(batch_X)).to(device)
    batch_A = torch.from_numpy(np.array(batch_A)).to(device)
    batch_y = torch.from_numpy(np.array(batch_y)).to(device).unsqueeze(1).float()
    
    return batch_X, batch_A, batch_y

# ============================================================================
# TRAINING
# ============================================================================
print("\n[4/6] Training GCN model...")

model = GCNModel(node_feature_dim=7, hidden_dim=64, output_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train/test split
train_idx, test_idx = train_test_split(
    range(len(graph_data)), test_size=0.15, random_state=42
)

train_graphs = [graph_data[i] for i in train_idx]
test_graphs = [graph_data[i] for i in test_idx]
y_test = np.array([graph_data[i]['logp'] for i in test_idx])

print(f"  Train: {len(train_graphs)} graphs | Test: {len(test_graphs)} graphs")

# Train epochs
num_epochs = 30
batch_size = 16
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # Shuffle training data
    train_indices = np.random.permutation(len(train_graphs))
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, len(train_indices), batch_size):
        batch_indices = train_indices[i:i+batch_size]
        batch = [train_graphs[idx] for idx in batch_indices]
        
        X_batch, A_batch, y_batch = pad_and_batch(batch)
        
        optimizer.zero_grad()
        y_pred = model(X_batch, A_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    epoch_loss /= num_batches
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1}/{num_epochs}: Loss = {epoch_loss:.6f}")
    
    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

print("  ✓ Training complete")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[5/6] Evaluating GCN model...")

model.eval()
with torch.no_grad():
    # Test set evaluation (process in batches)
    y_pred_list = []
    for i in range(0, len(test_graphs), batch_size):
        batch = test_graphs[i:i+batch_size]
        X_batch, A_batch, _ = pad_and_batch(batch)
        y_pred_batch = model(X_batch, A_batch).cpu().numpy()
        y_pred_list.extend(y_pred_batch.flatten())
    
    y_pred = np.array(y_pred_list)

# Metrics
gcn_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
gcn_mape = mean_absolute_percentage_error(y_test, y_pred)

def calculate_success_at_threshold(y_true, y_pred, threshold=0.20):
    errors = np.abs(y_true - y_pred) / np.abs(y_true)
    return (errors <= threshold).sum() / len(errors) * 100

gcn_success = calculate_success_at_threshold(y_test, y_pred, threshold=0.20)

# ============================================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================================
print("\n[6/6] Running 5-fold cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx_fold, val_idx_fold) in enumerate(kf.split(graph_data), 1):
    train_fold = [graph_data[i] for i in train_idx_fold]
    val_fold = [graph_data[i] for i in val_idx_fold]
    y_val = np.array([graph_data[i]['logp'] for i in val_idx_fold])
    
    # Train
    model_fold = GCNModel(node_feature_dim=7, hidden_dim=64, output_dim=32).to(device)
    optimizer_fold = optim.Adam(model_fold.parameters(), lr=0.001)
    
    for _ in range(30):
        train_indices = np.random.permutation(len(train_fold))
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            batch = [train_fold[idx] for idx in batch_indices]
            X_batch, A_batch, y_batch = pad_and_batch(batch)
            
            optimizer_fold.zero_grad()
            y_pred_batch = model_fold(X_batch, A_batch)
            loss = criterion(y_pred_batch, y_batch)
            loss.backward()
            optimizer_fold.step()
    
    # Evaluate
    model_fold.eval()
    with torch.no_grad():
        y_pred_fold_list = []
        for i in range(0, len(val_fold), batch_size):
            batch = val_fold[i:i+batch_size]
            X_batch, A_batch, _ = pad_and_batch(batch)
            y_pred_batch = model_fold(X_batch, A_batch).cpu().numpy()
            y_pred_fold_list.extend(y_pred_batch.flatten())
        y_pred_fold = np.array(y_pred_fold_list)
    
    fold_score = calculate_success_at_threshold(y_val, y_pred_fold, threshold=0.20)
    cv_scores.append(fold_score)
    print(f"  Fold {fold}: {fold_score:.1f}%")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3 APPROACH 2: GRAPH CONVOLUTIONAL NETWORK - FINAL RESULTS")
print("=" * 80)

results = {
    "approach": "Graph Convolutional Network (GCN)",
    "architecture": {
        "node_features": 7,
        "gcn1_output": 64,
        "gcn2_output": 32,
        "mlp_layers": "32→16→1",
        "graph_pooling": "mean"
    },
    "dataset": {
        "total_molecules": len(graph_data),
        "train_samples": len(train_graphs),
        "test_samples": len(test_graphs),
        "atom_count_range": [min([g['num_atoms'] for g in graph_data]), 
                            max([g['num_atoms'] for g in graph_data])]
    },
    "training": {
        "epochs": 30,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "criterion": "MSELoss",
        "device": str(device)
    },
    "test_performance": {
        "rmse": float(gcn_rmse),
        "mape": float(gcn_mape),
        "success_at_20percent": float(gcn_success),
        "predictions_sample": {
            "actual": y_test[:5].tolist(),
            "predicted": y_pred[:5].tolist(),
            "errors": (y_pred[:5] - y_test[:5]).tolist()
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
        "approach2_accuracy": float(gcn_success),
        "improvement_percentage_points": float(gcn_success - 69.3),
        "improvement_relative_percent": float((gcn_success - 69.3) / 69.3 * 100)
    }
}

print(f"\n▶ TEST SET PERFORMANCE")
print(f"  • Success@±20%: {gcn_success:.1f}%")
print(f"  • RMSE:         {gcn_rmse:.4f}")
print(f"  • MAPE:         {gcn_mape:.2f}%")

print(f"\n▶ 5-FOLD CROSS-VALIDATION")
print(f"  • Mean:     {cv_mean:.1f}%")
print(f"  • Std Dev:  ±{cv_std:.1f}%")
print(f"  • Range:    [{min(cv_scores):.1f}%, {max(cv_scores):.1f}%]")

print(f"\n▶ COMPARISON TO BASELINE (69.3%)")
print(f"  • Baseline (Handcrafted):   69.3%")
print(f"  • Approach 2 (Graph GCN):   {gcn_success:.1f}%")
print(f"  • Improvement:              {gcn_success - 69.3:+.1f} percentage points")
print(f"  • Relative improvement:     {(gcn_success - 69.3) / 69.3 * 100:+.1f}%")

print("\n" + "=" * 80)

# Save results
with open('phase3_approach2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved to phase3_approach2_results.json")
