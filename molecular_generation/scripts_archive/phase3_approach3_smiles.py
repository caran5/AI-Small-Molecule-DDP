#!/usr/bin/env python3
"""
PHASE 3 APPROACH 3: SMILES TRANSFORMER
======================================
Strategy: Tokenize SMILES strings, embed tokens (128D), apply 4-layer Transformer
with multi-head attention (8 heads), pool CLS token, add MLP head for LogP prediction.

What it captures: SMILES token patterns, attention to important atoms/bonds,
semantic meaning of molecular string representation, sequential dependencies

Expected improvement: +15-20 percentage points (69% → 85-92%)
Timeline: 3-4 hours
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
from rdkit.Chem import Descriptors
import sys
sys.path.insert(0, '/Users/ceejayarana/diffusion_model/molecular_generation')
from src.data.loader import DataLoader

print("=" * 80)
print("PHASE 3 APPROACH 3: SMILES TRANSFORMER")
print("=" * 80)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ============================================================================
# SMILES TOKENIZER
# ============================================================================
class SMILESTokenizer:
    """Simple SMILES character tokenizer with special tokens"""
    
    # Common SMILES characters
    VOCAB = list('CcNnOoPpSsFf[H]@+\\=/()#-123456789') + ['[CLS]', '[PAD]', '[UNK]']
    
    def __init__(self):
        self.char_to_idx = {char: idx for idx, char in enumerate(self.VOCAB)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.VOCAB)}
    
    def encode(self, smiles, max_length=100):
        """Encode SMILES string to token indices"""
        tokens = ['[CLS]']  # Add CLS token at start
        
        i = 0
        while i < len(smiles) and len(tokens) < max_length - 1:
            if i < len(smiles) - 1 and smiles[i:i+3] == '[H]':
                tokens.append('[H]')
                i += 3
            elif i < len(smiles) - 1 and smiles[i:i+2] in ['Cl', 'Br']:
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append('[PAD]')
        
        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.char_to_idx:
                indices.append(self.char_to_idx[token])
            else:
                indices.append(self.char_to_idx['[UNK]'])
        
        return np.array(indices[:max_length], dtype=np.int64)
    
    def get_vocab_size(self):
        return len(self.VOCAB)


# ============================================================================
# LOAD DATA & TOKENIZE
# ============================================================================
print("\n[1/6] Loading and tokenizing SMILES...")
loader = DataLoader()
molecules = loader._load_chembl_database('src/data/chembl_34_sqlite.tar.gz', limit=500)
print(f"  ✓ Loaded {len(molecules)} molecules")

tokenizer = SMILESTokenizer()
vocab_size = tokenizer.get_vocab_size()
print(f"  ✓ Vocabulary size: {vocab_size}")

smiles_list = [m['smiles'] for m in molecules]
encoded_smiles = []
logp_list = []

max_length = 100

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        encoded = tokenizer.encode(smi, max_length=max_length)
        logp = float(Descriptors.MolLogP(mol))
        encoded_smiles.append(encoded)
        logp_list.append(logp)

encoded_smiles = np.array(encoded_smiles)
logp_list = np.array(logp_list)

print(f"  ✓ Tokenized {len(encoded_smiles)} SMILES")
print(f"  ✓ Sequence length: {max_length}")
print(f"  ✓ Encoded shape: {encoded_smiles.shape}")

# ============================================================================
# TRANSFORMER MODEL
# ============================================================================
print("\n[2/6] Building Transformer model...")

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, num_layers=4, ff_dim=256):
        super(SMILESTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)  # [PAD] = 1
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, token_ids):
        # Embedding
        x = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        x = x + self.positional_encoding
        
        # Create padding mask
        pad_mask = (token_ids == 1)  # [PAD] token
        
        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask=pad_mask)
        
        # CLS token pooling (first token)
        cls_output = x[:, 0, :]  # (batch, embed_dim)
        
        # MLP head
        output = self.mlp(cls_output)  # (batch, 1)
        return output


# ============================================================================
# TRAINING
# ============================================================================
print("\n[3/6] Preparing data for training...")

# Convert to torch tensors
X_torch = torch.from_numpy(encoded_smiles).long()
y_torch = torch.from_numpy(logp_list).float().unsqueeze(1)

# Train/test split
train_idx, test_idx = train_test_split(
    range(len(encoded_smiles)), test_size=0.15, random_state=42
)

X_train = X_torch[train_idx].to(device)
y_train = y_torch[train_idx].to(device)
X_test = X_torch[test_idx].to(device)
y_test = y_torch[test_idx].cpu().numpy()

train_dataset = TensorDataset(X_train, y_train)
train_loader = TorchDataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

print("\n[4/6] Training Transformer model...")

model = SMILESTransformer(vocab_size=vocab_size, embed_dim=128, num_heads=8, 
                          num_layers=4, ff_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 40
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    
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
print("\n[5/6] Evaluating Transformer model...")

model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

y_pred = y_pred.flatten()

# Metrics
transformer_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
transformer_mape = mean_absolute_percentage_error(y_test, y_pred)

def calculate_success_at_threshold(y_true, y_pred, threshold=0.20):
    errors = np.abs(y_true - y_pred) / np.abs(y_true)
    return (errors <= threshold).sum() / len(errors) * 100

transformer_success = calculate_success_at_threshold(y_test, y_pred, threshold=0.20)

# ============================================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================================
print("\n[6/6] Running 5-fold cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx_fold, val_idx_fold) in enumerate(kf.split(encoded_smiles), 1):
    X_fold_train = X_torch[train_idx_fold].to(device)
    y_fold_train = y_torch[train_idx_fold].to(device)
    X_fold_val = X_torch[val_idx_fold].to(device)
    y_fold_val = y_torch[val_idx_fold].cpu().numpy()
    
    train_dataset_fold = TensorDataset(X_fold_train, y_fold_train)
    train_loader_fold = TorchDataLoader(train_dataset_fold, batch_size=16, shuffle=True)
    
    # Train
    model_fold = SMILESTransformer(vocab_size=vocab_size, embed_dim=128, num_heads=8,
                                   num_layers=4, ff_dim=256).to(device)
    optimizer_fold = optim.Adam(model_fold.parameters(), lr=0.001)
    
    for _ in range(40):
        for X_batch, y_batch in train_loader_fold:
            optimizer_fold.zero_grad()
            y_pred_batch = model_fold(X_batch)
            loss = criterion(y_pred_batch, y_batch)
            loss.backward()
            optimizer_fold.step()
    
    # Evaluate
    model_fold.eval()
    with torch.no_grad():
        y_pred_fold = model_fold(X_fold_val).cpu().numpy().flatten()
    
    fold_score = calculate_success_at_threshold(y_fold_val, y_pred_fold, threshold=0.20)
    cv_scores.append(fold_score)
    print(f"  Fold {fold}: {fold_score:.1f}%")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3 APPROACH 3: SMILES TRANSFORMER - FINAL RESULTS")
print("=" * 80)

results = {
    "approach": "SMILES Transformer",
    "architecture": {
        "tokenizer": "Character-based SMILES tokenizer",
        "vocab_size": vocab_size,
        "embedding_dim": 128,
        "transformer_heads": 8,
        "transformer_layers": 4,
        "feed_forward_dim": 256,
        "sequence_length": max_length,
        "pooling": "CLS token",
        "mlp_head": "64→32→1"
    },
    "dataset": {
        "total_molecules": len(encoded_smiles),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "sequence_length": max_length
    },
    "training": {
        "epochs": 40,
        "batch_size": 16,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "criterion": "MSELoss",
        "device": str(device)
    },
    "test_performance": {
        "rmse": float(transformer_rmse),
        "mape": float(transformer_mape),
        "success_at_20percent": float(transformer_success),
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
        "approach3_accuracy": float(transformer_success),
        "improvement_percentage_points": float(transformer_success - 69.3),
        "improvement_relative_percent": float((transformer_success - 69.3) / 69.3 * 100)
    }
}

print(f"\n▶ TEST SET PERFORMANCE")
print(f"  • Success@±20%: {transformer_success:.1f}%")
print(f"  • RMSE:         {transformer_rmse:.4f}")
print(f"  • MAPE:         {transformer_mape:.2f}%")

print(f"\n▶ 5-FOLD CROSS-VALIDATION")
print(f"  • Mean:     {cv_mean:.1f}%")
print(f"  • Std Dev:  ±{cv_std:.1f}%")
print(f"  • Range:    [{min(cv_scores):.1f}%, {max(cv_scores):.1f}%]")

print(f"\n▶ COMPARISON TO BASELINE (69.3%)")
print(f"  • Baseline (Handcrafted):      69.3%")
print(f"  • Approach 3 (SMILES Trans):   {transformer_success:.1f}%")
print(f"  • Improvement:                 {transformer_success - 69.3:+.1f} percentage points")
print(f"  • Relative improvement:        {(transformer_success - 69.3) / 69.3 * 100:+.1f}%")

print("\n" + "=" * 80)

# Save results
with open('phase3_approach3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved to phase3_approach3_results.json")
