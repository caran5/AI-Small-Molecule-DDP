#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/ceejayarana/diffusion_model/molecular_generation/src')

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json

# Test imports
try:
    from data.loader import ChEMBLDataLoader
    print("✅ ChEMBLDataLoader imported")
except Exception as e:
    print(f"❌ ChEMBLDataLoader import failed: {e}")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    print("✅ RDKit imported")
except Exception as e:
    print(f"❌ RDKit import failed: {e}")

# Simple test
print("\n" + "="*70)
print("PHASE 2 REBUILD: ChEMBL Data + Non-Linear Models")
print("="*70)

print("\n1. Testing ChEMBL data loading...")
try:
    loader = ChEMBLDataLoader(limit=100)
    molecules = loader.load_molecules()
    print(f"✅ Loaded {len(molecules)} molecules")
    
    if len(molecules) > 0:
        print(f"   First molec        print(f"   First molec        print(f"   First molec        print(f"   First molec        pport traceback
    traceb    trint_exc()

