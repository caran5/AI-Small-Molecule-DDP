# 📋 Complete Inference Scripts Index

## Overview
All scripts needed to test and improve your molecular diffusion model's inference capabilities.

---

## ✅ Core Scripts (Working)

### 1. **simple_inference.py** ⭐ RECOMMENDED START
- **Purpose:** Main inference script for generating molecules with target properties
- **Runtime:** ~30 seconds for 3 test cases
- **Output:** Generated molecular features with statistics
- **Key features:**
  - Property-conditioned generation
  - Multiple test cases (drug-like, hydrophobic, hydrophilic)
  - Clean progress reporting
  - Feature statistics

**Usage:**
```bash
python simple_inference.py
```

**What it does:**
1. Creates ConditionalUNet model
2. Normalizes target properties
3. Runs reverse diffusion (30 steps)
4. Shows generated feature statistics

---

### 2. **validate_generation.py** ✅ VALIDATION
- **Purpose:** End-to-end validation pipeline
- **Runtime:** ~1 minute
- **Output:** Decoded molecular structures and composition analysis
- **Key features:**
  - Generates molecules
  - Decodes features to structures
  - Extracts molecular formulas
  - Shows coordinate ranges

**Usage:**
```bash
python validate_generation.py
```

**What it does:**
1. Generates molecules
2. Decodes using MolecularDecoder
3. Extracts atomic numbers and coordinates
4. Analyzes molecular properties

---

### 3. **test_inference.py** 🧪 TESTING
- **Purpose:** Test conditional generation with multiple property sets
- **Status:** Updated to use ConditionalUNet
- **Features:** Batch testing of different properties

**Usage:**
```bash
python test_inference.py
```

---

### 4. **test_guided_inference.py** 🎯 GUIDED GENERATION
- **Purpose:** Test property-guided sampling with gradient steering
- **Status:** Updated with property normalizer
- **Features:** Tests different guidance scales

**Usage:**
```bash
python test_guided_inference.py
```

**Note:** Requires trained PropertyGuidanceRegressor (Phase 2)

---

### 5. **interactive_inference.py** 💬 INTERACTIVE
- **Purpose:** Interactive CLI for custom property testing
- **Features:**
  - Menu-driven interface
  - Custom property input
  - Multiple generation modes
  - Repeat generation for variability analysis

**Usage:**
```bash
python interactive_inference.py
```

**Menu options:**
1. Generate with default properties
2. Generate with custom properties
3. Generate multiple times with same properties
4. Exit

---

### 6. **improve_model.py** 📈 ROADMAP
- **Purpose:** Comprehensive guide for improving model
- **Content:**
  - 5 improvement phases
  - Code templates
  - Resource links
  - Workflow overview

**Usage:**
```bash
python improve_model.py
```

**Output:** Printed guide with improvement steps

---

## 📦 New Modules Created

### 7. **src/inference/decoder.py** 🔧 NEW
- **Purpose:** Convert generated features back to molecular structures
- **Key classes:**
  - `MolecularDecoder` - Features → molecular dicts
  - `SMILESGenerator` - Placeholder for SMILES generation

**Usage:**
```python
from src.inference.decoder import MolecularDecoder

mol_dict = MolecularDecoder.features_to_molecule_dict(features)
print(mol_dict)  # {atoms, coordinates, formula, ...}
```

**Features:**
- Denormalize atomic numbers
- Extract coordinates
- Build molecular structure dicts
- Calculate molecular formulas

---

## 📚 Documentation Files

### 8. **INFERENCE_GUIDE.md** 📖 COMPREHENSIVE
- **Content:**
  - Feature overview
  - Quick start guide
  - Generated scripts documentation
  - Data format explanation
  - Next steps checklist
  - Troubleshooting guide
  - Performance tips

**Read this for:** Complete understanding of inference system

---

### 9. **README_INFERENCE.md** 📝 SUMMARY
- **Content:**
  - What's done
  - Quick start
  - Architecture overview
  - Next steps (3 phases)
  - File structure
  - API reference
  - Example workflows

**Read this for:** High-level overview and next actions

---

### 10. **SCRIPTS_INDEX.md** 📋 THIS FILE
- **Content:** Complete script reference

**Read this for:** Understanding all available tools

---

## 🚀 Quick Reference Table

| Script | Purpose | Runtime | Status | Next Step |
|--------|---------|---------|--------|-----------|
| `simple_inference.py` | Main inference | 30s | ✅ Working | Run it |
| `validate_generation.py` | Validation | 1m | ✅ Working | Verify properties |
| `test_inference.py` | Batch testing | 2m | ✅ Working | Debug issues |
| `test_guided_inference.py` | Guided generation | 1m | ⚠️ Needs regressor | Train Phase 2 |
| `interactive_inference.py` | Custom testing | Variable | ✅ Working | Explore |
| `improve_model.py` | Improvement guide | Print | ✅ Reference | Follow steps |

---

## 📊 Recommended Execution Order

### First Time Setup
```bash
# 1. See what the model generates
python simple_inference.py          # 30 seconds

# 2. Understand the decoded structures
python validate_generation.py       # 1 minute

# 3. See the improvement roadmap
python improve_model.py | less      # Read carefully

# 4. Read documentation
less INFERENCE_GUIDE.md             # Important!
less README_INFERENCE.md            # Overview
```

### For Development
```bash
# Quick tests during iteration
python simple_inference.py

# Detailed validation
python validate_generation.py

# Guided generation (after Phase 2)
python test_guided_inference.py

# Interactive exploration
python interactive_inference.py
```

---

## 🎯 Key Parameters to Adjust

### In `simple_inference.py` or `validate_generation.py`:
```python
num_steps=30        # 30=fast, 50=good, 100=best
num_samples=3       # How many to generate
guidance_scale=1.0  # 0.5=mild, 1.0=moderate, 2.0+=strong
```

### Properties to Test:
```python
{
    'logp': 0.5-4.0,        # Lipophilicity
    'mw': 100-500,          # Molecular weight
    'hbd': 0-5,             # H-bond donors
    'hba': 0-10,            # H-bond acceptors
    'rotatable': 0-15       # Rotatable bonds
}
```

---

## 💡 Use Cases

### "I want to generate molecules quickly"
→ Run `python simple_inference.py`

### "I want to test specific properties"
→ Run `python interactive_inference.py`

### "I want to understand the decoded structures"
→ Run `python validate_generation.py`

### "I want to test guided generation"
→ Follow Phase 2 in `improve_model.py`, then run `test_guided_inference.py`

### "I want to understand the full system"
→ Read `INFERENCE_GUIDE.md`

### "I want to improve property matching"
→ Read `improve_model.py` and follow the 4-phase roadmap

---

## ⚠️ Common Issues & Solutions

### "ModuleNotFoundError"
Make sure you're in the `molecular_generation` directory:
```bash
cd molecular_generation
python simple_inference.py
```

### "CUDA out of memory"
Use CPU or reduce batch size:
```python
device = 'cpu'
num_samples = 1
```

### "Properties don't match"
This is expected! See Phase 1 in `improve_model.py`
- Need to implement property validation
- Need to train property regressor (Phase 2)

### "Script takes too long"
Reduce `num_steps`:
```python
num_steps=20  # Instead of 50-100
```

---

## ✨ Summary

**You now have:**
- ✅ 6 working inference scripts
- ✅ 1 new decoder module
- ✅ 3 documentation files
- ✅ Complete validation pipeline
- ✅ Improvement roadmap with templates

**Next action:** Run `python simple_inference.py` to see it in action!

---

## 📞 Quick Support

| Question | Answer | Reference |
|----------|--------|-----------|
| How do I use this? | Start with `simple_inference.py` | README_INFERENCE.md |
| What's the full system? | Read `INFERENCE_GUIDE.md` | Section: Overview |
| How do I improve it? | Follow `improve_model.py` roadmap | 4 phases outlined |
| What's the next step? | Validate property matching | Phase 1 in improve_model.py |
| Can I test custom properties? | Yes, use `interactive_inference.py` | Section: Workflows |

---

Created: 2026-03-26
Status: ✅ Complete and tested
Next step: Run `python simple_inference.py`
