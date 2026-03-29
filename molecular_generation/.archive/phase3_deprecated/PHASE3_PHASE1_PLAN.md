# PHASE 3 PHASE 1: RDKit Descriptors Enhancement

## Objective
Test if adding 30 RDKit descriptors (beyond the original 9 topology features) can jump accuracy from **50%** to **55-65%**.

## Experiment Design

### Feature Set Comparison

**Test 1: Original 9 Features (Phase 2 Baseline)**
- `NumAtoms` - Total atoms in molecule
- `NumHeavyAtoms` - Non-hydrogen atoms
- `NumRings` - Ring count
- `NumAromaticRings` - Aromatic rings
- `NumHeteroatoms` - Heteroatoms (N, O, S, P, etc.)
- `NumHDonors` - Hydrogen bond donors
- `NumHAcceptors` - Hydrogen bond acceptors
- `NumRotatableBonds` - Flexibility/rotatable bonds
- `TPSA` - Topological polar surface area

**Test 2: Enhanced 39 Features (9 + 30 RDKit)**
All 9 above + 30 additional RDKit descriptors:
- Molecular Weight
- Fraction sp3 carbons
- Aromatic atoms count
- Aliphatic ring count
- Ring complexity (BertzCT)
- Connectivity indices (Chi0)
- Labute surface area
- Heteroatom cycles
- Shape descriptors (Kappa1, Kappa2, Kappa3)
- Asphericity, Eccentricity
- Inertial shape factor
- Radius of gyration
- Lipinski rules descriptors
- And others...

### Model & Data
- **Algorithm**: Linear Regression (non-circular, proven winner at 50.7%)
- **Data**: 500 ChEMBL molecules
- **Train/Test Split**: 70% / 15% (validation) / 15% (test)
- **Target**: LogP (lipophilicity) prediction
- **Metric**: Success@±20% (predicted LogP within ±20% of actual LogP)

## Expected Outcomes

### If RDKit descriptors help
- Accuracy: 50% → **55-65%** (+5-15% absolute)
- Reason: Chemical descriptors (aromaticity, lipophilicity, complexity) correlate with LogP
- Decision: Proceed to Phase 3 Phase 2 (feature selection via correlation)

### If RDKit descriptors don't help
- Accuracy: ~50% (no change)
- Reason: Features still don't capture LogP prediction signal
- Decision: Investigate why; possibly need different approach (molecular fingerprints, 3D structures)

## Roadmap to 90%

```
Phase 3 Phase 1 (THIS): 50% → 55-65%  [Better features]
Phase 3 Phase 2:         55-65% → 70-75%  [Feature selection + correlation]
Phase 3 Phase 3:         70-75% → 85-90%  [Ensemble: Linear 40% + RF 60%]
```

## Key Hypothesis
**The problem is not the model, it's the features.** Adding chemistry-aware descriptors (RDKit) should unlock the signal that Linear regression can then capture. If not, we need to reconsider feature engineering approach (e.g., molecular fingerprints, 3D spatial features, or graph neural networks).

## Status
- ✅ Script created: [phase3_phase1_rdkit_descriptors.py](phase3_phase1_rdkit_descriptors.py)
- 🟡 Execution in progress (extracting 4.5GB ChEMBL tar file)
- 📊 Results will be saved to `phase3_phase1_results.json`
