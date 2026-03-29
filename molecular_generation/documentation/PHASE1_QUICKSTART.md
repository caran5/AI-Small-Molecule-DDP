# Phase 1 Quick Start Guide

## TL;DR - Get Started in 5 Minutes

### 1. Install & Setup
```bash
cd /Users/ceejayarana/diffusion_model/molecular_generation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### 2. Run Integration Tests
```bash
python tests/test_phase1.py
```

Expected output:
```
✓ ALL TESTS PASSED
Phase 1 components are ready for production use:
  ✓ Conditional generation with property steering
  ✓ Property normalization and dataloader
  ✓ Comprehensive evaluation metrics
  ✓ Ensemble inference with uncertainty
  ✓ Full drug candidate generation pipeline
```

### 3. Basic Usage

#### Generate Molecules with Target Properties
```python
from src.models.unet import ConditionalUNet
from src.inference.generate import generate_with_properties

# Load model
model = ConditionalUNet(input_dim=100, n_properties=5)

# Target properties
target = {'logp': 3.5, 'mw': 400, 'hbd': 2, 'hba': 4, 'rotatable': 5}

# Generate
samples = generate_with_properties(model, target, num_samples=50)
# → Tensor of shape [50, 100]
```

#### Check Generated Molecule Quality
```python
from src.eval.metrics import chemical_validity, diversity_metric

validity = chemical_validity(smiles_list)
diversity = diversity_metric(features)

print(f"Validity: {validity['validity']:.1%}")
print(f"Diversity: {diversity:.3f}")
```

#### Use Ensemble for Uncertainty
```python
from src.inference.ensemble import EnsembleModel

ensemble = EnsembleModel(['model_0.pt', 'model_1.pt', 'model_2.pt'])
results = ensemble.generate(target, num_samples=100)
filtered, conf, mask = ensemble.filter_by_confidence(results, threshold=0.8)
print(f"Kept {mask.sum()}/{len(mask)} high-confidence samples")
```

---

## Key Components

### 1. ConditionalUNet
**What it does**: Conditions diffusion on molecular properties
```python
from src.models.unet import ConditionalUNet

model = ConditionalUNet(
    in_channels=100,
    out_channels=100,
    hidden_channels=128,
    n_properties=5,  # logp, mw, hbd, hba, rotatable
    depth=3
)
```

### 2. PropertyNormalizer
**What it does**: Robust normalization for properties
```python
from src.data.preprocessing import PropertyNormalizer

norm = PropertyNormalizer()
norm.fit({'logp': [1, 2, 3], 'mw': [300, 400, 500]})
normalized = norm.normalize({'logp': 2.5, 'mw': 400})
```

### 3. ConditionalMoleculeDataLoader
**What it does**: Batches features with properties
```python
from src.data.loader import ConditionalMoleculeDataLoader

loader = ConditionalMoleculeDataLoader(features, properties, batch_size=32)
for batch_features, batch_properties in loader:
    print(batch_features.shape, batch_properties.shape)  # [32, 100], [32, 5]
```

### 4. Metrics
**What it does**: Evaluate generation quality
```python
from src.eval.metrics import (
    chemical_validity,      # % valid SMILES
    diversity_metric,       # Pairwise distance
    property_fidelity,      # Error vs target
    distribution_distance,  # MMD/Wasserstein
    compute_all_metrics     # All at once
)
```

### 5. Ensemble
**What it does**: Uncertainty quantification
```python
from src.inference.ensemble import EnsembleModel

ensemble = EnsembleModel(checkpoint_paths)
results = ensemble.generate(target_props)  # mean + std
filtered, conf, mask = ensemble.filter_by_confidence(results)
```

---

## Common Workflows

### Workflow A: Train & Evaluate Single Model
```python
from scripts.train_conditional import train_conditional_model
from src.eval.metrics import compute_all_metrics

# Train
model, history = train_conditional_model(
    train_loader, val_loader, epochs=20
)

# Evaluate
metrics = compute_all_metrics(
    generated_smiles,
    generated_features,
    training_features,
    target_properties
)
print(f"Validity: {metrics['validity']:.1%}")
print(f"Fidelity MSE: {metrics['fidelity_mse']:.4f}")
```

### Workflow B: Train Ensemble for Production
```python
from src.inference.ensemble import train_ensemble

checkpoints, metrics = train_ensemble(
    train_loader, val_loader,
    n_models=3,
    epochs=20
)
print(f"Saved 3 models to checkpoints/ensemble/")
```

### Workflow C: Generate Drug Candidates
```python
from scripts.generate_candidates import (
    generate_drug_candidates,
    rank_candidates,
    print_candidates
)

results = generate_drug_candidates(
    ensemble,
    target_properties={'logp': 3.5, 'mw': 400, ...},
    num_candidates=200
)

ranked = rank_candidates(results)
print_candidates(ranked, top_n=10)
```

---

## Metrics Interpretation

### ✅ Healthy Model
- Validity: **>85%**
- Diversity: **0.4–0.6**
- Fidelity MSE: **<0.1**
- MMD: **0.1–0.3**

### ⚠️ Warning Signs
- Validity drops to **70–85%** → overfitting
- Diversity <**0.2** → mode collapse
- Fidelity MSE >**0.5** → property conditioning not working
- MMD <**0.05** → memorizing training set

### ❌ Critical Issues
- Validity <**70%** → model severely degraded
- Diversity <**0.1** → only generating same molecule
- Fidelity MSE >**1.0** → complete failure

---

## Troubleshooting

**Q: Generated molecules have low validity?**
- A: Check property normalizer is fitted. Verify properties in training data are realistic.

**Q: High ensemble std (uncertainty)?**
- A: Train more models (5 instead of 3) or increase training epochs.

**Q: Generation is slow?**
- A: Reduce `num_steps` from 100 to 50. Use GPU. Parallelize across ensemble.

**Q: Models not learning properties?**
- A: Property values might be too similar. Add more property diversity in training data.

---

## Performance Tips

### For Speed
- Use 50 diffusion steps instead of 100 (2× faster, slightly lower quality)
- Enable GPU: `device='cuda'`
- Parallelize ensemble inference

### For Quality
- Train 5 models instead of 3 (better uncertainty)
- Increase training epochs to 30
- Use larger hidden dimension (256 instead of 128)

### For Memory
- Reduce batch size (16 instead of 32)
- Use lower resolution features (64-d instead of 100-d)
- Inference only: don't keep full gradients

---

## File Locations

| Component | File |
|-----------|------|
| Model architecture | `src/models/unet.py` |
| Data loading | `src/data/loader.py` |
| Preprocessing | `src/data/preprocessing.py` |
| Metrics | `src/eval/metrics.py` |
| Generation | `src/inference/generate.py` |
| Ensemble | `src/inference/ensemble.py` |
| Training | `scripts/train_conditional.py` |
| Candidates | `scripts/generate_candidates.py` |
| Tests | `tests/test_phase1.py` |

---

## Next: Phase 2

Phase 2 features (coming next):
- Guided sampling for directed exploration
- Energy filtering for chemical realism
- FastAPI for production deployment
- Model distillation for speed

---

**Status**: ✅ Phase 1 complete and tested. Ready for real-world deployment.

