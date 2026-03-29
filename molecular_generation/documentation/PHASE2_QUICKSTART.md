# Phase 2: Quick Start Guide

## Setup (5 minutes)

### 1. Train PropertyGuidanceRegressor

```python
from src.inference.guided_sampling import TrainableGuidance
from src.data.loader import ConditionalMoleculeDataLoader

# Create dataloader from your training data
loader = ConditionalMoleculeDataLoader(features, properties)
train, val = random_split(loader, [0.8, 0.2])

# Train regressor
trainer = TrainableGuidance(device='cuda')
history = trainer.train(train, val, epochs=50)
trainer.save('models/regressor.pt')

# ~5 minutes on GPU, ~30 minutes on CPU
```

### 2. Load for Inference

```python
import torch
from src.models.unet import ConditionalUNet
from src.inference.guided_sampling import GuidedGenerator, PropertyGuidanceRegressor
from src.data.preprocessing import PropertyNormalizer

# Load components
model = ConditionalUNet.load('checkpoint.pt')
regressor = PropertyGuidanceRegressor()
regressor.load_state_dict(torch.load('models/regressor.pt'))
normalizer = PropertyNormalizer.load('normalizer.pkl')

# Create generator
generator = GuidedGenerator(model, regressor, normalizer, device='cuda')
```

---

## Guided Sampling (1 minute)

```python
# Define target
target = {'logp': 3.5, 'mw': 400, 'hbd': 2, 'hba': 5, 'rotatable': 4}

# Generate with guidance
samples = generator.generate_guided(
    target,
    num_samples=100,
    num_steps=50,
    noise_schedule='cosine'
)

# Decode and use
smiles = decode_features(samples)
```

**Performance**: 10 seconds for 100 molecules

---

## Energy Filtering (5 minutes)

```python
from src.filtering.energy_filter import ConformationFilter

# Create filter
filter_obj = ConformationFilter(energy_threshold=100.0)

# Filter SMILES
filtered, results = filter_obj.filter_smiles(smiles, verbose=True)

print(f"Passed: {len(filtered)}/{len(smiles)}")
```

**Performance**: 8-15 minutes for 1000 molecules

---

## Full Workflow (15 minutes)

```python
from scripts.generate_candidates import generate_with_energy_filtering

result = generate_with_energy_filtering(
    ensemble_or_generator=generator,
    target_properties=target,
    energy_threshold=100.0,
    use_guided=True,
    num_samples=100
)

# Results
candidates = result['filtered']
properties = result['properties']
fidelity = result['fidelity']

print(f"✓ {len(candidates)} candidates generated")
print(f"✓ Fidelity: {fidelity['overall_mse']:.4f}")
```

---

## Key Parameters

### Guided Sampling

- `guidance_scale`: 0-10
  - 0 = no guidance
  - 1 = mild (default)
  - 5 = strong
  - 10+ = very strong

- `num_steps`: 1-100
  - 10 = very fast, lower quality
  - 50 = balanced (default)
  - 100 = slow, highest quality

### Energy Filtering

- `energy_threshold`: 50-150 kcal/mol
  - 50 = very strict (most filtering)
  - 100 = balanced (default)
  - 150 = lenient (least filtering)

---

## Decision Guide

**Just need to generate fast?**
→ Use Guided Sampling (10 sec)

**Need to validate structures?**
→ Use Energy Filtering (8 min)

**Want the best molecules?**
→ Use Both (13 min)

**Need uncertainty?**
→ Use Phase 1 Ensemble

---

## Troubleshooting

**Guided sampling not converging?**
```python
generator.set_guidance_scale(5.0)  # Increase from 1.0
```

**Energy filter rejecting too many?**
```python
filter_obj.set_energy_threshold(120.0)  # Relax from 100
```

**Regressor not training?**
```python
history = trainer.train(train, val, epochs=100, learning_rate=1e-4)
```

---

**For full documentation, see PHASE2_README.md**
