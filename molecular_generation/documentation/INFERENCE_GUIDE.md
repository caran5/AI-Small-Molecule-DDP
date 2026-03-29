# Molecular Diffusion Model - Inference Guide

## Overview

You now have a complete inference pipeline to test your diffusion model on specific molecular properties. The system generates molecular features conditioned on target properties like LogP, molecular weight, H-bond donors/acceptors, and rotatable bonds.

---

## Quick Start

### Run Basic Inference
```bash
python simple_inference.py
```

This generates molecules with different property targets and shows statistics about the generated samples.

### Run Validation Pipeline
```bash
python validate_generation.py
```

Decodes generated features into molecular structures and validates atom composition.

---

## Generated Scripts

### 1. **simple_inference.py** ⭐ Start here
The main working inference script.

**Usage:**
```python
from simple_inference import generate_conditional
import torch

model = ConditionalUNet(...)  # Your model
properties = {'logp': 2.5, 'mw': 350, 'hbd': 2, 'hba': 4, 'rotatable': 5}

generated = generate_conditional(
    model=model,
    target_properties=properties,
    num_samples=5,
    num_steps=50,
    device='cuda'
)
# Output shape: (5, 128, 5) - 5 molecules, 128 atoms, 5 features each
```

**Output:**
- `(num_samples, 128, 5)` tensor
- Features: `[atomic_number, x, y, z, distance_from_com]`

### 2. **src/inference/decoder.py**
Converts generated features back to molecular structures.

**Usage:**
```python
from src.inference.decoder import MolecularDecoder

mol_dict = MolecularDecoder.features_to_molecule_dict(features)
print(mol_dict)
# Output: {
#   'atoms': [6, 7, 8, ...],  # atomic numbers
#   'coordinates': [[x1,y1,z1], ...],
#   'formula': 'C12H14NO2',
#   'valid': True,
#   'n_atoms': 27
# }
```

### 3. **validate_generation.py**
End-to-end validation showing:
- Generation with target properties
- Decoding to molecular structures
- Property analysis

---

## Data Format

**Input Features** (shape: `[batch_size, 128, 5]`):
```
Feature 0: Atomic number (normalized to [0, 1])
Feature 1: X coordinate (normalized to [-1, 1])
Feature 2: Y coordinate (normalized to [-1, 1])
Feature 3: Z coordinate (normalized to [-1, 1])
Feature 4: Distance from center of mass (normalized)
```

**Target Properties** (dict):
```python
{
    'logp': float,        # Lipophilicity (0-5)
    'mw': float,          # Molecular weight (100-500)
    'hbd': int,           # H-bond donors (0-5)
    'hba': int,           # H-bond acceptors (0-10)
    'rotatable': int      # Rotatable bonds (0-15)
}
```

---

## Next Steps to Improve

### ✅ What Works Now
- ✓ Generate features conditioned on properties
- ✓ Denormalize atomic numbers and coordinates
- ✓ Decode to molecular structure dicts
- ✓ Extract molecular formula

### ⚠️ Still Needed

#### 1. **Property Validation** (High Priority)
Your model should learn to match target properties. To verify this:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

# 1. Infer connectivity from coordinates
def build_molecule_graph(atoms, coords):
    # Use distance thresholds to determine bonds
    # TODO: Implement connectivity inference
    pass

# 2. Calculate actual properties
mol = build_molecule_graph(mol_dict['atoms'], mol_dict['coordinates'])
actual_logp = Descriptors.MolLogP(mol)
actual_mw = Descriptors.MolWt(mol)

print(f"Target LogP: {target['logp']}, Actual: {actual_logp}")
```

#### 2. **Load Trained Checkpoint** (If Available)
```python
checkpoint_path = "checkpoints/diffusion_model.pt"
model = ConditionalUNet(...)
model.load_state_dict(torch.load(checkpoint_path))
```

#### 3. **Improve Property Control**
Use the `GuidedGenerator` for stronger property steering:

```python
from src.inference.guided_sampling import GuidedGenerator, PropertyGuidanceRegressor

generator = GuidedGenerator(
    model=model,
    property_regressor=PropertyGuidanceRegressor(),
    guidance_scale=2.0  # Higher = stronger guidance
)

samples = generator.generate_guided(
    target_properties={'logp': 3.0, 'mw': 400},
    num_samples=10,
    num_steps=100
)
```

#### 4. **Train Property Matching**
Add property loss to training:

```python
def property_loss(generated_features, target_properties, property_regressor):
    pred_props = property_regressor(generated_features)
    return MSE(pred_props, target_properties)

# Combined loss during training
total_loss = diffusion_loss + lambda_props * property_loss
```

---

## File Structure

```
molecular_generation/
├── simple_inference.py              ← Main inference script
├── validate_generation.py           ← Validation pipeline
├── test_inference.py                ← Conditional generation tests
├── test_guided_inference.py         ← Property-guided generation tests
├── interactive_inference.py         ← Interactive CLI testing
│
├── src/
│   ├── inference/
│   │   ├── generate.py              ← ConditionalGenerationPipeline
│   │   ├── decoder.py               ← MolecularDecoder (NEW)
│   │   ├── guided_sampling.py       ← GuidedGenerator
│   │   └── ensemble.py
│   │
│   ├── models/
│   │   ├── unet.py                  ← ConditionalUNet ✓
│   │   ├── diffusion.py             ← NoiseScheduler
│   │   ├── trainer.py
│   │   └── embeddings.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   └── eval/
│       └── metrics.py
```

---

## Key Concepts

### Noise Scheduler
Controls the forward/reverse diffusion process:
- **Cosine schedule**: Smooth decay (recommended for molecules)
- **Linear schedule**: Constant decay rate
- **Quadratic schedule**: Accelerated decay

### Conditional Generation
Model receives both noisy features AND target properties:
```
Model input: (x_t, timestep_t, target_properties)
Model output: predicted_noise
```

### Property Normalization
Maps properties to [0, 1] range for neural network:
```
normalized = (value - min) / (max - min)
```

---

## Troubleshooting

### "Shape mismatch" errors
Ensure features are shape `(batch_size, 128, 5)`, not flattened.

### Generated molecules don't match properties
- Train longer with property loss
- Use stronger guidance_scale
- Improve property regressor training

### Out of memory
- Reduce num_steps (fewer denoising iterations)
- Reduce batch size
- Use CPU if GPU memory limited

---

## Example: Full Pipeline

```python
import torch
from simple_inference import generate_conditional, PropertyNormalizer
from src.models.unet import ConditionalUNet
from src.inference.decoder import MolecularDecoder

# 1. Setup
device = 'cuda'
model = ConditionalUNet(...).to(device)

# 2. Define target
target = {'logp': 2.5, 'mw': 350, 'hbd': 2, 'hba': 4, 'rotatable': 5}

# 3. Generate
features = generate_conditional(
    model=model,
    target_properties=target,
    num_samples=10,
    num_steps=100,
    device=device
)  # Shape: (10, 128, 5)

# 4. Decode & Validate
for i, feat in enumerate(features):
    mol_dict = MolecularDecoder.features_to_molecule_dict(feat)
    print(f"Molecule {i}: {mol_dict['formula']}")
    # TODO: Calculate actual properties and validate
```

---

## Performance Tips

| Goal | Setting |
|------|---------|
| **Fast testing** | `num_steps=30` |
| **Good quality** | `num_steps=50-100` |
| **Best quality** | `num_steps=1000` |
| **No guidance** | `properties=None` |
| **Mild guidance** | `guidance_scale=0.5-1.0` |
| **Strong guidance** | `guidance_scale=2.0-5.0` |

---

## References

- **Paper**: Diffusion Models Beat GANs on Image Synthesis (Ho et al., 2021)
- **Architecture**: U-Net with time conditioning and property embedding
- **Data**: Molecular features from ChemBL database
- **Task**: Conditional molecular generation with property control

---

For questions or improvements, check the generated scripts and adapt to your specific needs!
