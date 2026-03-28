# Claude Code Guidelines for Molecular Diffusion Model

## Project Overview

This is a **molecular diffusion model** written in Python that generates drug-like molecules using a denoising diffusion process. The core domains include:
- **Molecular generation**: Using diffusion to generate valid molecular structures
- **Chemical validation**: Bond inference, valence checks, RDKit sanitization
- **Tensor-based diffusion sampling**: Iterative denoising with guided generation

### Key Project Structure

```
molecular_generation/
  ├── requirements.txt              # Dependencies (torch, rdkit, numpy, etc)
  ├── src/
  │   ├── models/                   # Core model definitions
  │   │   ├── diffusion_model.py   # Main diffusion model
  │   │   ├── embeddings.py        # Time/property embeddings
  │   │   └── decoder.py           # Structure decoding (critical for validation)
  │   ├── utils/
  │   │   ├── molecule_utils.py    # RDKit integration, SMILES handling
  │   │   └── validation.py        # Chemical validity checks
  │   └── data/
  │       └── datasets.py          # Data loading pipeline
  ├── tests/                        # Unit tests for all components
  ├── scripts/                      # Training and evaluation scripts
  └── interactive_inference.py      # Canonical inference entry point
```

### Dependency Management

**Always use this path for dependencies:**
- `molecular_generation/requirements.txt` — Contains torch, rdkit, numpy, pandas, and all ML dependencies

**Before installing dependencies:**
1. Check `molecular_generation/requirements.txt` for the exact file path
2. Verify the file exists before running pip install
3. Never assume requirements.txt is in the project root

---

## Python / ML Conventions

### Tensor Shape Validation (CRITICAL)

When writing inference or generation scripts:

1. **Print tensor shapes at every major step**:
   - After model forward pass
   - After any reshape/reshape operations
   - Before and after decoding

2. **Include explicit assertions for expected tensor dimensions**:
   ```python
   output = model(x)
   assert output.shape == (batch_size, n_atoms, n_features), f"Expected shape (...), got {output.shape}"
   ```

3. **Verify tensor shapes match decoder expectations** before passing to structure decoding:
   - The decoder expects specific input shapes for atom positions and bonds
   - Always check model output format against decoder input signature

4. **Test with a minimal example before presenting code**:
   - Create a dummy batch with correct input shapes
   - Run the inference script end-to-end
   - Fix any shape mismatches or runtime errors before submitting

**Why**: Previous inference scripts repeatedly failed with identical tensor shape mismatch bugs. Shape assertions catch these immediately rather than requiring iterative debugging cycles.

### Before Writing Inference Code

1. **Always read the model's forward pass first** to understand:
   - Input tensor shapes expected by the model
   - Output tensor shapes produced by the model
   - The decoder's input/output format

2. **Document expected shapes**:
   ```
   Model input: (batch_size, n_atoms, atom_features)
   Model output: (batch_size, n_atoms, n_features)
   Decoder input: (batch_size, n_atoms, n_features)
   Decoder output: SMILES strings
   ```

3. **Match the code to real tensor dimensions**, not assumptions

**Why**: Prevents the recurring bug of inference code assuming wrong dimensions.

### Molecular Structure Validation

When decoding model outputs to molecular structures:

1. Include bond inference logic that handles:
   - Single, double, triple bonds
   - Aromatic bonds
   - Chemical validity (valence checks, Kekulization)

2. Use RDKit for sanitization:
   ```python
   from rdkit import Chem
   mol = Chem.MolFromSmiles(smiles)
   if mol is not None:
       Chem.SanitizeMol(mol)  # Validates chemical structure
   ```

3. Log rejected molecules and why they failed validation for debugging

**Why**: Valid chemical structures require more than just graph decoding—bond inference and RDKit validation are non-negotiable.

---

## Environment Setup

When setting up the environment:

1. **Locate the requirements file** — it's in `molecular_generation/requirements.txt`, not the project root
2. Navigate to the correct directory before installing
3. Verify torch and rdkit install correctly (these are the heaviest dependencies)

```bash
cd molecular_generation
pip install -r requirements.txt
```

---

## Code Quality & Review

- Always run generated inference scripts with a small test input before presenting them as complete
- Shape validation is non-negotiable for ML code
- One canonical inference script is better than multiple scripts with duplicate bugs
- When debugging shape mismatches, print intermediate shapes rather than guessing

---

## Testing & Validation

- Unit tests are in `tests/` — run with pytest to validate changes
- Integration tests check end-to-end diffusion → decoding → validation pipeline
- When adding new decoder logic, include tests that validate:
  - Bond inference correctness
  - Chemical validity of output molecules
  - Tensor shape integrity through the decoding process
