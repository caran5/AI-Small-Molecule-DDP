# WHAT TO DO RIGHT NOW

Your system just failed the critical test. That's actually good news - you discovered it before production.

---

## What Just Happened

The validation framework you just ran shows:

```
TEST: Generate molecules with LogP=3.0 (normal drug target)
RESULT: 10% success rate (should be >70%)

Interpretation: Guidance is NOT working effectively
```

This is the **real gap** between "code works" and "shipping works."

---

## Why This Happened

The mock test framework revealed the core problem: **You haven't connected real guidance to the diffusion sampling yet.**

Here's what's missing:

```python
# What you have:
✅ regressor = PropertyGuidanceRegressor()
✅ regressor is trained (0.77x ratio)

# What you don't have:
❌ Actual gradient computation during sampling
❌ Integration into diffusion.py sample_guidance()
❌ Proof that regressor gradients steer the diffusion
```

---

## The Three Real Problems

### Problem 1: Regressor Gradients Aren't Connected
```python
# Currently, guided sampling probably looks like:
def sample_guidance(self, target_properties, ...):
    x_t = initial_noise
    for step in reversed(self.timesteps):
        # Do diffusion step
        x_t = self.denoise(x_t, step)
        # But NO gradient computation through regressor!
    return x_t

# What it needs to be:
def sample_guidance(self, regressor, target_properties, ...):
    x_t = initial_noise
    x_t.requires_grad = True  # <-- This is missing
    for step in reversed(self.timesteps):
        # Do diffusion step
        x_t = self.denoise(x_t, step)
        
        # Add guidance using regressor gradient <-- This is missing
        with torch.enable_grad():
            pred_props = regressor(x_t)
            loss = MSE(pred_props, target_properties)
            grad = torch.autograd.grad(loss, x_t)[0]
            x_t = x_t - guidance_scale * grad  # <-- This is missing
    return x_t
```

### Problem 2: Your Test Shows It's Not Working
```
Actual success rate: 10%
Expected success rate: >70%

That's a 7x gap. Your guidance is effectively random.
```

### Problem 3: You're About to Deploy Something Broken
If you'd deployed this, users would generate molecules that **don't match targets** and assume the system is broken. It is - but only the guidance part, not the foundation.

---

## The Actual Fix (In Order)

### Step 1: Verify Regressor Works Standalone
```python
# Test: Does the regressor predict properties correctly?
def test_regressor_accuracy():
    regressor = load('checkpoints/property_regressor_improved.pt')
    
    # Create test features
    test_features = torch.randn(100, 100)
    
    # Predict
    predictions = regressor(test_features)
    
    # Are predictions reasonable?
    assert predictions.shape == (100, 5)
    assert predictions.min() > -10 and predictions.max() < 10
    
    print(f"✓ Regressor works: predictions shape {predictions.shape}")
```

**Expected output**: ✓ Regressor works

### Step 2: Verify Gradients Flow Through Regressor
```python
# Test: Can we backprop through the regressor?
def test_gradient_flow():
    regressor = load('checkpoints/property_regressor_improved.pt')
    regressor.eval()
    
    # Create input that needs gradients
    x = torch.randn(10, 100, requires_grad=True)
    
    # Forward pass
    pred = regressor(x)
    
    # Create loss
    target = torch.randn(10, 5)
    loss = torch.nn.functional.mse_loss(pred, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert x.grad.abs().max() > 0
    
    print(f"✓ Gradients flow: max gradient = {x.grad.abs().max().item():.4f}")
```

**Expected output**: ✓ Gradients flow: max gradient = X.XXXX

### Step 3: Add Gradient-Based Guidance to Diffusion Sampler
```python
# In src/inference/guided_sampling.py (or create if missing):

class GuidedDiffusionSampler:
    def sample_guided(
        self,
        regressor,
        target_properties: Dict,
        guidance_scale: float = 1.0,
        num_steps: int = 50,
    ):
        """Sample with property guidance using regressor gradients."""
        
        x_t = torch.randn(1, latent_dim)  # Initial noise
        x_t.requires_grad_(True)
        
        for step_idx, t in enumerate(reversed(self.timesteps)):
            # 1. Denoise step
            with torch.no_grad():  # Don't track gradients for denoising
                x_t_denoised = self.model.denoise(x_t, t)
            
            x_t = x_t_denoised.clone().requires_grad_(True)
            
            # 2. Compute property guidance
            with torch.enable_grad():
                pred_props = regressor(x_t)
                
                # Loss toward target properties
                loss = 0
                for prop_name, target_val in target_properties.items():
                    # Map property names to indices (depends on your regressor)
                    prop_idx = {'logp': 0, 'mw': 1, 'hbd': 2, 'hba': 3, 'rotatable': 4}[prop_name]
                    loss += (pred_props[0, prop_idx] - target_val) ** 2
                
                # Compute gradients
                grad = torch.autograd.grad(loss, x_t, create_graph=False)[0]
            
            # 3. Move toward target
            x_t = x_t_denoised - guidance_scale * grad
        
        return x_t
```

### Step 4: Test Integration End-to-End
```python
# Create test: Does the integrated guidance work?
def test_end_to_end_guidance():
    regressor = load('checkpoints/property_regressor_improved.pt')
    sampler = GuidedDiffusionSampler()
    decoder = MolecularDecoder()
    
    # Target properties
    target = {'logp': 3.5, 'mw': 350, 'hbd': 2}
    
    # Sample with guidance
    features = sampler.sample_guided(regressor, target, num_steps=50)
    
    # Decode to molecule
    smiles = decoder.decode(features)
    
    # Compute actual properties
    actual_props = compute_properties_rdkit(smiles)
    
    # Check error
    logp_error = abs(actual_props['logp'] - target['logp'])
    
    if logp_error < 0.5:
        print(f"✅ SUCCESS: Generated molecule with LogP error {logp_error:.2f}")
    else:
        print(f"❌ FAILED: Generated molecule with LogP error {logp_error:.2f}")
        print(f"  Target: {target}")
        print(f"  Actual: {actual_props}")
```

---

## The Real Timeline Now

### Today
- [ ] Run test framework (shows guidance failing)
- [ ] Understand the gap

### Tomorrow (Day 1)
- [ ] Test regressor works
- [ ] Test gradients flow
- [ ] Identify where guidance integration is missing

### Day 2-3
- [ ] Implement gradient-based guidance in sampler
- [ ] Add to guided_sampling.py

### Day 4
- [ ] Test end-to-end
- [ ] Measure success rates
- [ ] Document what works/fails

### Day 5
- [ ] Fix edge cases from tests
- [ ] Run full validation suite

### Week 2
- [ ] Test on real molecules
- [ ] Measure performance
- [ ] Document limitations

### Weeks 3-4
- [ ] Production hardening
- [ ] Deployment runbook
- [ ] Go live

---

## The Honest Status Now

```
✅ Foundation: Solid
✅ Architecture: Sound  
❌ Guidance: Not working yet
⚠️  Tests: Failing (10% success rate)

Next: Implement gradient-based guidance integration
Timeline: 3-5 days to get guidance working
Then: 1-2 weeks to production-ready
```

---

## Bottom Line

You're not broken. You just found the real work that was hidden.

**What you have**: Good foundation
**What you need**: Connect the regressor to the sampler via gradients
**Estimated work**: 3-5 days to get it working, 2 weeks to production-ready

The test framework just saved you from shipping something that looked good but didn't work.

That's progress.

