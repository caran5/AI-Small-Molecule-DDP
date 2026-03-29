# PHASE 2: REDEFINED
## Property-Guided Molecular Generation (Not Just Prediction)

**Date**: Current Session  
**Status**: 🔄 TRANSITIONING  
**Previous Finding**: LogP prediction from structural features is ~50% accurate (hard problem, not bad model)

---

## Problem with Original Phase 2

Original Phase 2: "Train regressor to predict LogP from structural features"
- ❌ This was **backwards** thinking
- ❌ The regressor was being tested in isolation (not integrated with diffusion)
- ❌ We tested property prediction, NOT property-guided generation
- ❌ The real test should be: **"Can we generate molecules with target properties?"**

---

## NEW Phase 2: Property-Guided Generation

### What We Actually Want to Test

**The Real Question**: "Does the diffusion model + guidance regressor generate molecules with the target properties?"

**Concrete Test**:
1. Start with a random noise molecule
2. Use diffusion model + LogP guidance to steer generation
3. Sample final molecule
4. Compute ACTUAL LogP of generated molecule (using RDKit)
5. Compare to TARGETED LogP
6. Success = generated LogP matches target ±20%

### Why This is Different

| Aspect | Old Phase 2 | New Phase 2 |
|--------|-----------|-----------|
| **Task** | Predict LogP from features | Generate molecules with target LogP |
| **Test Input** | Fixed features of known molecules | Random noise / random molecules |
| **Test Output** | Predicted LogP value | Generated molecule + computed LogP |
| **Validation** | Prediction ≈ actual LogP | Generated molecule has target properties |
| **What Breaks** | Bad regressor | (1) Bad regressor, (2) Broken diffusion coupling, (3) Guidance too weak |
| **Success Metric** | Accuracy of prediction | % of generations matching target ±20% |

---

## Implementation Plan

### Phase 2a: Establish Guidance Model ✅ DONE
- Train regressor on structural features → LogP
- Result: Linear ~51%, MLPDeep ~41% on ±20% metric
- Finding: **This is actually honest/reasonable** (LogP is hard to predict)
- Decision: Use MLPDeep (competitive with Linear, learned nonlinearity)

### Phase 2b: Generate with Guidance (NEW)
- Load trained MLPDeep regressor
- Load trained diffusion model (from Phase 1)
- For target property `LogP = X`:
  1. Start: Random noise
  2. Denoise step 1000 → 0
  3. At each step, compute guidance loss from regressor
  4. Update noise to maximize guidance
  5. Sample final molecule
- Repeat 50 times per target

### Phase 2c: Validate Generation Quality
- For 10 target LogP values (sampled from -2 to +8):
  - Generate 50 molecules each
  - Extract LogP from each (using RDKit)
  - Compute % within ±20% of target
  - Success = 80%+ of generations hit target

- For each generated molecule:
  - Validity: Can it be parsed by RDKit? (should be 100%)
  - Novelty: Is it in training data? (should be <10%)
  - Diversity: Are all 50 unique? (should be >95% unique)

---

## Success Criteria

### Minimum (Phase 2 Passes)
- ✅ 10 target values tested
- ✅ 50 molecules generated per target
- ✅ ≥70% of generations within ±20% of target LogP
- ✅ ≥90% validity (parseable by RDKit)
- ✅ ≥80% novelty (not in training set)

### Strong (Phase 2 Strong)
- ✅ ≥80% of generations within ±20%
- ✅ ≥95% validity
- ✅ ≥95% novelty
- ✅ ≥90% diversity (unique molecules)

### Exceptional (Phase 2 Excellence)
- ✅ ≥85% of generations within ±20%
- ✅ ≥98% validity
- ✅ ≥99% novelty
- ✅ ≥95% diversity

---

## Metrics to Report

For each target LogP value:
```
Target LogP: 5.0
Generated 50 molecules

Success:  75% (37/50 within ±20%)
Error:    Mean ±1.2, Median ±0.8
Validity: 98% (49/50 parseable)
Novelty:  94% (47/50 not in training)
Diversity: 98% (49/50 unique)

Sampled molecules:
  1. C1CCCCC1 (LogP=3.18, error=-1.82)  ✗ Miss
  2. CCc1ccccc1Cl (LogP=4.95, error=-0.05)  ✓ Hit
  3. ...
```

---

## Timeline

| Phase | Task | Time | Blocker |
|-------|------|------|---------|
| **2a** | ✅ Train guidance model | DONE | None |
| **2b** | Implement guidance in diffusion | 2-3 hrs | Need diffusion + regressor integration |
| **2c** | Generate & validate | 1-2 hrs | Depends on 2b |
| **Total** | Phase 2 redefined + implemented | 3-5 hrs | None (critical path) |

---

## Rationale for Redefinition

**Original Phase 2 was flawed because**:
1. Tested regressor in isolation (not how it's used)
2. Tested on known molecules (not realistic)
3. Success metric was prediction accuracy (not generation quality)
4. Didn't actually test the diffusion+guidance coupling

**New Phase 2 tests the REAL capability**:
1. Regressor embedded in diffusion loop
2. Tested on generation from scratch (realistic)
3. Success metric is property match in generated molecules (real goal)
4. Tests full pipeline: diffusion + guidance + molecular validity

---

## Decision: Proceed with Phase 2 Redefinition

**User Intent**: "Do option 1 then redefine phase"
- ✅ Option 1: Fixed Phase 2 with non-circular features → MLPDeep is competitive
- ✅ Redefine: Phase 2 is property-guided generation, not just prediction

**Next Action**: Implement Phase 2b (guidance integration in diffusion loop)

---
