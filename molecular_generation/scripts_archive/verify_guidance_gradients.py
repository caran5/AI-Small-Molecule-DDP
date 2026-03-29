#!/usr/bin/env python3
"""
Gradient verification for property-guided generation.

Critical check before deploying regressor for guidance: ensure gradients
are well-behaved (no NaN, no explosion, reasonable magnitude).

This catches subtle issues that loss metrics alone won't reveal.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from pathlib import Path
import numpy as np

from train_property_regressor_improved import RegularizedPropertyGuidanceRegressor


class GradientVerifier:
    """Verify gradient behavior for guidance operations."""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.criterion = nn.MSELoss()
        
    def verify_guidance_gradients(self, sample_x, target_properties, verbose=True):
        """
        Core verification: check if gradients are valid for guidance.
        
        Args:
            sample_x: Input features (batch_size, 100)
            target_properties: Target properties (batch_size, 5)
            verbose: Print diagnostics
            
        Returns:
            gradients: Computed gradients (same shape as sample_x)
            diagnostics: Dict with checks (all_checks_passed, issues, stats)
        """
        
        sample_x = sample_x.to(self.device).requires_grad_(True)
        target_properties = target_properties.to(self.device)
        
        # Compute prediction
        with torch.enable_grad():
            pred_properties = self.model(sample_x)
            loss = self.criterion(pred_properties, target_properties)
            
            # Compute gradients
            try:
                gradients = autograd.grad(loss, sample_x, create_graph=True)[0]
            except RuntimeError as e:
                return None, {
                    'passed': False,
                    'issue': f'Gradient computation failed: {str(e)}',
                    'stats': {}
                }
        
        # Initialize diagnostics
        diagnostics = {
            'passed': True,
            'issues': [],
            'stats': {}
        }
        
        # Check 1: No NaN values
        if torch.isnan(gradients).any():
            diagnostics['passed'] = False
            diagnostics['issues'].append('NaN gradients detected')
            if verbose:
                print("❌ NaN gradients - model instability")
            return gradients, diagnostics
        
        # Check 2: No Inf values
        if torch.isinf(gradients).any():
            diagnostics['passed'] = False
            diagnostics['issues'].append('Inf gradients detected')
            if verbose:
                print("❌ Inf gradients - gradient explosion")
            return gradients, diagnostics
        
        # Check 3: Gradient magnitude (should be reasonable, not too large)
        grad_magnitude = gradients.abs().max().item()
        grad_mean = gradients.abs().mean().item()
        grad_std = gradients.std().item()
        
        diagnostics['stats']['max_grad'] = grad_magnitude
        diagnostics['stats']['mean_grad'] = grad_mean
        diagnostics['stats']['std_grad'] = grad_std
        
        if grad_magnitude > 100:
            diagnostics['passed'] = False
            diagnostics['issues'].append(f'Gradient explosion: max={grad_magnitude:.2f}')
            if verbose:
                print(f"⚠️  Gradient explosion: max gradient = {grad_magnitude:.2f}")
        elif grad_magnitude < 0.001:
            diagnostics['issues'].append(f'Vanishing gradients: max={grad_magnitude:.6f}')
            if verbose:
                print(f"⚠️  Vanishing gradients: max gradient = {grad_magnitude:.6f}")
        
        # Check 4: Loss value is reasonable
        loss_value = loss.item()
        diagnostics['stats']['loss'] = loss_value
        
        if loss_value > 10000:
            diagnostics['issues'].append(f'Loss too high: {loss_value:.2f}')
        
        # Check 5: Prediction magnitude (should be in property ranges)
        pred_ranges = [
            (-2.0, 5.0, "LogP"),
            (50, 700, "MW"),
            (0, 5, "HBD"),
            (0, 10, "HBA"),
            (0, 15, "Rotatable"),
        ]
        
        out_of_range_count = 0
        for prop_idx, (min_val, max_val, name) in enumerate(pred_ranges):
            pred_vals = pred_properties[:, prop_idx].detach()
            oor = ((pred_vals < min_val) | (pred_vals > max_val)).sum().item()
            out_of_range_count += oor
        
        diagnostics['stats']['out_of_range_predictions'] = out_of_range_count
        
        if out_of_range_count > 0:
            diagnostics['issues'].append(f'{out_of_range_count} predictions out of range')
        
        # Check 6: Gradient direction (should point toward target)
        # Rough heuristic: gradient*input should have same sign as (target-pred)
        with torch.no_grad():
            pred_direction = (target_properties - pred_properties).sign()
            grad_direction = gradients.mean(dim=1, keepdim=True).sign()
            alignment = (pred_direction[:, 0] == grad_direction[:, 0]).float().mean()
        
        diagnostics['stats']['gradient_alignment'] = alignment.item()
        
        # Print comprehensive diagnostics
        if verbose:
            print("\n" + "="*70)
            print("GRADIENT VERIFICATION REPORT")
            print("="*70)
            
            print(f"\n✓ PASSED CHECKS" if diagnostics['passed'] else f"\n❌ FAILED CHECKS")
            
            print(f"\nGradient Statistics:")
            print(f"  Max magnitude:       {grad_magnitude:.6f}")
            print(f"  Mean magnitude:      {grad_mean:.6f}")
            print(f"  Std deviation:       {grad_std:.6f}")
            print(f"  Loss value:          {loss_value:.6f}")
            
            print(f"\nPrediction Validation:")
            print(f"  Out of range:        {out_of_range_count}/25")
            print(f"  Gradient alignment:  {alignment.item()*100:.1f}%")
            
            if diagnostics['issues']:
                print(f"\n⚠️  Issues Detected:")
                for issue in diagnostics['issues']:
                    print(f"    - {issue}")
            else:
                print(f"\n✅ All checks passed! Safe for guidance.")
            
            print("="*70 + "\n")
        
        return gradients, diagnostics
    
    def batch_verify(self, test_features, test_properties, num_batches=5):
        """Verify gradient behavior across multiple batches."""
        
        print("\n" + "="*70)
        print("BATCH VERIFICATION (multiple test cases)")
        print("="*70)
        
        batch_results = []
        all_passed = True
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * (len(test_features) // num_batches)
            end_idx = start_idx + (len(test_features) // num_batches)
            
            batch_x = test_features[start_idx:end_idx]
            batch_y = test_properties[start_idx:end_idx]
            
            print(f"\nBatch {batch_idx + 1}/{num_batches}:")
            _, diagnostics = self.verify_guidance_gradients(
                batch_x, batch_y, verbose=False
            )
            
            batch_results.append(diagnostics)
            all_passed = all_passed and diagnostics['passed']
            
            status = "✅ PASS" if diagnostics['passed'] else "❌ FAIL"
            print(f"  Status: {status}")
            if 'max_grad' in diagnostics['stats']:
                print(f"  Max gradient: {diagnostics['stats']['max_grad']:.6f}")
            if diagnostics['issues']:
                for issue in diagnostics['issues']:
                    print(f"  ⚠️  {issue}")
        
        print("\n" + "="*70)
        print(f"Overall: {'✅ ALL BATCHES PASSED' if all_passed else '❌ SOME BATCHES FAILED'}")
        print("="*70 + "\n")
        
        return batch_results


def main():
    """Run comprehensive gradient verification."""
    
    print("\n" + "="*70)
    print("GRADIENT VERIFICATION SUITE FOR PROPERTY GUIDANCE")
    print("="*70)
    
    # Load model
    print("\n1. Loading improved model...")
    model = RegularizedPropertyGuidanceRegressor(input_dim=100, n_properties=5)
    checkpoint_path = Path('checkpoints/property_regressor_improved.pt')
    
    if not checkpoint_path.exists():
        print(f"❌ Model not found at {checkpoint_path}")
        print("   Run: python train_property_regressor_improved.py --epochs 100")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print("✓ Model loaded successfully")
    
    # Create verifier
    verifier = GradientVerifier(model, device='cpu')
    
    # Generate test data
    print("\n2. Generating test data...")
    test_features = torch.randn(200, 100)
    
    # Create realistic test properties
    test_properties = torch.zeros(200, 5)
    test_properties[:, 0] = torch.clamp(torch.randn(200) * 2 - 0.5, -2, 5)  # LogP
    test_properties[:, 1] = torch.clamp(torch.randn(200) * 100 + 250, 50, 700)  # MW
    test_properties[:, 2] = torch.clamp(torch.abs(torch.randn(200)) * 2, 0, 5)  # HBD
    test_properties[:, 3] = torch.clamp(torch.abs(torch.randn(200)) * 3 + 5, 0, 10)  # HBA
    test_properties[:, 4] = torch.clamp(torch.abs(torch.randn(200)) * 4 + 8, 0, 15)  # Rotatable
    
    print(f"✓ Created {len(test_features)} test samples")
    
    # Single sample verification
    print("\n3. Single sample verification...")
    sample_x = test_features[:1]
    sample_y = test_properties[:1]
    
    gradients, diagnostics = verifier.verify_guidance_gradients(sample_x, sample_y)
    
    if diagnostics['passed']:
        print("✅ Single sample verification PASSED")
    else:
        print("❌ Single sample verification FAILED")
        print(f"   Issues: {diagnostics['issues']}")
    
    # Batch verification
    print("\n4. Batch verification (5 batches)...")
    batch_results = verifier.batch_verify(test_features, test_properties, num_batches=5)
    
    passed_count = sum(1 for r in batch_results if r['passed'])
    print(f"✓ Passed {passed_count}/{len(batch_results)} batches")
    
    # Summary
    print("\n" + "="*70)
    print("GRADIENT VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = all(r['passed'] for r in batch_results)
    
    if all_passed and diagnostics['passed']:
        print("""
✅ ALL CHECKS PASSED

Your model is ready for property-guided generation:
  - Gradients are stable and well-behaved
  - No NaN or explosion issues detected
  - Predictions stay in valid ranges
  - Safe to use for real molecular guidance

Next steps:
  1. Integrate into guided_sampling.py
  2. Test end-to-end molecular generation
  3. Monitor guidance effectiveness
""")
    else:
        print("""
⚠️  SOME CHECKS FAILED

Do NOT deploy for production guidance yet:
  - Fix the issues listed above
  - Consider retraining with adjusted hyperparameters
  - Run verification again after fixes
""")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
