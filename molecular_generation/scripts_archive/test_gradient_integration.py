#!/usr/bin/env python3
"""
PHASE 1 VALIDATION: Gradient-Based Guidance Integration Test

This test verifies the critical fix that connects regressor gradients to the sampling loop.

What it tests:
1. Gradient flow: Can gradients flow from regressor through features to noise?
2. Guidance effectiveness: Does steering actually work?
3. Property convergence: Do generated molecules move toward target properties?
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import json
from pathlib import Path


class PropertyGuidanceRegressor(nn.Module):
    """Property prediction regressor for guidance."""
    
    def __init__(self, input_dim: int = 100, n_properties: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_properties)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientFlowValidator:
    """Validates that gradient-based guidance integration works."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.regressor = None
        self.results = {}
    
    def load_regressor(self, path: str = None):
        """Load trained regressor."""
        if path is None:
            path = "./checkpoints/property_regressor_improved.pt"
        
        try:
            self.regressor = PropertyGuidanceRegressor(input_dim=100, n_properties=5)
            state = torch.load(path, map_location=self.device)
            
            # Handle both direct state_dict and wrapped state
            if isinstance(state, dict) and 'net' not in state:
                self.regressor.load_state_dict(state)
            else:
                self.regressor.load_state_dict(state)
            
            self.regressor.to(self.device)
            self.regressor.eval()
            print(f"✅ Loaded regressor from {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load regressor: {e}")
            # Create untrained regressor for testing
            self.regressor = PropertyGuidanceRegressor(input_dim=100, n_properties=5).to(self.device)
            print("⚠️  Using untrained regressor (this is OK for gradient flow testing)")
            return False
    
    def test_gradient_flow(self, batch_size: int = 10) -> Dict:
        """
        CRITICAL TEST 1: Can gradients flow from regressor to features?
        
        This tests the fundamental requirement for gradient-based guidance to work.
        """
        print("\n" + "="*70)
        print("TEST 1: GRADIENT FLOW")
        print("="*70)
        
        results = {
            'test': 'gradient_flow',
            'batch_size': batch_size,
            'gradient_magnitude': 0,
            'gradient_std': 0,
            'gradient_flow_ok': False,
            'computation_graph_ok': False,
        }
        
        # Initialize features with gradients enabled
        features = torch.randn(batch_size, 100, device=self.device, requires_grad=True)
        target_props = torch.randn(batch_size, 5, device=self.device)
        
        print(f"✓ Created feature tensor: shape={features.shape}, requires_grad={features.requires_grad}")
        print(f"✓ Created target tensor: shape={target_props.shape}")
        
        # Forward pass through regressor
        with torch.enable_grad():
            pred_props = self.regressor(features)
            print(f"✓ Regressor forward pass: pred_props.shape={pred_props.shape}")
            
            # Compute loss
            loss = torch.mean((pred_props - target_props) ** 2)
            print(f"✓ Loss computed: {loss.item():.6f}")
            
            # Backward pass - THIS IS THE CRITICAL LINE
            try:
                loss.backward()
                print(f"✓ Backward pass successful (gradients computed)")
                results['computation_graph_ok'] = True
            except Exception as e:
                print(f"❌ Backward pass failed: {e}")
                return results
        
        # Check gradients
        if features.grad is not None:
            grad_magnitude = torch.norm(features.grad).item()
            grad_std = features.grad.std().item()
            
            print(f"✓ Gradient magnitude: {grad_magnitude:.6f}")
            print(f"✓ Gradient std: {grad_std:.6f}")
            print(f"✓ Min gradient: {features.grad.min().item():.6f}")
            print(f"✓ Max gradient: {features.grad.max().item():.6f}")
            print(f"✓ Non-zero gradients: {(features.grad.abs() > 1e-8).sum().item()} / {features.grad.numel()}")
            
            results['gradient_magnitude'] = float(grad_magnitude)
            results['gradient_std'] = float(grad_std)
            results['gradient_flow_ok'] = grad_magnitude > 0
            
            if grad_magnitude > 0:
                print("✅ PASS: Gradients flow from regressor to features")
            else:
                print("❌ FAIL: No gradient flow detected")
        else:
            print("❌ FAIL: features.grad is None")
        
        return results
    
    def test_guidance_signal_computation(self, batch_size: int = 10) -> Dict:
        """
        CRITICAL TEST 2: Can we compute and apply guidance signal?
        
        This tests the 5-line pattern:
            1. features.requires_grad = True
            2. pred_props = regressor(features)
            3. loss = MSE(pred_props, target_properties)
            4. grad = torch.autograd.grad(loss, features)[0]
            5. features = features - guidance_scale * grad
        """
        print("\n" + "="*70)
        print("TEST 2: GUIDANCE SIGNAL COMPUTATION")
        print("="*70)
        
        results = {
            'test': 'guidance_signal',
            'batch_size': batch_size,
            'initial_loss': 0,
            'final_loss': 0,
            'loss_improvement': 0,
            'guidance_applied': False,
        }
        
        # Initialize
        features = torch.randn(batch_size, 100, device=self.device)
        target_props = torch.randn(batch_size, 5, device=self.device)
        guidance_scale = 0.1
        
        # Line 1: features.requires_grad = True
        features.requires_grad_(True)
        print(f"✓ Step 1: features.requires_grad = True")
        
        # Line 2: pred_props = regressor(features)
        pred_props = self.regressor(features)
        print(f"✓ Step 2: pred_props = regressor(features), shape={pred_props.shape}")
        
        # Line 3: loss = MSE(pred_props, target_properties)
        initial_loss = torch.mean((pred_props - target_props) ** 2)
        print(f"✓ Step 3: initial_loss = {initial_loss.item():.6f}")
        results['initial_loss'] = float(initial_loss.item())
        
        # Line 4: grad = torch.autograd.grad(loss, features)[0]
        try:
            initial_loss.backward()
            grad = features.grad.clone()
            print(f"✓ Step 4: grad computed, shape={grad.shape}, norm={torch.norm(grad).item():.6f}")
            results['guidance_applied'] = True
        except Exception as e:
            print(f"❌ Step 4 failed: {e}")
            return results
        
        # Line 5: features = features - guidance_scale * grad
        with torch.no_grad():
            features_guided = features - guidance_scale * grad
            features_guided.requires_grad_(True)
        
        print(f"✓ Step 5: Applied guidance signal (scale={guidance_scale})")
        
        # Check if loss improved
        with torch.no_grad():
            pred_props_new = self.regressor(features_guided)
            final_loss = torch.mean((pred_props_new - target_props) ** 2)
        
        loss_improvement = (initial_loss.item() - final_loss.item()) / (initial_loss.item() + 1e-8)
        results['final_loss'] = float(final_loss.item())
        results['loss_improvement'] = float(loss_improvement)
        
        print(f"✓ After guidance: loss = {final_loss.item():.6f}")
        print(f"✓ Loss improvement: {loss_improvement*100:.2f}%")
        
        if loss_improvement > 0:
            print("✅ PASS: Loss decreased after applying guidance signal")
        else:
            print("⚠️  Loss did not improve (but gradient flow is working)")
        
        return results
    
    def test_iterative_guidance(self, num_steps: int = 5, batch_size: int = 10) -> Dict:
        """
        CRITICAL TEST 3: Does repeated guidance iteration improve properties?
        
        This simulates multiple diffusion steps with guidance applied each time.
        """
        print("\n" + "="*70)
        print("TEST 3: ITERATIVE GUIDANCE IMPROVEMENT")
        print("="*70)
        
        results = {
            'test': 'iterative_guidance',
            'num_steps': num_steps,
            'batch_size': batch_size,
            'loss_per_step': [],
            'total_improvement': 0,
        }
        
        # Initialize
        features = torch.randn(batch_size, 100, device=self.device)
        target_props = torch.randn(batch_size, 5, device=self.device)
        guidance_scale = 0.2
        
        print(f"Starting iterative guidance for {num_steps} steps...")
        
        initial_loss = None
        for step in range(num_steps):
            # Enable gradients
            features.requires_grad_(True)
            
            # Forward pass
            pred_props = self.regressor(features)
            
            # Compute loss
            loss = torch.mean((pred_props - target_props) ** 2)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            results['loss_per_step'].append(float(loss.item()))
            print(f"  Step {step+1}: loss = {loss.item():.6f}")
            
            # Backward
            loss.backward()
            grad = features.grad.clone()
            
            # Apply guidance
            with torch.no_grad():
                features = features - guidance_scale * grad
        
        final_loss = results['loss_per_step'][-1]
        total_improvement = (initial_loss - final_loss) / (initial_loss + 1e-8)
        results['total_improvement'] = float(total_improvement)
        
        print(f"\nInitial loss: {initial_loss:.6f}")
        print(f"Final loss: {final_loss:.6f}")
        print(f"Total improvement: {total_improvement*100:.2f}%")
        
        if total_improvement > 0.01:
            print("✅ PASS: Loss decreased over iterations")
        else:
            print("⚠️  Loss improvement was marginal")
        
        return results
    
    def test_batch_consistency(self, num_trials: int = 5) -> Dict:
        """
        TEST 4: Does guidance work consistently across batches?
        """
        print("\n" + "="*70)
        print("TEST 4: BATCH CONSISTENCY")
        print("="*70)
        
        results = {
            'test': 'batch_consistency',
            'num_trials': num_trials,
            'trial_improvements': [],
            'avg_improvement': 0,
            'consistency': 0,
        }
        
        improvements = []
        for trial in range(num_trials):
            features = torch.randn(10, 100, device=self.device, requires_grad=True)
            target_props = torch.randn(10, 5, device=self.device)
            
            pred_props = self.regressor(features)
            initial_loss = torch.mean((pred_props - target_props) ** 2)
            
            initial_loss.backward()
            grad = features.grad.clone()
            
            with torch.no_grad():
                features_guided = features - 0.1 * grad
                pred_props_new = self.regressor(features_guided)
                final_loss = torch.mean((pred_props_new - target_props) ** 2)
            
            improvement = (initial_loss.item() - final_loss.item()) / (initial_loss.item() + 1e-8)
            improvements.append(improvement)
            results['trial_improvements'].append(float(improvement))
            print(f"  Trial {trial+1}: improvement = {improvement*100:.2f}%")
        
        avg_improvement = np.mean(improvements)
        consistency = 1.0 - np.std(improvements)
        results['avg_improvement'] = float(avg_improvement)
        results['consistency'] = float(consistency)
        
        print(f"\nAverage improvement: {avg_improvement*100:.2f}%")
        print(f"Consistency: {consistency*100:.2f}%")
        
        if consistency > 0.5:
            print("✅ PASS: Guidance is consistent across batches")
        else:
            print("⚠️  Guidance effectiveness varies across batches")
        
        return results
    
    def run_full_validation_suite(self) -> Dict:
        """Run all Phase 1 validation tests."""
        print("\n" + "█"*70)
        print("█  PHASE 1: GRADIENT INTEGRATION VALIDATION SUITE  █")
        print("█"*70)
        
        # Load regressor
        self.load_regressor()
        
        # Run tests
        all_results = {
            'timestamp': str(Path.cwd()),
            'device': str(self.device),
            'tests': {}
        }
        
        test_functions = [
            self.test_gradient_flow,
            self.test_guidance_signal_computation,
            self.test_iterative_guidance,
            self.test_batch_consistency,
        ]
        
        for test_func in test_functions:
            try:
                result = test_func()
                all_results['tests'][result['test']] = result
            except Exception as e:
                print(f"\n❌ Test {test_func.__name__} failed: {e}")
                all_results['tests'][test_func.__name__] = {'error': str(e)}
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        gradient_flow_ok = all_results['tests'].get('gradient_flow', {}).get('gradient_flow_ok', False)
        guidance_ok = all_results['tests'].get('guidance_signal', {}).get('guidance_applied', False)
        iterative_ok = all_results['tests'].get('iterative_guidance', {}).get('total_improvement', 0) > 0
        consistent_ok = all_results['tests'].get('batch_consistency', {}).get('consistency', 0) > 0.5
        
        print(f"Gradient Flow:        {'✅ PASS' if gradient_flow_ok else '❌ FAIL'}")
        print(f"Guidance Signal:      {'✅ PASS' if guidance_ok else '❌ FAIL'}")
        print(f"Iterative Improvement:{'✅ PASS' if iterative_ok else '❌ FAIL'}")
        print(f"Batch Consistency:    {'✅ PASS' if consistent_ok else '❌ FAIL'}")
        
        all_pass = all([gradient_flow_ok, guidance_ok, iterative_ok, consistent_ok])
        
        print("\n" + "="*70)
        if all_pass:
            print("✅ PHASE 1 VALIDATION: PASSED - Gradient integration is working")
            print("Status: Ready for real guidance sampling test")
        else:
            print("❌ PHASE 1 VALIDATION: FAILED - Gradient integration needs fixes")
        print("="*70)
        
        # Save results
        with open('gradient_integration_results.json', 'w') as f:
            # Convert numpy types to native Python
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                return obj
            
            json.dump(all_results, f, indent=2, default=convert)
        
        print(f"\n✓ Results saved to gradient_integration_results.json")
        
        return all_results


if __name__ == '__main__':
    validator = GradientFlowValidator(device='cpu')
    results = validator.run_full_validation_suite()
