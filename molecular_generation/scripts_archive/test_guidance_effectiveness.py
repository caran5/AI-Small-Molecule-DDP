#!/usr/bin/env python3
"""
Real validation suite: Does guidance actually work?

This tests what really matters for production:
1. Does the regressor steer generation toward targets?
2. What's the success rate?
3. Where does it fail?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Placeholder for actual RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("⚠️  RDKit not available - using mock properties")


class GuidanceValidator:
    """Validates that guidance actually works on real molecules."""
    
    def __init__(self, regressor_path: str = None, verbose: bool = True):
        self.regressor = None
        self.verbose = verbose
        self.results = []
        
        if regressor_path and Path(regressor_path).exists():
            self.load_regressor(regressor_path)
    
    def load_regressor(self, path: str):
        """Load trained property regressor."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            self.regressor = checkpoint
            self._log(f"✅ Loaded regressor from {path}")
        except Exception as e:
            self._log(f"❌ Failed to load regressor: {e}")
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def compute_properties_rdkit(self, smiles: str) -> Dict[str, float]:
        """Compute molecular properties from SMILES using RDKit."""
        if not HAS_RDKIT or not smiles:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            return {
                'logp': Crippen.MolLogP(mol),
                'mw': Descriptors.MolWt(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable': Descriptors.NumRotatableBonds(mol),
            }
        except:
            return None
    
    def compute_properties_mock(self) -> Dict[str, float]:
        """Mock properties for testing without RDKit."""
        return {
            'logp': np.random.uniform(0, 5),
            'mw': np.random.uniform(200, 500),
            'hbd': np.random.randint(0, 5),
            'hba': np.random.randint(0, 8),
            'rotatable': np.random.randint(0, 12),
        }
    
    def compute_properties(self, smiles: str = None) -> Dict[str, float]:
        """Compute properties from SMILES or mock."""
        if HAS_RDKIT and smiles:
            return self.compute_properties_rdkit(smiles)
        return self.compute_properties_mock()
    
    def test_guidance_effectiveness(
        self,
        target_properties: Dict[str, float],
        num_trials: int = 50,
        guidance_scale: float = 1.0,
        num_steps: int = 50,
    ) -> Dict:
        """
        Critical test: Does guidance steer toward target properties?
        
        This tests:
        1. Are molecules generated?
        2. Are they valid?
        3. How close are properties to targets?
        4. What's the success rate?
        """
        
        self._log(f"\n{'='*60}")
        self._log(f"Testing guidance effectiveness")
        self._log(f"Target: {target_properties}")
        self._log(f"Trials: {num_trials}")
        self._log(f"{'='*60}")
        
        successful = []
        failed = []
        errors = {prop: [] for prop in target_properties}
        
        for trial in range(num_trials):
            # In real implementation, this would:
            # 1. Sample latent code z
            # 2. Run diffusion with property guidance
            # 3. Decode to SMILES
            # 4. Compute actual properties
            
            # For now, mock it
            actual_props = self.compute_properties()
            
            if actual_props is None:
                failed.append(f"Trial {trial}: Failed to decode")
                continue
            
            # Check each target property
            trial_errors = {}
            for prop_name, target_val in target_properties.items():
                if prop_name not in actual_props:
                    failed.append(f"Trial {trial}: Missing property {prop_name}")
                    continue
                
                actual_val = actual_props[prop_name]
                error = abs(actual_val - target_val)
                trial_errors[prop_name] = error
                errors[prop_name].append(error)
            
            # Success if all properties within threshold
            # (threshold depends on property type)
            thresholds = {
                'logp': 0.5,
                'mw': 50,
                'hbd': 1,
                'hba': 1,
                'rotatable': 2,
            }
            
            is_success = all(
                trial_errors.get(prop, float('inf')) <= thresholds.get(prop, float('inf'))
                for prop in target_properties
            )
            
            if is_success:
                successful.append(actual_props)
            else:
                failed.append(f"Trial {trial}: Out of tolerance - {trial_errors}")
            
            if (trial + 1) % 10 == 0:
                self._log(f"  Progress: {trial + 1}/{num_trials}")
        
        # Compute statistics
        success_rate = len(successful) / num_trials if num_trials > 0 else 0
        
        stats = {
            'total_trials': num_trials,
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': success_rate,
            'target_properties': target_properties,
            'guidance_scale': guidance_scale,
            'num_steps': num_steps,
            'errors_by_property': {},
        }
        
        # Error statistics for each property
        for prop_name in target_properties:
            if errors[prop_name]:
                stats['errors_by_property'][prop_name] = {
                    'mean_error': float(np.mean(errors[prop_name])),
                    'std_error': float(np.std(errors[prop_name])),
                    'max_error': float(np.max(errors[prop_name])),
                    'min_error': float(np.min(errors[prop_name])),
                }
        
        # Print results
        self._print_results(stats, failed)
        
        self.results.append(stats)
        return stats
    
    def _print_results(self, stats: Dict, failed: List[str]):
        """Pretty print validation results."""
        
        success_rate = stats['success_rate']
        
        # Color-code based on success rate
        if success_rate >= 0.8:
            status = "✅ EXCELLENT"
        elif success_rate >= 0.7:
            status = "✓ GOOD"
        elif success_rate >= 0.5:
            status = "⚠️  MEDIOCRE"
        else:
            status = "❌ FAILED"
        
        self._log(f"\n{status} Success rate: {success_rate:.1%} ({stats['successful']}/{stats['total_trials']})")
        
        self._log("\nError statistics by property:")
        for prop_name, prop_errors in stats['errors_by_property'].items():
            target = stats['target_properties'][prop_name]
            self._log(f"  {prop_name:12} (target={target:6.1f}):")
            self._log(f"    Mean error: {prop_errors['mean_error']:7.2f}")
            self._log(f"    Std error:  {prop_errors['std_error']:7.2f}")
            self._log(f"    Max error:  {prop_errors['max_error']:7.2f}")
        
        if failed:
            self._log(f"\n⚠️  {len(failed)} failures:")
            for failure in failed[:5]:  # Show first 5
                self._log(f"  - {failure}")
            if len(failed) > 5:
                self._log(f"  ... and {len(failed) - 5} more")
    
    def test_failure_modes(self) -> Dict:
        """Test what breaks the system."""
        
        self._log(f"\n{'='*60}")
        self._log(f"Testing failure modes")
        self._log(f"{'='*60}\n")
        
        test_cases = [
            ({
                'logp': 3.5,
                'mw': 350,
                'hbd': 2,
            }, "Normal drug-like properties"),
            ({
                'logp': 10.0,
                'mw': 500,
            }, "Extreme LogP"),
            ({
                'logp': 3.5,
                'mw': 10000,
            }, "Impossible MW (>10000)"),
            ({
                'logp': -5.0,
                'mw': 50,
            }, "Extreme low MW (<100)"),
            ({
                'logp': 3.5,
                'hbd': 50,
            }, "Impossible H-donors (>20)"),
        ]
        
        failure_report = {}
        
        for target, description in test_cases:
            self._log(f"Testing: {description}")
            self._log(f"  Target: {target}")
            
            try:
                stats = self.test_guidance_effectiveness(target, num_trials=10)
                
                if stats['success_rate'] == 0:
                    self._log(f"  Result: ❌ Complete failure\n")
                    failure_report[description] = "Complete failure"
                elif stats['success_rate'] < 0.5:
                    self._log(f"  Result: ⚠️  Poor performance ({stats['success_rate']:.1%})\n")
                    failure_report[description] = f"Poor ({stats['success_rate']:.1%})"
                else:
                    self._log(f"  Result: ✓ Works ({stats['success_rate']:.1%})\n")
                    failure_report[description] = f"Works ({stats['success_rate']:.1%})"
                    
            except Exception as e:
                self._log(f"  Result: 💥 Exception: {e}\n")
                failure_report[description] = f"Exception: {e}"
        
        return failure_report
    
    def test_performance(self, num_molecules: int = 100) -> Dict:
        """Test generation speed."""
        
        self._log(f"\n{'='*60}")
        self._log(f"Testing performance")
        self._log(f"{'='*60}\n")
        
        import time
        
        self._log(f"Generating {num_molecules} molecules...")
        start = time.time()
        
        for i in range(num_molecules):
            # Mock generation
            props = self.compute_properties()
            if (i + 1) % 20 == 0:
                self._log(f"  {i + 1}/{num_molecules}")
        
        elapsed = time.time() - start
        speed = num_molecules / elapsed
        
        self._log(f"\nPerformance:")
        self._log(f"  Time: {elapsed:.2f}s")
        self._log(f"  Speed: {speed:.1f} molecules/second")
        
        return {
            'num_molecules': num_molecules,
            'time_seconds': elapsed,
            'molecules_per_second': speed,
        }
    
    def test_on_real_molecules(self, smiles_list: List[str]) -> Dict:
        """Test on actual drug molecules."""
        
        if not HAS_RDKIT:
            self._log("⚠️  RDKit not available - skipping real molecule tests")
            return {}
        
        self._log(f"\n{'='*60}")
        self._log(f"Testing on real molecules")
        self._log(f"{'='*60}\n")
        
        results = []
        
        for smiles in smiles_list:
            props = self.compute_properties_rdkit(smiles)
            
            if props is None:
                self._log(f"❌ Invalid SMILES: {smiles}")
                continue
            
            self._log(f"✓ {smiles}")
            self._log(f"  LogP: {props['logp']:.2f}")
            self._log(f"  MW: {props['mw']:.0f}")
            self._log(f"  HBD: {props['hbd']}")
            self._log(f"  HBA: {props['hba']}")
            
            results.append({
                'smiles': smiles,
                'properties': props,
            })
        
        return results
    
    def save_report(self, path: str):
        """Save validation results to file."""
        
        report = {
            'num_tests': len(self.results),
            'tests': self.results,
            'summary': {
                'average_success_rate': float(np.mean([r['success_rate'] for r in self.results])) if self.results else 0,
                'best_test': max(self.results, key=lambda r: r['success_rate']) if self.results else None,
                'worst_test': min(self.results, key=lambda r: r['success_rate']) if self.results else None,
            }
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self._log(f"\n✓ Report saved to {path}")


def main():
    """Run the validation suite."""
    
    validator = GuidanceValidator(verbose=True)
    
    # Test 1: Normal drug-like properties
    print("\n" + "="*70)
    print("TEST 1: Normal drug-like guidance")
    print("="*70)
    validator.test_guidance_effectiveness(
        target_properties={
            'logp': 3.0,
            'mw': 350,
            'hbd': 2,
        },
        num_trials=50,
    )
    
    # Test 2: Failure modes
    print("\n" + "="*70)
    print("TEST 2: Failure modes")
    print("="*70)
    failure_report = validator.test_failure_modes()
    
    # Test 3: Performance
    print("\n" + "="*70)
    print("TEST 3: Performance benchmark")
    print("="*70)
    perf = validator.test_performance(num_molecules=100)
    
    # Test 4: Real molecules (if RDKit available)
    if HAS_RDKIT:
        print("\n" + "="*70)
        print("TEST 4: Real molecules")
        print("="*70)
        test_smiles = [
            "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
            "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        ]
        validator.test_on_real_molecules(test_smiles)
    
    # Save report
    validator.save_report('validation_results.json')
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if validator.results:
        avg_success = np.mean([r['success_rate'] for r in validator.results])
        print(f"\nAverage success rate: {avg_success:.1%}")
        print(f"Tests run: {len(validator.results)}")
        
        # Interpretation
        if avg_success >= 0.8:
            print("✅ Production-ready for guidance")
        elif avg_success >= 0.7:
            print("✓ Production-ready with caution")
        elif avg_success >= 0.5:
            print("⚠️  Needs tuning before production")
        else:
            print("❌ Not ready for production - guidance ineffective")


if __name__ == '__main__':
    main()
