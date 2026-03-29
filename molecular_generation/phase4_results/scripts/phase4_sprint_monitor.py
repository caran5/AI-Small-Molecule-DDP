#!/usr/bin/env python3
"""
PHASE 4 SPRINT MONITOR
======================
Monitor Path 1 execution and automatically queue Path 2 upon completion.

This script will:
1. Check if phase4_path1_grid_search_results.json exists and is complete
2. Extract best configuration and accuracy improvement
3. If successful, print summary and prompt for Path 2
4. Provide decision logic for next steps
"""

import json
import os
import time
from pathlib import Path
import subprocess

def check_path1_status():
    """Check if Path 1 is complete and show results"""
    results_file = Path('/Users/ceejayarana/diffusion_model/molecular_generation/phase4_path1_grid_search_results.json')
    
    if not results_file.exists():
        print("⏳ Path 1 still running... results file not yet generated")
        return False, None
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Verify file has complete data
        if 'best_configuration' not in results:
            print("⏳ Path 1 still running... results incomplete")
            return False, None
        
        print("\n" + "=" * 100)
        print("✅ PATH 1 COMPLETE!")
        print("=" * 100)
        
        best = results['best_configuration']
        baseline = results['baseline_accuracy']
        improvement = best['test_accuracy'] - baseline
        
        print(f"\n▶ RESULTS SUMMARY:")
        print(f"  Baseline (Approach 1):    {baseline:.1f}%")
        print(f"  Best configuration found: {best['test_accuracy']:.1f}%")
        print(f"  Improvement:              {improvement:+.1f} percentage points ({improvement/baseline*100:+.1f}%)")
        print(f"\n▶ BEST CONFIGURATION:")
        print(f"  • Fingerprint radius:     {best['radius']}")
        print(f"  • Morgan nBits:           {best['n_bits']}")
        print(f"  • PCA components:         {best['pca_components']}")
        print(f"  • Cross-validation:       {best['cv_mean']:.1f}% ± {best['cv_std']:.1f}%")
        
        print(f"\n▶ TOP 3 CONFIGURATIONS:")
        for i, config in enumerate(results['top_10_configurations'][:3], 1):
            print(f"  {i}. r={config['radius']} b={config['n_bits']:4d} pca={config['pca_components']:3d} " +
                  f"→ {config['test_accuracy']:.1f}%")
        
        # Decision logic
        print(f"\n▶ DECISION:")
        if improvement >= 3:
            print(f"  ✅ SIGNIFICANT IMPROVEMENT FOUND (+{improvement:.1f}pp)")
            print(f"  → Proceed to Path 2 with confidence")
            return True, results
        elif improvement >= 0:
            print(f"  ⚠️  MODEST IMPROVEMENT ({improvement:+.1f}pp)")
            print(f"  → Still proceed, but may hit diminishing returns")
            return True, results
        else:
            print(f"  ❌ NO IMPROVEMENT FOUND ({improvement:+.1f}pp)")
            print(f"  → Consider alternative strategies or data expansion")
            return False, results
        
    except json.JSONDecodeError:
        print("⏳ Path 1 results incomplete or corrupted, waiting...")
        return False, None
    except Exception as e:
        print(f"❌ Error reading Path 1 results: {e}")
        return False, None


def estimate_time_remaining():
    """Estimate time remaining for Path 1"""
    try:
        # Check log file if available
        import subprocess
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if 'phase4_path1' in result.stdout:
            print("  Process still running...")
            return True
        else:
            print("  Process not found in ps output")
            return False
    except:
        return None


def suggest_next_action(path1_results=None):
    """Suggest next action based on Path 1 results"""
    if path1_results is None:
        return """
Next step: Continue monitoring Path 1

When Path 1 completes, run:
  $ python /Users/ceejayarana/diffusion_model/molecular_generation/phase4_path2_features.py
"""
    
    best = path1_results['best_configuration']
    return f"""
Next step: Execute Path 2 (Feature Engineering)

Use Path 1 best configuration:
  Radius: {best['radius']}
  nBits: {best['n_bits']}
  PCA components: {best['pca_components']}
  
Command:
  $ python /Users/ceejayarana/diffusion_model/molecular_generation/phase4_path2_features.py

Path 2 will:
  - Extract 200+ new descriptors
  - Select top 150 via correlation
  - Combine with Path 1 best features
  - Target: {best['test_accuracy'] + 5:.0f}-{best['test_accuracy'] + 8:.0f}%
"""


# Main monitoring loop
print("\n" + "=" * 100)
print("PHASE 4 SPRINT MONITOR - Path 1 Progress Check")
print("=" * 100)

print("\nChecking Path 1 status...")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

is_complete, results = check_path1_status()

if is_complete:
    print(suggest_next_action(results))
else:
    print(suggest_next_action(None))
    print("\n" + "-" * 100)
    print("Checking process...")
    is_running = estimate_time_remaining()
    if is_running:
        print("✅ Path 1 process confirmed running")
        print("   Estimated completion: 1.5-2.5 hours from start")
    elif is_running is None:
        print("⚠️  Could not verify process status")
    else:
        print("❌ Path 1 process not found - may have crashed or completed")

print("\n" + "=" * 100)
print("Tip: Check this periodically with:")
print("  $ python phase4_sprint_monitor.py")
print("=" * 100 + "\n")
