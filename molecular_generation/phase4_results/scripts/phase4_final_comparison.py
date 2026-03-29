#!/usr/bin/env python3
"""
PHASE 4 COMPREHENSIVE COMPARISON REPORT
========================================
After all three paths complete, this script generates a final report
comparing all approaches and providing actionable recommendations.

Generates: phase4_final_comparison_report.json
"""

import json
from pathlib import Path

def load_results(filename):
    """Load results JSON file if it exists"""
    path = Path(f'/Users/ceejayarana/diffusion_model/molecular_generation/{filename}')
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

def generate_comparison_report():
    """Generate final comparison across all paths"""
    
    # Load all results
    path1_results = load_results('phase4_path1_grid_search_results.json')
    path2_results = load_results('phase4_path2_feature_engineering_results.json')
    path3_results = load_results('phase4_path3_stacking_results.json')
    
    baseline = 76.0
    
    print("\n" + "=" * 100)
    print("PHASE 4 FINAL COMPARISON REPORT")
    print("=" * 100)
    
    print("\n▶ PATH COMPLETION STATUS:\n")
    
    status_p1 = "✅ COMPLETE" if path1_results else "⏳ PENDING"
    status_p2 = "✅ COMPLETE" if path2_results else "⏳ PENDING"
    status_p3 = "✅ COMPLETE" if path3_results else "⏳ PENDING"
    
    print(f"  Path 1 (Hyperparameters): {status_p1}")
    print(f"  Path 2 (Features):        {status_p2}")
    print(f"  Path 3 (Stacking):        {status_p3}")
    
    if not path1_results:
        print("\n⏳ Waiting for Path 1 to complete...")
        return {
            "status": "incomplete",
            "message": "Waiting for Path 1 results"
        }
    
    # Baseline
    print(f"\n▶ BASELINE & IMPROVEMENTS:\n")
    print(f"  Baseline (Approach 1):    {baseline:.1f}%")
    
    # Path 1
    p1_acc = path1_results['best_configuration']['test_accuracy']
    p1_imp = p1_acc - baseline
    print(f"\n  ✅ Path 1 Result:")
    print(f"     Accuracy:              {p1_acc:.1f}%")
    print(f"     Improvement:           {p1_imp:+.1f}pp ({p1_imp/baseline*100:+.1f}%)")
    print(f"     Config:                r={path1_results['best_configuration']['radius']}, " +
          f"b={path1_results['best_configuration']['n_bits']}, " +
          f"pca={path1_results['best_configuration']['pca_components']}")
    
    # Path 2 (if available)
    if path2_results:
        p2_acc = path2_results['performance']['test_accuracy']
        p2_imp = p2_acc - baseline
        print(f"\n  ✅ Path 2 Result:")
        print(f"     Accuracy:              {p2_acc:.1f}%")
        print(f"     Improvement:           {p2_imp:+.1f}pp ({p2_imp/baseline*100:+.1f}%)")
        print(f"     Features:              {path2_results['features']['combined']}D")
    else:
        p2_acc = None
        print(f"\n  ⏳ Path 2: Pending")
    
    # Path 3 (if available)
    if path3_results:
        p3_acc = path3_results['performance']['test_accuracy']
        p3_imp = p3_acc - baseline
        print(f"\n  ✅ Path 3 Result:")
        print(f"     Accuracy:              {p3_acc:.1f}%")
        print(f"     Improvement:           {p3_imp:+.1f}pp ({p3_imp/baseline*100:+.1f}%)")
        print(f"     Base Models:           {path3_results['architecture']['base_models']}")
    else:
        p3_acc = None
        print(f"\n  ⏳ Path 3: Pending")
    
    # Summary table
    print(f"\n▶ PERFORMANCE SUMMARY TABLE:\n")
    print("  Approach               │ Accuracy │ Improvement │ Notes")
    print("  ─────────────────────┼──────────┼─────────────┼──────────────────")
    print(f"  Baseline (Approach 1)  │  {baseline:5.1f}%  │     -       │ Morgan FP + PCA + ensemble")
    print(f"  Path 1 (Hyperparams)   │  {p1_acc:5.1f}%  │   {p1_imp:+5.1f}pp   │ Best: r={path1_results['best_configuration']['radius']}, b={path1_results['best_configuration']['n_bits']}, pca={path1_results['best_configuration']['pca_components']}")
    if p2_acc:
        print(f"  Path 2 (Features)      │  {p2_acc:5.1f}%  │   {p2_acc-baseline:+5.1f}pp   │ {path2_results['features']['combined']}D features")
    else:
        print(f"  Path 2 (Features)      │    ??   │   (pending)  │ 250D extended features")
    if p3_acc:
        print(f"  Path 3 (Stacking)      │  {p3_acc:5.1f}%  │   {p3_acc-baseline:+5.1f}pp   │ {path3_results['architecture']['base_models']} base models")
    else:
        print(f"  Path 3 (Stacking)      │    ??   │   (pending)  │ 10 base models + meta-learner")
    
    # Goal assessment
    print(f"\n▶ SPRINT GOAL ASSESSMENT:\n")
    
    current_best = p3_acc if p3_acc else (p2_acc if p2_acc else p1_acc)
    goal_min = 85.0
    goal_max = 90.0
    
    if current_best:
        gap_to_goal = goal_min - current_best
        if current_best >= goal_max:
            print(f"  🏆 SPRINT COMPLETE!")
            print(f"     {current_best:.1f}% > {goal_max:.1f}% goal")
        elif current_best >= goal_min:
            print(f"  ✅ GOAL ACHIEVED!")
            print(f"     {current_best:.1f}% ≥ {goal_min:.1f}% goal")
        else:
            print(f"  ⚠️  Gap to goal: {gap_to_goal:+.1f}pp")
            print(f"     Need {gap_to_goal:.1f} more to reach {goal_min:.1f}%")
            if gap_to_goal <= 5:
                print(f"     → Close! Consider data expansion or Path 3 tuning")
            else:
                print(f"     → May need 1000+ molecules (~{int(gap_to_goal*2)} more needed)")
    
    # Recommendations
    print(f"\n▶ RECOMMENDATIONS:\n")
    
    if p3_acc and p3_acc >= 85:
        print(f"  ✅ Path 1-3 successful! Final accuracy: {p3_acc:.1f}%")
        print(f"     → Production ready")
    elif p3_acc and p3_acc >= 82:
        print(f"  ⚠️  Path 1-3 yielded {p3_acc:.1f}% (gap: {goal_min - p3_acc:.1f}pp)")
        print(f"     Recommendation: Collect 1000+ molecules from ChemBL")
        print(f"     → Expected boost: +5-10pp → {p3_acc + 7:.0f}% (estimated)")
    elif p1_acc < 77:
        print(f"  ⚠️  Path 1 underperforming ({p1_acc:.1f}%)")
        print(f"     Recommendation: Review feature engineering or try Graph approaches")
    else:
        print(f"  → Continue to next path")
    
    # Save report
    report = {
        "status": "complete" if (path1_results and path2_results and path3_results) else "in-progress",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "baseline": {
            "approach": "Approach 1: Morgan FP + PCA + Ensemble",
            "accuracy": baseline
        },
        "results": {
            "path_1": {
                "accuracy": p1_acc,
                "improvement_pp": p1_imp,
                "best_config": path1_results['best_configuration'] if path1_results else None
            } if path1_results else None,
            "path_2": {
                "accuracy": p2_acc,
                "improvement_pp": p2_acc - baseline,
                "features": path2_results['features'] if path2_results else None
            } if path2_results else None,
            "path_3": {
                "accuracy": p3_acc,
                "improvement_pp": p3_acc - baseline,
                "base_models": path3_results['architecture']['base_models'] if path3_results else None
            } if path3_results else None
        },
        "summary": {
            "best_accuracy": current_best,
            "total_improvement": current_best - baseline,
            "goal_achieved": (current_best >= goal_min) if current_best else False,
            "gap_to_goal": goal_min - current_best if current_best else goal_min - baseline
        }
    }
    
    report_path = Path('/Users/ceejayarana/diffusion_model/molecular_generation/phase4_final_comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: phase4_final_comparison_report.json")
    print("=" * 100 + "\n")
    
    return report


if __name__ == '__main__':
    generate_comparison_report()
