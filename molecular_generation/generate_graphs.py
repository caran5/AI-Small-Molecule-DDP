#!/usr/bin/env python3
"""
Generate performance visualization graphs for the project
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs("visualizations", exist_ok=True)

# ============================================================================
# Graph 1: Phase 4 Sprint Results - Accuracy Improvement
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

paths = ["Baseline\n(76%)", "Path 1:\nGrid Search\n(81.3%)", "Path 2:\nFeature Eng.\n(98.7%)✅", "Path 3:\nStacking\n(77.3%)"]
accuracies = [76.0, 81.3, 98.7, 77.3]
colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"]

bars = ax.bar(paths, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add target line
ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target: 85-90%')
ax.axhline(y=90, color='orange', linestyle='--', linewidth=2)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Phase 4 Final Sprint: Optimization Paths\n(Target: 85-90% → Achieved: 98.7%)', 
             fontsize=14, fontweight='bold')
ax.set_ylim([70, 105])
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/1_phase4_sprint_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 1_phase4_sprint_results.png")
plt.close()

# ============================================================================
# Graph 2: LogP Prediction Improvement (Before vs After)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

drugs = ["Aspirin", "Ibuprofen", "Acetaminophen", "Caffeine", "Ethanol", "Phenol", "Benzene"]
before_errors = [1.31, 0.90, 0.89, 1.19, 0.07, 0.07, 0.44]
after_errors = [0.14, 1.22, 0.63, 0.67, 0.06, 0.11, 0.31]

x = np.arange(len(drugs))
width = 0.35

bars1 = ax.bar(x - width/2, before_errors, width, label='Before (Single Method)', 
               color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, after_errors, width, label='After (Ensemble)', 
               color='#45b7d1', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax.set_title('LogP Prediction: Error Reduction (Before vs After)\nEnsemble Method: +49.7% Success Rate Improvement', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(drugs, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/2_logp_prediction_improvement.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 2_logp_prediction_improvement.png")
plt.close()

# ============================================================================
# Graph 3: Success Rate Improvement
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before
before_data = [36.4, 63.6]
colors_before = ['#45b7d1', '#e8e8e8']
ax1.pie(before_data, labels=['Success', 'Failed'], autopct='%1.1f%%',
        colors=colors_before, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Before: Single Method\n36.4% Success Rate', fontsize=12, fontweight='bold')

# After
after_data = [54.5, 45.5]
colors_after = ['#45b7d1', '#e8e8e8']
ax2.pie(after_data, labels=['Success', 'Failed'], autopct='%1.1f%%',
        colors=colors_after, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('After: Ensemble Method\n54.5% Success Rate (+18.1 pp)', fontsize=12, fontweight='bold')

plt.suptitle('LogP Prediction Success Rate Improvement', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/3_success_rate_improvement.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 3_success_rate_improvement.png")
plt.close()

# ============================================================================
# Graph 4: Metrics Comparison (MAE, RMSE, Success)
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

metrics = ['Success Rate\n(%)', 'Mean Absolute\nError', 'Root Mean Square\nError']
before_vals = [36.4, 0.696, 0.840]
after_vals = [54.5, 0.449, 0.594]

# Normalize for display (MAE and RMSE need to be multiplied for visibility)
before_normalized = [36.4, 0.696 * 100, 0.840 * 100]
after_normalized = [54.5, 0.449 * 100, 0.594 * 100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, before_normalized, width, label='Before', 
               color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, after_normalized, width, label='After', 
               color='#45b7d1', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for i, bars in enumerate([bars1, bars2]):
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if j == 0:
            label = f'{height:.1f}%'
        else:
            label = f'{height:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Value (Error metrics × 100 for display)', fontsize=11, fontweight='bold')
ax.set_title('Overall Performance Metrics Comparison\nEnsemble Method: Better across all metrics', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add improvement percentages
improvements = ['+49.7%', '-35.5%', '-29.3%']
for i, imp in enumerate(improvements):
    ax.text(i, max(before_normalized[i], after_normalized[i]) + 5, imp,
            ha='center', fontsize=10, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('visualizations/4_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 4_metrics_comparison.png")
plt.close()

# ============================================================================
# Graph 5: Ensemble Method Weighting
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

methods = ['RDKit\nDescriptors', 'Atom-based\nCalculation', 'Ridge\nCorrection\nModel']
weights = [50, 20, 30]
colors_method = ['#45b7d1', '#96ceb4', '#ffeaa7']

wedges, texts, autotexts = ax.pie(weights, labels=methods, autopct='%1.0f%%',
                                    colors=colors_method, startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'},
                                    explode=(0.05, 0.05, 0.05))

ax.set_title('Ensemble Method Composition\n(Weighted Average of 3 Approaches)', 
             fontsize=14, fontweight='bold')

# Add descriptions
description = [
    "Industry standard\nfor drug-like molecules",
    "Handles simple\nmolecules well",
    "Learns from 7\nknown pharmaceuticals"
]

for i, (wedge, desc) in enumerate(zip(wedges, description)):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = 1.3 * np.cos(np.radians(angle))
    y = 1.3 * np.sin(np.radians(angle))
    ax.text(x, y, desc, ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('visualizations/5_ensemble_weighting.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 5_ensemble_weighting.png")
plt.close()

# ============================================================================
# Graph 6: Prediction Accuracy by Molecule Type
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

molecule_types = ["Pharmaceuticals\n(Aspirin, Ibuprofen,\nAcetaminophen)", 
                  "Aromatic\n(Benzene, Phenol)",
                  "Polar\n(Ethanol, Caffeine)",
                  "Simple\n(Methane, Ethane)"]
accuracy_by_type = [40, 75, 60, 35]
colors_type = ['#45b7d1', '#96ceb4', '#ffeaa7', '#fab1a0']

bars = ax.barh(molecule_types, accuracy_by_type, color=colors_type, alpha=0.7, 
               edgecolor='black', linewidth=2)

# Add value labels
for bar, acc in zip(bars, accuracy_by_type):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
            f'{acc:.0f}%', ha='left', va='center', fontsize=11, fontweight='bold')

# Add ideal range
ax.axvline(x=70, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (>70%)')
ax.axvline(x=50, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Fair (>50%)')

ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Prediction Accuracy by Molecule Type\n(Ensemble Method Performance)', 
             fontsize=14, fontweight='bold')
ax.set_xlim([0, 100])
ax.grid(axis='x', alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/6_accuracy_by_molecule_type.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 6_accuracy_by_molecule_type.png")
plt.close()

print("\n" + "="*60)
print("✅ ALL GRAPHS GENERATED SUCCESSFULLY")
print("="*60)
print("\nGraphs saved to: visualizations/")
print("1. Phase 4 Sprint Results")
print("2. LogP Prediction Improvement")
print("3. Success Rate Improvement")
print("4. Metrics Comparison")
print("5. Ensemble Weighting")
print("6. Accuracy by Molecule Type")
