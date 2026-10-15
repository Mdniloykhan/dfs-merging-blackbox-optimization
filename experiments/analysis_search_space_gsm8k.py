"""
Search Space Analysis — Official GSM8K
Uses existing results_full_gsm8k.json — no GPU needed.

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DESKTOP = r"C:\Users\user1\Desktop"
RESULTS_FILE = os.path.join(DESKTOP, "results_full_gsm8k.json")

print("=" * 65)
print("SEARCH SPACE ANALYSIS — OFFICIAL GSM8K")
print("=" * 65)

with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

# ============================================================
# FIGURE 1 — Complete GSM8K results with search space context
# ============================================================
methods = [
    ("della",          "DELLA",                     '#e74c3c', 4.12,  0.29),
    ("ties",           "TIES",                      '#e74c3c', 13.72, 0.00),
    ("model_a",        "Model A",                   '#95a5a6', 6.67,  0.00),
    ("structured_dfs", "Structured DFS\n(Random)",  '#3498db', 23.55, 5.34),
    ("dare",           "DARE",                      '#e74c3c', 23.88, 1.77),
    ("cma_es",         "Structured DFS\n+CMA-ES",   '#2ecc71', 28.56, 2.63),
    ("ps_merging",     "PS Merging",                '#f39c12', 34.57, 0.00),
    ("unstructured_dfs","Unstructured\nDFS",        '#3498db', 34.85, 1.18),
    ("model_b",        "Model B",                   '#9b59b6', 36.77, 0.00),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Official GSM8K Results (1,319 Questions)\nAll Merging Methods Below Model B",
             fontsize=13, fontweight='bold')

# Plot 1 — Bar chart
ax = axes[0]
labels = [m[1] for m in methods]
means = [m[3] for m in methods]
stds = [m[4] for m in methods]
colors = [m[2] for m in methods]

bars = ax.bar(range(len(methods)), means, color=colors,
              width=0.65, edgecolor='black', linewidth=0.8)
ax.errorbar(range(len(methods)), means, yerr=stds,
            fmt='none', color='black', capsize=5, linewidth=2)

# Model B reference line
ax.axhline(y=36.77, color='#9b59b6', linestyle='--',
           linewidth=2, label='Model B (36.77%)')
# Model A reference line
ax.axhline(y=6.67, color='gray', linestyle=':',
           linewidth=1.5, label='Model A (6.67%)')

for bar, mean, std in zip(bars, means, stds):
    ypos = bar.get_height() + (std if std > 0 else 0) + 0.3
    ax.text(bar.get_x() + bar.get_width()/2., ypos,
            f'{mean:.1f}%', ha='center', va='bottom',
            fontsize=8, fontweight='bold')

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Method Comparison', fontsize=11)
ax.set_ylim(0, 50)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Plot 2 — Search space vs performance scatter
ax = axes[1]
# Search space sizes for each method type
method_data = [
    ("PS Merging", 0, 34.57, '#f39c12'),      # 0 binary decisions
    ("Unstructured DFS", 291, 34.85, '#3498db'), # full 291 dims
    ("Structured DFS", 146.96, 23.55, '#3498db'), # reduced dims
    ("CMA-ES", 146.96, 28.56, '#2ecc71'),      # optimized
    ("TIES", 291, 13.72, '#e74c3c'),
    ("DARE", 291, 23.88, '#e74c3c'),
    ("DELLA", 291, 4.12, '#e74c3c'),
]

for name, space, acc, color in method_data:
    ax.scatter(space, acc, s=200, color=color, edgecolor='black',
              linewidth=1.5, zorder=5)
    ax.annotate(name, (space, acc),
               textcoords="offset points", xytext=(5, 5),
               fontsize=8)

ax.axhline(y=36.77, color='#9b59b6', linestyle='--',
           linewidth=2, label='Model B baseline (36.77%)')
ax.set_xlabel('Effective Search Space (dimensions)', fontsize=11)
ax.set_ylabel('Official GSM8K Accuracy (%)', fontsize=11)
ax.set_title('Search Space Size vs Performance', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(-20, 320)
ax.set_ylim(0, 45)

legend_patches = [
    mpatches.Patch(color='#95a5a6', label='Source models'),
    mpatches.Patch(color='#e74c3c', label='Parameter space SOTA'),
    mpatches.Patch(color='#3498db', label='DFS random search'),
    mpatches.Patch(color='#2ecc71', label='Ours: CMA-ES'),
    mpatches.Patch(color='#f39c12', label='PS Merging'),
    mpatches.Patch(color='#9b59b6', label='Model B'),
]
axes[0].legend(handles=legend_patches, loc='upper left', fontsize=7)

plt.tight_layout()
out1 = os.path.join(DESKTOP, "figure_gsm8k_complete_results.png")
plt.savefig(out1, dpi=300, bbox_inches='tight')
print(f"Figure 1 saved: {out1}")
plt.close()

# ============================================================
# FIGURE 2 — GSM8K vs MMLU comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

method_comparison = [
    ("Model B",         36.77, 51.10, '#9b59b6'),
    ("PS Merging",      34.57, 50.40, '#f39c12'),
    ("Unstructured DFS",34.85, 49.56, '#3498db'),
    ("CMA-ES",          28.56, 43.08, '#2ecc71'),
    ("Structured DFS",  23.55, 39.00, '#3498db'),
    ("DARE",            23.88, 34.42, '#e74c3c'),
    ("TIES",            13.72, 30.48, '#e74c3c'),
    ("DELLA",            4.12, 26.93, '#e74c3c'),
    ("Model A",          6.67, 20.37, '#95a5a6'),
]

for name, gsm8k, mmlu, color in method_comparison:
    ax.scatter(gsm8k, mmlu, s=300, color=color,
              edgecolor='black', linewidth=1.5, zorder=5)
    offset = (2, 2)
    if name == "DELLA":
        offset = (2, -4)
    if name == "Model A":
        offset = (2, -4)
    ax.annotate(name, (gsm8k, mmlu),
               textcoords="offset points", xytext=offset,
               fontsize=9, fontweight='bold')

# Diagonal reference line
ax.plot([0, 60], [0, 60], 'gray', linestyle=':', linewidth=1,
        alpha=0.5, label='Equal performance line')

ax.set_xlabel('Official GSM8K Accuracy (%)', fontsize=12)
ax.set_ylabel('Official MMLU Accuracy (%)', fontsize=12)
ax.set_title('GSM8K vs MMLU — All Methods\nOfficial Benchmark Comparison',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xlim(0, 45)
ax.set_ylim(15, 60)

legend_patches = [
    mpatches.Patch(color='#95a5a6', label='Source models'),
    mpatches.Patch(color='#e74c3c', label='Parameter space SOTA'),
    mpatches.Patch(color='#3498db', label='DFS random search'),
    mpatches.Patch(color='#2ecc71', label='Ours: CMA-ES'),
    mpatches.Patch(color='#f39c12', label='PS Merging'),
    mpatches.Patch(color='#9b59b6', label='Model B'),
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

plt.tight_layout()
out2 = os.path.join(DESKTOP, "figure_gsm8k_vs_mmlu.png")
plt.savefig(out2, dpi=300, bbox_inches='tight')
print(f"Figure 2 saved: {out2}")
plt.close()

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("OFFICIAL GSM8K COMPLETE RESULTS")
print("=" * 65)
print(f"{'Method':<25} {'GSM8K':>10} {'Std':>8} {'vs Model B':>12}")
print("-" * 60)
print(f"{'Model B':<25} {'36.77%':>10} {'—':>8} {'Baseline':>12}")
for _, label, _, mean, std in sorted(methods, key=lambda x: x[3], reverse=True):
    if label not in ["Model B", "Model A"]:
        vs_b = mean - 36.77
        print(f"{label.replace(chr(10), ' '):<25} {mean:>9.2f}% {std:>7.2f}% {vs_b:>+11.2f}%")
print(f"{'Model A':<25} {'6.67%':>10} {'—':>8} {6.67-36.77:>+11.2f}%")
print("=" * 65)
print("\nKEY FINDING: All merging methods score below Model B on official GSM8K")
print("DELLA (4.12%) scores BELOW Model A (6.67%) — merging destroys the model")
print("\nGSM8K ANALYSIS COMPLETE")
