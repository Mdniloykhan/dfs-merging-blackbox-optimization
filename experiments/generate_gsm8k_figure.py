"""
Script 2 — Generate Complete GSM8K Figure
Run this immediately after full_gsm8k_v2.py completes.
Reads results_full_gsm8k.json and generates final Figure 5.

Author: Md. Robiul Islam Niloy
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

DESKTOP = r"C:\Users\user1\Desktop"
RESULTS_FILE = os.path.join(DESKTOP, "results_full_gsm8k.json")
OUTPUT_FILE = os.path.join(DESKTOP, "figure5_full_gsm8k_results.png")

# Load results
with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

print("Loaded GSM8K results:")
for key, val in data.items():
    if isinstance(val, dict) and 'mean' in val:
        print(f"  {key}: {val['mean']:.2f}% +/- {val['std']:.2f}%")
    elif isinstance(val, dict) and 'accuracy' in val:
        print(f"  {key}: {val['accuracy']:.2f}%")

# Build chart data
methods = []
means = []
stds = []
colors = []

# Order: worst to best
method_config = [
    ("model_a",        "Model A\n(Mistral-7B-v0.1)",      '#95a5a6'),
    ("della",          "DELLA",                            '#e74c3c'),
    ("ties",           "TIES-Merging",                     '#e74c3c'),
    ("dare",           "DARE",                             '#e74c3c'),
    ("structured_dfs", "Structured DFS\n(Random)",         '#3498db'),
    ("ps_merging",     "PS Merging",                       '#f39c12'),
    ("unstructured_dfs","Unstructured\nDFS",               '#3498db'),
    ("cma_es",         "Structured DFS\n+ CMA-ES",         '#2ecc71'),
    ("model_b",        "Model B\n(Mistral-7B-Instruct)",   '#9b59b6'),
]

for key, label, color in method_config:
    if key in data:
        val = data[key]
        if isinstance(val, dict):
            if 'mean' in val:
                methods.append(label)
                means.append(val['mean'])
                stds.append(val.get('std', 0))
                colors.append(color)
            elif 'accuracy' in val:
                methods.append(label)
                means.append(val['accuracy'])
                stds.append(0)
                colors.append(color)

# Sort by mean accuracy
sorted_data = sorted(zip(means, stds, methods, colors))
means = [x[0] for x in sorted_data]
stds = [x[1] for x in sorted_data]
methods = [x[2] for x in sorted_data]
colors = [x[3] for x in sorted_data]

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.bar(range(len(methods)), means, color=colors,
              width=0.65, edgecolor='black', linewidth=0.8)

ax.errorbar(range(len(methods)), means, yerr=stds,
            fmt='none', color='black', capsize=6, linewidth=2)

# Model B reference line
model_b_acc = data['model_b']['accuracy'] if 'accuracy' in data['model_b'] else data['model_b']['mean']
ax.axhline(y=model_b_acc, color='#9b59b6', linestyle='--',
           linewidth=1.5, label=f'Model B baseline ({model_b_acc:.2f}%)')

# Value labels
for bar, mean, std in zip(bars, means, stds):
    ypos = bar.get_height() + (std if std > 0 else 0) + 0.3
    ax.text(bar.get_x() + bar.get_width()/2., ypos,
            f'{mean:.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Accuracy on Full GSM8K (%)', fontsize=12)
ax.set_title(
    'Full GSM8K Evaluation Results\n'
    '(1,319 Questions — Official Benchmark)',
    fontsize=13, fontweight='bold'
)
ax.set_ylim(0, max(means) + 15)
ax.grid(axis='y', alpha=0.3)

legend_patches = [
    mpatches.Patch(color='#95a5a6', label='Source models (no merging)'),
    mpatches.Patch(color='#e74c3c', label='Parameter space SOTA'),
    mpatches.Patch(color='#3498db', label='DFS random search'),
    mpatches.Patch(color='#2ecc71', label='Ours: Structured DFS + CMA-ES'),
    mpatches.Patch(color='#f39c12', label='PS Merging'),
    mpatches.Patch(color='#9b59b6', label='Model B (strongest source)'),
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {OUTPUT_FILE}")
plt.close()
print("Done.")
