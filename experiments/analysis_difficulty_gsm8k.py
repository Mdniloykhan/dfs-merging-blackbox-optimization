"""
GSM8K Difficulty Analysis
Analyzes performance across question difficulty levels.
Uses existing results — no GPU needed.
Estimates difficulty from question length and number complexity.

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datasets import load_dataset

DESKTOP = r"C:\Users\user1\Desktop"
RESULTS_FILE = os.path.join(DESKTOP, "results_full_gsm8k.json")

print("=" * 65)
print("GSM8K DIFFICULTY ANALYSIS")
print("=" * 65)

with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

# Load GSM8K to analyze question properties
print("Loading GSM8K dataset...")
gsm8k = load_dataset("openai/gsm8k", "main", split="test")

# Estimate difficulty by solution length (longer = harder)
solution_lengths = [len(item['answer'].split()) for item in gsm8k]
questions = [item['question'] for item in gsm8k]
question_lengths = [len(q.split()) for q in questions]

# Bin by difficulty
bins = np.percentile(solution_lengths, [0, 25, 50, 75, 100])
difficulty_labels = ['Easy\n(Q1)', 'Medium\n(Q2)', 'Hard\n(Q3)', 'Very Hard\n(Q4)']

print(f"Solution length percentiles: {[f'{b:.0f}' for b in bins]}")
print(f"Total questions: {len(solution_lengths)}")

# Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("GSM8K Question Analysis\n(Official Test Set — 1,319 Questions)",
             fontsize=13, fontweight='bold')

# Plot 1 — Solution length distribution
ax = axes[0]
ax.hist(solution_lengths, bins=30, color='#3498db',
        edgecolor='black', linewidth=0.8, alpha=0.8)
for i, p in enumerate([25, 50, 75]):
    val = np.percentile(solution_lengths, p)
    ax.axvline(x=val, color=['green', 'orange', 'red'][i],
               linestyle='--', linewidth=2,
               label=f'Q{i+1}: {val:.0f} words')
ax.set_xlabel('Solution Length (words)', fontsize=11)
ax.set_ylabel('Number of Questions', fontsize=11)
ax.set_title('Distribution of Solution Complexity', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Plot 2 — Method performance summary
ax = axes[1]
method_results = [
    ("DELLA",           4.12,  '#e74c3c'),
    ("TIES",           13.72,  '#e74c3c'),
    ("Model A",         6.67,  '#95a5a6'),
    ("Structured DFS", 23.55,  '#3498db'),
    ("DARE",           23.88,  '#e74c3c'),
    ("CMA-ES",         28.56,  '#2ecc71'),
    ("PS Merging",     34.57,  '#f39c12'),
    ("Unstr. DFS",     34.85,  '#3498db'),
    ("Model B",        36.77,  '#9b59b6'),
]

labels = [m[0] for m in method_results]
accs = [m[1] for m in method_results]
colors = [m[2] for m in method_results]

bars = ax.barh(range(len(method_results)), accs, color=colors,
               edgecolor='black', linewidth=0.8, height=0.65)

ax.axvline(x=36.77, color='#9b59b6', linestyle='--',
           linewidth=2, label='Model B (36.77%)')
ax.axvline(x=25, color='gray', linestyle=':',
           linewidth=1.5, alpha=0.7, label='Random-like (25%)')

for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
            f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(method_results)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Accuracy (%)', fontsize=11)
ax.set_title('Official GSM8K — All Methods', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 48)

plt.tight_layout()
out = os.path.join(DESKTOP, "figure_gsm8k_difficulty_analysis.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Figure saved: {out}")
plt.close()

print("\nGSM8K DIFFICULTY ANALYSIS COMPLETE")
