"""
Search Space Visualization
Shows distribution of active layers and binary selection patterns.
Uses existing experiment results — no GPU needed.

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

DESKTOP = r"C:\Users\user1\Desktop"

print("=" * 65)
print("SEARCH SPACE VISUALIZATION")
print("=" * 65)

# ============================================================
# FIGURE 1 — Active layer distribution across seeds
# ============================================================
# Simulate from our known results (selection prob = 0.5, 291 layers)
np.random.seed(42)
n_simulations = 1000
active_counts = []
for _ in range(n_simulations):
    z = np.random.binomial(1, 0.5, 291)
    active_counts.append(z.sum())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Search Space Analysis — Structured DFS Merging\n"
             "Mistral-7B (291 Parameter Groups)",
             fontsize=13, fontweight='bold')

# Plot 1 — Distribution of active layers
ax = axes[0]
ax.hist(active_counts, bins=30, color='#3498db', edgecolor='black',
        linewidth=0.8, alpha=0.8)
ax.axvline(x=np.mean(active_counts), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(active_counts):.1f}')
ax.axvline(x=146.96, color='green', linestyle='-',
           linewidth=2, label=f'Measured: 146.96')
ax.set_xlabel('Number of Active Layers', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Active Layers\n(1,000 random seeds)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Plot 2 — Search space reduction visualization
ax = axes[1]
categories = ['Total\nParameter\nGroups', 'Active\nLayers\n(Mean)', 'Inactive\nLayers\n(Mean)']
values = [291, 146.96, 144.04]
colors = ['#95a5a6', '#2ecc71', '#e74c3c']
bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2.,
           bar.get_height() + 2,
           f'{val:.0f}', ha='center', va='bottom',
           fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Parameter Groups', fontsize=11)
ax.set_title('Search Space Reduction\n49.5% reduction confirmed', fontsize=10)
ax.set_ylim(0, 350)
ax.grid(axis='y', alpha=0.3)

# Add annotation
ax.annotate('49.5%\nreduction', xy=(1.5, 73), xytext=(1.5, 200),
           fontsize=11, fontweight='bold', color='red', ha='center',
           arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Plot 3 — Selection probability vs active layers
ax = axes[2]
probs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
expected_active = [p * 291 for p in probs]
measured_active = [29.3, 56.3, 80.3, 143.0, 210.7, 261.7]

ax.plot(probs, expected_active, 'b--', linewidth=2, label='Expected (theoretical)')
ax.plot(probs, measured_active, 'go-', linewidth=2, markersize=8, label='Measured (experimental)')
ax.fill_between(probs, expected_active, measured_active, alpha=0.2, color='gray')
ax.set_xlabel('Selection Probability', fontsize=11)
ax.set_ylabel('Active Layers', fontsize=11)
ax.set_title('Selection Probability vs\nActive Layers', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 310)

plt.tight_layout()
out1 = os.path.join(DESKTOP, "figure_search_space_analysis.png")
plt.savefig(out1, dpi=300, bbox_inches='tight')
print(f"Figure 1 saved: {out1}")
plt.close()

# ============================================================
# FIGURE 2 — Type-I interaction visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Type-I Interaction in DFS Merging\n"
             "Conditional Dependency Between Binary and Continuous Variables",
             fontsize=13, fontweight='bold')

# Plot 1 — Variance vs selection probability
ax = axes[0]
probs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
stds = [6.9, 9.0, 7.3, 3.5, 0.5, 0.5]
ax.plot(probs, stds, 'ro-', linewidth=2, markersize=10)
ax.fill_between(probs, stds, alpha=0.2, color='red')
ax.set_xlabel('Binary Selection Probability (p)', fontsize=11)
ax.set_ylabel('Performance Std Dev (%)', fontsize=11)
ax.set_title('Type-I Interaction Evidence\nHigher variance at lower selection probability',
            fontsize=10)
ax.grid(alpha=0.3)
ax.annotate('High variance\n(few active layers\n= strong dependency)',
           xy=(0.1, 6.9), xytext=(0.3, 8.5),
           fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('Low variance\n(many active layers\n= weak dependency)',
           xy=(0.9, 0.5), xytext=(0.6, 3.0),
           fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

# Plot 2 — Effective search space
ax = axes[1]
theta = np.linspace(0, 2*np.pi, 100)

# Unstructured DFS search space (full)
circle_full = plt.Circle((0.3, 0.5), 0.25, color='#e74c3c',
                         alpha=0.3, label='Unstructured DFS\n(Full space, N=291)')
ax.add_patch(circle_full)
ax.text(0.3, 0.5, '291\ndims', ha='center', va='center',
       fontsize=11, fontweight='bold', color='#c0392b')

# Structured DFS search space (reduced)
circle_reduced = plt.Circle((0.72, 0.5), 0.18, color='#2ecc71',
                            alpha=0.4, label='Structured DFS\n(Reduced space, ~147)')
ax.add_patch(circle_reduced)
ax.text(0.72, 0.5, '~147\ndims', ha='center', va='center',
       fontsize=11, fontweight='bold', color='#27ae60')

ax.set_xlim(0, 1)
ax.set_ylim(0.1, 0.9)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Search Space Reduction\n49.5% fewer effective dimensions',
            fontsize=10)
ax.axis('off')
ax.annotate('49.5%\nsmaller', xy=(0.72, 0.32), xytext=(0.5, 0.2),
           fontsize=10, fontweight='bold', color='green', ha='center',
           arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.tight_layout()
out2 = os.path.join(DESKTOP, "figure_type1_interaction.png")
plt.savefig(out2, dpi=300, bbox_inches='tight')
print(f"Figure 2 saved: {out2}")
plt.close()

print("\n" + "=" * 65)
print("SEARCH SPACE ANALYSIS COMPLETE")
print("=" * 65)
print(f"Figures saved to Desktop:")
print(f"  1. figure_search_space_analysis.png")
print(f"  2. figure_type1_interaction.png")
