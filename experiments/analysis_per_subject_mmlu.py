"""
Per-Subject MMLU Analysis
Analyzes which of 57 subjects each method performs best/worst on.
Uses existing results_full_mmlu.json — no GPU needed.

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
RESULTS_FILE = os.path.join(DESKTOP, "results_full_mmlu.json")

print("=" * 65)
print("PER-SUBJECT MMLU ANALYSIS")
print("=" * 65)

with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

# Extract per-subject accuracies
methods_subj = {}
method_names = {
    "model_a": "Model A",
    "model_b": "Model B",
    "ps_merging": "PS Merging",
    "unstructured_dfs": "Unstructured DFS",
    "structured_dfs": "Structured DFS",
    "ties": "TIES",
    "dare": "DARE",
    "della": "DELLA",
    "cma_es": "CMA-ES (Ours)"
}

for key, label in method_names.items():
    if key in data and isinstance(data[key], dict):
        if "by_subject" in data[key]:
            methods_subj[label] = data[key]["by_subject"]
        elif "subject_accuracies" in data[key]:
            methods_subj[label] = data[key]["subject_accuracies"]

if not methods_subj:
    print("No per-subject data found in results file.")
    print("Per-subject data requires running full MMLU with subject tracking.")
    exit()

# Get all subjects
all_subjects = set()
for subj_data in methods_subj.values():
    all_subjects.update(subj_data.keys())
all_subjects = sorted(all_subjects)
print(f"Found {len(all_subjects)} subjects")

# ============================================================
# FIGURE 1 — Subject categories heatmap
# ============================================================
# Group subjects by category
categories = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "college_biology",
             "college_chemistry", "college_computer_science", "college_mathematics",
             "college_physics", "computer_security", "conceptual_physics",
             "electrical_engineering", "elementary_mathematics", "formal_logic",
             "high_school_biology", "high_school_chemistry", "high_school_computer_science",
             "high_school_mathematics", "high_school_physics", "high_school_statistics",
             "machine_learning"],
    "Humanities": ["formal_logic", "high_school_european_history",
                   "high_school_us_history", "high_school_world_history",
                   "international_law", "jurisprudence", "logical_fallacies",
                   "moral_disputes", "moral_scenarios", "philosophy",
                   "prehistory", "professional_law", "world_religions"],
    "Social Sciences": ["econometrics", "high_school_geography",
                        "high_school_government_and_politics",
                        "high_school_macroeconomics", "high_school_microeconomics",
                        "high_school_psychology", "human_sexuality",
                        "political_science", "professional_psychology",
                        "public_relations", "security_studies", "sociology",
                        "us_foreign_policy"],
    "Other": ["business_ethics", "clinical_knowledge", "college_medicine",
              "global_facts", "human_aging", "management", "marketing",
              "medical_genetics", "miscellaneous", "nutrition",
              "professional_accounting", "professional_medicine",
              "virology"]
}

# Calculate category averages for key methods
key_methods = ["Model A", "Model B", "PS Merging", "CMA-ES (Ours)"]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Per-Category MMLU Performance\n(Official Benchmark — 14,042 Questions)",
             fontsize=14, fontweight='bold')

colors = ['#95a5a6', '#9b59b6', '#f39c12', '#2ecc71']

for cat_idx, (cat_name, cat_subjects) in enumerate(categories.items()):
    ax = axes[cat_idx // 2][cat_idx % 2]

    cat_data = {}
    for method in key_methods:
        if method in methods_subj:
            accs = [methods_subj[method].get(s, 0) for s in cat_subjects
                   if s in methods_subj[method]]
            if accs:
                cat_data[method] = np.mean(accs)

    if cat_data:
        methods_list = list(cat_data.keys())
        accs_list = list(cat_data.values())
        method_colors = [colors[key_methods.index(m)] for m in methods_list]

        bars = ax.bar(range(len(methods_list)), accs_list,
                     color=method_colors, edgecolor='black', linewidth=0.8)

        for bar, acc in zip(bars, accs_list):
            ax.text(bar.get_x() + bar.get_width()/2.,
                   bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(methods_list)))
        ax.set_xticklabels(methods_list, fontsize=9, rotation=15)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title(f'{cat_name} ({len(cat_subjects)} subjects)', fontsize=11)
        ax.set_ylim(0, max(accs_list) + 15)
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out1 = os.path.join(DESKTOP, "figure_per_category_mmlu.png")
plt.savefig(out1, dpi=300, bbox_inches='tight')
print(f"Figure 1 saved: {out1}")
plt.close()

# ============================================================
# FIGURE 2 — Top and Bottom subjects for CMA-ES
# ============================================================
if "CMA-ES (Ours)" in methods_subj:
    cmaes_subj = methods_subj["CMA-ES (Ours)"]
    model_b_subj = methods_subj.get("Model B", {})

    # Calculate CMA-ES advantage over Model B per subject
    advantages = {}
    for subj in all_subjects:
        if subj in cmaes_subj and subj in model_b_subj:
            advantages[subj] = cmaes_subj[subj] - model_b_subj[subj]

    if advantages:
        sorted_adv = sorted(advantages.items(), key=lambda x: x[1])
        bottom_10 = sorted_adv[:10]
        top_10 = sorted_adv[-10:]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("CMA-ES vs Model B: Subject-Level Performance Difference\n"
                    "(Official MMLU — 14,042 Questions)",
                    fontsize=13, fontweight='bold')

        # Top 10 subjects where CMA-ES helps most
        subjs = [x[0].replace('_', ' ').title()[:25] for x in top_10]
        advs = [x[1] for x in top_10]
        colors_top = ['#2ecc71' if a >= 0 else '#e74c3c' for a in advs]
        ax1.barh(range(len(subjs)), advs, color=colors_top, edgecolor='black', linewidth=0.8)
        ax1.set_yticks(range(len(subjs)))
        ax1.set_yticklabels(subjs, fontsize=9)
        ax1.set_xlabel('CMA-ES accuracy - Model B accuracy (%)', fontsize=10)
        ax1.set_title('Top 10 Subjects — CMA-ES Advantage', fontsize=11)
        ax1.axvline(x=0, color='black', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)

        # Bottom 10 subjects where CMA-ES hurts most
        subjs = [x[0].replace('_', ' ').title()[:25] for x in bottom_10]
        advs = [x[1] for x in bottom_10]
        colors_bot = ['#2ecc71' if a >= 0 else '#e74c3c' for a in advs]
        ax2.barh(range(len(subjs)), advs, color=colors_bot, edgecolor='black', linewidth=0.8)
        ax2.set_yticks(range(len(subjs)))
        ax2.set_yticklabels(subjs, fontsize=9)
        ax2.set_xlabel('CMA-ES accuracy - Model B accuracy (%)', fontsize=10)
        ax2.set_title('Bottom 10 Subjects — CMA-ES Disadvantage', fontsize=11)
        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        out2 = os.path.join(DESKTOP, "figure_subject_advantage_mmlu.png")
        plt.savefig(out2, dpi=300, bbox_inches='tight')
        print(f"Figure 2 saved: {out2}")
        plt.close()

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("OVERALL METHOD COMPARISON BY CATEGORY")
print("=" * 65)
print(f"{'Category':<20}", end="")
for m in key_methods:
    print(f"{m[:12]:>14}", end="")
print()
print("-" * (20 + 14 * len(key_methods)))

for cat_name, cat_subjects in categories.items():
    print(f"{cat_name:<20}", end="")
    for method in key_methods:
        if method in methods_subj:
            accs = [methods_subj[method].get(s, 0) for s in cat_subjects
                   if s in methods_subj[method]]
            mean = np.mean(accs) if accs else 0
            print(f"{mean:>13.1f}%", end="")
        else:
            print(f"{'N/A':>14}", end="")
    print()

print("=" * 65)
print("\nPER-SUBJECT ANALYSIS COMPLETE")
