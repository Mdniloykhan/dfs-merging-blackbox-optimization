"""
Statistical Significance Tests — Official GSM8K Results
Run this AFTER full_gsm8k_v2.py completes.
No GPU needed — pure Python/scipy.

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import json
import numpy as np
from scipy import stats
import os

DESKTOP = r"C:\Users\DCO\Desktop"
RESULTS_FILE = os.path.join(DESKTOP, "results_full_gsm8k.json")

print("=" * 65)
print("STATISTICAL SIGNIFICANCE TESTS — OFFICIAL GSM8K")
print("=" * 65)

# Load results
with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

print(f"\nLoaded results from {RESULTS_FILE}")

# Extract runs
methods = {}
for key in ["ps_merging", "unstructured_dfs", "structured_dfs",
            "ties", "dare", "della", "cma_es"]:
    if key in data and "all_runs" in data[key]:
        methods[key] = data[key]["all_runs"]
        print(f"{key}: {np.mean(data[key]['all_runs']):.2f}% "
              f"+/- {np.std(data[key]['all_runs']):.2f}% "
              f"({len(data[key]['all_runs'])} runs)")

model_b_acc = data["model_b"]["accuracy"] if "accuracy" in data["model_b"] \
              else data["model_b"]["mean"]
print(f"Model B baseline: {model_b_acc:.2f}%")

if "cma_es" not in methods:
    print("\nCMA-ES results not yet available. Run after GSM8K completes.")
    exit()

cma_es_runs = methods["cma_es"]

print("\n" + "=" * 65)
print("CMA-ES vs All Methods (Wilcoxon Signed-Rank Test)")
print("=" * 65)
print(f"{'Comparison':<35} {'CMA-ES':>8} {'Other':>8} "
      f"{'Diff':>8} {'p-value':>10} {'Effect d':>10} {'Sig?':>6}")
print("-" * 95)

for method_name, method_runs in methods.items():
    if method_name == "cma_es":
        continue

    cma_mean = np.mean(cma_es_runs)
    other_mean = np.mean(method_runs)
    diff = cma_mean - other_mean

    # Pad shorter array if needed
    n = min(len(cma_es_runs), len(method_runs))
    cma_arr = np.array(cma_es_runs[:n])
    other_arr = np.array(method_runs[:n])

    # Wilcoxon signed-rank test
    try:
        if np.all(cma_arr == other_arr):
            p_val = 1.0
            stat = 0
        else:
            stat, p_val = stats.wilcoxon(cma_arr, other_arr,
                                          alternative='greater')
    except Exception:
        _, p_val = stats.ttest_rel(cma_arr, other_arr)

    # Cohen's d
    pooled_std = np.sqrt((np.std(cma_arr)**2 + np.std(other_arr)**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    sig = "✓ Yes" if p_val < 0.05 else ("~ Mar" if p_val < 0.1 else "✗ No")

    print(f"{'CMA-ES vs ' + method_name:<35} "
          f"{cma_mean:>7.2f}% {other_mean:>7.2f}% "
          f"{diff:>+7.2f}% {p_val:>10.4f} {cohens_d:>10.2f} {sig:>6}")

print("\n" + "=" * 65)
print("CMA-ES vs Model B Baseline")
print("=" * 65)
cma_mean = np.mean(cma_es_runs)
diff = cma_mean - model_b_acc
print(f"CMA-ES: {cma_mean:.2f}%")
print(f"Model B: {model_b_acc:.2f}%")
print(f"Difference: {diff:+.2f}%")
if diff > 0:
    print("✓ CMA-ES BEATS Model B on official GSM8K")
else:
    print(f"✗ CMA-ES is {abs(diff):.2f}% below Model B")

print("\n" + "=" * 65)
print("SUMMARY TABLE — Official GSM8K")
print("=" * 65)
print(f"{'Method':<25} {'Mean':>10} {'Std':>8} {'vs Model B':>12}")
print("-" * 60)
print(f"{'Model B':<25} {model_b_acc:>9.2f}%  {'—':>7}  {'Baseline':>12}")
for key in ["cma_es", "unstructured_dfs", "structured_dfs",
            "ps_merging", "dare", "ties", "della"]:
    if key in methods:
        mean = np.mean(methods[key])
        std = np.std(methods[key])
        vs_b = mean - model_b_acc
        print(f"{key:<25} {mean:>9.2f}% {std:>7.2f}% {vs_b:>+11.2f}%")
print("=" * 60)

# Save results
output = {
    "date": __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "benchmark": "Official GSM8K (1,319 questions)",
    "model_b_baseline": model_b_acc,
    "comparisons": {}
}

for method_name, method_runs in methods.items():
    if method_name == "cma_es":
        continue
    n = min(len(cma_es_runs), len(method_runs))
    cma_arr = np.array(cma_es_runs[:n])
    other_arr = np.array(method_runs[:n])
    diff = np.mean(cma_arr) - np.mean(other_arr)
    try:
        if np.all(cma_arr == other_arr):
            p_val = 1.0
        else:
            _, p_val = stats.wilcoxon(cma_arr, other_arr, alternative='greater')
    except Exception:
        _, p_val = stats.ttest_rel(cma_arr, other_arr)
    pooled_std = np.sqrt((np.std(cma_arr)**2 + np.std(other_arr)**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    output["comparisons"][f"cma_es_vs_{method_name}"] = {
        "mean_diff": float(diff),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "significant": bool(p_val < 0.05)
    }

out_file = os.path.join(DESKTOP, "statistical_tests_gsm8k.json")
with open(out_file, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_file}")
print("\nSTATISTICAL TESTS COMPLETE")
