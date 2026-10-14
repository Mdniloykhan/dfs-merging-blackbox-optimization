"""
Statistical Significance Tests — Official MMLU Results
Runs immediately — no GPU needed.
Uses results_full_mmlu.json already on Desktop.

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import json
import numpy as np
from scipy import stats
import os

DESKTOP = r"C:\Users\user1\Desktop"
RESULTS_FILE = os.path.join(DESKTOP, "results_full_mmlu.json")

print("=" * 65)
print("STATISTICAL SIGNIFICANCE TESTS — OFFICIAL MMLU")
print("14,042 questions, 57 subjects")
print("=" * 65)

# Load results
with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

print(f"\nLoaded results from {RESULTS_FILE}")

# Extract runs
methods = {}
method_order = ["ps_merging", "unstructured_dfs", "structured_dfs",
                "ties", "dare", "della", "cma_es"]

for key in method_order:
    if key in data and isinstance(data[key], dict):
        if "mean" in data[key] and "all_runs" in data[key]:
            methods[key] = data[key]["all_runs"]
            print(f"{key}: {data[key]['mean']:.2f}% "
                  f"+/- {data[key]['std']:.2f}% "
                  f"({len(data[key]['all_runs'])} runs)")

model_b_acc = data["model_b"]["overall_accuracy"] \
    if "overall_accuracy" in data["model_b"] \
    else data["model_b"].get("accuracy", data["model_b"].get("mean", 51.10))

model_a_acc = data["model_a"]["overall_accuracy"] \
    if "overall_accuracy" in data["model_a"] \
    else data["model_a"].get("accuracy", data["model_a"].get("mean", 20.37))

print(f"Model A baseline: {model_a_acc:.2f}%")
print(f"Model B baseline: {model_b_acc:.2f}%")

if "cma_es" not in methods:
    print("\nCMA-ES results not found. Check results file.")
    exit()

cma_es_runs = methods["cma_es"]

print("\n" + "=" * 65)
print("CMA-ES vs All Methods (Wilcoxon Signed-Rank Test)")
print("=" * 65)
print(f"{'Comparison':<35} {'CMA-ES':>8} {'Other':>8} "
      f"{'Diff':>8} {'p-value':>10} {'Effect d':>10} {'Sig?':>6}")
print("-" * 95)

comparison_results = {}

for method_name, method_runs in methods.items():
    if method_name == "cma_es":
        continue

    cma_mean = np.mean(cma_es_runs)
    other_mean = np.mean(method_runs)
    diff = cma_mean - other_mean

    n = min(len(cma_es_runs), len(method_runs))
    cma_arr = np.array(cma_es_runs[:n])
    other_arr = np.array(method_runs[:n])

    # Wilcoxon signed-rank test
    try:
        if np.all(cma_arr == other_arr):
            p_val = 1.0
        else:
            _, p_val = stats.wilcoxon(cma_arr, other_arr,
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

    comparison_results[f"cma_es_vs_{method_name}"] = {
        "cma_es_mean": float(cma_mean),
        "other_mean": float(other_mean),
        "mean_diff": float(diff),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "significant": bool(p_val < 0.05)
    }

print("\n" + "=" * 65)
print("CMA-ES vs Model B Baseline")
print("=" * 65)
cma_mean = np.mean(cma_es_runs)
diff = cma_mean - model_b_acc
print(f"CMA-ES: {cma_mean:.2f}%")
print(f"Model B: {model_b_acc:.2f}%")
print(f"Difference: {diff:+.2f}%")
if diff > 0:
    print("✓ CMA-ES BEATS Model B on official MMLU")
else:
    print(f"✗ CMA-ES is {abs(diff):.2f}% below Model B on official MMLU")

# PS Merging vs Model B
ps_mean = np.mean(methods["ps_merging"]) if "ps_merging" in methods else None
if ps_mean:
    print(f"\nPS Merging vs Model B: {ps_mean:.2f}% vs {model_b_acc:.2f}% "
          f"({ps_mean-model_b_acc:+.2f}%)")

print("\n" + "=" * 65)
print("COMPLETE RANKING — Official MMLU")
print("=" * 65)
print(f"{'Method':<25} {'Mean':>10} {'Std':>8} {'vs Model B':>12}")
print("-" * 60)
print(f"{'Model B':<25} {model_b_acc:>9.2f}%  {'—':>7}  {'Baseline':>12}")

all_results = []
for key in method_order:
    if key in methods:
        mean = np.mean(methods[key])
        std = np.std(methods[key])
        all_results.append((mean, key, std))

all_results.sort(reverse=True)
for mean, key, std in all_results:
    vs_b = mean - model_b_acc
    print(f"{key:<25} {mean:>9.2f}% {std:>7.2f}% {vs_b:>+11.2f}%")

print(f"{'Model A':<25} {model_a_acc:>9.2f}%  {'—':>7}  "
      f"{model_a_acc-model_b_acc:>+11.2f}%")
print("=" * 60)

print("\n" + "=" * 65)
print("KEY FINDINGS")
print("=" * 65)
ties_mean = np.mean(methods["ties"]) if "ties" in methods else 0
della_mean = np.mean(methods["della"]) if "della" in methods else 0
random_chance = 25.0

if ties_mean < random_chance + 10:
    print(f"✗ TIES ({ties_mean:.2f}%) near random chance ({random_chance}%)")
if della_mean < random_chance + 5:
    print(f"✗ DELLA ({della_mean:.2f}%) near/below random chance ({random_chance}%)")
if ps_mean and ps_mean > model_b_acc - 2:
    print(f"✓ PS Merging ({ps_mean:.2f}%) nearly matches Model B ({model_b_acc:.2f}%)")
if cma_mean > ties_mean and cma_mean > della_mean:
    print(f"✓ CMA-ES ({cma_mean:.2f}%) outperforms TIES and DELLA")
if cma_mean < model_b_acc:
    print(f"⚠ CMA-ES ({cma_mean:.2f}%) below Model B ({model_b_acc:.2f}%) — "
          f"proxy optimization gap identified")

# Save results
output = {
    "date": __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "benchmark": "Official MMLU (14,042 questions, 57 subjects)",
    "model_a_baseline": float(model_a_acc),
    "model_b_baseline": float(model_b_acc),
    "method_means": {k: float(np.mean(v)) for k, v in methods.items()},
    "method_stds": {k: float(np.std(v)) for k, v in methods.items()},
    "comparisons": comparison_results
}

out_file = os.path.join(DESKTOP, "statistical_tests_mmlu.json")
with open(out_file, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_file}")
print("\nSTATISTICAL TESTS MMLU COMPLETE")
