# Black-Box Optimization of Mixed Binary-Continuous Variables: Challenges and Opportunities in Evolutionary Model Merging

[![arXiv](https://img.shields.io/badge/arXiv-2605.12326-b31b1b.svg)](https://arxiv.org/abs/2605.12326)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20123507-blue)](https://doi.org/10.5281/zenodo.20123506)

**Author:** Md. Robiul Islam Niloy  
**Institution:** BRAC University, Dhaka, Bangladesh  
**Email:** md.niloy26643@gmail.com  
**arXiv:** https://arxiv.org/abs/2605.12326  
**Zenodo:** https://doi.org/10.5281/zenodo.20123506

---

## Abstract

Model merging has emerged as a cost-effective paradigm for combining capabilities of multiple large language models (LLMs) without additional training. This work formally characterizes the Data Flow Space (DFS) merging problem as a **mixed binary-continuous black-box optimization problem**, identifying three fundamental challenges: mixed variable types, high dimensionality (291 parameter groups for Mistral-7B), and conditional variable dependencies corresponding to Type-I interaction as characterized by Akimoto et al. [GECCO 2025].

We evaluate seven merging methods on Mistral-7B models across official GSM8K (1,319 questions) and MMLU (14,042 questions, 57 subjects) benchmarks. Key findings: all tested merging methods fail to surpass the stronger source model on official benchmarks. DELLA (4.12%) scores below even the base model (6.67%) on GSM8K — demonstrating complete model degradation. These results validate the fundamental difficulty of the Type-I interaction structure and motivate algorithms specifically designed for this challenge, such as the concurrent IGBD work of Ong et al. [arXiv:2606.12885].

---

## Key Results

### Official GSM8K (1,319 questions — 3 runs)

| Method | Accuracy | Std | vs Model B |
|--------|----------|-----|------------|
| Model B (Mistral-7B-Instruct) | **36.77%** | — | Baseline |
| Unstructured DFS | 34.85% | 1.18% | -1.92% |
| PS Merging | 34.57% | 0.00% | -2.20% |
| Structured DFS + CMA-ES (Ours) | 28.56% | 2.63% | -8.21% |
| DARE | 23.88% | 1.77% | -12.89% |
| Structured DFS (Random) | 23.55% | 5.34% | -13.22% |
| TIES | 13.72% | 0.00% | -23.05% |
| Model A (Mistral-7B-v0.1) | 6.67% | — | -30.10% |
| DELLA | 4.12% | 0.29% | -32.65% |

### Official MMLU (14,042 questions, 57 subjects — 3 runs)

| Method | Accuracy | Std | vs Model B |
|--------|----------|-----|------------|
| Model B (Mistral-7B-Instruct) | **51.10%** | — | Baseline |
| PS Merging | 50.40% | 0.00% | -0.70% |
| Unstructured DFS | 49.56% | 0.88% | -1.54% |
| Structured DFS + CMA-ES (Ours) | 43.08% | 2.83% | -8.02% |
| Structured DFS (Random) | 39.00% | 4.32% | -12.10% |
| DARE | 34.42% | 4.19% | -16.68% |
| TIES | 30.48% | 0.00% | -20.62% |
| DELLA | 26.93% | 0.21% | -24.17% |
| Model A (Mistral-7B-v0.1) | 20.37% | — | -30.73% |

### Development Subset Results (used for ablation only)

> ⚠️ Note: These results use controlled development subsets. **Not directly comparable to official benchmarks.**

**GSM8K Subset (97 questions, 5 runs):**

| Method | Accuracy | Std |
|--------|----------|-----|
| **Structured DFS + CMA-ES (Ours)** | **88.1%** | **1.1%** |
| Structured DFS (Random) | 86.5% | 1.1% |
| Unstructured DFS | 86.0% | 0.5% |
| Model B | 85.4% | — |
| PS Merging | 84.4% | 0.0% |
| DARE | 77.3% | 5.2% |
| TIES | 53.1% | 0.0% |
| DELLA | 53.1% | 2.4% |
| Model A | 43.8% | — |

**MMLU Subset (100 questions, 10 subjects, 5 runs):**

| Method | Accuracy | Std |
|--------|----------|-----|
| **Structured DFS + CMA-ES (Ours)** | **93.2%** | **1.2%** |
| PS Merging | 93.0% | 0.0% |
| DARE | 92.2% | 2.5% |
| Structured DFS (Random) | 91.6% | 1.2% |
| TIES | 90.0% | 0.0% |
| Model A | 89.0% | — |
| Model B | 86.0% | — |
| DELLA | 69.2% | 4.8% |

---

## Key Findings

1. **All merging methods fail to beat Model B on official benchmarks** — the optimization problem is harder than previously reported
2. **DELLA catastrophically fails on GSM8K (4.12%)** — below Model A (6.67%) — magnitude-based dropping destroys instruction-following weights
3. **TIES and DELLA approach random chance on MMLU** — 30.48% and 26.93% vs 25% random
4. **PS Merging surprisingly strong on MMLU (50.40%)** — simple averaging preserves general knowledge
5. **49.5% search space reduction** — confirmed on real 7B models — validates Type-I interaction formulation
6. **CMA-ES learned alpha beats fixed values on GSM8K subset** — but proxy task dependency limits official benchmark generalization
7. **Type-I interaction directly validated** — ablation shows variance decreases monotonically with selection probability

---

## Models

- **Model A:** `mistralai/Mistral-7B-v0.1` (base, 7.24B parameters)
- **Model B:** `mistralai/Mistral-7B-Instruct-v0.2` (instruction-tuned, 7.24B parameters)

---

## Hardware

- **Primary:** NVIDIA GeForce RTX 4070 Ti SUPER (16GB VRAM), Windows 11
- **Secondary:** NVIDIA RTX Pro 6000 Blackwell (96GB VRAM), Windows 11
- **Python:** 3.12, PyTorch 2.5, Transformers 4.45, CMA 4.4.4

---

## Installation

```bash
git clone https://github.com/Mdniloykhan/dfs-merging-blackbox-optimization.git
cd dfs-merging-blackbox-optimization
pip install torch transformers datasets accelerate cma matplotlib numpy scipy
```

---

## Reproducing Experiments

### Development Subset Experiments (Phases 1-5)

```bash
python experiments/phase1_baseline_gsm8k_experiment.py
python experiments/phase2_cmaes_experiment.py
python experiments/phase3_sota_comparison.py
python experiments/phase4_baseline_mmlu_experiment.py
python experiments/phase5_ablation.py
```

### Official Benchmark Evaluation

```bash
# Official MMLU (14,042 questions)
python experiments/full_mmlu_v2.py

# Official GSM8K (1,319 questions)
python experiments/full_gsm8k_evaluation.py
```

### Ablation Studies

```bash
# Fixed vs learned alpha — both subsets
python experiments/ablation_fixed_vs_learned.py

# Full benchmark ablation (requires high-end GPU)
python experiments/full_benchmark_ablation.py
```

### Statistical Tests (no GPU needed)

```bash
python experiments/statistical_tests_mmlu.py
python experiments/statistical_tests_gsm8k.py
```

### Analysis and Figures (no GPU needed)

```bash
python experiments/analysis_search_space.py
python experiments/analysis_search_space_gsm8k.py
python experiments/analysis_per_subject_mmlu.py
python experiments/analysis_difficulty_gsm8k.py
python experiments/generate_gsm8k_figure.py
```

---

## Repository Structure

```
dfs-merging-blackbox-optimization/
├── README.md
├── requirements.txt
├── LICENSE
├── experiments/
│   ├── phase1_baseline_gsm8k_experiment.py
│   ├── phase2_cmaes_experiment.py
│   ├── phase3_sota_comparison.py
│   ├── phase4_baseline_mmlu_experiment.py
│   ├── phase5_ablation.py
│   ├── full_mmlu_v2.py
│   ├── full_gsm8k_evaluation.py
│   ├── ablation_fixed_vs_learned.py
│   ├── full_benchmark_ablation.py
│   ├── generate_gsm8k_figure.py
│   ├── statistical_tests_gsm8k.py
│   ├── statistical_tests_mmlu.py
│   ├── analysis_per_subject_mmlu.py
│   ├── analysis_search_space.py
│   ├── analysis_search_space_gsm8k.py
│   └── analysis_difficulty_gsm8k.py
├── results/
│   ├── experiment_results_phase1.json
│   ├── experiment_results_phase2.json
│   ├── experiment_results_phase3.json
│   ├── experiment_results_phase4.json
│   ├── experiment_results_phase5.json
│   ├── experiment_results_full_mmlu.json
│   ├── results_full_gsm8k.json
│   ├── experiment_results_ablation_fixed_vs_learned.json
│   ├── statistical_tests_mmlu.json
│   └── statistical_tests_gsm8k.json
└── figures/
    ├── figure_gsm8k_subset.png
    ├── figure_mmlu_subset.png
    ├── figure_cmaes_convergence.png
    ├── figure_full_mmlu_results.png
    ├── figure_gsm8k_complete_results.png
    ├── figure_gsm8k_vs_mmlu.png
    ├── figure_ablation_study.png
    ├── figure_ablation_fixed_vs_learned.png
    ├── figure_search_space_analysis.png
    ├── figure_type1_interaction.png
    ├── figure_per_category_mmlu.png
    └── figure_gsm8k_difficulty_analysis.png
```

---

## Problem Formulation

DFS merging formalized as:

**minimize** f(**x**, **z**)  
**subject to** **x** ∈ [0,1]^N, **z** ∈ {0,1}^N

Three fundamental challenges:
1. **Mixed binary-continuous variables** — binary selection **z** and continuous weights **x**
2. **High dimensionality** — 291 parameter groups for Mistral-7B
3. **Type-I interaction** — **x** only meaningful when **z** = 1

---

## Related Work

- **Akimoto et al. (GECCO 2025)** — Type-I interaction in mixed categorical-continuous optimization
- **Ong, Shirakawa, Akimoto (2026)** — IGBD: bilevel algorithm for mixed binary-continuous optimization (arXiv:2606.12885)
- **Akiba et al. (2024)** — Evolutionary optimization of model merging recipes (Sakana AI)
- **Yadav et al. (2023)** — TIES-Merging
- **Yu et al. (2023)** — DARE
- **Deep et al. (2024)** — DELLA

---

## Citation

```bibtex
@misc{niloy2026blackbox,
  title={Black-Box Optimization of Mixed Binary-Continuous Variables:
         Challenges and Opportunities in Evolutionary Model Merging},
  author={Niloy, Md. Robiul Islam},
  year={2026},
  eprint={2605.12326},
  archivePrefix={arXiv},
  primaryClass={cs.NE},
  url={https://arxiv.org/abs/2605.12326}
}
```

---

## Acknowledgments

The author thanks Professor Youhei Akimoto for valuable discussions and feedback on earlier versions of this research that shaped the direction of this work, and for his foundational contributions to mixed categorical-continuous optimization theory.

---

## Contact

**Md. Robiul Islam Niloy**  
Department Coordinator, CSE  
BRAC University, Bangladesh  
md.niloy26643@gmail.com
