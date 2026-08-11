# Black-Box Optimization of Mixed Binary-Continuous Variables: Challenges and Opportunities in Evolutionary Model Merging

[![arXiv](https://img.shields.io/badge/arXiv-2605.12326-b31b1b.svg)](https://arxiv.org/abs/2605.12326)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

**Author:** Md. Robiul Islam Niloy  
**Institution:** BRAC University, Bangladesh  
**Email:** md.niloy26643@gmail.com  
**arXiv:** https://arxiv.org/abs/2605.12326  
**Zenodo:** https://zenodo.org/records/20123507

---

## Abstract

Model merging has emerged as a cost-effective alternative to training large language models from scratch. Evolutionary approaches to model merging have shown particular promise, but the optimization challenges underlying these approaches — particularly in Data Flow Space (DFS) merging — remain poorly understood.

This work formally characterizes the DFS merging problem as a **black-box optimization problem with mixed binary-continuous variables**, high-dimensional search spaces, and conditional dependencies between variable types. We demonstrate that CMA-ES, specifically designed for continuous optimization, outperforms both naive random search and published SOTA merging methods when applied to this structured problem.

---

## Key Results

### GSM8K Benchmark (Math Reasoning)

| Method | Mean Accuracy | Std | vs CMA-ES |
|--------|--------------|-----|-----------|
| Model A (Mistral-7B-v0.1) | 43.8% | N/A | -44.3% |
| TIES-Merging | 53.1% | 0.0% | -35.0%* |
| DELLA | 53.1% | 2.4% | -35.0%* |
| DARE | 77.3% | 5.2% | -10.8%* |
| Model B (Mistral-7B-Instruct) | 85.4% | N/A | -2.7% |
| PS Merging | 84.4% | 0.0% | -3.7% |
| Unstructured DFS | 86.0% | 0.5% | -2.1% |
| Structured DFS (Random) | 86.5% | 1.1% | -1.6% |
| **Structured DFS + CMA-ES** | **88.1%** | **1.1%** | **Best** |

*Statistically significant (p < 0.01)

### MMLU Benchmark (General Knowledge — 100 questions, 10 subjects)

| Method | Mean Accuracy | Std |
|--------|--------------|-----|
| Model A (Mistral-7B-v0.1) | 89.0% | N/A |
| Model B (Mistral-7B-Instruct) | 86.0% | N/A |
| TIES-Merging | 90.0% | 0.0% |
| DELLA | 69.2% | 4.8% |
| DARE | 92.2% | 2.5% |
| PS Merging | 93.0% | 0.0% |
| Structured DFS (Random) | 91.6% | 1.2% |
| **Structured DFS + CMA-ES** | **93.2%** | **1.2%** |

### Key Findings

1. **CMA-ES outperforms all SOTA methods** on GSM8K with statistically significant margins against TIES (p<0.0001, d=46.70), DARE (p=0.006, d=2.86), and DELLA (p<0.0001, d=19.13)
2. **49.5% search space reduction** — Structured DFS automatically focuses on ~147/291 active layers, directly validating the conditional dependency formulation
3. **PS Merging degrades below source models** — 84.4% vs Model B alone (85.4%), showing naive weight averaging loses task-specific knowledge
4. **Wider alpha range improves CMA-ES** — [0.1,0.9] gives 86.1% ± 1.0% vs fixed α=0.5 giving 74.0% ± 5.6%
5. **CMA-ES learned alpha beats all fixed alpha values** — proving optimization is genuinely necessary

---

## Models Used

- **Model A:** `mistralai/Mistral-7B-v0.1` (base model, 7.2B parameters)
- **Model B:** `mistralai/Mistral-7B-Instruct-v0.2` (instruction tuned, 7.2B parameters)

---

## Hardware

All experiments conducted on:
- **GPU:** NVIDIA GeForce RTX 4070 Ti SUPER (16GB VRAM)
- **CPU:** Intel Core i7-14700
- **RAM:** 64GB DDR5
- **OS:** Windows 11

---

## Installation

```bash
# Clone repository
git clone https://github.com/Mdniloykhan/dfs-merging-blackbox-optimization.git
cd dfs-merging-blackbox-optimization

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install cma matplotlib numpy scipy
pip install mergekit --no-deps
```

---

## Reproducing Experiments

### Phase 1: Baseline Comparison (PS Merging, Unstructured DFS, Structured DFS)

```bash
python experiments/phase1_baseline_gsm8k_experiment.py
```

Results saved to: `results/experiment_results_final.json`

### Phase 2: CMA-ES Experiment

```bash
python experiments/phase2_cmaes_experiment.py
```

Results saved to: `results/experiment_results_phase2.json`

### Phase 3: SOTA Comparison (TIES, DARE, DELLA)

```bash
python experiments/phase3_sota_comparison.py
```

Results saved to: `results/experiment_results_phase3.json`

### Phase 4: MMLU Benchmark

```bash
python experiments/phase4_baseline_mmlu_experiment.py
```

Results saved to: `results/experiment_results_phase4.json`

### Phase 5: Ablation Study

```bash
python experiments/phase5_ablation.py
```

Results saved to: `results/experiment_results_phase5.json`

---

## Repository Structure

```
dfs-merging-blackbox-optimization/
├── README.md
├── requirements.txt
├── experiments/
│   ├── phase1_baseline_gsm8k_experiment.py
│   ├── phase2_cmaes_experiment.py
│   ├── phase3_sota_comparison.py
│   ├── phase4_baseline_mmlu_experiment.py
│   └── phase5_ablation.py
├── results/
│   ├── experiment_results_final.json
│   ├── experiment_results_phase2.json
│   ├── experiment_results_phase3.json
│   ├── experiment_results_phase4.json
│   └── experiment_results_phase5.json
└── figures/
    ├── phase1_results.png
    ├── phase2_results.png
    └── phase5_ablation.png
```

---

## Problem Formulation

The Data Flow Space (DFS) merging problem is formalized as:

**minimize** f(**x**, **z**)  
**subject to** **x** ∈ ℝⁿ, **z** ∈ {0,1}ᵐ

Where:
- **z** ∈ {0,1}^(N×L) — binary vector selecting which layers to include
- **x** ∈ ℝ^(N×L) — continuous scaling weights for selected layers
- f is a black-box objective (model performance on target task)

This formulation reveals three fundamental challenges:
1. **Mixed binary-continuous variables** — standard CMA-ES cannot handle binary variables directly
2. **High dimensionality** — 291 parameter groups for Mistral-7B
3. **Conditional variable dependencies** — x is only relevant when z=1 (Type-I interaction per Akimoto et al. GECCO 2025)

---

## Connection to Prior Work

This work connects directly to:

- **Akiba et al. (2024)** — Evolutionary optimization of model merging recipes (Sakana AI)
- **Akimoto et al. (GECCO 2025)** — Challenges of interaction in optimizing mixed categorical-continuous variables
- **Hamano et al. (2024)** — CatCMA: Stochastic optimization for mixed-category problems

---

## Citation

If you use this code or find this work helpful, please cite:

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

## Requirements

```
torch>=2.5.0
transformers>=4.45.0
datasets
accelerate
cma>=4.4.4
matplotlib
numpy
scipy
mergekit
```

---

## Contact

**Md. Robiul Islam Niloy**  
Department Coordinator, CSE  
BRAC University, Bangladesh  
md.niloy26643@gmail.com  
arXiv: https://arxiv.org/abs/2605.12326
