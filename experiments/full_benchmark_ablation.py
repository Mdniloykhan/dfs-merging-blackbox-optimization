"""
Full Benchmark Ablation Study
Tests all 3 ablations on Official GSM8K (1,319Q) and Official MMLU (14,042Q)

Ablation 1: Binary Selection Probability (6 configs)
Ablation 2: Alpha Range (5 configs)  
Ablation 3: Fixed Alpha vs CMA-ES Learned Alpha (4 configs)

Optimized for RTX Pro 6000 (96GB VRAM)
- In-place weight loading — no new model creation
- 3 seeds per config
- CMA-ES: 8 iterations, population 4

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import copy
import json
import gc
import re
import cma as cma_lib
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.2"
DESKTOP = r"C:\Users\DCO\Desktop"
CHECKPOINT_FILE = os.path.join(DESKTOP, "checkpoint_full_ablation.json")
RESULTS_FILE = os.path.join(DESKTOP, "results_full_ablation.json")
SEEDS = [0, 42, 84]
CMA_ITERATIONS = 8
CMA_POPSIZE = 4

print("=" * 65)
print("FULL BENCHMARK ABLATION STUDY")
print("Official GSM8K (1,319Q) + Official MMLU (14,042Q)")
print("=" * 65)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# CHECKPOINT
# ============================================================
def save_ckpt(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print("[CHECKPOINT SAVED]")

def load_ckpt():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

ckpt = load_ckpt()
if ckpt:
    print(f"Resuming. Completed configs: {len(ckpt.get('results', {}))}")
else:
    print("Starting fresh.")
    ckpt = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": {}
    }
    save_ckpt(ckpt)

# ============================================================
# LOAD DATASETS
# ============================================================
print("\nLoading datasets...")

# GSM8K
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
GSM8K_QUESTIONS = []
for item in gsm8k:
    match = re.search(r'####\s*([0-9,.\-]+)', item['answer'])
    answer = match.group(1).replace(',', '').strip() if match else item['answer'].strip().split()[-1].replace(',', '')
    GSM8K_QUESTIONS.append({"question": item['question'], "answer": answer})
print(f"GSM8K: {len(GSM8K_QUESTIONS)} questions")

# MMLU
mmlu = load_dataset("cais/mmlu", "all", split="test")
MMLU_QUESTIONS = []
for item in mmlu:
    choices = [f"A) {item['choices'][0]}", f"B) {item['choices'][1]}",
               f"C) {item['choices'][2]}", f"D) {item['choices'][3]}"]
    MMLU_QUESTIONS.append({
        "question": item['question'],
        "choices": choices,
        "answer": ["A", "B", "C", "D"][item['answer']],
        "subject": item['subject']
    })
print(f"MMLU: {len(MMLU_QUESTIONS)} questions")

# Proxy for CMA-ES
GSM8K_PROXY = GSM8K_QUESTIONS[:100]

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def eval_gsm8k(model, tokenizer, questions, device):
    model.eval()
    correct = 0
    total = len(questions)
    with torch.no_grad():
        for idx, item in enumerate(questions):
            if idx % 200 == 0 and total > 200:
                print(f"    GSM8K: {idx}/{total} ({100*idx/total:.1f}%)")
            prompt = (f"Solve step by step. Write final answer after ####.\n"
                     f"Problem: {item['question']}\nSolution:")
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=512, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=256,
                                    do_sample=False, temperature=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            m = re.search(r'####\s*([0-9,.\-]+)', pred)
            pred_ans = m.group(1).replace(',', '').strip() if m else (
                re.findall(r'-?[0-9]+(?:\.[0-9]+)?', pred) or [''])[-1]
            if pred_ans == str(item['answer']).replace(',', '').strip():
                correct += 1
    return correct / total * 100

def eval_mmlu(model, tokenizer, questions, device):
    model.eval()
    correct = 0
    total = len(questions)
    with torch.no_grad():
        for idx, item in enumerate(questions):
            if idx % 2000 == 0:
                print(f"    MMLU: {idx}/{total} ({100*idx/total:.1f}%)")
            prompt = (f"Question: {item['question']}\nChoices:\n" +
                     "\n".join(item['choices']) +
                     "\nAnswer with only the letter A, B, C, or D:\n")
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=512, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=5,
                                    do_sample=False, temperature=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            pred_letter = next((c for c in pred.upper() if c in "ABCD"), "")
            if pred_letter == item['answer']:
                correct += 1
    return correct / total * 100

# ============================================================
# IN-PLACE WEIGHT OPERATIONS
# ============================================================
def apply_weights(model, sd, device):
    """Apply state dict weights to model in-place."""
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        for k, v in sd.items():
            if k in model_params:
                model_params[k].data.copy_(v.to(device))
            elif k in model_buffers:
                model_buffers[k].data.copy_(v.to(device))

def restore_weights(model, sd_a, device):
    """Restore model to original Model A weights."""
    apply_weights(model, sd_a, device)

# ============================================================
# LOAD MODELS
# ============================================================
print("\nLoading models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_A)
tokenizer.pad_token = tokenizer.eos_token

model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model A: {sum(p.numel() for p in model_a.parameters()):,} params")

model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model B: {sum(p.numel() for p in model_b.parameters()):,} params")

sd_a = {k: v.cpu().clone() for k, v in model_a.state_dict().items()}
sd_b = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
total_layers = len(sd_a)
layer_keys = list(sd_a.keys())
print(f"Total layers: {total_layers}")

del model_b
torch.cuda.empty_cache()
gc.collect()
print("Model B freed from GPU.")

# ============================================================
# MERGE AND EVALUATE FUNCTION
# ============================================================
def run_config(config_name, merged_sd, seed):
    """Evaluate one merged configuration on both benchmarks."""
    key = f"{config_name}_seed{seed}"
    if key in ckpt["results"]:
        r = ckpt["results"][key]
        print(f"  [SKIP] {key}: GSM8K={r['gsm8k']:.1f}% MMLU={r['mmlu']:.1f}%")
        return r

    print(f"\n  Evaluating {key}...")

    # Apply merged weights
    apply_weights(model_a, merged_sd, device)

    # Evaluate GSM8K
    print(f"  GSM8K evaluation...")
    gsm8k_acc = eval_gsm8k(model_a, tokenizer, GSM8K_QUESTIONS, device)

    # Evaluate MMLU
    print(f"  MMLU evaluation...")
    mmlu_acc = eval_mmlu(model_a, tokenizer, MMLU_QUESTIONS, device)

    # Restore original weights
    restore_weights(model_a, sd_a, device)
    torch.cuda.empty_cache()
    gc.collect()

    result = {"gsm8k": gsm8k_acc, "mmlu": mmlu_acc}
    ckpt["results"][key] = result
    save_ckpt(ckpt)
    print(f"  {key}: GSM8K={gsm8k_acc:.2f}% MMLU={mmlu_acc:.2f}% [SAVED]")
    return result

def make_fixed_sd(alpha, seed):
    """Create merged state dict with fixed alpha."""
    np.random.seed(seed)
    z = np.random.binomial(1, 0.5, total_layers)
    active = [layer_keys[i] for i in range(total_layers) if z[i] == 1]
    sd = {k: v.clone() for k, v in sd_a.items()}
    for k in active:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            sd[k] = (alpha * sd_a[k].float() + (1-alpha) * sd_b[k].float()).half()
    return sd

def make_prob_sd(prob, seed):
    """Create merged state dict with given selection probability."""
    np.random.seed(seed)
    z = np.random.binomial(1, prob, total_layers)
    active = [layer_keys[i] for i in range(total_layers) if z[i] == 1]
    sd = {k: v.clone() for k, v in sd_a.items()}
    for k in active:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            alpha = np.random.uniform(0.3, 0.7)
            sd[k] = (alpha * sd_a[k].float() + (1-alpha) * sd_b[k].float()).half()
    return sd, len(active)

def make_range_sd(low, high, seed):
    """Create merged state dict with given alpha range."""
    np.random.seed(seed)
    z = np.random.binomial(1, 0.5, total_layers)
    active = [layer_keys[i] for i in range(total_layers) if z[i] == 1]
    sd = {k: v.clone() for k, v in sd_a.items()}
    for k in active:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            alpha = np.random.uniform(low, high)
            sd[k] = (alpha * sd_a[k].float() + (1-alpha) * sd_b[k].float()).half()
    return sd

def make_cmaes_sd(seed):
    """Create merged state dict using CMA-ES optimization."""
    np.random.seed(seed)
    z = np.random.binomial(1, 0.5, total_layers)
    active = [layer_keys[i] for i in range(total_layers) if z[i] == 1]
    dim = len(active)

    opts = cma_lib.CMAOptions()
    opts['seed'] = seed
    opts['popsize'] = CMA_POPSIZE
    opts['maxiter'] = CMA_ITERATIONS
    opts['bounds'] = [0.1, 0.9]
    opts['verbose'] = -9

    es = cma_lib.CMAEvolutionStrategy([0.5] * dim, 0.2, opts)
    best_sd = {k: v.clone() for k, v in sd_a.items()}
    best_acc = 0
    it = 0

    while not es.stop():
        it += 1
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            alphas = np.clip(sol, 0.1, 0.9)
            candidate = {k: v.clone() for k, v in sd_a.items()}
            for i, k in enumerate(active):
                if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                    candidate[k] = (alphas[i] * sd_a[k].float() +
                                   (1-alphas[i]) * sd_b[k].float()).half()

            apply_weights(model_a, candidate, device)
            acc = eval_gsm8k(model_a, tokenizer, GSM8K_PROXY, device)
            restore_weights(model_a, sd_a, device)
            torch.cuda.empty_cache()
            gc.collect()
            del candidate

            fitnesses.append(-acc)
            if acc > best_acc:
                best_acc = acc
                best_sd = {k: v.clone() for k, v in sd_a.items()}
                for i, k in enumerate(active):
                    if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                        best_sd[k] = (alphas[i] * sd_a[k].float() +
                                     (1-alphas[i]) * sd_b[k].float()).half()

        es.tell(solutions, fitnesses)
        print(f"    CMA-ES iter {it}: proxy best={best_acc:.1f}%")

    return best_sd

# ============================================================
# ABLATION 1 — SELECTION PROBABILITY
# ============================================================
print("\n" + "=" * 65)
print("ABLATION 1: Binary Selection Probability")
print("=" * 65)

probs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
abl1_results = {}

for prob in probs:
    name = f"abl1_prob_{prob}"
    print(f"\nSelection probability: {prob}")
    runs_gsm8k = []
    runs_mmlu = []
    active_counts = []

    for seed in SEEDS:
        key = f"{name}_seed{seed}"
        if key in ckpt["results"]:
            r = ckpt["results"][key]
            runs_gsm8k.append(r['gsm8k'])
            runs_mmlu.append(r['mmlu'])
            active_counts.append(r.get('active', 0))
            print(f"  [SKIP] seed={seed}: GSM8K={r['gsm8k']:.1f}% MMLU={r['mmlu']:.1f}%")
            continue

        merged_sd, active_count = make_prob_sd(prob, seed)
        active_counts.append(active_count)

        apply_weights(model_a, merged_sd, device)
        gsm8k_acc = eval_gsm8k(model_a, tokenizer, GSM8K_QUESTIONS, device)
        mmlu_acc = eval_mmlu(model_a, tokenizer, MMLU_QUESTIONS, device)
        restore_weights(model_a, sd_a, device)
        torch.cuda.empty_cache()
        gc.collect()
        del merged_sd

        result = {"gsm8k": gsm8k_acc, "mmlu": mmlu_acc, "active": active_count}
        ckpt["results"][key] = result
        save_ckpt(ckpt)
        runs_gsm8k.append(gsm8k_acc)
        runs_mmlu.append(mmlu_acc)
        print(f"  seed={seed}: GSM8K={gsm8k_acc:.2f}% MMLU={mmlu_acc:.2f}% active={active_count} [SAVED]")

    if runs_gsm8k:
        abl1_results[str(prob)] = {
            "gsm8k_mean": float(np.mean(runs_gsm8k)),
            "gsm8k_std": float(np.std(runs_gsm8k)),
            "mmlu_mean": float(np.mean(runs_mmlu)),
            "mmlu_std": float(np.std(runs_mmlu)),
            "avg_active": float(np.mean(active_counts))
        }
        print(f"prob={prob}: GSM8K={np.mean(runs_gsm8k):.2f}%±{np.std(runs_gsm8k):.2f}% "
              f"MMLU={np.mean(runs_mmlu):.2f}%±{np.std(runs_mmlu):.2f}%")

# ============================================================
# ABLATION 2 — ALPHA RANGE
# ============================================================
print("\n" + "=" * 65)
print("ABLATION 2: Alpha Range")
print("=" * 65)

ranges = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]
abl2_results = {}

for low, high in ranges:
    name = f"abl2_range_{low}_{high}"
    label = f"[{low},{high}]"
    print(f"\nAlpha range: {label}")
    runs_gsm8k = []
    runs_mmlu = []

    for seed in SEEDS:
        key = f"{name}_seed{seed}"
        if key in ckpt["results"]:
            r = ckpt["results"][key]
            runs_gsm8k.append(r['gsm8k'])
            runs_mmlu.append(r['mmlu'])
            print(f"  [SKIP] seed={seed}: GSM8K={r['gsm8k']:.1f}% MMLU={r['mmlu']:.1f}%")
            continue

        merged_sd = make_range_sd(low, high, seed)
        apply_weights(model_a, merged_sd, device)
        gsm8k_acc = eval_gsm8k(model_a, tokenizer, GSM8K_QUESTIONS, device)
        mmlu_acc = eval_mmlu(model_a, tokenizer, MMLU_QUESTIONS, device)
        restore_weights(model_a, sd_a, device)
        torch.cuda.empty_cache()
        gc.collect()
        del merged_sd

        result = {"gsm8k": gsm8k_acc, "mmlu": mmlu_acc}
        ckpt["results"][key] = result
        save_ckpt(ckpt)
        runs_gsm8k.append(gsm8k_acc)
        runs_mmlu.append(mmlu_acc)
        print(f"  seed={seed}: GSM8K={gsm8k_acc:.2f}% MMLU={mmlu_acc:.2f}% [SAVED]")

    if runs_gsm8k:
        abl2_results[label] = {
            "gsm8k_mean": float(np.mean(runs_gsm8k)),
            "gsm8k_std": float(np.std(runs_gsm8k)),
            "mmlu_mean": float(np.mean(runs_mmlu)),
            "mmlu_std": float(np.std(runs_mmlu))
        }
        print(f"range={label}: GSM8K={np.mean(runs_gsm8k):.2f}%±{np.std(runs_gsm8k):.2f}% "
              f"MMLU={np.mean(runs_mmlu):.2f}%±{np.std(runs_mmlu):.2f}%")

# ============================================================
# ABLATION 3 — FIXED vs LEARNED ALPHA
# ============================================================
print("\n" + "=" * 65)
print("ABLATION 3: Fixed Alpha vs CMA-ES Learned Alpha")
print("=" * 65)

fixed_alphas = [0.3, 0.5, 0.7]
abl3_results = {}

for alpha in fixed_alphas:
    name = f"abl3_fixed_{alpha}"
    print(f"\nFixed alpha={alpha}")
    runs_gsm8k = []
    runs_mmlu = []

    for seed in SEEDS:
        key = f"{name}_seed{seed}"
        if key in ckpt["results"]:
            r = ckpt["results"][key]
            runs_gsm8k.append(r['gsm8k'])
            runs_mmlu.append(r['mmlu'])
            print(f"  [SKIP] seed={seed}: GSM8K={r['gsm8k']:.1f}% MMLU={r['mmlu']:.1f}%")
            continue

        merged_sd = make_fixed_sd(alpha, seed)
        apply_weights(model_a, merged_sd, device)
        gsm8k_acc = eval_gsm8k(model_a, tokenizer, GSM8K_QUESTIONS, device)
        mmlu_acc = eval_mmlu(model_a, tokenizer, MMLU_QUESTIONS, device)
        restore_weights(model_a, sd_a, device)
        torch.cuda.empty_cache()
        gc.collect()
        del merged_sd

        result = {"gsm8k": gsm8k_acc, "mmlu": mmlu_acc}
        ckpt["results"][key] = result
        save_ckpt(ckpt)
        runs_gsm8k.append(gsm8k_acc)
        runs_mmlu.append(mmlu_acc)
        print(f"  seed={seed}: GSM8K={gsm8k_acc:.2f}% MMLU={mmlu_acc:.2f}% [SAVED]")

    if runs_gsm8k:
        abl3_results[f"fixed_{alpha}"] = {
            "gsm8k_mean": float(np.mean(runs_gsm8k)),
            "gsm8k_std": float(np.std(runs_gsm8k)),
            "mmlu_mean": float(np.mean(runs_mmlu)),
            "mmlu_std": float(np.std(runs_mmlu))
        }

# CMA-ES learned
print(f"\nCMA-ES learned alpha")
runs_gsm8k = []
runs_mmlu = []

for seed in SEEDS:
    key = f"abl3_cmaes_seed{seed}"
    if key in ckpt["results"]:
        r = ckpt["results"][key]
        runs_gsm8k.append(r['gsm8k'])
        runs_mmlu.append(r['mmlu'])
        print(f"  [SKIP] seed={seed}: GSM8K={r['gsm8k']:.1f}% MMLU={r['mmlu']:.1f}%")
        continue

    merged_sd = make_cmaes_sd(seed)
    apply_weights(model_a, merged_sd, device)
    gsm8k_acc = eval_gsm8k(model_a, tokenizer, GSM8K_QUESTIONS, device)
    mmlu_acc = eval_mmlu(model_a, tokenizer, MMLU_QUESTIONS, device)
    restore_weights(model_a, sd_a, device)
    torch.cuda.empty_cache()
    gc.collect()
    del merged_sd

    result = {"gsm8k": gsm8k_acc, "mmlu": mmlu_acc}
    ckpt["results"][key] = result
    save_ckpt(ckpt)
    runs_gsm8k.append(gsm8k_acc)
    runs_mmlu.append(mmlu_acc)
    print(f"  seed={seed}: GSM8K={gsm8k_acc:.2f}% MMLU={mmlu_acc:.2f}% [SAVED]")

if runs_gsm8k:
    abl3_results["cma_es_learned"] = {
        "gsm8k_mean": float(np.mean(runs_gsm8k)),
        "gsm8k_std": float(np.std(runs_gsm8k)),
        "mmlu_mean": float(np.mean(runs_mmlu)),
        "mmlu_std": float(np.std(runs_mmlu))
    }

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 65)
print("FULL BENCHMARK ABLATION COMPLETE")
print("=" * 65)

print("\nAblation 1 — Selection Probability:")
print(f"{'Prob':<10} {'GSM8K':>12} {'MMLU':>12} {'Active':>8}")
for prob, r in abl1_results.items():
    print(f"{prob:<10} {r['gsm8k_mean']:>8.2f}%±{r['gsm8k_std']:.2f} "
          f"{r['mmlu_mean']:>8.2f}%±{r['mmlu_std']:.2f} {r['avg_active']:>7.0f}")

print("\nAblation 2 — Alpha Range:")
print(f"{'Range':<12} {'GSM8K':>12} {'MMLU':>12}")
for rng, r in abl2_results.items():
    print(f"{rng:<12} {r['gsm8k_mean']:>8.2f}%±{r['gsm8k_std']:.2f} "
          f"{r['mmlu_mean']:>8.2f}%±{r['mmlu_std']:.2f}")

print("\nAblation 3 — Fixed vs Learned Alpha:")
print(f"{'Method':<20} {'GSM8K':>12} {'MMLU':>12}")
for method, r in abl3_results.items():
    print(f"{method:<20} {r['gsm8k_mean']:>8.2f}%±{r['gsm8k_std']:.2f} "
          f"{r['mmlu_mean']:>8.2f}%±{r['mmlu_std']:.2f}")

# Save results
results = {
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "benchmarks": "Official GSM8K (1,319Q) + Official MMLU (14,042Q)",
    "seeds": SEEDS,
    "ablation1_selection_prob": abl1_results,
    "ablation2_alpha_range": abl2_results,
    "ablation3_fixed_vs_learned": abl3_results
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {RESULTS_FILE}")
print("\nFULL BENCHMARK ABLATION COMPLETE")
