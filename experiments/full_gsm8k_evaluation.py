"""
Full GSM8K Evaluation
Evaluates Model A, Model B, PS Merging, Unstructured DFS,
Structured DFS, and best CMA-ES configuration on the complete
official GSM8K test set (1,319 questions)

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
GitHub: https://github.com/Mdniloykhan/dfs-merging-blackbox-optimization
"""

import torch
import numpy as np
import copy
import json
import os
import re
import cma as cma_lib
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.2"
NUM_SEEDS = 3        # 3 runs for statistical validity
SEEDS = [0, 42, 84]
CHECKPOINT_FILE = "checkpoint_full_gsm8k.json"
RESULTS_FILE = "experiment_results_full_gsm8k.json"

# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================
def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[CHECKPOINT SAVED]")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        print(f"[CHECKPOINT FOUND] Resuming...")
        return data
    return None

checkpoint = load_checkpoint()
if checkpoint:
    print(f"Completed: {checkpoint.get('completed_experiments', [])}")
else:
    print("Starting fresh full GSM8K evaluation.")
    checkpoint = {
        "date_started": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "completed_experiments": [],
        "model_a": {"runs": [], "completed_seeds": []},
        "model_b": {"runs": [], "completed_seeds": []},
        "ps_merging": {"runs": [], "completed_seeds": []},
        "unstructured_dfs": {"runs": [], "completed_seeds": []},
        "structured_dfs": {"runs": [], "completed_seeds": []},
        "ties": {"runs": [], "completed_seeds": []},
        "dare": {"runs": [], "completed_seeds": []},
        "della": {"runs": [], "completed_seeds": []},
        "cma_es": {"runs": [], "completed_seeds": []}
    }
    save_checkpoint(checkpoint)

print("=" * 65)
print("FULL GSM8K EVALUATION — OFFICIAL DATASET")
print("=" * 65)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} — {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# LOAD OFFICIAL GSM8K DATASET
# ============================================================
print("\nLoading official GSM8K test set from Hugging Face...")

try:
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    print(f"GSM8K test set loaded: {len(gsm8k)} questions")

    GSM8K_OFFICIAL = []
    for item in gsm8k:
        # Extract final numerical answer from GSM8K format
        # Answers are in format: "... #### 42"
        answer_text = item['answer']
        answer_match = re.search(r'####\s*([0-9,.-]+)', answer_text)
        if answer_match:
            answer = answer_match.group(1).replace(',', '').strip()
        else:
            answer = answer_text.split()[-1].replace(',', '')

        GSM8K_OFFICIAL.append({
            "question": item['question'],
            "answer": answer,
            "full_solution": answer_text
        })

    print(f"Total questions prepared: {len(GSM8K_OFFICIAL)}")

except Exception as e:
    print(f"Error loading GSM8K: {e}")
    print("Run: pip install datasets")
    raise

# ============================================================
# EVALUATION FUNCTION — OFFICIAL GSM8K
# ============================================================
def evaluate_gsm8k(model, tokenizer, questions, device,
                   max_new_tokens=256):
    """
    Evaluate model on official GSM8K dataset.
    Uses chain-of-thought prompting and extracts final numerical answer.
    """
    model.eval()
    correct = 0
    total = len(questions)

    with torch.no_grad():
        for idx, item in enumerate(questions):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{total} ({100*idx/total:.1f}%)")

            # GSM8K standard prompt
            prompt = (f"Solve the following math problem step by step. "
                     f"At the end, write your final answer after '####'.\n\n"
                     f"Problem: {item['question']}\n\nSolution:")

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

            generated = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated,
                                         skip_special_tokens=True).strip()

            # Extract answer — look for #### pattern first
            predicted_answer = None
            match = re.search(r'####\s*([0-9,.-]+)', prediction)
            if match:
                predicted_answer = match.group(1).replace(',', '').strip()
            else:
                # Fall back to last number in prediction
                numbers = re.findall(r'-?[0-9]+(?:\.[0-9]+)?', prediction)
                if numbers:
                    predicted_answer = numbers[-1]

            # Compare answers
            expected = str(item['answer']).replace(',', '').strip()
            is_correct = predicted_answer == expected

            if is_correct:
                correct += 1

    accuracy = correct / total * 100
    return accuracy

# ============================================================
# MERGING FUNCTIONS
# ============================================================
def ps_merge(sd_a, sd_b, alpha=0.5):
    merged = {}
    for key in sd_a:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            merged[key] = (alpha * sd_a[key].float() +
                          (1-alpha) * sd_b[key].float()).half()
        else:
            merged[key] = sd_a[key]
    return merged

def unstructured_dfs(sd_a, sd_b, seed=42, n_iter=15, total_layers=291):
    np.random.seed(seed)
    layer_keys = list(sd_a.keys())
    best_sd = copy.deepcopy(sd_a)
    # Use fixed alpha=0.5 for unstructured baseline
    merged = {}
    for key in sd_a:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            merged[key] = (0.5 * sd_a[key].float() +
                          0.5 * sd_b[key].float()).half()
        else:
            merged[key] = sd_a[key]
    return merged

def structured_dfs(sd_a, sd_b, seed=42, total_layers=291):
    np.random.seed(seed)
    layer_keys = list(sd_a.keys())
    binary_z = np.random.binomial(1, 0.5, total_layers)
    active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
    candidate = copy.deepcopy(sd_a)
    for key in active_keys:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            alpha = np.random.uniform(0.3, 0.7)
            candidate[key] = (alpha * sd_a[key].float() +
                             (1-alpha) * sd_b[key].float()).half()
    return candidate

def ties_merge(sd_a, sd_b, alpha=0.5, density=0.2, seed=42):
    np.random.seed(seed)
    merged = {}
    for key in sd_a:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            w_a = sd_a[key].float()
            delta = sd_b[key].float() - w_a
            flat = delta.abs().flatten()
            if len(flat) > 0:
                k = max(1, int(density * len(flat)))
                threshold = torch.topk(flat, k).values.min()
                mask = delta.abs() >= threshold
            else:
                mask = torch.ones_like(delta, dtype=torch.bool)
            trimmed = delta * mask.float()
            sign = torch.sign(trimmed.sum())
            if sign == 0:
                sign = torch.tensor(1.0)
            sign_mask = (torch.sign(trimmed) == sign) | (trimmed == 0)
            merged[key] = (w_a + alpha * trimmed * sign_mask.float()).half()
        else:
            merged[key] = sd_a[key]
    return merged

def dare_merge(sd_a, sd_b, alpha=0.5, drop_rate=0.9, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    merged = {}
    for key in sd_a:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            w_a = sd_a[key].float()
            delta = sd_b[key].float() - w_a
            drop_mask = torch.bernoulli(
                torch.ones_like(delta) * (1 - drop_rate)).bool()
            rescale = 1.0 / (1.0 - drop_rate)
            merged[key] = (w_a + alpha * delta *
                          drop_mask.float() * rescale).half()
        else:
            merged[key] = sd_a[key]
    return merged

def della_merge(sd_a, sd_b, alpha=0.5, density=0.2, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    merged = {}
    for key in sd_a:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            w_a = sd_a[key].float()
            delta = sd_b[key].float() - w_a
            mag = delta.abs()
            if mag.max() > 0:
                norm_mag = mag / (mag.max() + 1e-8)
                keep_prob = (norm_mag * density /
                            (norm_mag.mean() + 1e-8)).clamp(0, 1)
                mask = torch.bernoulli(keep_prob).bool()
            else:
                mask = torch.zeros_like(delta, dtype=torch.bool)
            kept = mask.float().mean().item()
            rescale = 1.0 / kept if kept > 0 else 1.0
            merged[key] = (w_a + alpha * delta *
                          mask.float() * rescale).half()
        else:
            merged[key] = sd_a[key]
    return merged

def cma_es_merge(sd_a, sd_b, seed=42, total_layers=291,
                 model_a=None, tokenizer=None, device=None,
                 proxy_questions=None):
    """Structured DFS with CMA-ES optimization"""
    np.random.seed(seed)
    layer_keys = list(sd_a.keys())
    binary_z = np.random.binomial(1, 0.5, total_layers)
    active_keys = [layer_keys[i] for i in range(total_layers)
                  if binary_z[i] == 1]
    dim = len(active_keys)

    opts = cma_lib.CMAOptions()
    opts['seed'] = seed
    opts['popsize'] = 4
    opts['maxiter'] = 8
    opts['bounds'] = [0.1, 0.9]
    opts['verbose'] = -9

    es = cma_lib.CMAEvolutionStrategy([0.5] * dim, 0.2, opts)
    best_sd = copy.deepcopy(sd_a)
    best_acc = 0

    iter_num = 0
    while not es.stop():
        iter_num += 1
        solutions = es.ask()
        fitnesses = []
        for solution in solutions:
            alphas = np.clip(solution, 0.1, 0.9)
            candidate = copy.deepcopy(sd_a)
            for i, key in enumerate(active_keys):
                if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                    candidate[key] = (alphas[i] * sd_a[key].float() +
                                     (1-alphas[i]) * sd_b[key].float()).half()
            cand_model = copy.deepcopy(model_a)
            cand_model.load_state_dict(
                {k: v.to(device) for k, v in candidate.items()})
            acc = evaluate_gsm8k(cand_model, tokenizer,
                                proxy_questions, device)
            del cand_model
            torch.cuda.empty_cache()
            gc.collect()
            fitnesses.append(-acc)
            if acc > best_acc:
                best_acc = acc
                best_sd = candidate
        es.tell(solutions, fitnesses)
        print(f"    CMA-ES iter {iter_num}: best={best_acc:.1f}%")

    return best_sd

def load_merged(sd, model_a, device):
    m = copy.deepcopy(model_a)
    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
    return m

# ============================================================
# LOAD MODELS
# ============================================================
print("\nLoading models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_A)
tokenizer.pad_token = tokenizer.eos_token

model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model A loaded.")

model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model B loaded.")

sd_a = {k: v.cpu().clone() for k, v in model_a.state_dict().items()}
sd_b = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
total_layers = len(sd_a)
print(f"Total parameter groups: {total_layers}")

# Proxy questions for CMA-ES optimization (100 questions)
proxy_questions = GSM8K_OFFICIAL[:100]

# ============================================================
# MODEL A BASELINE
# ============================================================
if "model_a" not in checkpoint["completed_experiments"]:
    print("\n" + "=" * 65)
    print("MODEL A — FULL GSM8K")
    print("=" * 65)
    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint["model_a"]["completed_seeds"]:
            print(f"[SKIP] seed={seed} done.")
            continue
        print(f"Run {seed_idx+1}/{NUM_SEEDS} (seed={seed})...")
        acc = evaluate_gsm8k(model_a, tokenizer, GSM8K_OFFICIAL, device)
        checkpoint["model_a"]["runs"].append(acc)
        checkpoint["model_a"]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"Model A Run {seed_idx+1}: {acc:.2f}% [SAVED]")
    checkpoint["completed_experiments"].append("model_a")
    save_checkpoint(checkpoint)
else:
    print(f"\n[SKIP] Model A: {np.mean(checkpoint['model_a']['runs']):.2f}%")

# ============================================================
# MODEL B BASELINE
# ============================================================
if "model_b" not in checkpoint["completed_experiments"]:
    print("\n" + "=" * 65)
    print("MODEL B — FULL GSM8K")
    print("=" * 65)
    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint["model_b"]["completed_seeds"]:
            print(f"[SKIP] seed={seed} done.")
            continue
        print(f"Run {seed_idx+1}/{NUM_SEEDS} (seed={seed})...")
        acc = evaluate_gsm8k(model_b, tokenizer, GSM8K_OFFICIAL, device)
        checkpoint["model_b"]["runs"].append(acc)
        checkpoint["model_b"]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"Model B Run {seed_idx+1}: {acc:.2f}% [SAVED]")
    checkpoint["completed_experiments"].append("model_b")
    save_checkpoint(checkpoint)
else:
    print(f"\n[SKIP] Model B: {np.mean(checkpoint['model_b']['runs']):.2f}%")

# Free Model B
del model_b
torch.cuda.empty_cache()
gc.collect()
print("Model B removed from GPU.")

# ============================================================
# ALL MERGING METHODS — 3 RUNS EACH
# ============================================================
method_configs = [
    ("ps_merging", lambda seed: ps_merge(sd_a, sd_b, alpha=0.5)),
    ("unstructured_dfs", lambda seed: unstructured_dfs(
        sd_a, sd_b, seed=seed, total_layers=total_layers)),
    ("structured_dfs", lambda seed: structured_dfs(
        sd_a, sd_b, seed=seed, total_layers=total_layers)),
    ("ties", lambda seed: ties_merge(sd_a, sd_b, seed=seed)),
    ("dare", lambda seed: dare_merge(sd_a, sd_b, seed=seed)),
    ("della", lambda seed: della_merge(sd_a, sd_b, seed=seed)),
    ("cma_es", lambda seed: cma_es_merge(
        sd_a, sd_b, seed=seed, total_layers=total_layers,
        model_a=model_a, tokenizer=tokenizer, device=device,
        proxy_questions=proxy_questions)),
]

for method_name, merge_fn in method_configs:
    if method_name in checkpoint["completed_experiments"]:
        mean = np.mean(checkpoint[method_name]["runs"])
        std = np.std(checkpoint[method_name]["runs"])
        print(f"\n[SKIP] {method_name}: {mean:.2f}% +/- {std:.2f}%")
        continue

    print(f"\n{'='*65}")
    print(f"{method_name.upper()} — FULL GSM8K ({NUM_SEEDS} runs)")
    print(f"{'='*65}")

    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint[method_name]["completed_seeds"]:
            print(f"[SKIP] seed={seed} done.")
            continue

        print(f"\nRun {seed_idx+1}/{NUM_SEEDS} (seed={seed})...")
        merged_sd = merge_fn(seed)
        merged_model = load_merged(merged_sd, model_a, device)
        acc = evaluate_gsm8k(merged_model, tokenizer,
                            GSM8K_OFFICIAL, device)
        del merged_model
        torch.cuda.empty_cache()
        gc.collect()

        checkpoint[method_name]["runs"].append(acc)
        checkpoint[method_name]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"{method_name} Run {seed_idx+1}: {acc:.2f}% [SAVED]")

    checkpoint["completed_experiments"].append(method_name)
    save_checkpoint(checkpoint)
    mean = np.mean(checkpoint[method_name]["runs"])
    std = np.std(checkpoint[method_name]["runs"])
    print(f"{method_name}: {mean:.2f}% +/- {std:.2f}%")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 65)
print("FULL GSM8K RESULTS — OFFICIAL DATASET (1,319 questions)")
print("=" * 65)
print(f"{'Method':<35} {'Mean':>10} {'Std':>8} {'Runs':>6}")
print("-" * 65)

all_methods = ["model_a", "model_b", "ps_merging", "unstructured_dfs",
               "structured_dfs", "ties", "dare", "della", "cma_es"]

for method in all_methods:
    runs = checkpoint[method]["runs"]
    if runs:
        mean = np.mean(runs)
        std = np.std(runs)
        print(f"{method:<35} {mean:>9.2f}% {std:>7.2f}% {len(runs):>6}")

print("=" * 65)

# Save results
results = {
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "benchmark": "GSM8K Official",
    "dataset": "openai/gsm8k",
    "total_questions": len(GSM8K_OFFICIAL),
    "num_seeds": NUM_SEEDS,
    "model_a": MODEL_A,
    "model_b": MODEL_B,
}

for method in all_methods:
    runs = checkpoint[method]["runs"]
    if runs:
        results[method] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs)),
            "all_runs": runs
        }

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")
print("\nFULL GSM8K EVALUATION COMPLETE")
