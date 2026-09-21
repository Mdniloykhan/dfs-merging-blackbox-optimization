import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import copy
import json
import gc
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.2"
DESKTOP = r"C:\Users\user1\Desktop"
CHECKPOINT_FILE = os.path.join(DESKTOP, "checkpoint_full_gsm8k.json")
RESULTS_FILE = os.path.join(DESKTOP, "results_full_gsm8k.json")
SEEDS = [0, 42, 84]

print("=" * 65)
print("FULL GSM8K EVALUATION — OFFICIAL DATASET (1,319 questions)")
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
    print(f"Resuming. Completed: {ckpt.get('completed', [])}")
else:
    print("Starting fresh.")
    ckpt = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "completed": [],
        "model_a": None,
        "model_b": None,
        "ps_merging": {"runs": [], "seeds_done": []},
        "unstructured_dfs": {"runs": [], "seeds_done": []},
        "structured_dfs": {"runs": [], "seeds_done": []},
        "ties": {"runs": [], "seeds_done": []},
        "dare": {"runs": [], "seeds_done": []},
        "della": {"runs": [], "seeds_done": []},
        "cma_es": {"runs": [], "seeds_done": []}
    }
    save_ckpt(ckpt)

# ============================================================
# LOAD OFFICIAL GSM8K DATASET
# ============================================================
print("\nLoading official GSM8K test set from Hugging Face...")
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
print(f"Loaded {len(gsm8k)} questions")

QUESTIONS = []
for item in gsm8k:
    answer_text = item['answer']
    match = re.search(r'####\s*([0-9,.\-]+)', answer_text)
    if match:
        answer = match.group(1).replace(',', '').strip()
    else:
        answer = answer_text.strip().split()[-1].replace(',', '')
    QUESTIONS.append({
        "question": item['question'],
        "answer": answer,
        "full_solution": answer_text
    })

print(f"Total questions: {len(QUESTIONS)}")
PROXY = QUESTIONS[:100]

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, tokenizer, questions, device, max_new_tokens=256):
    model.eval()
    correct = 0
    total = len(questions)

    with torch.no_grad():
        for idx, item in enumerate(questions):
            if idx % 200 == 0:
                print(f"  {idx}/{total} ({100*idx/total:.1f}%)")

            prompt = (
                f"Solve the following math problem step by step. "
                f"At the end write your final answer after ####.\n\n"
                f"Problem: {item['question']}\n\nSolution:"
            )

            inputs = tokenizer(
                prompt, return_tensors="pt",
                max_length=512, truncation=True
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

            generated = outputs[0][inputs['input_ids'].shape[1]:]
            pred_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Extract answer after ####
            pred_answer = None
            m = re.search(r'####\s*([0-9,.\-]+)', pred_text)
            if m:
                pred_answer = m.group(1).replace(',', '').strip()
            else:
                nums = re.findall(r'-?[0-9]+(?:\.[0-9]+)?', pred_text)
                if nums:
                    pred_answer = nums[-1]

            expected = str(item['answer']).replace(',', '').strip()
            if pred_answer == expected:
                correct += 1

    return correct / total * 100, correct, total

# ============================================================
# MERGING FUNCTIONS
# ============================================================
def ps_merge(sd_a, sd_b):
    merged = {}
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            merged[k] = (0.5 * sd_a[k].float() + 0.5 * sd_b[k].float()).half()
        else:
            merged[k] = sd_a[k]
    return merged

def unstructured_merge(sd_a, sd_b, seed):
    np.random.seed(seed)
    merged = {}
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            alpha = np.random.uniform(0.2, 0.8)
            merged[k] = (alpha * sd_a[k].float() + (1-alpha) * sd_b[k].float()).half()
        else:
            merged[k] = sd_a[k]
    return merged

def structured_merge(sd_a, sd_b, seed, total_layers):
    np.random.seed(seed)
    keys = list(sd_a.keys())
    z = np.random.binomial(1, 0.5, total_layers)
    active = [keys[i] for i in range(total_layers) if z[i] == 1]
    candidate = {k: v.clone() for k, v in sd_a.items()}
    for k in active:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            alpha = np.random.uniform(0.3, 0.7)
            candidate[k] = (alpha * sd_a[k].float() + (1-alpha) * sd_b[k].float()).half()
    return candidate

def ties_merge(sd_a, sd_b, seed, alpha=0.5, density=0.2):
    np.random.seed(seed)
    merged = {}
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            w_a = sd_a[k].float()
            delta = sd_b[k].float() - w_a
            flat = delta.abs().flatten()
            if len(flat) > 0:
                thresh = torch.topk(flat, max(1, int(density*len(flat)))).values.min()
                mask = delta.abs() >= thresh
            else:
                mask = torch.ones_like(delta, dtype=torch.bool)
            trimmed = delta * mask.float()
            sign = torch.sign(trimmed.sum())
            if sign == 0:
                sign = torch.tensor(1.0)
            smask = (torch.sign(trimmed) == sign) | (trimmed == 0)
            merged[k] = (w_a + alpha * trimmed * smask.float()).half()
        else:
            merged[k] = sd_a[k]
    return merged

def dare_merge(sd_a, sd_b, seed, alpha=0.5, drop=0.9):
    np.random.seed(seed)
    torch.manual_seed(seed)
    merged = {}
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            w_a = sd_a[k].float()
            delta = sd_b[k].float() - w_a
            mask = torch.bernoulli(torch.ones_like(delta) * (1-drop)).bool()
            merged[k] = (w_a + alpha * delta * mask.float() / (1-drop)).half()
        else:
            merged[k] = sd_a[k]
    return merged

def della_merge(sd_a, sd_b, seed, alpha=0.5, density=0.2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    merged = {}
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            w_a = sd_a[k].float()
            delta = sd_b[k].float() - w_a
            mag = delta.abs()
            if mag.max() > 0:
                norm = mag / (mag.max() + 1e-8)
                prob = (norm * density / (norm.mean() + 1e-8)).clamp(0, 1)
                mask = torch.bernoulli(prob).bool()
            else:
                mask = torch.zeros_like(delta, dtype=torch.bool)
            kept = mask.float().mean().item()
            scale = 1.0 / kept if kept > 0 else 1.0
            merged[k] = (w_a + alpha * delta * mask.float() * scale).half()
        else:
            merged[k] = sd_a[k]
    return merged

def cma_es_merge(sd_a, sd_b, seed, total_layers, model_a_ref, tokenizer, device):
    import cma as cma_lib
    np.random.seed(seed)
    keys = list(sd_a.keys())
    z = np.random.binomial(1, 0.5, total_layers)
    active = [keys[i] for i in range(total_layers) if z[i] == 1]
    dim = len(active)
    print(f"  Active layers: {dim}/{total_layers}")

    opts = cma_lib.CMAOptions()
    opts['seed'] = seed
    opts['popsize'] = 4
    opts['maxiter'] = 5
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
                    candidate[k] = (
                        alphas[i] * sd_a[k].float() +
                        (1-alphas[i]) * sd_b[k].float()
                    ).half()

            torch.cuda.empty_cache()
            gc.collect()
            cand = copy.deepcopy(model_a_ref)
            cand.load_state_dict({k: v.to(device) for k, v in candidate.items()})
            acc, _, _ = evaluate(cand, tokenizer, PROXY, device, max_new_tokens=100)
            del cand
            del candidate
            torch.cuda.empty_cache()
            gc.collect()

            fitnesses.append(-acc)
            if acc > best_acc:
                best_acc = acc
                best_sd = {k: v.clone() for k, v in sd_a.items()}
                for i, k in enumerate(active):
                    if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                        best_sd[k] = (
                            alphas[i] * sd_a[k].float() +
                            (1-alphas[i]) * sd_b[k].float()
                        ).half()

        es.tell(solutions, fitnesses)
        print(f"  CMA-ES iter {it}: proxy best={best_acc:.1f}%")

    return best_sd

def load_model(sd, model_a, device):
    torch.cuda.empty_cache()
    gc.collect()
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
print(f"Total layers: {total_layers}")

# ============================================================
# MODEL A
# ============================================================
if "model_a" not in ckpt["completed"]:
    print("\n[MODEL A] Full GSM8K evaluation...")
    acc, correct, total = evaluate(model_a, tokenizer, QUESTIONS, device)
    ckpt["model_a"] = {"accuracy": acc, "correct": correct, "total": total}
    ckpt["completed"].append("model_a")
    save_ckpt(ckpt)
    print(f"Model A: {acc:.2f}% ({correct}/{total})")
else:
    print(f"[SKIP] Model A: {ckpt['model_a']['accuracy']:.2f}%")

# ============================================================
# MODEL B
# ============================================================
if "model_b" not in ckpt["completed"]:
    print("\n[MODEL B] Full GSM8K evaluation...")
    acc, correct, total = evaluate(model_b, tokenizer, QUESTIONS, device)
    ckpt["model_b"] = {"accuracy": acc, "correct": correct, "total": total}
    ckpt["completed"].append("model_b")
    save_ckpt(ckpt)
    print(f"Model B: {acc:.2f}% ({correct}/{total})")
else:
    print(f"[SKIP] Model B: {ckpt['model_b']['accuracy']:.2f}%")

del model_b
torch.cuda.empty_cache()
gc.collect()
print("Model B freed.")

# ============================================================
# ALL MERGING METHODS
# ============================================================
methods = [
    ("ps_merging",       lambda seed: ps_merge(sd_a, sd_b)),
    ("unstructured_dfs", lambda seed: unstructured_merge(sd_a, sd_b, seed)),
    ("structured_dfs",   lambda seed: structured_merge(sd_a, sd_b, seed, total_layers)),
    ("ties",             lambda seed: ties_merge(sd_a, sd_b, seed)),
    ("dare",             lambda seed: dare_merge(sd_a, sd_b, seed)),
    ("della",            lambda seed: della_merge(sd_a, sd_b, seed)),
    ("cma_es",           lambda seed: cma_es_merge(
        sd_a, sd_b, seed, total_layers, model_a, tokenizer, device)),
]

for name, fn in methods:
    runs = ckpt[name].get("runs", [])
    seeds_done = ckpt[name].get("seeds_done", [])

    if len(runs) == 3:
        print(f"\n[SKIP] {name}: {np.mean(runs):.2f}% +/- {np.std(runs):.2f}%")
        continue

    print(f"\n{'='*65}")
    print(f"{name.upper()} — Full GSM8K (3 runs)")
    print(f"{'='*65}")

    for seed_idx, seed in enumerate(SEEDS):
        if seed in seeds_done:
            print(f"[SKIP] seed={seed} done.")
            continue

        print(f"\nRun {seed_idx+1}/3 (seed={seed})...")
        merged_sd = fn(seed)
        torch.cuda.empty_cache()
        gc.collect()
        merged_model = load_model(merged_sd, model_a, device)
        acc, correct, total = evaluate(merged_model, tokenizer, QUESTIONS, device)
        del merged_model
        del merged_sd
        torch.cuda.empty_cache()
        gc.collect()

        ckpt[name]["runs"].append(acc)
        ckpt[name]["seeds_done"].append(seed)
        save_ckpt(ckpt)
        print(f"{name} Run {seed_idx+1}: {acc:.2f}% [SAVED]")

    mean = np.mean(ckpt[name]["runs"])
    std = np.std(ckpt[name]["runs"])
    print(f"{name}: {mean:.2f}% +/- {std:.2f}%")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 65)
print("FINAL RESULTS — Full GSM8K (1,319 questions)")
print("=" * 65)
print(f"{'Method':<30} {'Mean':>10} {'Std':>8}")
print("-" * 55)

if ckpt["model_a"]:
    print(f"{'Model A':<30} {ckpt['model_a']['accuracy']:>9.2f}%")
if ckpt["model_b"]:
    print(f"{'Model B':<30} {ckpt['model_b']['accuracy']:>9.2f}%")

for name, _ in methods:
    runs = ckpt[name].get("runs", [])
    if runs:
        print(f"{name:<30} {np.mean(runs):>9.2f}% {np.std(runs):>7.2f}%")

print("=" * 65)

# Save results
results = {
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "benchmark": "GSM8K Official",
    "dataset": "openai/gsm8k",
    "total_questions": len(QUESTIONS),
    "model_a": ckpt["model_a"],
    "model_b": ckpt["model_b"],
}
for name, _ in methods:
    runs = ckpt[name].get("runs", [])
    if runs:
        results[name] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs)),
            "all_runs": runs
        }

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {RESULTS_FILE}")
print("\nFULL GSM8K EVALUATION COMPLETE")
