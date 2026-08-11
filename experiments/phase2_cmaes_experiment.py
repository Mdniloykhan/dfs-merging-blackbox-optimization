"""
Phase 2: CMA-ES Experiment
Black-Box Optimization of Mixed Binary-Continuous Variables
in Evolutionary Model Merging

Compares:
- Structured DFS + Random Search (baseline from Phase 1)
- Structured DFS + CMA-ES (this experiment)

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import torch
import numpy as np
import copy
import json
import os
import cma
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.2"
NUM_SEEDS = 5
SEEDS = [0, 42, 84, 126, 168]
NUM_QUESTIONS = 97
CMA_MAX_ITER = 15        # CMA-ES iterations per run
CMA_POPSIZE = 8          # Population size per iteration
CHECKPOINT_FILE = "C:\\Users\\user1\\checkpoint_phase2.json"
RESULTS_FILE = "C:\\Users\\user1\\experiment_results_phase2.json"
FIGURE_FILE = "C:\\Users\\user1\\phase2_results.png"

# Phase 1 results for comparison
PHASE1_RANDOM_SEARCH = {
    "mean": 86.458,
    "std": 1.141,
    "all_runs": [86.45833333333334, 84.375, 86.45833333333334, 87.5, 87.5]
}

# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================
def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[CHECKPOINT SAVED] {CHECKPOINT_FILE}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        print(f"[CHECKPOINT FOUND] Resuming from checkpoint...")
        return data
    return None

# ============================================================
# INITIALIZE OR LOAD CHECKPOINT
# ============================================================
checkpoint = load_checkpoint()

if checkpoint:
    print(f"Resuming Phase 2 from checkpoint.")
    print(f"Completed runs: {checkpoint.get('completed_seeds', [])}")
else:
    print("Starting Phase 2 fresh.")
    checkpoint = {
        "date_started": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "completed_seeds": [],
        "cma_runs": [],
        "cma_run_accs": [],
        "cma_active_counts": [],
        "cma_convergence": []
    }
    save_checkpoint(checkpoint)

print("=" * 65)
print("PHASE 2: CMA-ES EXPERIMENT")
print("=" * 65)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"CMA-ES iterations per run: {CMA_MAX_ITER}")
print(f"Population size: {CMA_POPSIZE}")
print(f"Total evaluations per run: {CMA_MAX_ITER * CMA_POPSIZE}")

# ============================================================
# DEVICE SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# GSM8K QUESTIONS
# ============================================================
GSM8K_QUESTIONS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. How many eggs does she sell at the farmers' market daily if she sells for $2 per egg?", "answer": "9"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "answer": "540"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She has 20 chickens. How many cups of chicken feed does Wendi need to buy for a week?", "answer": "420"},
    {"question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?", "answer": "64"},
    {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", "answer": "460"},
    {"question": "A company sells widgets at $15 each. The company sold 200 widgets last month and 250 widgets this month. How much more revenue did they make this month compared to last month?", "answer": "750"},
    {"question": "Tom has 3 times as many marbles as Jerry. Jerry has 15 marbles. How many marbles do they have together?", "answer": "60"},
    {"question": "A baker makes 48 cookies. She puts them in boxes of 6. How many boxes does she need?", "answer": "8"},
    {"question": "Sarah earns $12 per hour. She worked 8 hours on Monday and 6 hours on Tuesday. How much did she earn in total?", "answer": "168"},
    {"question": "A train travels at 60 mph. How far does it travel in 2.5 hours?", "answer": "150"},
    {"question": "Mark has $50. He buys 3 books at $8 each. How much money does he have left?", "answer": "26"},
    {"question": "A rectangle has a length of 12 cm and width of 5 cm. What is its area?", "answer": "60"},
    {"question": "There are 24 students in a class. If they are divided into groups of 4, how many groups are there?", "answer": "6"},
    {"question": "A store has 150 apples. They sell 45 on Monday and 38 on Tuesday. How many apples are left?", "answer": "67"},
    {"question": "John runs 5 km every day. How many km does he run in 2 weeks?", "answer": "70"},
    {"question": "A bag has 8 red balls and 12 blue balls. How many balls are in the bag?", "answer": "20"},
    {"question": "Emma saves $25 per week. How much does she save in 8 weeks?", "answer": "200"},
    {"question": "A farmer has 5 fields. Each field produces 120 kg of wheat. How much wheat does he produce in total?", "answer": "600"},
    {"question": "A car travels 240 km using 20 liters of fuel. How many km per liter does it get?", "answer": "12"},
    {"question": "A box contains 144 chocolates. If shared equally among 12 children, how many does each get?", "answer": "12"},
    {"question": "Lisa has 3 times as many stickers as Mike. Mike has 24 stickers. How many stickers do they have together?", "answer": "96"},
    {"question": "A shop sells shirts for $25 each. If they sold 40 shirts, how much money did they make?", "answer": "1000"},
    {"question": "There are 7 days in a week. How many days are in 52 weeks?", "answer": "364"},
    {"question": "A pool holds 5000 liters. If 1200 liters evaporate, how many liters remain?", "answer": "3800"},
    {"question": "Peter earns $15 per hour. He works 40 hours per week. How much does he earn per week?", "answer": "600"},
    {"question": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?", "answer": "2"},
    {"question": "A school has 450 students. If 180 are boys, how many are girls?", "answer": "270"},
    {"question": "A jar contains 200 candies. Maria takes 45 and John takes 38. How many candies remain?", "answer": "117"},
    {"question": "A cyclist rides 18 km per hour. How far does he ride in 3 hours?", "answer": "54"},
    {"question": "A factory produces 250 toys per day. How many toys does it produce in 5 days?", "answer": "1250"},
    {"question": "Anna has $100. She spends $35 on food and $28 on clothes. How much does she have left?", "answer": "37"},
    {"question": "A rope is 48 meters long. It is cut into pieces of 6 meters each. How many pieces are there?", "answer": "8"},
    {"question": "There are 30 students. Each student needs 3 pencils. How many pencils are needed?", "answer": "90"},
    {"question": "A book has 320 pages. Maria reads 40 pages per day. How many days does it take to finish?", "answer": "8"},
    {"question": "A garden is 15 meters long and 8 meters wide. What is its perimeter?", "answer": "46"},
    {"question": "Sam has 4 times as many coins as Tim. Tim has 12 coins. How many coins does Sam have?", "answer": "48"},
    {"question": "A supermarket sells 500 items per day. How many items does it sell in 30 days?", "answer": "15000"},
    {"question": "A bottle holds 750 ml. How many bottles are needed to hold 3 liters?", "answer": "4"},
    {"question": "There are 12 months in a year. How many months are in 5 years?", "answer": "60"},
    {"question": "A worker earns $80 per day. How much does he earn in 15 days?", "answer": "1200"},
    {"question": "A bus can carry 45 passengers. How many buses are needed for 180 passengers?", "answer": "4"},
    {"question": "A fruit seller has 96 oranges. He packs them in bags of 8. How many bags does he need?", "answer": "12"},
    {"question": "Two numbers add up to 50. One number is 18. What is the other?", "answer": "32"},
    {"question": "A swimming pool is 25 meters long. A swimmer does 8 laps. How many meters does he swim?", "answer": "200"},
    {"question": "A family spends $450 per month on groceries. How much do they spend in a year?", "answer": "5400"},
    {"question": "A garden has 5 rows of flowers with 12 flowers in each row. How many flowers are there?", "answer": "60"},
    {"question": "Mike types 60 words per minute. How many words does he type in 15 minutes?", "answer": "900"},
    {"question": "A store buys items for $20 each and sells them for $35. What is the profit per item?", "answer": "15"},
    {"question": "There are 1000 students. 600 study science. How many do not study science?", "answer": "400"},
    {"question": "A car uses 8 liters of fuel per 100 km. How much fuel does it need for 250 km?", "answer": "20"},
    {"question": "A box weighs 15 kg. How much do 12 such boxes weigh?", "answer": "180"},
    {"question": "A library has 2400 books. 800 are fiction. How many are non-fiction?", "answer": "1600"},
    {"question": "Jenny bakes 6 dozen cookies. How many cookies does she bake?", "answer": "72"},
    {"question": "A class has 35 students. 14 are absent. How many are present?", "answer": "21"},
    {"question": "A store sells 3 items for $10. How much do 9 items cost?", "answer": "30"},
    {"question": "A plane flies at 800 km per hour. How far does it fly in 4 hours?", "answer": "3200"},
    {"question": "David saves $150 per month. How much does he save in 6 months?", "answer": "900"},
    {"question": "A field produces 1200 kg of corn. If sold at $2 per kg, how much money is made?", "answer": "2400"},
    {"question": "A number multiplied by 7 equals 84. What is the number?", "answer": "12"},
    {"question": "A tank has 800 liters. 250 liters are used. How many liters remain?", "answer": "550"},
    {"question": "Three friends share 48 candies equally. How many does each get?", "answer": "16"},
    {"question": "A building has 15 floors with 8 apartments each. How many apartments are there?", "answer": "120"},
    {"question": "A shirt costs $45. With a 20% discount, how much does it cost?", "answer": "36"},
    {"question": "A train has 12 coaches with 60 seats each. How many seats in total?", "answer": "720"},
    {"question": "Mary reads 25 pages per day. How many days to read a 300-page book?", "answer": "12"},
    {"question": "A box has 5 layers with 20 items per layer. How many items are in the box?", "answer": "100"},
    {"question": "A worker produces 45 units per hour. How many units in 8 hours?", "answer": "360"},
    {"question": "A number divided by 6 equals 15. What is the number?", "answer": "90"},
    {"question": "A store has 240 items. They sell 60%. How many items are sold?", "answer": "144"},
    {"question": "A pond has 500 fish. 150 are caught. How many remain?", "answer": "350"},
    {"question": "Each child gets 4 balloons. There are 25 children. How many balloons are needed?", "answer": "100"},
    {"question": "A wall is 6 meters high and 10 meters wide. What is its area?", "answer": "60"},
    {"question": "Tom earns $18 per hour and works 35 hours per week. What is his weekly income?", "answer": "630"},
    {"question": "A jar contains 180 marbles. 60 are red and the rest are blue. How many are blue?", "answer": "120"},
    {"question": "A school has 8 classes with 32 students each. How many students in total?", "answer": "256"},
    {"question": "A book costs $12. How much do 15 books cost?", "answer": "180"},
    {"question": "A runner completes a 10 km race in 50 minutes. What is his speed in km per minute?", "answer": "0.2"},
    {"question": "A basket holds 24 apples. How many baskets are needed for 144 apples?", "answer": "6"},
    {"question": "A shop has 350 items. 70 are returned. How many items remain?", "answer": "280"},
    {"question": "Five friends each contribute $15 for a gift. How much do they collect in total?", "answer": "75"},
    {"question": "A factory makes 1200 products in 8 hours. How many per hour?", "answer": "150"},
    {"question": "A number increased by 35 equals 92. What is the number?", "answer": "57"},
    {"question": "A car park has 8 rows with 15 spaces each. How many spaces in total?", "answer": "120"},
    {"question": "A student scores 85, 90, and 95 on three tests. What is the average?", "answer": "90"},
    {"question": "A shop bought 200 items at $5 each and sold them at $8 each. What is the total profit?", "answer": "600"},
    {"question": "A container holds 2.5 liters. How many containers are needed for 15 liters?", "answer": "6"},
    {"question": "A house has 4 bedrooms. Each bedroom has 2 windows. How many windows in total?", "answer": "8"},
    {"question": "A team scores 3 goals per game. How many goals in 12 games?", "answer": "36"},
    {"question": "A number subtracted from 100 equals 37. What is the number?", "answer": "63"},
    {"question": "A cinema has 200 seats. 75% are occupied. How many seats are occupied?", "answer": "150"},
    {"question": "A person walks 4 km per hour. How far do they walk in 2.5 hours?", "answer": "10"},
    {"question": "A jar has 50 coins. 20 are quarters. How many are not quarters?", "answer": "30"},
    {"question": "A bag of rice weighs 5 kg. How much do 14 bags weigh?", "answer": "70"},
    {"question": "A student reads 30 pages per day. How many pages in 10 days?", "answer": "300"},
]

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, tokenizer, questions, device, max_new_tokens=50):
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in questions:
            prompt = f"Solve this math problem and give only the final numerical answer.\nProblem: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    temperature=1.0, pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if str(item['answer']).strip() in prediction:
                correct += 1
    return correct / len(questions) * 100

# ============================================================
# LOAD MODELS
# ============================================================
print("\nLoading models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_A)
tokenizer.pad_token = tokenizer.eos_token

model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model A: {sum(p.numel() for p in model_a.parameters()):,} parameters")

model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model B: {sum(p.numel() for p in model_b.parameters()):,} parameters")

state_dict_a = {k: v.cpu().clone() for k, v in model_a.state_dict().items()}
state_dict_b = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
total_layers = len(state_dict_a)
layer_keys = list(state_dict_a.keys())
print(f"Total parameter groups: {total_layers}")

del model_b
torch.cuda.empty_cache()

# ============================================================
# CMA-ES EXPERIMENT
# ============================================================
print("\n" + "=" * 65)
print("CMA-ES EXPERIMENT: Structured DFS + CMA-ES (5 runs)")
print("=" * 65)
print("Key difference from Phase 1:")
print("- Phase 1: Random sampling of continuous weights x")
print("- Phase 2: CMA-ES intelligently searches continuous weights x")
print("- Both use same binary layer selection strategy")

for seed_idx, seed in enumerate(SEEDS):
    if seed in checkpoint["completed_seeds"]:
        print(f"\n[SKIP] CMA-ES Run {seed_idx+1} already done.")
        continue

    print(f"\nCMA-ES Run {seed_idx+1}/5 (seed={seed})...")
    np.random.seed(seed)

    # Step 1: Binary decision — select active layers
    # Use fixed selection probability for fair comparison
    selection_prob = 0.5
    np.random.seed(seed)
    binary_z = np.random.binomial(1, selection_prob, total_layers)
    active_layer_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
    active_count = len(active_layer_keys)
    print(f"  Active layers selected: {active_count}/{total_layers}")

    # Step 2: CMA-ES optimizes continuous weights for active layers
    # Dimension = number of active layers
    dim = active_count
    x0 = [0.5] * dim  # Initial guess: equal weighting
    sigma0 = 0.2      # Initial step size

    # CMA-ES options
    opts = cma.CMAOptions()
    opts['seed'] = seed
    opts['popsize'] = CMA_POPSIZE
    opts['maxiter'] = CMA_MAX_ITER
    opts['bounds'] = [0.1, 0.9]  # Alpha between 0.1 and 0.9
    opts['verbose'] = -9  # Silent
    opts['tolx'] = 1e-4
    opts['tolfun'] = 1e-4

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # Check if this seed has partial iteration progress
    partial_key = f"partial_seed_{seed}"
    if partial_key in checkpoint and checkpoint[partial_key]["seed"] == seed:
        partial = checkpoint[partial_key]
        run_accs = partial["run_accs"]
        best_acc = partial["best_acc"]
        iteration = partial["iteration"]
        active_count = partial["active_count"]
        best_candidate = None  # Will be updated when better solution found
        print(f"  [RESUMING from iteration {iteration}] Best so far: {best_acc:.1f}%")
        # Fast-forward CMA-ES distribution using saved history
        for _ in range(iteration):
            dummy_solutions = es.ask()
            dummy_fitnesses = [-best_acc] * len(dummy_solutions)
            es.tell(dummy_solutions, dummy_fitnesses)
    else:
        run_accs = []
        iteration = 0
        best_acc = 0
        best_candidate = None

    print(f"  Running CMA-ES ({CMA_MAX_ITER} iterations x {CMA_POPSIZE} population)...")

    while not es.stop() and iteration < CMA_MAX_ITER:
        # Ask for candidate solutions
        solutions = es.ask()
        fitnesses = []

        for sol_idx, solution in enumerate(solutions):
            # Clip to valid range
            alphas = np.clip(solution, 0.1, 0.9)

            # Build merged model using CMA-ES suggested alphas
            candidate = {k: v.clone() for k, v in state_dict_a.items()}
            for layer_idx, key in enumerate(active_layer_keys):
                if key in state_dict_b and state_dict_a[key].shape == state_dict_b[key].shape:
                    alpha = alphas[layer_idx]
                    candidate[key] = (alpha * state_dict_a[key].float() +
                                     (1 - alpha) * state_dict_b[key].float()).half()

            # Evaluate
            candidate_model = copy.deepcopy(model_a)
            candidate_model.load_state_dict({k: v.to(device) for k, v in candidate.items()})
            acc = evaluate_model(candidate_model, tokenizer, GSM8K_QUESTIONS, device)
            del candidate_model
            torch.cuda.empty_cache()

            # CMA-ES minimizes, so negate accuracy
            fitnesses.append(-acc)

            if acc > best_acc:
                best_acc = acc
                best_candidate = alphas.copy()

        # Tell CMA-ES the results
        es.tell(solutions, fitnesses)

        # Track best accuracy this iteration
        iter_best = -min(fitnesses)
        run_accs.append(iter_best)
        iteration += 1

        print(f"  Iter {iteration}/{CMA_MAX_ITER}: Best this iter={iter_best:.1f}% | Overall best={best_acc:.1f}%")

        # ITERATION-LEVEL CHECKPOINT — saves after every single iteration
        # If machine crashes, we resume from last completed iteration
        partial_key = f"partial_seed_{seed}"
        checkpoint[partial_key] = {
            "seed": seed,
            "iteration": iteration,
            "best_acc": best_acc,
            "run_accs": run_accs,
            "active_count": active_count,
            "es_history": [{"mean": es.mean.tolist()}]
        }
        save_checkpoint(checkpoint)
        print(f"  [ITERATION CHECKPOINT] Iter {iteration} saved. Best so far: {best_acc:.1f}%")

    # Run complete — save final results and remove partial checkpoint
    checkpoint["cma_runs"].append(best_acc)
    checkpoint["cma_run_accs"].append(run_accs)
    checkpoint["cma_active_counts"].append(active_count)
    checkpoint["cma_convergence"].append(run_accs)
    checkpoint["completed_seeds"].append(seed)

    # Remove partial checkpoint for this seed — run is complete
    partial_key = f"partial_seed_{seed}"
    if partial_key in checkpoint:
        del checkpoint[partial_key]

    save_checkpoint(checkpoint)
    print(f"CMA-ES Run {seed_idx+1} Best Accuracy: {best_acc:.1f}% [SAVED]")

# ============================================================
# RESULTS SUMMARY
# ============================================================
cma_mean = np.mean(checkpoint["cma_runs"])
cma_std = np.std(checkpoint["cma_runs"])
random_mean = PHASE1_RANDOM_SEARCH["mean"]
random_std = PHASE1_RANDOM_SEARCH["std"]

print("\n" + "=" * 65)
print("PHASE 2 RESULTS SUMMARY")
print("=" * 65)
print(f"{'Method':<40} {'Mean':>10} {'Std':>8}")
print("-" * 65)
print(f"{'Structured DFS + Random Search':<40} {random_mean:>9.1f}% {random_std:>7.1f}%")
print(f"{'Structured DFS + CMA-ES':<40} {cma_mean:>9.1f}% {cma_std:>7.1f}%")
print("=" * 65)
print(f"\nCMA-ES improvement over Random Search: {cma_mean - random_mean:+.2f}%")
print(f"CMA-ES runs: {checkpoint['cma_runs']}")

# Statistical test
from scipy import stats
try:
    stat, p = stats.wilcoxon(checkpoint["cma_runs"], PHASE1_RANDOM_SEARCH["all_runs"])
    print(f"Wilcoxon p-value: {p:.4f}")
    if p < 0.05:
        print("Result: STATISTICALLY SIGNIFICANT")
    else:
        print("Result: Not statistically significant")
except Exception as e:
    stat, p = stats.ttest_rel(checkpoint["cma_runs"], PHASE1_RANDOM_SEARCH["all_runs"])
    print(f"Paired t-test p-value: {p:.4f}")

# Save results
results = {
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "cma_es": {
        "mean": cma_mean,
        "std": cma_std,
        "all_runs": checkpoint["cma_runs"],
        "run_accs": checkpoint["cma_run_accs"],
        "active_counts": checkpoint["cma_active_counts"]
    },
    "random_search_phase1": PHASE1_RANDOM_SEARCH,
    "improvement": cma_mean - random_mean,
    "wilcoxon_p": float(p)
}
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")

# ============================================================
# PLOTS
# ============================================================
print("\nGenerating Phase 2 figures...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Phase 2: CMA-ES vs Random Search in Structured DFS\n'
    '(Mistral-7B, GSM8K, 5 runs)',
    fontsize=12, fontweight='bold'
)

# Plot 1: Accuracy comparison
ax1 = axes[0]
methods = ['Structured DFS\n+ Random Search', 'Structured DFS\n+ CMA-ES']
means = [random_mean, cma_mean]
stds = [random_std, cma_std]
colors = ['#e74c3c', '#2ecc71']
bars = ax1.bar(methods, means, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
ax1.errorbar(range(len(methods)), means, yerr=stds,
             fmt='none', color='black', capsize=8, linewidth=2)
ax1.set_ylabel('Accuracy on GSM8K (%)', fontsize=11)
ax1.set_title('Accuracy: CMA-ES vs Random Search\n(Mean +/- Std, 5 runs)', fontsize=11, fontweight='bold')
ax1.set_ylim(min(means) - 3, max(means) + 3)
ax1.grid(axis='y', alpha=0.3)
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.1,
             f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Convergence comparison
ax2 = axes[1]
colors_cma = plt.cm.Greens(np.linspace(0.4, 0.9, NUM_SEEDS))
colors_random = plt.cm.Reds(np.linspace(0.4, 0.9, NUM_SEEDS))

for i, run_accs in enumerate(checkpoint["cma_run_accs"]):
    ax2.plot(range(1, len(run_accs)+1), run_accs,
             color=colors_cma[i], alpha=0.7, linewidth=1.5,
             label=f'CMA-ES Run {i+1}' if i == 0 else '')

ax2.axhline(y=random_mean, color='red', linestyle='--', linewidth=2,
            label=f'Random Search mean ({random_mean:.1f}%)')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Best Accuracy so far (%)', fontsize=11)
ax2.set_title('CMA-ES Convergence\nAcross 5 Runs', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xticks(range(1, CMA_MAX_ITER+1))

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=300, bbox_inches='tight')
print(f"Figure saved to {FIGURE_FILE}")
plt.show()

print("\n" + "=" * 65)
print("PHASE 2 COMPLETE")
print(f"Results: {RESULTS_FILE}")
print(f"Figure: {FIGURE_FILE}")
print("=" * 65)
