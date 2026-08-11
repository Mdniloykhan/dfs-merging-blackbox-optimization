"""
Phase 5: Ablation Study
Tests sensitivity of Structured DFS + CMA-ES to key hyperparameters

Ablation 1: Effect of binary selection probability (sparsity of z)
Ablation 2: Effect of CMA-ES population size
Ablation 3: Effect of alpha range for continuous weights
Ablation 4: Structured DFS with fixed alpha vs learned alpha (CMA-ES)

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
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.2"
NUM_QUESTIONS = 97
SEEDS = [0, 42, 84]  # 3 seeds for ablation (faster)
CHECKPOINT_FILE = "C:\\Users\\user1\\checkpoint_phase5.json"
RESULTS_FILE = "C:\\Users\\user1\\experiment_results_phase5.json"
FIGURE_FILE = "C:\\Users\\user1\\phase5_ablation.png"

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
    print(f"Resuming Phase 5. Completed: {checkpoint.get('completed', [])}")
else:
    print("Starting Phase 5 fresh.")
    checkpoint = {
        "date_started": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "completed": [],
        "ablation1_selection_prob": {},
        "ablation2_cma_popsize": {},
        "ablation3_alpha_range": {},
        "ablation4_fixed_vs_learned": {}
    }
    save_checkpoint(checkpoint)

print("=" * 65)
print("PHASE 5: ABLATION STUDY")
print("=" * 65)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Seeds: {SEEDS}")

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} — {torch.cuda.get_device_name(0)}")

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
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=256, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, temperature=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
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

model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})

sd_a = {k: v.cpu().clone() for k, v in model_a.state_dict().items()}
sd_b = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
total_layers = len(sd_a)
layer_keys = list(sd_a.keys())
print(f"Models loaded. Total layers: {total_layers}")

del model_b
torch.cuda.empty_cache()
gc.collect()

def load_merged(sd, model_a, device):
    m = copy.deepcopy(model_a)
    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
    return m

def run_structured_dfs(selection_prob, alpha_low, alpha_high, seed):
    """Core structured DFS with configurable parameters"""
    np.random.seed(seed)
    binary_z = np.random.binomial(1, selection_prob, total_layers)
    active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
    candidate = copy.deepcopy(sd_a)
    for key in active_keys:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            alpha = np.random.uniform(alpha_low, alpha_high)
            candidate[key] = (alpha * sd_a[key].float() +
                             (1-alpha) * sd_b[key].float()).half()
    return candidate, len(active_keys)

def run_cma_es(selection_prob, alpha_low, alpha_high, popsize, seed, n_iter=8):
    """CMA-ES with configurable parameters"""
    np.random.seed(seed)
    binary_z = np.random.binomial(1, selection_prob, total_layers)
    active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
    dim = len(active_keys)

    opts = cma.CMAOptions()
    opts['seed'] = seed
    opts['popsize'] = popsize
    opts['maxiter'] = n_iter
    opts['bounds'] = [alpha_low, alpha_high]
    opts['verbose'] = -9

    x0 = [(alpha_low + alpha_high) / 2] * dim
    sigma0 = (alpha_high - alpha_low) / 4
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_sd = copy.deepcopy(sd_a)
    best_acc = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for solution in solutions:
            alphas = np.clip(solution, alpha_low, alpha_high)
            candidate = copy.deepcopy(sd_a)
            for i, key in enumerate(active_keys):
                if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                    candidate[key] = (alphas[i] * sd_a[key].float() +
                                     (1-alphas[i]) * sd_b[key].float()).half()
            m = load_merged(candidate, model_a, device)
            acc = evaluate_model(m, tokenizer, GSM8K_QUESTIONS, device)
            del m
            torch.cuda.empty_cache()
            gc.collect()
            fitnesses.append(-acc)
            if acc > best_acc:
                best_acc = acc
                best_sd = candidate
        es.tell(solutions, fitnesses)

    return best_sd, best_acc, len(active_keys)

# ============================================================
# ABLATION 1: EFFECT OF BINARY SELECTION PROBABILITY
# ============================================================
if "ablation1" not in checkpoint["completed"]:
    print("\n" + "=" * 65)
    print("ABLATION 1: Effect of Binary Selection Probability (z sparsity)")
    print("=" * 65)
    print("Tests: How does the fraction of selected layers affect accuracy?")
    print("Fixed: alpha_range=[0.3,0.7], CMA-ES popsize=4, 5 iterations")

    selection_probs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results = {}

    for prob in selection_probs:
        prob_key = str(prob)
        if prob_key in checkpoint["ablation1_selection_prob"]:
            print(f"[SKIP] prob={prob} already done.")
            continue

        # Load partial progress for this probability if exists
        partial_key = f"partial_abl1_{prob}"
        if partial_key in checkpoint:
            accs = checkpoint[partial_key]["accs"]
            active_counts = checkpoint[partial_key]["active_counts"]
            done_seeds = checkpoint[partial_key]["done_seeds"]
        else:
            accs = []
            active_counts = []
            done_seeds = []
        print(f"\nSelection prob={prob}... (completed seeds: {done_seeds})")

        for seed in SEEDS:
            if seed in done_seeds:
                print(f"  [SKIP] seed={seed} already done.")
                continue
            np.random.seed(seed)
            binary_z = np.random.binomial(1, prob, total_layers)
            active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
            active_count = len(active_keys)
            active_counts.append(active_count)

            if len(active_keys) == 0:
                accs.append(43.8)
                done_seeds.append(seed)
                checkpoint[partial_key] = {"accs": accs, "active_counts": active_counts, "done_seeds": done_seeds}
                save_checkpoint(checkpoint)
                continue

            # Aggressive memory clearing before each CMA-ES run
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()

            dim = len(active_keys)
            opts = cma.CMAOptions()
            opts['seed'] = seed
            opts['popsize'] = 4
            opts['maxiter'] = 3
            opts['bounds'] = [0.3, 0.7]
            opts['verbose'] = -9

            es = cma.CMAEvolutionStrategy([0.5]*dim, 0.1, opts)
            best_acc = 0

            while not es.stop():
                solutions = es.ask()
                fitnesses = []
                for sol in solutions:
                    alphas = np.clip(sol, 0.3, 0.7)
                    candidate = copy.deepcopy(sd_a)
                    for i, key in enumerate(active_keys):
                        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                            candidate[key] = (alphas[i] * sd_a[key].float() +
                                             (1-alphas[i]) * sd_b[key].float()).half()
                    m = load_merged(candidate, model_a, device)
                    acc = evaluate_model(m, tokenizer, GSM8K_QUESTIONS, device)
                    del m
                    del candidate
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    fitnesses.append(-acc)
                    if acc > best_acc:
                        best_acc = acc
                es.tell(solutions, fitnesses)
                torch.cuda.empty_cache()
                gc.collect()

            accs.append(best_acc)
            done_seeds.append(seed)
            checkpoint[partial_key] = {"accs": accs, "active_counts": active_counts, "done_seeds": done_seeds}
            save_checkpoint(checkpoint)
            print(f"  seed={seed}: active={active_count}/{total_layers} acc={best_acc:.1f}% [SAVED]")

        checkpoint["ablation1_selection_prob"][prob_key] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "runs": accs,
            "avg_active": float(np.mean(active_counts))
        }
        save_checkpoint(checkpoint)
        print(f"prob={prob}: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}% [SAVED]")

    checkpoint["completed"].append("ablation1")
    save_checkpoint(checkpoint)
else:
    print("\n[SKIP] Ablation 1 already done.")

# ============================================================
# ABLATION 2: EFFECT OF ALPHA RANGE
# ============================================================
if "ablation2" not in checkpoint["completed"]:
    print("\n" + "=" * 65)
    print("ABLATION 2: Effect of Alpha Range for Continuous Weights")
    print("=" * 65)
    print("Tests: How does the allowed alpha range affect merging quality?")
    print("Fixed: selection_prob=0.5, CMA-ES popsize=4, 5 iterations")

    alpha_ranges = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]
    results = {}

    for alpha_low, alpha_high in alpha_ranges:
        range_key = f"{alpha_low}_{alpha_high}"
        if range_key in checkpoint["ablation3_alpha_range"]:
            print(f"[SKIP] alpha=[{alpha_low},{alpha_high}] already done.")
            continue

        accs = []
        print(f"\nAlpha range=[{alpha_low},{alpha_high}]...")

        for seed in SEEDS:
            np.random.seed(seed)
            binary_z = np.random.binomial(1, 0.5, total_layers)
            active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
            dim = len(active_keys)

            if alpha_low == alpha_high:
                # Fixed alpha — no optimization needed
                candidate = copy.deepcopy(sd_a)
                for key in active_keys:
                    if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                        candidate[key] = (alpha_low * sd_a[key].float() +
                                         (1-alpha_low) * sd_b[key].float()).half()
                m = load_merged(candidate, model_a, device)
                acc = evaluate_model(m, tokenizer, GSM8K_QUESTIONS, device)
                del m
                torch.cuda.empty_cache()
                gc.collect()
                accs.append(acc)
            else:
                opts = cma.CMAOptions()
                opts['seed'] = seed
                opts['popsize'] = 4
                opts['maxiter'] = 5
                opts['bounds'] = [alpha_low, alpha_high]
                opts['verbose'] = -9

                mid = (alpha_low + alpha_high) / 2
                sigma = (alpha_high - alpha_low) / 4
                es = cma.CMAEvolutionStrategy([mid]*dim, sigma, opts)
                best_acc = 0

                while not es.stop():
                    solutions = es.ask()
                    fitnesses = []
                    for sol in solutions:
                        alphas = np.clip(sol, alpha_low, alpha_high)
                        candidate = copy.deepcopy(sd_a)
                        for i, key in enumerate(active_keys):
                            if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                                candidate[key] = (alphas[i] * sd_a[key].float() +
                                                 (1-alphas[i]) * sd_b[key].float()).half()
                        m = load_merged(candidate, model_a, device)
                        acc = evaluate_model(m, tokenizer, GSM8K_QUESTIONS, device)
                        del m
                        torch.cuda.empty_cache()
                        gc.collect()
                        fitnesses.append(-acc)
                        if acc > best_acc:
                            best_acc = acc
                    es.tell(solutions, fitnesses)
                accs.append(best_acc)

            print(f"  seed={seed}: acc={accs[-1]:.1f}%")

        checkpoint["ablation3_alpha_range"][range_key] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "runs": accs
        }
        save_checkpoint(checkpoint)
        print(f"alpha=[{alpha_low},{alpha_high}]: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}% [SAVED]")

    checkpoint["completed"].append("ablation2")
    save_checkpoint(checkpoint)
else:
    print("\n[SKIP] Ablation 2 already done.")

# ============================================================
# ABLATION 3: FIXED vs LEARNED ALPHA (KEY ABLATION)
# ============================================================
if "ablation3" not in checkpoint["completed"]:
    print("\n" + "=" * 65)
    print("ABLATION 3: Fixed Alpha vs CMA-ES Learned Alpha")
    print("=" * 65)
    print("Key question: Does CMA-ES actually help vs fixed alpha=0.5?")

    configs = [
        ("fixed_alpha_0.3", False, 0.3),
        ("fixed_alpha_0.5", False, 0.5),
        ("fixed_alpha_0.7", False, 0.7),
        ("cma_es_learned", True, None),
    ]

    for config_name, use_cma, fixed_alpha in configs:
        if config_name in checkpoint["ablation4_fixed_vs_learned"]:
            print(f"[SKIP] {config_name} already done.")
            continue

        accs = []
        print(f"\n{config_name}...")

        for seed in SEEDS:
            np.random.seed(seed)
            binary_z = np.random.binomial(1, 0.5, total_layers)
            active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]

            if not use_cma:
                candidate = copy.deepcopy(sd_a)
                for key in active_keys:
                    if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                        candidate[key] = (fixed_alpha * sd_a[key].float() +
                                         (1-fixed_alpha) * sd_b[key].float()).half()
                m = load_merged(candidate, model_a, device)
                acc = evaluate_model(m, tokenizer, GSM8K_QUESTIONS, device)
                del m
                torch.cuda.empty_cache()
                gc.collect()
            else:
                dim = len(active_keys)
                opts = cma.CMAOptions()
                opts['seed'] = seed
                opts['popsize'] = 4
                opts['maxiter'] = 8
                opts['bounds'] = [0.1, 0.9]
                opts['verbose'] = -9
                es = cma.CMAEvolutionStrategy([0.5]*dim, 0.2, opts)
                acc = 0
                while not es.stop():
                    solutions = es.ask()
                    fitnesses = []
                    for sol in solutions:
                        alphas = np.clip(sol, 0.1, 0.9)
                        candidate = copy.deepcopy(sd_a)
                        for i, key in enumerate(active_keys):
                            if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                                candidate[key] = (alphas[i] * sd_a[key].float() +
                                                 (1-alphas[i]) * sd_b[key].float()).half()
                        m = load_merged(candidate, model_a, device)
                        a = evaluate_model(m, tokenizer, GSM8K_QUESTIONS, device)
                        del m
                        torch.cuda.empty_cache()
                        gc.collect()
                        fitnesses.append(-a)
                        if a > acc:
                            acc = a
                    es.tell(solutions, fitnesses)

            accs.append(acc)
            print(f"  seed={seed}: acc={acc:.1f}%")

        checkpoint["ablation4_fixed_vs_learned"][config_name] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "runs": accs
        }
        save_checkpoint(checkpoint)
        print(f"{config_name}: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}% [SAVED]")

    checkpoint["completed"].append("ablation3")
    save_checkpoint(checkpoint)
else:
    print("\n[SKIP] Ablation 3 already done.")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 65)
print("PHASE 5 COMPLETE — ABLATION RESULTS")
print("=" * 65)

print("\nAblation 1: Binary Selection Probability")
print(f"{'Prob':>6} {'Active Layers':>14} {'Mean Acc':>10} {'Std':>8}")
print("-" * 45)
for prob_key, data in sorted(checkpoint["ablation1_selection_prob"].items(),
                              key=lambda x: float(x[0])):
    avg_active = data.get("avg_active", 0)
    print(f"{float(prob_key):>6.1f} {avg_active:>14.0f} {data['mean']:>9.1f}% {data['std']:>7.1f}%")

print("\nAblation 2: Alpha Range")
print(f"{'Alpha Range':>15} {'Mean Acc':>10} {'Std':>8}")
print("-" * 38)
for range_key, data in checkpoint["ablation3_alpha_range"].items():
    low, high = range_key.split("_")
    print(f"[{low},{high}]:>15 {data['mean']:>9.1f}% {data['std']:>7.1f}%")

print("\nAblation 3: Fixed Alpha vs CMA-ES Learned Alpha")
print(f"{'Method':>25} {'Mean Acc':>10} {'Std':>8}")
print("-" * 48)
for config_name, data in checkpoint["ablation4_fixed_vs_learned"].items():
    print(f"{config_name:>25} {data['mean']:>9.1f}% {data['std']:>7.1f}%")

# Save results
results = {
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "ablation1_selection_prob": checkpoint["ablation1_selection_prob"],
    "ablation2_alpha_range": checkpoint["ablation3_alpha_range"],
    "ablation3_fixed_vs_learned": checkpoint["ablation4_fixed_vs_learned"]
}
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")

# ============================================================
# PLOTS
# ============================================================
print("\nGenerating ablation figures...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Phase 5: Ablation Study — Structured DFS + CMA-ES\n(Mistral-7B, GSM8K)',
             fontsize=12, fontweight='bold')

# Plot 1: Selection probability
ax1 = axes[0]
probs = sorted([float(k) for k in checkpoint["ablation1_selection_prob"].keys()])
means = [checkpoint["ablation1_selection_prob"][str(p)]["mean"] for p in probs]
stds = [checkpoint["ablation1_selection_prob"][str(p)]["std"] for p in probs]
active = [checkpoint["ablation1_selection_prob"][str(p)].get("avg_active", 0) for p in probs]

ax1.plot(probs, means, 'b-o', linewidth=2, markersize=8)
ax1.fill_between(probs,
                 [m-s for m,s in zip(means,stds)],
                 [m+s for m,s in zip(means,stds)],
                 alpha=0.2, color='blue')
ax1.set_xlabel('Binary Selection Probability', fontsize=11)
ax1.set_ylabel('Accuracy on GSM8K (%)', fontsize=11)
ax1.set_title('Ablation 1: Effect of Layer\nSelection Probability', fontsize=11, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_xticks(probs)

# Plot 2: Alpha range
ax2 = axes[1]
ranges = list(checkpoint["ablation3_alpha_range"].keys())
range_means = [checkpoint["ablation3_alpha_range"][r]["mean"] for r in ranges]
range_stds = [checkpoint["ablation3_alpha_range"][r]["std"] for r in ranges]
range_labels = [f"[{r.split('_')[0]},{r.split('_')[1]}]" for r in ranges]

bars = ax2.bar(range(len(ranges)), range_means, color='#3498db',
               edgecolor='black', linewidth=0.8)
ax2.errorbar(range(len(ranges)), range_means, yerr=range_stds,
             fmt='none', color='black', capsize=5, linewidth=2)
ax2.set_xticks(range(len(ranges)))
ax2.set_xticklabels(range_labels, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Accuracy on GSM8K (%)', fontsize=11)
ax2.set_title('Ablation 2: Effect of\nAlpha Range', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, mean in zip(bars, range_means):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Fixed vs learned alpha
ax3 = axes[2]
configs = list(checkpoint["ablation4_fixed_vs_learned"].keys())
config_means = [checkpoint["ablation4_fixed_vs_learned"][c]["mean"] for c in configs]
config_stds = [checkpoint["ablation4_fixed_vs_learned"][c]["std"] for c in configs]
colors3 = ['#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71']
config_labels = ['Fixed\nα=0.3', 'Fixed\nα=0.5', 'Fixed\nα=0.7', 'CMA-ES\nLearned']

bars3 = ax3.bar(range(len(configs)), config_means, color=colors3,
                edgecolor='black', linewidth=0.8)
ax3.errorbar(range(len(configs)), config_means, yerr=config_stds,
             fmt='none', color='black', capsize=5, linewidth=2)
ax3.set_xticks(range(len(configs)))
ax3.set_xticklabels(config_labels, fontsize=10)
ax3.set_ylabel('Accuracy on GSM8K (%)', fontsize=11)
ax3.set_title('Ablation 3: Fixed Alpha\nvs CMA-ES Learned Alpha', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, mean in zip(bars3, config_means):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=300, bbox_inches='tight')
print(f"Figure saved to {FIGURE_FILE}")
plt.show()

print("\n" + "=" * 65)
print("PHASE 5 COMPLETE")
print("=" * 65)
