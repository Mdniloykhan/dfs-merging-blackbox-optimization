"""
Phase 3: SOTA Comparison Experiment
Compares TIES-Merging, DARE, and DELLA against our methods

Methods compared:
- TIES-Merging: Resolves weight conflicts by trimming and electing signs
- DARE: Drop And REscale — randomly drops and rescales delta weights
- DELLA: Magnitude-based sampling for interference reduction

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
"""

import torch
import numpy as np
import copy
import json
import os
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
NUM_SEEDS = 5
SEEDS = [0, 42, 84, 126, 168]
NUM_QUESTIONS = 97
CHECKPOINT_FILE = "C:\\Users\\user1\\checkpoint_phase3.json"
RESULTS_FILE = "C:\\Users\\user1\\experiment_results_phase3.json"

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
    print(f"Resuming Phase 3. Completed: {checkpoint.get('completed_experiments', [])}")
else:
    print("Starting Phase 3 fresh.")
    checkpoint = {
        "date_started": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "completed_experiments": [],
        "ties": {"runs": [], "completed_seeds": []},
        "dare": {"runs": [], "completed_seeds": []},
        "della": {"runs": [], "completed_seeds": []}
    }
    save_checkpoint(checkpoint)

print("=" * 65)
print("PHASE 3: SOTA COMPARISON")
print("=" * 65)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Methods: TIES-Merging, DARE, DELLA")
print(f"Runs per method: {NUM_SEEDS}")

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
# SOTA MERGING IMPLEMENTATIONS
# ============================================================

def ties_merge(state_dict_a, state_dict_b, alpha=0.5, density=0.2, seed=42):
    """
    TIES-Merging: Trim, Elect Sign, Disjoint Merge
    1. Trim: Keep only top-k% of delta weights by magnitude
    2. Elect Sign: Resolve sign conflicts by majority vote
    3. Merge: Average remaining parameters
    
    Reference: Yadav et al. NeurIPS 2023
    """
    np.random.seed(seed)
    merged = {}
    
    for key in state_dict_a:
        if key in state_dict_b and state_dict_a[key].shape == state_dict_b[key].shape:
            w_a = state_dict_a[key].float()
            w_b = state_dict_b[key].float()
            
            # Compute task vectors (deltas from base model)
            # We use model_a as base
            delta_b = w_b - w_a
            
            # Step 1: TRIM — keep only top density% of delta by magnitude
            flat_delta = delta_b.abs().flatten()
            if len(flat_delta) > 0:
                k = max(1, int(density * len(flat_delta)))
                threshold = torch.topk(flat_delta, k).values.min()
                mask = delta_b.abs() >= threshold
            else:
                mask = torch.ones_like(delta_b, dtype=torch.bool)
            
            trimmed_delta = delta_b * mask.float()
            
            # Step 2: ELECT SIGN — use sign of trimmed delta
            elected_sign = torch.sign(trimmed_delta.sum())
            if elected_sign == 0:
                elected_sign = torch.tensor(1.0)
            
            # Keep only parameters matching elected sign
            sign_mask = (torch.sign(trimmed_delta) == elected_sign) | (trimmed_delta == 0)
            final_delta = trimmed_delta * sign_mask.float()
            
            # Step 3: DISJOINT MERGE
            merged[key] = (w_a + alpha * final_delta).half()
        else:
            merged[key] = state_dict_a[key]
    
    return merged


def dare_merge(state_dict_a, state_dict_b, alpha=0.5, drop_rate=0.9, seed=42):
    """
    DARE: Drop And REscale
    1. Randomly drop delta weights with probability drop_rate
    2. Rescale remaining deltas by 1/(1-drop_rate)
    3. Add to base model
    
    Reference: Yu et al. 2023
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    merged = {}
    
    for key in state_dict_a:
        if key in state_dict_b and state_dict_a[key].shape == state_dict_b[key].shape:
            w_a = state_dict_a[key].float()
            w_b = state_dict_b[key].float()
            
            # Compute delta
            delta = w_b - w_a
            
            # Step 1: DROP — randomly zero out delta weights
            drop_mask = torch.bernoulli(
                torch.ones_like(delta) * (1 - drop_rate)
            ).bool()
            
            # Step 2: RESCALE — compensate for dropped weights
            rescale_factor = 1.0 / (1.0 - drop_rate)
            dare_delta = delta * drop_mask.float() * rescale_factor
            
            # Step 3: ADD to base
            merged[key] = (w_a + alpha * dare_delta).half()
        else:
            merged[key] = state_dict_a[key]
    
    return merged


def della_merge(state_dict_a, state_dict_b, alpha=0.5, density=0.2, seed=42):
    """
    DELLA: Drop via Magnitude-based Sampling
    Similar to DARE but uses magnitude-based sampling instead of uniform random
    Higher magnitude deltas are more likely to be kept
    
    Reference: Deep et al. 2024
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    merged = {}
    
    for key in state_dict_a:
        if key in state_dict_b and state_dict_a[key].shape == state_dict_b[key].shape:
            w_a = state_dict_a[key].float()
            w_b = state_dict_b[key].float()
            
            # Compute delta
            delta = w_b - w_a
            
            # Magnitude-based sampling probability
            # Higher magnitude = higher probability of being kept
            delta_magnitude = delta.abs()
            
            if delta_magnitude.max() > 0:
                # Normalize magnitudes to [0,1] range
                norm_magnitude = delta_magnitude / (delta_magnitude.max() + 1e-8)
                
                # Sample based on magnitude
                # density controls how many parameters to keep on average
                keep_prob = norm_magnitude * density / (norm_magnitude.mean() + 1e-8)
                keep_prob = keep_prob.clamp(0, 1)
                
                # Sample mask
                magnitude_mask = torch.bernoulli(keep_prob).bool()
            else:
                magnitude_mask = torch.zeros_like(delta, dtype=torch.bool)
            
            # Rescale kept deltas
            kept_fraction = magnitude_mask.float().mean().item()
            if kept_fraction > 0:
                rescale = 1.0 / kept_fraction
            else:
                rescale = 1.0
            
            della_delta = delta * magnitude_mask.float() * rescale
            
            merged[key] = (w_a + alpha * della_delta).half()
        else:
            merged[key] = state_dict_a[key]
    
    return merged

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
print(f"Total parameter groups: {total_layers}")

del model_b
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# HELPER: LOAD MERGED MODEL
# ============================================================
def load_merged_model(merged_state_dict, model_a, device):
    merged = copy.deepcopy(model_a)
    merged.load_state_dict({k: v.to(device) for k, v in merged_state_dict.items()})
    return merged

# ============================================================
# EXPERIMENT: TIES-MERGING
# ============================================================
if "ties" not in checkpoint["completed_experiments"]:
    print("\n" + "=" * 65)
    print("TIES-MERGING (5 runs)")
    print("=" * 65)

    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint["ties"]["completed_seeds"]:
            print(f"[SKIP] TIES Run {seed_idx+1} already done.")
            continue

        print(f"\nTIES Run {seed_idx+1}/5 (seed={seed})...")
        merged_sd = ties_merge(state_dict_a, state_dict_b,
                               alpha=0.5, density=0.2, seed=seed)
        merged_model = load_merged_model(merged_sd, model_a, device)
        acc = evaluate_model(merged_model, tokenizer, GSM8K_QUESTIONS, device)
        del merged_model
        torch.cuda.empty_cache()
        gc.collect()

        checkpoint["ties"]["runs"].append(acc)
        checkpoint["ties"]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"TIES Run {seed_idx+1}: {acc:.1f}% [SAVED]")

    checkpoint["completed_experiments"].append("ties")
    save_checkpoint(checkpoint)
else:
    print(f"\n[SKIP] TIES already done.")

ties_mean = np.mean(checkpoint["ties"]["runs"])
ties_std = np.std(checkpoint["ties"]["runs"])
print(f"TIES-Merging: {ties_mean:.1f}% +/- {ties_std:.1f}%")

# ============================================================
# EXPERIMENT: DARE
# ============================================================
if "dare" not in checkpoint["completed_experiments"]:
    print("\n" + "=" * 65)
    print("DARE MERGING (5 runs)")
    print("=" * 65)

    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint["dare"]["completed_seeds"]:
            print(f"[SKIP] DARE Run {seed_idx+1} already done.")
            continue

        print(f"\nDARE Run {seed_idx+1}/5 (seed={seed})...")
        merged_sd = dare_merge(state_dict_a, state_dict_b,
                               alpha=0.5, drop_rate=0.9, seed=seed)
        merged_model = load_merged_model(merged_sd, model_a, device)
        acc = evaluate_model(merged_model, tokenizer, GSM8K_QUESTIONS, device)
        del merged_model
        torch.cuda.empty_cache()
        gc.collect()

        checkpoint["dare"]["runs"].append(acc)
        checkpoint["dare"]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"DARE Run {seed_idx+1}: {acc:.1f}% [SAVED]")

    checkpoint["completed_experiments"].append("dare")
    save_checkpoint(checkpoint)
else:
    print(f"\n[SKIP] DARE already done.")

dare_mean = np.mean(checkpoint["dare"]["runs"])
dare_std = np.std(checkpoint["dare"]["runs"])
print(f"DARE: {dare_mean:.1f}% +/- {dare_std:.1f}%")

# ============================================================
# EXPERIMENT: DELLA
# ============================================================
if "della" not in checkpoint["completed_experiments"]:
    print("\n" + "=" * 65)
    print("DELLA MERGING (5 runs)")
    print("=" * 65)

    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint["della"]["completed_seeds"]:
            print(f"[SKIP] DELLA Run {seed_idx+1} already done.")
            continue

        print(f"\nDELLA Run {seed_idx+1}/5 (seed={seed})...")
        merged_sd = della_merge(state_dict_a, state_dict_b,
                                alpha=0.5, density=0.2, seed=seed)
        merged_model = load_merged_model(merged_sd, model_a, device)
        acc = evaluate_model(merged_model, tokenizer, GSM8K_QUESTIONS, device)
        del merged_model
        torch.cuda.empty_cache()
        gc.collect()

        checkpoint["della"]["runs"].append(acc)
        checkpoint["della"]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"DELLA Run {seed_idx+1}: {acc:.1f}% [SAVED]")

    checkpoint["completed_experiments"].append("della")
    save_checkpoint(checkpoint)
else:
    print(f"\n[SKIP] DELLA already done.")

della_mean = np.mean(checkpoint["della"]["runs"])
della_std = np.std(checkpoint["della"]["runs"])
print(f"DELLA: {della_mean:.1f}% +/- {della_std:.1f}%")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 65)
print("PHASE 3 COMPLETE — SOTA COMPARISON RESULTS")
print("=" * 65)
print(f"{'Method':<35} {'Mean':>10} {'Std':>8}")
print("-" * 65)

# Previous results for reference
print(f"{'Model A (Mistral-7B-v0.1)':<35} {'43.8%':>10} {'N/A':>8}")
print(f"{'Model B (Mistral-7B-Instruct)':<35} {'85.4%':>10} {'N/A':>8}")
print(f"{'PS Merging':<35} {'84.4%':>10} {'0.0%':>8}")
print(f"{'Unstructured DFS':<35} {'86.0%':>10} {'0.5%':>8}")
print(f"{'Structured DFS (Random)':<35} {'86.5%':>10} {'1.1%':>8}")
print(f"{'Structured DFS + CMA-ES':<35} {'88.1%':>10} {'1.1%':>8}")
print("-" * 65)
print(f"{'TIES-Merging':<35} {ties_mean:>9.1f}% {ties_std:>7.1f}%")
print(f"{'DARE':<35} {dare_mean:>9.1f}% {dare_std:>7.1f}%")
print(f"{'DELLA':<35} {della_mean:>9.1f}% {della_std:>7.1f}%")
print("=" * 65)

# Save results
results = {
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "ties": {"mean": ties_mean, "std": ties_std, "all_runs": checkpoint["ties"]["runs"]},
    "dare": {"mean": dare_mean, "std": dare_std, "all_runs": checkpoint["dare"]["runs"]},
    "della": {"mean": della_mean, "std": della_std, "all_runs": checkpoint["della"]["runs"]}
}
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")
print("\nPHASE 3 COMPLETE")
