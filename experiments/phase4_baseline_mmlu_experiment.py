"""
Phase 4: MMLU Benchmark Experiment
Black-Box Optimization of Mixed Binary-Continuous Variables
in Evolutionary Model Merging

Tests all methods on MMLU (Massive Multitask Language Understanding)
100 questions across 10 subjects, 5 runs per method.

Model B evaluation is integrated directly into this script.
Checkpoint saves after every completed run.

Methods compared:
- Model A alone (Mistral-7B-v0.1)
- Model B alone (Mistral-7B-Instruct-v0.2)
- PS Merging
- TIES-Merging
- DARE
- DELLA
- Structured DFS
- Structured DFS + CMA-ES

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
import cma
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
CHECKPOINT_FILE = "checkpoint_phase4.json"
RESULTS_FILE = "experiment_results_phase4.json"

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
    print(f"Resuming Phase 4. Completed: {checkpoint.get('completed_experiments', [])}")
else:
    print("Starting Phase 4 fresh.")
    checkpoint = {
        "date_started": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "completed_experiments": [],
        "model_a": {"runs": [], "completed_seeds": []},
        "model_b": {"runs": [], "completed_seeds": []},
        "ps_merging": {"runs": [], "completed_seeds": []},
        "ties": {"runs": [], "completed_seeds": []},
        "dare": {"runs": [], "completed_seeds": []},
        "della": {"runs": [], "completed_seeds": []},
        "structured_dfs": {"runs": [], "completed_seeds": []},
        "cma_es": {"runs": [], "completed_seeds": []}
    }
    save_checkpoint(checkpoint)

print("=" * 65)
print("PHASE 4: MMLU BENCHMARK")
print("=" * 65)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# DEVICE SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# MMLU QUESTIONS — 100 questions across 10 subjects
# ============================================================
MMLU_QUESTIONS = [
    # Mathematics (10)
    {"question": "What is the derivative of x^3?", "choices": ["A) 3x", "B) 3x^2", "C) x^2", "D) 2x^3"], "answer": "B"},
    {"question": "What is the integral of 2x?", "choices": ["A) 2", "B) x^2 + C", "C) 2x^2 + C", "D) x + C"], "answer": "B"},
    {"question": "What is log base 2 of 8?", "choices": ["A) 2", "B) 4", "C) 3", "D) 8"], "answer": "C"},
    {"question": "What is the sum of angles in a triangle?", "choices": ["A) 90 degrees", "B) 180 degrees", "C) 270 degrees", "D) 360 degrees"], "answer": "B"},
    {"question": "What is the value of pi approximately?", "choices": ["A) 3.14", "B) 2.71", "C) 1.41", "D) 1.73"], "answer": "A"},
    {"question": "What is 2 to the power of 10?", "choices": ["A) 512", "B) 256", "C) 1024", "D) 2048"], "answer": "C"},
    {"question": "What is the quadratic formula solution for ax^2+bx+c=0?", "choices": ["A) x = -b/2a", "B) x = (-b+-sqrt(b^2-4ac))/2a", "C) x = b/2a", "D) x = -b+-sqrt(b)/2a"], "answer": "B"},
    {"question": "What is the Pythagorean theorem?", "choices": ["A) a+b=c", "B) a^2+b^2=c^2", "C) a^2-b^2=c^2", "D) a*b=c^2"], "answer": "B"},
    {"question": "What is the factorial of 5?", "choices": ["A) 25", "B) 60", "C) 120", "D) 720"], "answer": "C"},
    {"question": "What is the value of sin(90 degrees)?", "choices": ["A) 0", "B) 0.5", "C) sqrt(2)/2", "D) 1"], "answer": "D"},
    # Physics (10)
    {"question": "What is Newton's second law?", "choices": ["A) F=ma", "B) E=mc^2", "C) F=mv", "D) P=mv"], "answer": "A"},
    {"question": "What is the speed of light in vacuum?", "choices": ["A) 3x10^6 m/s", "B) 3x10^8 m/s", "C) 3x10^10 m/s", "D) 3x10^4 m/s"], "answer": "B"},
    {"question": "What is the unit of electric current?", "choices": ["A) Volt", "B) Ohm", "C) Ampere", "D) Watt"], "answer": "C"},
    {"question": "What is Ohm's law?", "choices": ["A) V=IR", "B) P=IV", "C) E=mc^2", "D) F=ma"], "answer": "A"},
    {"question": "What is the unit of force?", "choices": ["A) Joule", "B) Watt", "C) Newton", "D) Pascal"], "answer": "C"},
    {"question": "What is the boiling point of water at sea level in Celsius?", "choices": ["A) 90", "B) 95", "C) 100", "D) 105"], "answer": "C"},
    {"question": "What is kinetic energy formula?", "choices": ["A) KE=mgh", "B) KE=mv", "C) KE=0.5mv^2", "D) KE=mv^2"], "answer": "C"},
    {"question": "What is the gravitational acceleration on Earth?", "choices": ["A) 8.9 m/s^2", "B) 9.8 m/s^2", "C) 10.8 m/s^2", "D) 11.2 m/s^2"], "answer": "B"},
    {"question": "What is the first law of thermodynamics?", "choices": ["A) Energy cannot be created or destroyed", "B) Entropy always increases", "C) Equal action and reaction", "D) Force equals mass times acceleration"], "answer": "A"},
    {"question": "What is the unit of energy?", "choices": ["A) Newton", "B) Watt", "C) Pascal", "D) Joule"], "answer": "D"},
    # Chemistry (10)
    {"question": "What is the chemical symbol for gold?", "choices": ["A) Go", "B) Gd", "C) Au", "D) Ag"], "answer": "C"},
    {"question": "What is the atomic number of carbon?", "choices": ["A) 4", "B) 6", "C) 8", "D) 12"], "answer": "B"},
    {"question": "What is the chemical formula for water?", "choices": ["A) HO", "B) H2O", "C) H2O2", "D) H3O"], "answer": "B"},
    {"question": "What is the pH of a neutral solution?", "choices": ["A) 0", "B) 5", "C) 7", "D) 14"], "answer": "C"},
    {"question": "What is the most abundant element in Earth's atmosphere?", "choices": ["A) Oxygen", "B) Carbon dioxide", "C) Argon", "D) Nitrogen"], "answer": "D"},
    {"question": "What is the chemical symbol for sodium?", "choices": ["A) So", "B) Sd", "C) Na", "D) Ni"], "answer": "C"},
    {"question": "What is Avogadro's number approximately?", "choices": ["A) 6.02x10^21", "B) 6.02x10^23", "C) 6.02x10^25", "D) 6.02x10^18"], "answer": "B"},
    {"question": "What type of bond involves sharing electrons?", "choices": ["A) Ionic bond", "B) Metallic bond", "C) Covalent bond", "D) Hydrogen bond"], "answer": "C"},
    {"question": "What is the chemical formula for table salt?", "choices": ["A) KCl", "B) NaCl", "C) CaCl2", "D) MgCl2"], "answer": "B"},
    {"question": "What is the lightest element?", "choices": ["A) Helium", "B) Lithium", "C) Hydrogen", "D) Carbon"], "answer": "C"},
    # Biology (10)
    {"question": "What is the powerhouse of the cell?", "choices": ["A) Nucleus", "B) Ribosome", "C) Mitochondria", "D) Golgi apparatus"], "answer": "C"},
    {"question": "What is the basic unit of life?", "choices": ["A) Tissue", "B) Organ", "C) Atom", "D) Cell"], "answer": "D"},
    {"question": "What molecule carries genetic information?", "choices": ["A) RNA", "B) DNA", "C) Protein", "D) Lipid"], "answer": "B"},
    {"question": "How many chromosomes do humans have?", "choices": ["A) 23", "B) 44", "C) 46", "D) 48"], "answer": "C"},
    {"question": "What process do plants use to make food?", "choices": ["A) Respiration", "B) Fermentation", "C) Photosynthesis", "D) Digestion"], "answer": "C"},
    {"question": "What is the largest organ in the human body?", "choices": ["A) Liver", "B) Brain", "C) Heart", "D) Skin"], "answer": "D"},
    {"question": "What blood type is the universal donor?", "choices": ["A) A", "B) B", "C) AB", "D) O"], "answer": "D"},
    {"question": "What is the function of red blood cells?", "choices": ["A) Fight infection", "B) Carry oxygen", "C) Clot blood", "D) Produce antibodies"], "answer": "B"},
    {"question": "What is the study of heredity called?", "choices": ["A) Ecology", "B) Genetics", "C) Taxonomy", "D) Physiology"], "answer": "B"},
    {"question": "What is the process by which cells divide?", "choices": ["A) Meiosis only", "B) Mitosis only", "C) Both mitosis and meiosis", "D) Osmosis"], "answer": "C"},
    # Computer Science (10)
    {"question": "What does CPU stand for?", "choices": ["A) Central Processing Unit", "B) Computer Processing Unit", "C) Central Program Unit", "D) Core Processing Unit"], "answer": "A"},
    {"question": "What is the binary representation of decimal 10?", "choices": ["A) 1000", "B) 1010", "C) 1100", "D) 0110"], "answer": "B"},
    {"question": "What does RAM stand for?", "choices": ["A) Read Access Memory", "B) Random Access Memory", "C) Read And Memory", "D) Random Allocation Memory"], "answer": "B"},
    {"question": "What is the time complexity of binary search?", "choices": ["A) O(n)", "B) O(n^2)", "C) O(log n)", "D) O(n log n)"], "answer": "C"},
    {"question": "What does HTML stand for?", "choices": ["A) Hyper Text Markup Language", "B) High Text Markup Language", "C) Hyper Transfer Markup Language", "D) Hyper Text Making Language"], "answer": "A"},
    {"question": "What is the base of hexadecimal number system?", "choices": ["A) 2", "B) 8", "C) 10", "D) 16"], "answer": "D"},
    {"question": "What does SQL stand for?", "choices": ["A) Structured Query Language", "B) Simple Query Language", "C) Standard Query Language", "D) Sequential Query Language"], "answer": "A"},
    {"question": "What is a compiler?", "choices": ["A) Runs programs line by line", "B) Translates high-level code to machine code", "C) Manages memory", "D) Connects to internet"], "answer": "B"},
    {"question": "What is the purpose of an operating system?", "choices": ["A) Browse internet", "B) Write code", "C) Manage hardware and software resources", "D) Store data permanently"], "answer": "C"},
    {"question": "What is machine learning?", "choices": ["A) Programming robots manually", "B) Systems that learn patterns from data", "C) Computer hardware design", "D) Network security"], "answer": "B"},
    # History (10)
    {"question": "In which year did World War II end?", "choices": ["A) 1943", "B) 1944", "C) 1945", "D) 1946"], "answer": "C"},
    {"question": "Who was the first President of the United States?", "choices": ["A) John Adams", "B) Thomas Jefferson", "C) George Washington", "D) Benjamin Franklin"], "answer": "C"},
    {"question": "In which year did the French Revolution begin?", "choices": ["A) 1776", "B) 1789", "C) 1799", "D) 1815"], "answer": "B"},
    {"question": "Who wrote the Communist Manifesto?", "choices": ["A) Lenin", "B) Stalin", "C) Marx and Engels", "D) Trotsky"], "answer": "C"},
    {"question": "In which year did India gain independence?", "choices": ["A) 1945", "B) 1946", "C) 1947", "D) 1948"], "answer": "C"},
    {"question": "What was the Cold War primarily between?", "choices": ["A) USA and China", "B) USA and USSR", "C) UK and Germany", "D) France and Russia"], "answer": "B"},
    {"question": "Who was Napoleon Bonaparte?", "choices": ["A) Russian tsar", "B) German emperor", "C) French military and political leader", "D) British general"], "answer": "C"},
    {"question": "In which year did the Berlin Wall fall?", "choices": ["A) 1987", "B) 1988", "C) 1989", "D) 1990"], "answer": "C"},
    {"question": "Which country was first to put a man on the moon?", "choices": ["A) USSR", "B) USA", "C) China", "D) UK"], "answer": "B"},
    {"question": "What was the Renaissance?", "choices": ["A) A war in Europe", "B) A cultural and intellectual movement", "C) A religious movement", "D) An economic crisis"], "answer": "B"},
    # Geography (10)
    {"question": "What is the capital of Australia?", "choices": ["A) Sydney", "B) Melbourne", "C) Canberra", "D) Brisbane"], "answer": "C"},
    {"question": "What is the longest river in the world?", "choices": ["A) Amazon", "B) Nile", "C) Yangtze", "D) Mississippi"], "answer": "B"},
    {"question": "Which is the largest ocean?", "choices": ["A) Atlantic", "B) Indian", "C) Arctic", "D) Pacific"], "answer": "D"},
    {"question": "What is the capital of Japan?", "choices": ["A) Osaka", "B) Kyoto", "C) Tokyo", "D) Hiroshima"], "answer": "C"},
    {"question": "Which country has the largest population?", "choices": ["A) China", "B) India", "C) USA", "D) Indonesia"], "answer": "B"},
    {"question": "What is the smallest country in the world?", "choices": ["A) Monaco", "B) San Marino", "C) Liechtenstein", "D) Vatican City"], "answer": "D"},
    {"question": "What is the highest mountain in the world?", "choices": ["A) K2", "B) Kangchenjunga", "C) Mount Everest", "D) Lhotse"], "answer": "C"},
    {"question": "Which continent has the most countries?", "choices": ["A) Asia", "B) Europe", "C) Africa", "D) Americas"], "answer": "C"},
    {"question": "What is the capital of Brazil?", "choices": ["A) Rio de Janeiro", "B) Sao Paulo", "C) Brasilia", "D) Salvador"], "answer": "C"},
    {"question": "Which desert is the largest hot desert?", "choices": ["A) Gobi", "B) Arabian", "C) Sahara", "D) Kalahari"], "answer": "C"},
    # Economics (10)
    {"question": "What does GDP stand for?", "choices": ["A) Gross Domestic Product", "B) General Domestic Product", "C) Gross Development Product", "D) Global Domestic Product"], "answer": "A"},
    {"question": "What is inflation?", "choices": ["A) Decrease in prices", "B) Increase in general price level over time", "C) Stable prices", "D) Government spending"], "answer": "B"},
    {"question": "What is a monopoly?", "choices": ["A) Many sellers", "B) Two sellers", "C) Single seller dominating market", "D) Government-owned market"], "answer": "C"},
    {"question": "What does fiscal policy refer to?", "choices": ["A) Central bank interest rates", "B) Government spending and taxation", "C) Exchange rate management", "D) Trade tariffs"], "answer": "B"},
    {"question": "What is opportunity cost?", "choices": ["A) Cost of production", "B) Value of next best alternative given up", "C) Total project cost", "D) Profit from decision"], "answer": "B"},
    {"question": "What is a recession?", "choices": ["A) Period of economic growth", "B) Period of stable growth", "C) Two consecutive quarters of negative GDP growth", "D) High inflation period"], "answer": "C"},
    {"question": "What is the stock market?", "choices": ["A) Where goods are traded", "B) Where currencies are exchanged", "C) Where company shares are traded", "D) Where bonds are issued"], "answer": "C"},
    {"question": "What is supply and demand?", "choices": ["A) Government price control", "B) Market forces determining price and quantity", "C) Fixed pricing system", "D) Import export balance"], "answer": "B"},
    {"question": "What is comparative advantage?", "choices": ["A) Producing more than others", "B) Lower opportunity cost in producing a good", "C) Having better technology", "D) Exporting more than importing"], "answer": "B"},
    {"question": "What is the purpose of a central bank?", "choices": ["A) Provide personal loans", "B) Control monetary policy and money supply", "C) Sell products to consumers", "D) Regulate stock markets only"], "answer": "B"},
    # Psychology (10)
    {"question": "Who is the father of psychoanalysis?", "choices": ["A) Carl Jung", "B) William James", "C) Sigmund Freud", "D) B.F. Skinner"], "answer": "C"},
    {"question": "What is classical conditioning?", "choices": ["A) Learning through rewards", "B) Learning through punishment", "C) Learning through association of stimuli", "D) Learning through observation"], "answer": "C"},
    {"question": "What does IQ stand for?", "choices": ["A) Intelligence Quotient", "B) Intellectual Quality", "C) Intelligence Quality", "D) Intellectual Quotient"], "answer": "A"},
    {"question": "What is cognitive dissonance?", "choices": ["A) Memory loss", "B) Mental discomfort from contradictory beliefs", "C) Learning disability", "D) Attention disorder"], "answer": "B"},
    {"question": "What is operant conditioning?", "choices": ["A) Learning through association", "B) Learning through observation", "C) Learning through consequences", "D) Learning through repetition"], "answer": "C"},
    {"question": "What does PTSD stand for?", "choices": ["A) Post Traumatic Stress Disorder", "B) Post Therapy Stress Disorder", "C) Primary Traumatic Stress Disorder", "D) Post Traumatic Stress Disease"], "answer": "A"},
    {"question": "What is the placebo effect?", "choices": ["A) Side effect of medication", "B) Improvement from inactive treatment due to belief", "C) Negative drug reaction", "D) Drug overdose effect"], "answer": "B"},
    {"question": "What is social learning theory associated with?", "choices": ["A) Freud", "B) Skinner", "C) Bandura", "D) Piaget"], "answer": "C"},
    {"question": "What is Maslow's highest level of needs?", "choices": ["A) Safety", "B) Love and belonging", "C) Esteem", "D) Self-actualization"], "answer": "D"},
    {"question": "What is the unconscious mind?", "choices": ["A) Conscious thoughts", "B) Memories and desires outside awareness", "C) Rational thinking", "D) Short-term memory"], "answer": "B"},
    # Literature (10)
    {"question": "Who wrote Romeo and Juliet?", "choices": ["A) Charles Dickens", "B) William Shakespeare", "C) Jane Austen", "D) Mark Twain"], "answer": "B"},
    {"question": "Who wrote 1984?", "choices": ["A) Aldous Huxley", "B) Ray Bradbury", "C) George Orwell", "D) H.G. Wells"], "answer": "C"},
    {"question": "Who wrote Pride and Prejudice?", "choices": ["A) Charlotte Bronte", "B) Emily Bronte", "C) Jane Austen", "D) George Eliot"], "answer": "C"},
    {"question": "Who wrote The Great Gatsby?", "choices": ["A) Ernest Hemingway", "B) F. Scott Fitzgerald", "C) John Steinbeck", "D) William Faulkner"], "answer": "B"},
    {"question": "Who wrote the Iliad and the Odyssey?", "choices": ["A) Virgil", "B) Plato", "C) Aristotle", "D) Homer"], "answer": "D"},
    {"question": "Who wrote Crime and Punishment?", "choices": ["A) Tolstoy", "B) Dostoevsky", "C) Chekhov", "D) Turgenev"], "answer": "B"},
    {"question": "Who wrote To Kill a Mockingbird?", "choices": ["A) Truman Capote", "B) Harper Lee", "C) John Steinbeck", "D) William Faulkner"], "answer": "B"},
    {"question": "What literary device is a comparison using like or as?", "choices": ["A) Metaphor", "B) Simile", "C) Personification", "D) Alliteration"], "answer": "B"},
    {"question": "Who wrote Don Quixote?", "choices": ["A) Lope de Vega", "B) Garcia Lorca", "C) Miguel de Cervantes", "D) Pablo Neruda"], "answer": "C"},
    {"question": "What is the first book of the Bible?", "choices": ["A) Exodus", "B) Leviticus", "C) Genesis", "D) Numbers"], "answer": "C"},
]

print(f"Loaded {len(MMLU_QUESTIONS)} MMLU questions across 10 subjects")

# ============================================================
# EVALUATION FUNCTION — MULTIPLE CHOICE
# ============================================================
def evaluate_model_mmlu(model, tokenizer, questions, device, max_new_tokens=10):
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in questions:
            prompt = (f"Question: {item['question']}\n"
                     f"Choices:\n" + "\n".join(item['choices']) +
                     f"\nAnswer with only the letter (A, B, C, or D):\n")
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=512, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, temperature=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            pred_letter = ""
            for char in prediction.upper():
                if char in ["A", "B", "C", "D"]:
                    pred_letter = char
                    break
            if pred_letter == item['answer']:
                correct += 1
    return correct / len(questions) * 100

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
            merged[key] = (w_a + alpha * delta * drop_mask.float() * rescale).half()
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
                keep_prob = (norm_mag * density / (norm_mag.mean() + 1e-8)).clamp(0, 1)
                mask = torch.bernoulli(keep_prob).bool()
            else:
                mask = torch.zeros_like(delta, dtype=torch.bool)
            kept = mask.float().mean().item()
            rescale = 1.0 / kept if kept > 0 else 1.0
            merged[key] = (w_a + alpha * delta * mask.float() * rescale).half()
        else:
            merged[key] = sd_a[key]
    return merged

def structured_dfs_merge(sd_a, sd_b, seed=42, total_layers=291):
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

def cma_es_merge(sd_a, sd_b, seed=42, total_layers=291, model_a=None,
                 tokenizer=None, device=None):
    """
    Structured DFS with CMA-ES optimization.
    Uses reduced iterations for MMLU to keep computation manageable.
    """
    np.random.seed(seed)
    layer_keys = list(sd_a.keys())
    binary_z = np.random.binomial(1, 0.5, total_layers)
    active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
    dim = len(active_keys)

    # Proxy evaluation on 20 questions for CMA-ES iterations
    proxy_questions = MMLU_QUESTIONS[:20]

    opts = cma.CMAOptions()
    opts['seed'] = seed
    opts['popsize'] = 4
    opts['maxiter'] = 5
    opts['bounds'] = [0.1, 0.9]
    opts['verbose'] = -9

    es = cma.CMAEvolutionStrategy([0.5] * dim, 0.2, opts)
    best_sd = copy.deepcopy(sd_a)
    best_acc = 0

    while not es.stop():
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
            cand_model.load_state_dict({k: v.to(device) for k, v in candidate.items()})
            acc = evaluate_model_mmlu(cand_model, tokenizer, proxy_questions, device)
            del cand_model
            torch.cuda.empty_cache()
            gc.collect()
            fitnesses.append(-acc)
            if acc > best_acc:
                best_acc = acc
                best_sd = candidate
        es.tell(solutions, fitnesses)

    return best_sd

def load_merged(merged_sd, model_a, device):
    merged = copy.deepcopy(model_a)
    merged.load_state_dict({k: v.to(device) for k, v in merged_sd.items()})
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
print(f"Model A loaded: {sum(p.numel() for p in model_a.parameters()):,} parameters")

model_b_standalone = AutoModelForCausalLM.from_pretrained(
    MODEL_B, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model B loaded: {sum(p.numel() for p in model_b_standalone.parameters()):,} parameters")

state_dict_a = {k: v.cpu().clone() for k, v in model_a.state_dict().items()}
state_dict_b = {k: v.cpu().clone() for k, v in model_b_standalone.state_dict().items()}
total_layers = len(state_dict_a)
print(f"Total parameter groups: {total_layers}")

# ============================================================
# MODEL A BASELINE
# ============================================================
if "model_a" not in checkpoint["completed_experiments"]:
    print("\nEvaluating Model A on MMLU...")
    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint["model_a"]["completed_seeds"]:
            continue
        acc = evaluate_model_mmlu(model_a, tokenizer, MMLU_QUESTIONS, device)
        checkpoint["model_a"]["runs"].append(acc)
        checkpoint["model_a"]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"Model A Run {seed_idx+1}: {acc:.1f}% [SAVED]")
    checkpoint["completed_experiments"].append("model_a")
    save_checkpoint(checkpoint)
else:
    print(f"[SKIP] Model A: {np.mean(checkpoint['model_a']['runs']):.1f}%")

# ============================================================
# MODEL B BASELINE — Integrated from eval_modelb_mmlu.py
# ============================================================
if "model_b" not in checkpoint["completed_experiments"]:
    print("\nEvaluating Model B on MMLU...")
    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint["model_b"]["completed_seeds"]:
            continue
        acc = evaluate_model_mmlu(model_b_standalone, tokenizer, MMLU_QUESTIONS, device)
        checkpoint["model_b"]["runs"].append(acc)
        checkpoint["model_b"]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"Model B Run {seed_idx+1}: {acc:.1f}% [SAVED]")
    checkpoint["completed_experiments"].append("model_b")
    save_checkpoint(checkpoint)
else:
    print(f"[SKIP] Model B: {np.mean(checkpoint['model_b']['runs']):.1f}%")

# Free Model B from GPU after evaluation
del model_b_standalone
torch.cuda.empty_cache()
gc.collect()
print("Model B removed from GPU to free VRAM.")

# ============================================================
# RUN ALL MERGING METHODS
# ============================================================
method_configs = [
    ("ps_merging", lambda seed: ps_merge(state_dict_a, state_dict_b, alpha=0.5)),
    ("ties", lambda seed: ties_merge(state_dict_a, state_dict_b, seed=seed)),
    ("dare", lambda seed: dare_merge(state_dict_a, state_dict_b, seed=seed)),
    ("della", lambda seed: della_merge(state_dict_a, state_dict_b, seed=seed)),
    ("structured_dfs", lambda seed: structured_dfs_merge(
        state_dict_a, state_dict_b, seed=seed, total_layers=total_layers)),
    ("cma_es", lambda seed: cma_es_merge(
        state_dict_a, state_dict_b, seed=seed, total_layers=total_layers,
        model_a=model_a, tokenizer=tokenizer, device=device)),
]

for method_name, merge_fn in method_configs:
    if method_name in checkpoint["completed_experiments"]:
        mean = np.mean(checkpoint[method_name]["runs"])
        std = np.std(checkpoint[method_name]["runs"])
        print(f"\n[SKIP] {method_name}: {mean:.1f}% +/- {std:.1f}%")
        continue

    print(f"\n{'='*65}")
    print(f"{method_name.upper()} ON MMLU ({NUM_SEEDS} runs)")
    print(f"{'='*65}")

    for seed_idx, seed in enumerate(SEEDS):
        if seed in checkpoint[method_name]["completed_seeds"]:
            print(f"[SKIP] {method_name} Run {seed_idx+1} done.")
            continue

        print(f"\nRun {seed_idx+1}/{NUM_SEEDS} (seed={seed})...")
        merged_sd = merge_fn(seed)
        merged_model = load_merged(merged_sd, model_a, device)
        acc = evaluate_model_mmlu(merged_model, tokenizer, MMLU_QUESTIONS, device)
        del merged_model
        torch.cuda.empty_cache()
        gc.collect()

        checkpoint[method_name]["runs"].append(acc)
        checkpoint[method_name]["completed_seeds"].append(seed)
        save_checkpoint(checkpoint)
        print(f"{method_name} Run {seed_idx+1}: {acc:.1f}% [SAVED]")

    checkpoint["completed_experiments"].append(method_name)
    save_checkpoint(checkpoint)
    mean = np.mean(checkpoint[method_name]["runs"])
    std = np.std(checkpoint[method_name]["runs"])
    print(f"{method_name}: {mean:.1f}% +/- {std:.1f}%")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 65)
print("PHASE 4 COMPLETE — MMLU RESULTS")
print("=" * 65)
print(f"{'Method':<35} {'Mean':>10} {'Std':>8}")
print("-" * 65)

all_methods = ["model_a", "model_b", "ps_merging", "ties", "dare",
               "della", "structured_dfs", "cma_es"]
for method in all_methods:
    if checkpoint[method]["runs"]:
        mean = np.mean(checkpoint[method]["runs"])
        std = np.std(checkpoint[method]["runs"])
        print(f"{method:<35} {mean:>9.1f}% {std:>7.1f}%")

print("=" * 65)

# Save results
results = {
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "benchmark": "MMLU",
    "num_questions": len(MMLU_QUESTIONS),
}
for method in all_methods:
    if checkpoint[method]["runs"]:
        results[method] = {
            "mean": float(np.mean(checkpoint[method]["runs"])),
            "std": float(np.std(checkpoint[method]["runs"])),
            "all_runs": checkpoint[method]["runs"]
        }

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")
print("\nPHASE 4 COMPLETE")
