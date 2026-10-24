print("Step 1: Basic imports...")
import os
import torch
import numpy as np
import json
import gc
import re
print("Step 1 OK")

print("Step 2: CMA...")
import cma as cma_lib
print("Step 2 OK")

print("Step 3: Datasets...")
from datasets import load_dataset
print("Step 3 OK")

print("Step 4: Transformers...")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Step 4 OK")

print("Step 5: Matplotlib...")
import matplotlib.pyplot as plt
print("Step 5 OK")

print("Step 6: GPU check...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("ALL IMPORTS OK")
