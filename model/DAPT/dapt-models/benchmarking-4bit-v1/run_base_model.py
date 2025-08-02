# run_base_model.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import logging
import csv

# -------------------- Config --------------------
INPUT_CSV = "benchmark_v1.csv"
OUTPUT_CSV = "results_base.csv"
MODEL_ID = "Dev9124/qwen3-finance-model"
OUTPUT_COLUMN = "base_output"
MAX_NEW_TOKENS = 1024

# ------------------ Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------ Load Model ------------------
logging.info("🔄 Loading tokenizer and base model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

generation_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "pad_token_id": pad_token_id,
    "eos_token_id": eos_token_id,
    "max_new_tokens": MAX_NEW_TOKENS,
    "enable_thinking": False
}

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto")
model.eval()

# ------------------ Load Data -------------------
# Read only the prompt column to avoid unnamed columns
df = pd.read_csv(INPUT_CSV, usecols=['prompt'])

# Add the output column
df[OUTPUT_COLUMN] = ""

# ------------------ Generate --------------------
def format_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou are a financial analyst. Output ONLY your final answer.<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def generate(prompt):
    try:
        inputs = tokenizer(format_prompt(prompt), return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if text.startswith(prompt.strip()):
            text = text[len(prompt.strip()):].strip()
        return text
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return "[ERROR]"

# ------------------ Inference -------------------
logging.info("🚀 Starting inference...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        continue

    if not row[OUTPUT_COLUMN] or row[OUTPUT_COLUMN] == "[ERROR]":
        logging.info(f"📝 [{idx+1}] Generating base output...")
        result = generate(prompt)
        df.at[idx, OUTPUT_COLUMN] = result

    if idx % 5 == 0:
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

# ------------------ Final Save ------------------
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"✅ Base model responses saved to {OUTPUT_CSV}")