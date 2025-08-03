# run_dapt_model.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm
import logging
import csv

# -------------------- Config --------------------
INPUT_CSV = "benchmark_v2.csv"
OUTPUT_CSV = "results_dapt_v2.csv"
MODEL_ID = "blakeziegler/llama_8b_dapt-600k_v1"
OUTPUT_COLUMN = "dapt_output"
MAX_NEW_TOKENS = 1024
N_CTX = 4096  # Match GGUF context window

# ------------------ Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------ Load Model ------------------
logging.info("🔄 Loading tokenizer and DAPT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

# Quantization config (4-bit for efficiency)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    trust_remote_code=True,
    device_map="auto"
)
model.eval()

# ------------------ Load Data -------------------
df = pd.read_csv(INPUT_CSV, usecols=['prompt'])
df[OUTPUT_COLUMN] = ""

# ------------------ Prompt Formatting -------------------
def format_prompt(prompt: str) -> str:
    return f"You are a finance professional tasked with answering this question:\n{prompt.strip()}\n\n Use ONLY the data provided in the question to answer the question. Do not make up any data."

# ------------------ Generate --------------------
def generate(prompt):
    try:
        full_prompt = format_prompt(prompt)
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=N_CTX).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded[len(full_prompt):].strip()
        return response

    except Exception as e:
        logging.error(f"❌ Generation failed: {e}")
        return "[ERROR]"

# ------------------ Inference -------------------
logging.info("🚀 Starting DAPT model inference...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        continue

    if not row[OUTPUT_COLUMN] or row[OUTPUT_COLUMN] == "[ERROR]":
        logging.info(f"📝 [{idx+1}] Generating DAPT output...")
        result = generate(prompt)
        df.at[idx, OUTPUT_COLUMN] = result

    if idx % 5 == 0:
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

# ------------------ Final Save ------------------
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"✅ DAPT model responses saved to {OUTPUT_CSV}")