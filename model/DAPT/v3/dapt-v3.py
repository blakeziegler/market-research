# run_dapt_model.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import logging
import csv

# -------------------- Config --------------------
INPUT_CSV = "benchmark_v2.csv"
OUTPUT_CSV = "results_dapt_v3.csv"
MODEL_ID = "blakeziegler/qwen3_4b_dapt-700k_v3"
OUTPUT_COLUMN = "dapt_output"
MAX_NEW_TOKENS = 1024

# ------------------ Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------ Load Model ------------------
logging.info("Loading tokenizer and DAPT model...")

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
        "<|im_start|>system\n"
        "You are a financial analyst at a professional investment firm. "
        "Your role is to evaluate the financial data below strictly using sound reasoning and accepted valuation principles."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "TASK:\n"
        f"{prompt.strip()}\n\n"
        "CONSTRAINTS:\n"
        "- Use only the data provided in the prompt. Do not make up or assume any additional numbers.\n"
        "- Avoid giving user instructions (e.g., 'write a paragraph' or 'submit your answer').\n"
        "- Focus on delivering clear financial analysis and valuation insights.\n"
        "- If calculations are required, show them step-by-step and explain their meaning.\n"
        "- Answer as if you were preparing an internal investment memo, not a student submission.\n"
        "/no_think"
        "<|im_end|>\n"
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
logging.info("Starting inference...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        continue

    if not row[OUTPUT_COLUMN] or row[OUTPUT_COLUMN] == "[ERROR]":
        logging.info(f"[{idx+1}] Generating DAPT output...")
        result = generate(prompt)
        df.at[idx, OUTPUT_COLUMN] = result

    if idx % 5 == 0:
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

# ------------------ Final Save ------------------
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"DAPT model responses saved to {OUTPUT_CSV}")