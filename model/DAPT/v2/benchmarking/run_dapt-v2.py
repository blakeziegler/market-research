# run_dapt_model.py

import pandas as pd
import torch
import logging
import csv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------- Config --------------------
INPUT_CSV = "benchmark_v2.csv"
OUTPUT_CSV = "results_dapt_v2.csv"
MODEL_ID = "blakeziegler/llama_8b_dapt-600k_v1"
OUTPUT_COLUMN = "dapt_output"
MAX_NEW_TOKENS = 1024
N_CTX = 4096
RESERVED_INPUT_TOKENS = N_CTX - MAX_NEW_TOKENS

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------- Load Tokenizer & Model --------------------
logging.info("🔄 Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

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

# -------------------- Load Data --------------------
df = pd.read_csv(INPUT_CSV, usecols=["prompt"])
if OUTPUT_COLUMN not in df.columns:
    df[OUTPUT_COLUMN] = ""

# -------------------- Prompt Formatter --------------------
def format_prompt(prompt: str) -> str:
    return (
        "You are a financial analyst at a professional investment firm. "
        "Your role is to evaluate the financial data below strictly using sound reasoning and accepted valuation principles.\n\n"
        "TASK:\n"
        f"{prompt.strip()}\n\n"
        "CONSTRAINTS:\n"
        "- Use only the data provided in the prompt. Do not make up or assume any additional numbers.\n"
        "- Avoid giving user instructions (e.g., 'write a paragraph' or 'submit your answer').\n"
        "- Focus on delivering clear financial analysis and valuation insights.\n"
        "- If calculations are required, show them step-by-step and explain their meaning.\n"
        "- Answer as if you were preparing an internal investment memo, not a student submission."
    )

# -------------------- Inference Function --------------------
def generate_response(prompt: str) -> str:
    try:
        formatted = format_prompt(prompt)
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=RESERVED_INPUT_TOKENS
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
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

        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if not response:
            logging.warning("⚠️ Empty response generated.")
        return response

    except Exception as e:
        logging.error(f"❌ Generation failed: {e}")
        return "[ERROR]"

# -------------------- Main Inference Loop --------------------
logging.info("🚀 Starting inference...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        continue

    if not row[OUTPUT_COLUMN] or row[OUTPUT_COLUMN] == "[ERROR]":
        logging.info(f"📝 Generating output for row {idx+1}...")
        result = generate_response(prompt)
        df.at[idx, OUTPUT_COLUMN] = result

    if idx % 5 == 0:
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

# -------------------- Final Save --------------------
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"✅ Finished. Saved results to {OUTPUT_CSV}")