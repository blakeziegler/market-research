import pandas as pd
import requests
from tqdm import tqdm
import logging
import csv

# -------------------- Config --------------------
INPUT_CSV = "benchmark_v2.csv"
OUTPUT_CSV = "results_base_v2.csv"
MODEL_NAME = "martain7r/finance-llama-8b:q4_k_m"
OUTPUT_COLUMN = "base_output"
MAX_TOKENS = 1024
OLLAMA_URL = "http://localhost:11434/api/generate"

# ------------------ Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------ Load Data -------------------
df = pd.read_csv(INPUT_CSV, usecols=["prompt"])
df[OUTPUT_COLUMN] = ""

# ------------------ Prompt Formatting -------------------
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

# ------------------ Inference -------------------
def generate(prompt: str) -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "num_predict": MAX_TOKENS,
            "stop": ["###"]
        })
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        logging.error(f"❌ Generation failed: {e}")
        return "[ERROR]"

# ------------------ Run Inference -------------------
logging.info("🚀 Starting Ollama model inference...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        continue

    if not row[OUTPUT_COLUMN] or row[OUTPUT_COLUMN] == "[ERROR]":
        logging.info(f"📝 [{idx+1}] Generating Ollama output...")
        formatted_prompt = format_prompt(prompt)
        result = generate(formatted_prompt)
        df.at[idx, OUTPUT_COLUMN] = result

    # Periodic save
    if idx % 5 == 0:
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

# ------------------ Final Save ------------------
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"✅ Ollama model responses saved to {OUTPUT_CSV}")