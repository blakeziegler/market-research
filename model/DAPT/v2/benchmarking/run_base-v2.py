# run_base_model.py

import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from tqdm import tqdm
import logging
import csv
import os

# -------------------- Config --------------------
INPUT_CSV = "benchmark_v2.csv"
OUTPUT_CSV = "results_base_v2.csv"
MODEL_REPO = "tarun7r/Finance-Llama-8B-q4_k_m-GGUF"
MODEL_FILE = "Finance-Llama-8B-GGUF-q4_K_M.gguf"
OUTPUT_COLUMN = "base_output"
MAX_NEW_TOKENS = 1024
N_CTX = 4096
N_GPU_LAYERS = -1  # Set to -1 for full GPU offload (if supported)
N_THREADS = 8

# ------------------ Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------ Load Model ------------------
logging.info("🔄 Downloading and loading GGUF model...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    local_dir="./models",
    local_dir_use_symlinks=False
)

llm = Llama(
    model_path=model_path,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    f16_kv=True,
    verbose=False
)

# ------------------ Load Data -------------------
df = pd.read_csv(INPUT_CSV, usecols=["prompt"])
df[OUTPUT_COLUMN] = ""

# ------------------ Prompt Formatting -------------------
def format_prompt(prompt: str) -> str:
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n"
        "You are a highly knowledgeable finance chatbot. Your purpose is to provide accurate, insightful, and actionable financial advice.\n"
        f"### Input:\n{prompt.strip()}\n"
        "### Response:\n"
    )

# ------------------ Inference -------------------
def generate(prompt: str) -> str:
    try:
        full_prompt = format_prompt(prompt)
        output = llm(
            full_prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            echo=False,
            stop=["###"]
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        logging.error(f"❌ Generation failed: {e}")
        return "[ERROR]"

# ------------------ Run Inference -------------------
logging.info("🚀 Starting GGUF model inference...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        continue

    if not row[OUTPUT_COLUMN] or row[OUTPUT_COLUMN] == "[ERROR]":
        logging.info(f"📝 [{idx+1}] Generating GGUF base output...")
        result = generate(prompt)
        df.at[idx, OUTPUT_COLUMN] = result

    # Periodic save
    if idx % 5 == 0:
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

# ------------------ Final Save ------------------
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
logging.info(f"✅ GGUF model responses saved to {OUTPUT_CSV}")