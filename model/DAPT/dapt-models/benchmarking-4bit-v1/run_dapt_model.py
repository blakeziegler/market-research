# run_dapt_model.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Config
input_csv = "benchmark_v1.csv"
output_csv = "results.csv"

dapt_model_id = "blakeziegler/qwen3-4b-dapt-v1"
max_new_tokens = 512

print("🔄 Loading tokenizer and DAPT model...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(dapt_model_id, trust_remote_code=True)

# Set token IDs
pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

# Generation config
generation_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "pad_token_id": pad_token_id,
    "eos_token_id": eos_token_id,
    "max_new_tokens": max_new_tokens
}

# Load DAPT model
dapt_model = AutoModelForCausalLM.from_pretrained(dapt_model_id, trust_remote_code=True, device_map="auto")
dapt_model.eval()

# Load CSV
df = pd.read_csv(input_csv)

# Ensure dapt_output column exists
if "dapt_output" not in df.columns:
    df["dapt_output"] = ""

# Generation function
def generate(model, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, **generation_kwargs)
        text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if text.startswith(prompt.strip()):
            text = text[len(prompt.strip()):].strip()
        return text
    except Exception as e:
        print(f"⚠️ Generation error: {e}")
        return "[ERROR]"

# Inference loop
print("🚀 Generating DAPT model outputs...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"]
    tqdm.write(f"\n📝 Prompt {idx+1}: {prompt[:100]}...")

    if not row["dapt_output"] or row["dapt_output"] == "[ERROR]":
        tqdm.write("🔸 Generating DAPT model response...")
        df.at[idx, "dapt_output"] = generate(dapt_model, prompt)

    # Optional: save progress every 5 rows
    if idx % 5 == 0:
        df.to_csv(output_csv, index=False)

# Final save
df.to_csv(output_csv, index=False)
print(f"\n✅ DAPT model responses saved to {output_csv}")