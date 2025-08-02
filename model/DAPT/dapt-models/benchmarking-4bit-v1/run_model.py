# run_models.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Config
input_csv = "benchmark_v1.csv"
output_csv = "results.csv"

base_model_id = "Dev9124/qwen3-finance-model"
dapt_model_id = "blakeziegler/qwen3-4b-dapt-v1"

max_new_tokens = 512

print("🔄 Loading tokenizer and models...")

# Load tokenizer with trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# Use safe defaults for EOS and PAD if available
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
    "max_new_tokens": max_new_tokens
}

# Load models
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, device_map="auto")
base_model.eval()

dapt_model = AutoModelForCausalLM.from_pretrained(dapt_model_id, trust_remote_code=True, device_map="auto")
dapt_model.eval()

# Load CSV
df = pd.read_csv(input_csv)

# Initialize output columns if missing
for col in ["base_output", "dapt_output"]:
    if col not in df.columns:
        df[col] = ""

# Generate function
def generate(model, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, **generation_kwargs)
        text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Optional: remove the prompt from the output if echoed
        if text.startswith(prompt.strip()):
            text = text[len(prompt.strip()):].strip()
        return text
    except Exception as e:
        print(f"⚠️ Error during generation: {e}")
        return "[ERROR]"

# Inference loop
print("🚀 Generating model outputs...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"]
    tqdm.write(f"\n📝 Prompt {idx+1}: {prompt[:100]}...")

    if not row["base_output"] or row["base_output"] == "[ERROR]":
        tqdm.write("🔹 Generating base model response...")
        df.at[idx, "base_output"] = generate(base_model, prompt)

    if not row["dapt_output"] or row["dapt_output"] == "[ERROR]":
        tqdm.write("🔸 Generating DAPT model response...")
        df.at[idx, "dapt_output"] = generate(dapt_model, prompt)

    # Save progress every N rows (optional)
    if idx % 5 == 0:
        df.to_csv(output_csv, index=False)

# Final save
df.to_csv(output_csv, index=False)
print(f"\n✅ All outputs saved to {output_csv}")