# run_models.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

input_csv = "benchmark_v1.csv"
output_csv = "results.csv"

base_model_id = "Dev9124/qwen3-finance-model"
dapt_model_id = "blakeziegler/qwen3-4b-dapt-v1"

max_new_tokens = 512
generation_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "pad_token_id": 151643,  # For Qwen tokenizer
    "eos_token_id": 151643
}

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, device_map="auto")
base_model.eval()

dapt_model = AutoModelForCausalLM.from_pretrained(dapt_model_id, trust_remote_code=True, device_map="auto")
dapt_model.eval()

df = pd.read_csv(input_csv)

if "base_output" not in df.columns:
    df["base_output"] = ""
if "dapt_output" not in df.columns:
    df["dapt_output"] = ""

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

print("Generating model outputs for prompts...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"]

    base_output = generate(base_model, prompt)
    dapt_output = generate(dapt_model, prompt)

    df.at[idx, "base_output"] = base_output
    df.at[idx, "dapt_output"] = dapt_output

df.to_csv(output_csv, index=False)
print(f"Saved model outputs to {output_csv}")