# === MERGE SCRIPT ===
# File: merge_fp16_model.py

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Dev9124/qwen3-finance-model"
adapter_path = "checkpoint-XXXX"  # <- replace with final checkpoint directory

# Load base model in FP16
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",  # or torch.float16 explicitly
    device_map="auto",
    trust_remote_code=True,
)

# Merge LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

# Save final merged model in FP16
output_path = "qwen3_4b_dapt-700k_v3"
model.save_pretrained(output_path, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.save_pretrained(output_path)