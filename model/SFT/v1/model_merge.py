from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Paths ===
base_model_name = "blakeziegler/qwen3_4b_dapt-700k_v3"              # Your DAPT model
lora_adapter_path = "./qwen3_4b_dapt-700k_SFT-260_v1"               # LoRA adapter from SFT
output_dir = "./qwen3_4b_dapt-700k_SFT-260_merged"                  # Final merged model output

# === Load Base Model ===
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

# === Load LoRA Adapter ===
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# === Merge LoRA Weights ===
merged_model = model.merge_and_unload()

# === Save Merged Model ===
merged_model.save_pretrained(output_dir)

# === Save Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"✅ Merged model saved to: {output_dir}")