from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "tarun7r/Finance-Llama-8B"
adapter_model = "checkpoint-0000"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)

model = model.merge_and_unload()

model.save_pretrained("llama_8b_dapt-600k_v1")
tokenizer.save_pretrained("llama_8b_dapt-600k_v1")