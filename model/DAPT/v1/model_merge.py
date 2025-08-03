from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Dev9124/qwen3-finance-model"
adapter_model = "checkpoint-5604"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)

model = model.merge_and_unload()

model.save_pretrained("qwen3_4b_dapt_v1")
tokenizer.save_pretrained("qwen3_4b_dapt_v1")