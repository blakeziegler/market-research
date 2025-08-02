from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Updated model paths
base_model_id = "Dev9124/qwen3-finance-model"
dapt_model_id = "blakeziegler/qwen3-4b-dapt-v1"

# Use the DAPT tokenizer for consistency
tokenizer = AutoTokenizer.from_pretrained(dapt_model_id, trust_remote_code=True)

# Load models
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, device_map="auto")
dapt_model = AutoModelForCausalLM.from_pretrained(dapt_model_id, trust_remote_code=True, device_map="auto")

base_model.eval()
dapt_model.eval()

# Handle token IDs safely
eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("<|endoftext|>")
pad_token_id = tokenizer.pad_token_id or eos_token_id

prompt = (
    "Assume you're a buy-side equity analyst evaluating a high-growth SaaS company with slowing top-line expansion. "
    "Walk through a full DCF valuation model in technical detail. Begin by forecasting free cash flows over a 5-year explicit period, "
    "taking into account margin compression, capex ramp, and working capital drag. Then justify your terminal value approach — choose between "
    "Gordon Growth or Exit Multiple and explain why. Describe how you'd select the appropriate WACC for discounting, including how you'd derive "
    "the cost of equity using CAPM with adjustments for beta instability in emerging tech. Finally, explain how you'd stress test the valuation "
    "and sanity check it using trading comps and precedent transactions."
)

def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🔍 Raw output:", repr(decoded))  # Optional debug
    return decoded

print("\n📊 Prompt:\n", prompt)
print("\n🧠 Base Model Response:\n", generate_response(base_model, prompt))
print("\n🚀 DAPT Model Response:\n", generate_response(dapt_model, prompt))