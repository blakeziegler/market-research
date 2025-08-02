from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load merged model directly from HF
model_id = "blakeziegler/qwen3-4b-dapt-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
model.eval()

# Prompt
prompt = (
    "You're a buy-side equity analyst evaluating a high-growth SaaS company with slowing top-line expansion.\n\n"
    "Walk through a full discounted cash flow (DCF) valuation model in technical detail. Forecast free cash flows over a 5-year explicit period, "
    "taking into account margin compression, capex ramp, and working capital drag. Then, justify your terminal value approach — Gordon Growth vs. Exit Multiple. "
    "Explain how you'd determine the appropriate WACC using CAPM, adjusting for beta instability in emerging tech. Finally, describe how you'd stress test the valuation, "
    "and sanity-check it using trading comps and precedent transactions.\n\n"
    "Answer:"
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode
text = tokenizer.decode(output[0], skip_special_tokens=True)

# Strip the prompt
response = text[len(prompt):].strip()

print("\n📊 Prompt:\n", prompt)
print("\n🚀 DAPT Model Response:\n", response)