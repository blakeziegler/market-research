from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = "Dev9124/qwen3-finance-model"
dapt_model = "blakeziegler/qwen3-4b-dapt-v1"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
base_model.eval()

dapt_model = AutoModelForCausalLM.from_pretrained(dapt_model, trust_remote_code=True, device_map="auto")
dapt_model.eval()

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
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Prompt:", prompt)
print("\nBase Model Response:\n", generate_response(base_model, prompt))
print("\nDAPT Model Response:\n", generate_response(dapt_model, prompt))



