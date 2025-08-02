from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load merged model directly from Hugging Face
model_id = "Dev9124/qwen3-4b-dapt-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
model.eval()

# Refined prompt with realistic valuation metrics
prompt = (
    "You're an equity analyst evaluating the intrinsic value of a software company. "
    "The company has $1.2 billion in trailing twelve-month revenue, a net income margin of 18%, and a free cash flow margin of 22%. "
    "Free cash flow is expected to grow at 10% annually over the next five years. Comparable firms in the sector trade at a P/E ratio of 30. "
    "The company holds $150 million in cash and has $100 million in debt. The appropriate discount rate for this business is 9%. "
    "Using only this information, forecast the company's free cash flows for the next five years and discount them to present value. "
    "Then calculate the terminal value using a P/E-based approach on year 5 earnings. Add the present value of cash flows and terminal value to arrive at the total firm value. "
    "Subtract net debt to get the equity value. Conclude by interpreting whether this business appears undervalued or overvalued if it currently trades at a $6.8 billion market cap, "
    "and briefly discuss how sensitive the valuation is to assumptions like P/E ratio, FCF growth, and profit margins.\n\n"
    "Answer:"
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode
text = tokenizer.decode(output[0], skip_special_tokens=True)

# Strip the prompt
response = text[len(prompt):].strip()

print("\n📊 Prompt:\n", prompt)
print("\n🚀 DAPT Model Response:\n", response)