from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Model ID
model_id = "blakeziegler/llama_8b_dapt-600k_v1"

# Define 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Prompt template
finance_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:
"""

# Prompt components
system_message = (
    "You are a highly knowledgeable finance chatbot. Your purpose is to provide accurate, insightful, and actionable financial advice. "
    "Ensure your response is formatted in a readable and professional manner."
)
user_question = (
    "You're an equity specialist evaluating the intrinsic value of a software company. "
    "The company has $1.2 billion in trailing twelve-month revenue, a net income margin of 18%, and a free cash flow margin of 22%. "
    "Free cash flow is expected to grow at 10% annually over the next five years. Comparable firms in the sector trade at a P/E ratio of 30. "
    "The company holds $150 million in cash and has $100 million in debt. The appropriate discount rate for this business is 9%. "
    "Using only this information, forecast the company's free cash flows for the next five years and discount them to present value. "
    "Then calculate the terminal value using a P/E-based approach on year 5 earnings. Add the present value of cash flows and terminal value to arrive at the total firm value. "
    "Subtract net debt to get the equity value. Conclude by interpreting whether this business appears undervalued or overvalued if it currently trades at a $6.8 billion market cap, "
    "and briefly discuss how sensitive the valuation is to assumptions like P/E ratio, FCF growth, and profit margins."
)

# Format prompt
prompt = finance_prompt_template.format(
    instruction=system_message,
    input=user_question
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
response = generated_text[len(prompt):].strip()

# Output
print("\n📊 Prompt:\n", prompt)
print("\n🚀 DAPT Model Response:\n", response)