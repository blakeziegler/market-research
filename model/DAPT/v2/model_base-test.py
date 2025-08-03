from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# GGUF model info
model_name = "tarun7r/Finance-Llama-8B-q4_k_m-GGUF"
model_file = "Finance-Llama-8B-GGUF-q4_K_M.gguf"  # Double-check this name matches exactly on HF

# Download model to local directory if not already
model_path = hf_hub_download(
    repo_id=model_name,
    filename=model_file,
    local_dir="./models"
)

# Initialize LLaMA GGUF model
llm = Llama(
    model_path=model_path,
    n_ctx=4096,         # You can increase this to 32768+ if needed
    n_threads=8,        # Adjust based on CPU
    n_gpu_layers=-1,    # Offload to GPU if possible
    verbose=False
)

# Prompt
valuation_prompt = (
    "You're an equity specialist evaluating the intrinsic value of a software company. "
    "The company has $1.2 billion in trailing twelve-month revenue, a net income margin of 18%, and a free cash flow margin of 22%. "
    "Free cash flow is expected to grow at 10% annually over the next five years. Comparable firms in the sector trade at a P/E ratio of 30. "
    "The company holds $150 million in cash and has $100 million in debt. The appropriate discount rate for this business is 9%. "
    "Using only this information, forecast the company's free cash flows for the next five years and discount them to present value. "
    "Then calculate the terminal value using a P/E-based approach on year 5 earnings. Add the present value of cash flows and terminal value to arrive at the total firm value. "
    "Subtract net debt to get the equity value. Conclude by interpreting whether this business appears undervalued or overvalued if it currently trades at a $6.8 billion market cap, "
    "and briefly discuss how sensitive the valuation is to assumptions like P/E ratio, FCF growth, and profit margins.\n\n"
    "Answer:"
)

# Use optional system/instruction wrapper for better formatting
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Ensure your response is formatted in a readable and professional manner.
### Instruction:
You are a highly knowledgeable finance chatbot. Your purpose is to provide accurate, insightful, and actionable financial advice.
### Input:
{valuation_prompt}
### Response:
"""

formatted_prompt = prompt_template.format(valuation_prompt=valuation_prompt)

# Run inference
response = llm(
    formatted_prompt,
    max_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    echo=False,
    stop=["###"]
)

# Extract text
answer = response["choices"][0]["text"].strip()

# Print results
print("\n📊 Prompt:\n", valuation_prompt)
print("\n🚀 GGUF Model Response:\n", answer)