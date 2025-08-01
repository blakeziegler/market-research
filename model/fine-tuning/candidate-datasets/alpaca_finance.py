from datasets import load_dataset
from collections import defaultdict

ds = load_dataset("gbharti/finance-alpaca")

metrics = [
    "P/E", "PEG", "EV/EBITDA", "book value", "valuation", "beta", "stock", "moving average", "Analyst",
    "EPS", "ROE", "profit", "loss", "income", "dividend", "retained earnings", "margin",
    "revenue", "YoY", "QoQ", "sector", "inventory", "goodwill", "depreciation", "amortization",
    "liabilities", "receivables", "current ratio", "debt", "D/E", "debt to equity", "cash flow", "FCF", "investments"
]

metrics_lower = [m.lower() for m in metrics]

# count storage
metric_counts = defaultdict(int)

# Loop over dataset
for example in ds["train"]:
    combined_text = f"{example['instruction']} {example['output']}".lower()
    for metric in metrics_lower:
        if metric in combined_text:
            metric_counts[metric] += 1

print("Metric Occurrences in Dataset:\n")
for metric in metrics_lower:
    print(f"{metric:12}: {metric_counts[metric]}")

# ---------------------------
# Change this keyword to get examples for another metric
# ---------------------------
keyword = "d/e"

print(f"\n3 Examples Containing '{keyword}':\n")
examples = []

for example in ds["train"]:
    combined_text = f"{example['instruction']} {example['output']}".lower()
    if keyword in combined_text:
        examples.append(example)
    if len(examples) >= 3:
        break

for i, ex in enumerate(examples, 1):
    print(f"\n--- Example {i} ---")
    print("Instruction:\n", ex["instruction"])
    print("\nOutput:\n", ex["output"])