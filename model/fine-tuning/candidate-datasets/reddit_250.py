from datasets import load_dataset
from collections import defaultdict

ds = load_dataset("winddude/reddit_finance_43_250k")

input = "selftext"
output = "body"

# Target metrics
metrics = [
    "P/E", "PEG", "EV/EBITDA", "book value", "valuation", "beta", "stock", "moving average", "Analyst",
    "EPS", "ROE", "profit", "loss", "income", "dividend", "retained earnings", "margin",
    "revenue", "YoY", "QoQ", "sector", "inventory", "goodwill", "depreciation", "amortization",
    "liabilities", "receivables", "current ratio", "debt", "D/E", "debt to equity", "cash flow", "FCF", "investments"
]
metrics_lower = [m.lower() for m in metrics]

# Print all unique subreddits in the dataset
subreddits = set()
for example in ds["train"]:
    if "subreddit" in example:
        subreddits.add(example["subreddit"])

print("\nUnique Subreddits in Dataset:")
for sr in sorted(subreddits):
    print("-", sr)

target_subreddits = {
    "ValueInvesting", "investing", "stocks", "options", "StockMarket", 
    "ETFs", "AskEconomics", "Money", "financialindependence",
    "UKInvesting", "CanadianInvestor", "economy", "UKPersonalFinance",
    "dividends", "finance", "pennystocks", "algotrading", "Trading",
    "wallstreetbets", "Economics", "Forex"
}

# Count metrics
metric_counts = defaultdict(int)

for example in ds["train"]:
    if example.get("subreddit") not in target_subreddits:
        continue

    combined_text = f"{example.get(input, '')} {example.get(output, '')}".lower()
    for metric in metrics_lower:
        if metric in combined_text:
            metric_counts[metric] += 1

print("\nMetric Occurrences in Dataset:\n")
for metric in metrics_lower:
    print(f"{metric:16}: {metric_counts[metric]}")

# keyword examples
keyword = "ev/ebitda"

print(f"\n3 Examples Containing '{keyword}':\n")
examples = []

for example in ds["train"]:
    if example.get("subreddit") not in target_subreddits:
        continue

    combined_text = f"{example.get(input, '')} {example.get(output, '')}".lower()
    if keyword in combined_text:
        examples.append(example)
    if len(examples) >= 3:
        break

for i, ex in enumerate(examples, 1):
    print(f"\n--- Example {i} ---")
    print("Instruction:\n", ex.get(input, ''))
    print("\nOutput:\n", ex.get(output, ''))


# Data to be used:
# 300 p/e
# 150 peg
# 100 ev/ebitda
# 85 book value
# 200 beta
# 200 moving average
# 150 roe
# 100 dividend
# 20 retained earnings
# 200 margin
# 50 yoy
# 20 qoq
# 50 sector
# 50 inventory
# 50 goodwill
# 50 depreciation
# 20 amortization
# 15 receivables
# 50 liabilities
# 20 current ratio
# 10 debt
# 30 d/e
# 10 debt to equity
# 50 cash flow
# 30 fcf
