from datasets import load_dataset
from collections import defaultdict

ds = load_dataset("Josephgflowers/Finance-Instruct-500k")

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
    combined_text = f"{example['user']} {example['assistant']}".lower()
    for metric in metrics_lower:
        if metric in combined_text:
            metric_counts[metric] += 1

print("Metric Occurrences in Dataset:\n")
for metric in metrics_lower:
    print(f"{metric:12}: {metric_counts[metric]}")



# p/e         : 393
# peg         : 784
# ev/ebitda   : 5
# book value  : 258
# valuation   : 5310
# beta        : 857
# stock       : 28075
# moving average: 260
# analyst     : 3804
# eps         : 11665
# roe         : 3930
# profit      : 24579
# loss        : 18855
# income      : 18968
# dividend    : 5295
# retained earnings: 357
# margin      : 6123
# revenue     : 21477
# yoy         : 291
# qoq         : 6
# sector      : 10477
# inventory   : 1815
# goodwill    : 1032
# depreciation: 1347
# amortization: 1123
# liabilities : 3763
# receivables : 627
# current ratio: 280
# debt        : 15704
# d/e         : 64
# debt to equity: 81
# cash flow   : 4019
# fcf         : 58
# investments : 10896

# Analysis: Seems to be fine tuned for public equites and earnings analysis. 
# Includes some general commentarty on finance/accounting but lacks advanced
# analytics like fcf, d/e. current ratio, etc.