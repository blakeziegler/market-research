# Curated dataset for fine tuning

### Problem

The finance llama 8b model focuses on on multiturn conversational
finance and news sentiment analysis. This is great for a base model
but it missing key valuation knowledge of popular metrics like P/E,
EBITA, ROE, etc. To see where the model was lacking, we searched the
dataset for key valuation metrics and counted the nuber of occurences
for each metric. We found:

**Occurrences in dataset**

p/e         : 393 \
peg         : 784 \
ev/ebitda   : 5 \
book value  : 258 \ 
valuation   : 5310 \ 
beta        : 857 \ 
stock       : 28075 \ 
moving average: 260 \
analyst     : 3804 \
eps         : 11665 \
roe         : 3930 \
profit      : 24579 \
loss        : 18855 \
income      : 18968 \
dividend    : 5295 \
retained earnings: 357 \
margin      : 6123 \
revenue     : 21477 \
yoy         : 291 \
qoq         : 6 \
sector      : 10477 \
inventory   : 1815 \
goodwill    : 1032 \
depreciation: 1347 \
amortization: 1123 \
liabilities : 3763 \
receivables : 627 \
current ratio: 280 \
debt        : 15704 \
d/e         : 64 \
debt to equity: 81 \
cash flow   : 4019 \
fcf         : 58 \
investments : 10896

These keywords are based off of data the model will revieve from various
API's                             


### Solution

To remidy this, we will fine tune the finance llama model using a curated
dataset from other hugging face models. We aim to bolster the models valuation
ability through this fine-tuning process.

**LACKING METRICS**
- p/e
- peg
- ev/ebita
- book value
- beta
- moving average
- roe
- retained earnings
- yoy/qoq
- inventory
- goodwill
- appreciation
- amortizzation
- liabilities
- recivables
- current ratio
- d/e & debt to equity
- cash flow
- fcf

### Candidate Datasets

alpaca_finance.py: Very similar to base model, will not be used in data curation

winddude/reddit_finance_43_250k: Has many more examples of financial metrics and traditional valuation, but result quality varies greatly. Will be used in dataset, but heavily curated.