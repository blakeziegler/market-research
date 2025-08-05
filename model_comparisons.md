# Model Comparisons

The main method of comparison between models was 40 CFA level I, II, III valuation questions. The results were graded by ChatGPT 4o-turbo, and all grading was validated by a human. Each prompt-result pair was graded 3 times to account grading variation from run-to-run.

Questions $PATH: model/DAPT/v3/benchmark_v2.csv \
ChatGPT Grader $PATH : model/DAPT/v3/gpt_grader.py

### Base Model
Dev9124/qwen3-finance-model \
https://huggingface.co/Dev9124/qwen3-finance-model

The base model we uses a Qwen3-4B-Base derivative, and is fine tuned with over 500k I/O pairs of financial data. The data generally consists of rudementary financial information and news articles, so we wanted to see if targeted DAPT + SFT could result in increased business valuation
capabilities.

Base model derivative: https://huggingface.co/Qwen/Qwen3-4B \
Base model SFT Dataset: https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k

### DAPT Data (700k)
The DAPT data consisted of a few main categories:
- Company 10-ks (371.4k)
- Retail Investor Reports (98.5k)
- Professional Investor Reports (85.1k)
- CFA Guides (61.4k)
- Research Papers (55.1k)
- Valuation/General Finance Articles (12.6k)
- Valuation Strategies (11.1k)
- Calculation Examples (3.1k)

Total Tokens: 699,844 \
Total Documents: 73

### Base Model vs Base Model + DAPT (700k Tokens)

*Link*: https://huggingface.co/blakeziegler/qwen3_4b_dapt-700k_v3

**Round 1 Results:** \
Base Results Summary: \
Tone: 7.85 ± 0.36 \
Accuracy: 8.70 ± 0.46 \
Structure: 8.03 ± 0.53 \
Total hallucinations: 0/40 (0.0%)

DAPT Results Summary: \
Tone: 7.65 ± 0.48 \
Accuracy: 8.12 ± 1.02 \
Structure: 7.95 ± 0.64 \
Total hallucinations: 8/40 (20.0%) \

Hallucination Comparison: \
Base model: 0 hallucinations \
DAPT model: 8 hallucinations \
Regression: 8 more hallucinations with DAPT model

****
**Round 2 Results** \
Base Results Summary:\
Tone: 7.90 ± 0.30 \
Accuracy: 8.68 ± 0.57 \
Structure: 8.10 ± 0.55 \
Total hallucinations: 1/40 (2.5%) \

DAPT Results Summary: \
Tone: 7.67 ± 0.47 \
Accuracy: 8.10 ± 1.08 \
Structure: 7.95 ± 0.64 \
Total hallucinations: 9/40 (22.5%)

Hallucination Comparison: \
Base model: 1 hallucinations \
DAPT model: 9 hallucinations

****
**Round 3 Results** \
Base Results Summary: \
Tone: 7.85 ± 0.36 \
Accuracy: 8.72 ± 0.51 \
Structure: 8.00 ± 0.51 \
Total hallucinations: 1/40 (2.5%)

DAPT Results Summary: \
Tone: 7.65 ± 0.48 \
Accuracy: 8.00 ± 1.09 \
Structure: 7.97 ± 0.66 \
Total hallucinations: 9/40 (22.5%)

Hallucination Comparison: \
Base model: 1 hallucinations \
DAPT model: 9 hallucinations \
Regression: 8 more hallucinations with DAPT model

****
**Analysis & Conclusion** \
As seen in the results, the DAPT 700k model performed worse and hallucinated at a greater rate than the base model. While we believe this is correct, we see evidence in the responses that the DAPT model gave more creative answers and expanded into deeper valuation concepts compared to the base model.

We believe the hallucinations were the main cause of lackluster scores (as it should be), mostly derived from artifacts within the DAPT text, primarily the 10-k documents. We hypothesize the hallucinations derived from:
- Not enough DAPT tokens causing memorization, not generalization
- No additional SFT post-DAPT to redirect model behavior and vocabulary.

We hope SFT on 500 - 1000 I/O pairs will help mitigate hallucinations. If not, we will go back to the drawing board and add an additional 500k - 1M tokens and perform DAPT again. 

### Base Model vs Base Model + DAPT + SFT
*Coming Soon*