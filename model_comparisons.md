# Model Comparisons

The main method of comparison between models was 40 CFA level I, II, III valuation questions. The results were graded by ChatGPT 4o, and all grading was validated by a human. Each prompt-result pair was graded 3 times to account grading variation from run-to-run.

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

### DAPT (1.54M)
- Finance Textbooks (657k)
- Company 10-ks (371.4k)
- Earnings Call Transcripts (164.4k)
- Retail Investor Reports (101.5k)
- Professional Investor Reports (85.1k)
- CFA Guides (61.4k)
- Research Papers (55.1k)
- AI valuation Reports (15.3k)
- Valuation Strategies (13.5k)
- Valuation/General Finance Articles (12.6k)
- Calculation Examples (3.1k)

### Base Model vs Base Model + DAPT (700k Tokens)

*Link*: https://huggingface.co/blakeziegler/qwen3_4b_dapt-700k_v3

Base Results Summary: \
Tone: 8.15 ± 0.36 \
Accuracy: 7.10 ± 0.55 \
Structure: 8.10 ± 0.50 \
Total hallucinations: 15/40 (37.5%) 

DAPT Results Summary: \
Tone: 7.97 ± 0.53 \
Accuracy: 6.62 ± 1.03 \
Structure: 7.62 ± 0.90 \
Total hallucinations: 28/40 (70.0%)

****
**Analysis & Conclusion** \
As seen in the results, the DAPT 700k model performed worse and hallucinated at a greater rate than the base model. While we believe this is correct, we see evidence in the responses that the DAPT model gave more creative answers and expanded into deeper valuation concepts compared to the base model.

We believe the hallucinations were the main cause of lackluster scores (as it should be), mostly derived from artifacts within the DAPT text, primarily the 10-k documents. We hypothesize the hallucinations derived from:
- Not enough DAPT tokens causing memorization, not generalization
- No additional SFT post-DAPT to redirect model behavior and vocabulary.

We hope SFT on 500 - 1000 I/O pairs will help mitigate hallucinations. If not, we will go back to the drawing board and add an additional 500k - 1M tokens and perform DAPT again.

Additionally, the DAPT model used language like "we believe the company ..." reflecting too heavy weighting for 10-k filings. If re-doing dapt is necessary, we will try to balance this out with more diverse document types to achieve the desired language and tonality. 

### Base Model vs Base Model + DAPT + SFT

Base Results Summary:
Tone: 8.15 ± 0.36
Accuracy: 7.08 ± 0.80
Structure: 8.00 ± 0.55
Total hallucinations: 12/40 (30.0%)

SFT Results Summary:
Tone: 7.88 ± 0.56
Accuracy: 6.47 ± 1.11
Structure: 7.47 ± 0.93
Total hallucinations: 30/40 (75.0%)