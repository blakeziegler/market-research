# Model Comparisons

The main method of comparison between models was 40 CFA level I, II, III valuation questions. The results were graded by ChatGPT 4o, and all grading was validated by a human. Each prompt-result pair was graded 3 times to account grading variation from run-to-run.

Questions $PATH: model/DAPT/v3/benchmark_v2.csv \
ChatGPT Grader $PATH : model/DAPT/v3/gpt_grader.py

## Base Model
Dev9124/qwen3-finance-model \
https://huggingface.co/Dev9124/qwen3-finance-model

The base model we uses a Qwen3-4B-Base derivative, and is fine tuned with over 500k I/O pairs of financial data. The data generally consists of rudementary financial information and news articles, so we wanted to see if targeted DAPT + SFT could result in increased business valuation
capabilities.

Base model derivative: https://huggingface.co/Qwen/Qwen3-4B \
Base model SFT Dataset: https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k

## DAPT 700k Token Model
*Link: https://huggingface.co/blakeziegler/qwen_4b_dapt-700k*

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

Base Results Summary: \
Tone: 8.30 ± 0.46 \
Accuracy: 8.32 ± 1.07 \
Creativity: 7.17 ± 0.71 \
Total hallucinations: 7/40 (17.5%)

DAPT Results Summary: \
Tone: 7.85 ± 0.77 \
Accuracy: 6.53 ± 2.01 \
Creativity: 6.45 ± 1.04 \
Total hallucinations: 20/40 (50.0%)

****
**Analysis & Conclusion** \
As seen in the results, the DAPT 700k model performed worse and hallucinated at a greater rate than the base model. While we believe this is correct, we see evidence in the responses that the DAPT model gave more creative answers and expanded into deeper valuation concepts compared to the base model.

We believe the hallucinations were the main cause of lackluster scores (as it should be), mostly derived from artifacts within the DAPT text, primarily the 10-k documents. We hypothesize the hallucinations derived from:
- Not enough DAPT tokens causing memorization, not generalization
- No additional SFT post-DAPT to redirect model behavior and vocabulary.

We hope SFT on 250 - 500 I/O pairs will help mitigate hallucinations. If not, we will go back to the drawing board and add an additional 500k - 1M tokens and perform DAPT again.

Additionally, the DAPT model used language like "we believe the company ..." reflecting too heavy weighting for 10-k filings. If re-doing dapt is necessary, we will try to balance this out with more diverse document types to achieve the desired language and tonality.

## DAPT 700k Token + 260 I/O SFT Model
*Link: https://huggingface.co/blakeziegler/qwen3_4b_dapt-700k_260-sft*

### SFT Data 
- 100 I/O Company Analysis
- 100 I/O CFA level I, II, III Questions
- 60 Balance Sheet Analysis

### Base Model vs Base Model + DAPT (700k) + SFT (260 I/O)

Base Results Summary: \
Tone: 8.30 ± 0.52 \
Accuracy: 7.97 ± 1.37 \
Creativity: 7.17 ± 0.81 \
Total hallucinations: 9/40 (22.5%)

SFT Results Summary: \
Tone: 7.88 ± 0.88 \
Accuracy: 6.35 ± 1.85 \
Creativity: 6.40 ± 1.22 \
Total hallucinations: 26/40 (65.0%)

****
**Analysis & Conclusion** \
The results show that not only did the DAPT + SFT model perform worse than the base model, it performed worse than the DAPT only model as well. While this was unexpected, we believe the culprit of lackluster performance was more human error than error in methodology. We hypothesize the decrease in performace is likely because of:
- The DAPT only model was already overfitting, so the small SFT dataset amplified hallucinations. 
- Not enough diversity of STF data caused rote memorization, leading to more artifacts from the training data in the responses.

For the next round of SFT, we will use a greater diversity of data and mix in examples from the base model SFT dataset. We hope this will reduce the number of artifacts present in the model and aid in generalization. Additionally, we will bump up the number of examples from 260 to ~ 750 for a broader range of context. 

## DAPT 1.5M Token Model
*Link: https://huggingface.co/blakeziegler/qwen3_4b_dapt-1.5M*
### DAPT Data (1.5M)
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


### Base Model vs Base Model + DAPT (1.5M Tokens)

Base Results Summary: \
Tone: 8.38 ± 0.67 \
Accuracy: 8.22 ± 1.48 \
Creativity: 7.10 ± 0.96 \
Total hallucinations: 9/40 (22.5%)

DAPT Results Summary: \
Tone: 7.92 ± 0.62 \
Accuracy: 6.85 ± 1.70 \
Creativity: 6.42 ± 0.93 \
Total hallucinations: 17/40 (42.5%)

****
**Analysis & Conclusion** \
We found it was necessary to re-do DAPT because of the high rate of hallucination found in the 700k token model. We added an additional 800k tokens of high quality valuation data, mostly in the form of textbooks and earnings call transcripts.

We assumed textbooks would help dampen the hallucination rate since most of the information is conceptual and not specific examples of particular companies. Our hypothesis appeared to have some validity with:
- A -7.5% reduction in hallucination rate
- A +0.04 increase in tone score
- A +0.50 increase in accuracy score
- A -0.02 decrease in creativity score.

*Compared with the 700k DAPT only model*

The increase in tone score and decrease in creativity score are fairly trivial and cannot be attributed to model capability. The difference in scores is likely due to differences in grading rather than ability of either model. On the other hand, the reduction in hallucination and increase in accuracy suggest the model is improving in generalization.

Another interesting thing to point out is the decrease in standard deviation across all scoring categories, alluding to the model producing more consistent outputs.

While the 1.5M token DAPT model did improve, it was not as much as we were expecting. We found no significant benefit in tone or creativity, and only slight improvements in accuracy and hallucination rate. For the next round of DAPT pre-training, we will increase the total token count by an additional ~ 1.5 - 3M tokens depending on how the DAPT + SFT model performs.

## DAPT 1.5M Token + 750 I/O SFT Model

### SFT Data
- 55 Company Overview Analysis
- 50 CFA Level I, II, II Questions
- 20 Balance Sheet Analysis

*Coming soon*

## DAPT 2.5M Token Model

### DAPT Data (1.5M)
- Finance Textbooks (1.037M)
- Company 10-ks (371.4k)
- Valuation, General Finance, and Economics Articles (250.9k)
- Professional Investor Reports (218.1k)
- Earnings Call Transcripts (164.4k)
- CFA Guides (131.4k)
- Research Papers (117.1k)
- Retail Investor Reports (101.5k)
- AI valuation Reports (15.3k)
- Valuation Strategies (13.5k)
- Calculation Examples (3.1k)
