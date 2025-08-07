import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import time

# Get the project root directory (4 levels up from this script)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
env_path = os.path.join(project_root, '.env')

load_dotenv(env_path)

openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def evaluate_output(output_text):
    """
    Evaluate a single output using OpenAI API
    Returns a dictionary with tone, accuracy, creativity, hallucinated, justification, and scoring_reasoning
    """
    system_prompt = """You are an expert financial writing evaluator. You will receive one financial analysis response and must rate it across four dimensions, each on a scale from 1 to 10 (except Hallucination Detection, which is binary). Your ratings should reflect the standards of a professional investment analyst. Be strict — only award 10s for truly exceptional, flawless performance.

	1.	Tone – Evaluate how clear, concise, structured, and professional the response is. The tone should resemble that of a financial analyst. Writing should be free of fluff and well-organized, with appropriate use of technical financial language.
	2.	Accuracy – Assess how factually correct and numerically precise the analysis is, based on the data provided. Conclusions should logically follow from the input.
	3.	Creativity – Measure the originality and sophistication of the financial reasoning. Did the response go beyond textbook answers? Did it apply valuation methods, metrics, or frameworks in a unique or insightful way that a skilled analyst might use in a real-world setting?
	4.	Hallucination Detection – This is a binary score (0 or 1).
	•	Score 1 if the response includes any made-up facts, figures, ratios, comparisons, or company details that are not present or derivable from the prompt.
	•	Score 0 if the entire response remains grounded in the information given.
    - Speculative statements DO NOT count as hallucinations unless they have no basis in reality.

    Additionally, provide detailed scoring reasoning in 3-4 sentences that explains:
    - What aspects of the response were particularly strong or well-executed
    - What areas need improvement or were problematic
    - Specific examples of good analysis or concerning elements
    - Overall assessment of the response quality
"""

    # Few-shot examples
    few_shot_examples = [
        {
            "role": "user", 
            "content": "TASK: Estimate sustainable growth using the provided ROE and retention ratio and evaluate reinvestment efficiency.\n\nDATA:\nROE: 15%\nRetention Ratio: 50%\nROIC: 12%\nWACC: 9%\nCapEx/Revenue: 8%"
        },
        {
            "role": "assistant",
            "content": "This is a basic response: 'Sustainable growth rate = ROE * Retention Ratio = 7.5%. Since ROIC > WACC, reinvestment is value-accretive.'\n\nScores:\nTone: 7\nAccuracy: 10\nCreativity: 2 (Textbook method, no added insight)\nHallucination: 0"
        },
        {
            "role": "assistant",
            "content": "This is a strong, creative response: 'Using the formula, sustainable growth is 7.5%. However, given the firm's ROIC-WACC spread of 3%, we can infer marginal reinvestment creates value. The CapEx-to-Revenue ratio of 8% aligns with moderate reinvestment intensity. If this is funding high-ROIC projects, the firm may compound value at a higher effective rate than the baseline 7.5%. Scenario analysis or reinvestment runway should be explored further.'\n\nScores:\nTone: 9\nAccuracy: 10\nCreativity: 9\nHallucination: 0"
        },
        {
            "role": "assistant",
            "content": "This is a mid-level response: 'The sustainable growth rate is calculated as ROE × Retention Ratio = 15% × 50% = 7.5%. The ROIC of 12% exceeds the WACC of 9%, indicating that reinvestment generates positive economic value. However, with a relatively low CapEx-to-Revenue ratio of 8%, the reinvestment intensity may not be sufficient to fully exploit high-return opportunities. The firm may benefit from increasing investment levels, provided the ROIC remains above the cost of capital.'\n\nScores:\nTone: 8\nAccuracy: 10\nCreativity: 6 (Goes beyond basic math to evaluate capital deployment intensity)\nHallucination: 0"
        },
        {
            "role": "assistant",
            "content": "This is a low-accuracy response: 'The sustainable growth rate is 12% since that's the ROIC, which represents how fast the company can grow sustainably. The reinvestment efficiency is strong because WACC is only slightly lower than ROIC, so the firm is growing efficiently.'\n\nScores:\nTone: 6\nAccuracy: 3 (Incorrect use of ROIC as a proxy for sustainable growth; misunderstands the ROIC-WACC spread)\nCreativity: 4 (Attempts insight but built on a flawed foundation)\nHallucination: 0"
        }
    ]

    user_prompt = f"""Evaluate the following financial analysis response. Return your answer in JSON format as:
{{"tone": X, "accuracy": Y, "creativity": Z, "hallucinated": W, "justification": "Brief explanation of what was hallucinated (max 2 sentences, or 'No hallucinations detected' if W=0)", "scoring_reasoning": "3-4 sentences providing detailed feedback on what was good and what was bad in the response"}}

Response:
{output_text}"""

    try:
        # Build messages with few-shot examples
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(few_shot_examples)
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.30,
            top_p=0.85
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                print(f"Could not find JSON in response: {content}")
                return {"tone": 5, "accuracy": 5, "creativity": 5, "hallucinated": 0, "justification": "Error parsing response", "scoring_reasoning": "Error parsing response - unable to evaluate"}
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {content}")
            return {"tone": 5, "accuracy": 5, "creativity": 5, "hallucinated": 0, "justification": "Error parsing response", "scoring_reasoning": "Error parsing response - unable to evaluate"}
            
    except Exception as e:
        print(f"Error evaluating output: {e}")
        return {"tone": 5, "accuracy": 5, "creativity": 5, "hallucinated": 0, "justification": "Error during evaluation", "scoring_reasoning": "Error during evaluation - unable to provide feedback"}

def process_csv_file(file_path, output_column):
    print(f"Processing {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Initialize new columns
    if output_column == "sft_dapt_output":
        prefix = "sft"
    else:
        prefix = output_column.split('_')[0]
    tone_col = f"{prefix}_tone"
    accuracy_col = f"{prefix}_accuracy"
    creativity_col = f"{prefix}_creativity"
    hallucinated_col = f"{prefix}_hallucinated"
    justification_col = f"{prefix}_justification"
    scoring_reasoning_col = f"{prefix}_scoring_reasoning"
    
    df[tone_col] = None
    df[accuracy_col] = None
    df[creativity_col] = None
    df[hallucinated_col] = None
    df[justification_col] = None
    df[scoring_reasoning_col] = None
    
    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)}")
        
        output_text = row[output_column]
        if pd.isna(output_text) or output_text == "":
            print(f"Empty output for row {idx + 1}, skipping...")
            continue
            
        # Evaluate the output
        evaluation = evaluate_output(output_text)
        
        # Store results
        df.at[idx, tone_col] = evaluation.get("tone", 5)
        df.at[idx, accuracy_col] = evaluation.get("accuracy", 5)
        df.at[idx, creativity_col] = evaluation.get("creativity", 5)
        df.at[idx, hallucinated_col] = evaluation.get("hallucinated", 0)
        df.at[idx, justification_col] = evaluation.get("justification", "No justification provided")
        df.at[idx, scoring_reasoning_col] = evaluation.get("scoring_reasoning", "No scoring reasoning provided")
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Save the updated DataFrame
    df.to_csv(file_path, index=False)
    print(f"Completed processing {file_path}")
    
    return df

def main():
    base_file = "results_base.csv"
    sft_file = "results_dapt_sft-v1.csv"
    
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found")
        return
    
    if not os.path.exists(sft_file):
        print(f"Error: {sft_file} not found")
        return
    
    print("=" * 50)
    print("Processing base results...")
    print("=" * 50)
    base_df = process_csv_file(base_file, "base_output")
    
    print("=" * 50)
    print("Processing SFT results...")
    print("=" * 50)
    sft_df = process_csv_file(sft_file, "sft_dapt_output")
    
    # Print summary statistics
    print("\nBase Results Summary:")
    print(f"Tone: {base_df['base_tone'].mean():.2f} ± {base_df['base_tone'].std():.2f}")
    print(f"Accuracy: {base_df['base_accuracy'].mean():.2f} ± {base_df['base_accuracy'].std():.2f}")
    print(f"Creativity: {base_df['base_creativity'].mean():.2f} ± {base_df['base_creativity'].std():.2f}")
    base_hallucinations = base_df['base_hallucinated'].sum()
    base_total = len(base_df)
    print(f"Total hallucinations: {base_hallucinations}/{base_total} ({base_hallucinations/base_total*100:.1f}%)")
    
    print("\nSFT Results Summary:")
    print(f"Tone: {sft_df['sft_tone'].mean():.2f} ± {sft_df['sft_tone'].std():.2f}")
    print(f"Accuracy: {sft_df['sft_accuracy'].mean():.2f} ± {sft_df['sft_accuracy'].std():.2f}")
    print(f"Creativity: {sft_df['sft_creativity'].mean():.2f} ± {sft_df['sft_creativity'].std():.2f}")
    sft_hallucinations = sft_df['sft_hallucinated'].sum()
    sft_total = len(sft_df)
    print(f"Total hallucinations: {sft_hallucinations}/{sft_total} ({sft_hallucinations/sft_total*100:.1f}%)")
    
    print("\nHallucination Comparison:")
    print(f"Base model: {base_hallucinations} hallucinations")
    print(f"SFT model: {sft_hallucinations} hallucinations")
    if base_hallucinations > 0 or sft_hallucinations > 0:
        improvement = base_hallucinations - sft_hallucinations
        if improvement > 0:
            print(f"Improvement: {improvement} fewer hallucinations with SFT model")
        elif improvement < 0:
            print(f"Regression: {abs(improvement)} more hallucinations with SFT model")
        else:
            print("No change in hallucination count")

if __name__ == "__main__":
    main()
