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
    Returns a dictionary with tone, accuracy, structure, hallucinated, and justification
    """
    system_prompt = """You are an expert financial writing evaluator. You will receive one financial analysis 
    response, and must rate it in four dimensions: tone (how clear and professional it sounds), accuracy 
    (how factually correct and logically sound it is), structure (how well-organized and easy to follow it is), 
    and hallucination detection (whether the response contains made-up facts, numbers, or claims not supported by 
    the prompt). Be strict, do not give 10s unless it's flawless. For hallucination, use 1 if any facts/numbers are 
    made up, 0 if everything appears factual."""

    user_prompt = f"""Evaluate the following financial analysis response. Return your answer in JSON format as:
{{"tone": X, "accuracy": Y, "structure": Z, "hallucinated": W, "justification": "Brief explanation of what was hallucinated (max 2 sentences, or 'No hallucinations detected' if W=0)"}}

Response:
{output_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
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
                return {"tone": 5, "accuracy": 5, "structure": 5, "hallucinated": 0, "justification": "Error parsing response"}
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {content}")
            return {"tone": 5, "accuracy": 5, "structure": 5, "hallucinated": 0, "justification": "Error parsing response"}
            
    except Exception as e:
        print(f"Error evaluating output: {e}")
        return {"tone": 5, "accuracy": 5, "structure": 5, "hallucinated": 0, "justification": "Error during evaluation"}

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
    structure_col = f"{prefix}_structure"
    hallucinated_col = f"{prefix}_hallucinated"
    justification_col = f"{prefix}_justification"
    
    df[tone_col] = None
    df[accuracy_col] = None
    df[structure_col] = None
    df[hallucinated_col] = None
    df[justification_col] = None
    
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
        df.at[idx, structure_col] = evaluation.get("structure", 5)
        df.at[idx, hallucinated_col] = evaluation.get("hallucinated", 0)
        df.at[idx, justification_col] = evaluation.get("justification", "No justification provided")
        
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
    print(f"Structure: {base_df['base_structure'].mean():.2f} ± {base_df['base_structure'].std():.2f}")
    base_hallucinations = base_df['base_hallucinated'].sum()
    base_total = len(base_df)
    print(f"Total hallucinations: {base_hallucinations}/{base_total} ({base_hallucinations/base_total*100:.1f}%)")
    
    print("\nSFT Results Summary:")
    print(f"Tone: {sft_df['sft_tone'].mean():.2f} ± {sft_df['sft_tone'].std():.2f}")
    print(f"Accuracy: {sft_df['sft_accuracy'].mean():.2f} ± {sft_df['sft_accuracy'].std():.2f}")
    print(f"Structure: {sft_df['sft_structure'].mean():.2f} ± {sft_df['sft_structure'].std():.2f}")
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
