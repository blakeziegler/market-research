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
    Returns a dictionary with tone, accuracy, and structure scores
    """
    system_prompt = """You are an expert financial writing evaluator. You will receive one financial analysis response, and must rate it in three dimensions from 1 to 10: tone (how clear and professional it sounds), accuracy (how factually correct and logically sound it is), and structure (how well-organized and easy to follow it is). Be strict, do not give 10s unless it's flawless."""

    user_prompt = f"""Evaluate the following financial analysis response. Return your answer in JSON format as:
{{"tone": X, "accuracy": Y, "structure": Z}}

Response:
{output_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        # Extract and parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                print(f"Could not find JSON in response: {content}")
                return {"tone": 5, "accuracy": 5, "structure": 5}
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {content}")
            return {"tone": 5, "accuracy": 5, "structure": 5}
            
    except Exception as e:
        print(f"Error evaluating output: {e}")
        return {"tone": 5, "accuracy": 5, "structure": 5}

def process_csv_file(file_path, output_column):
    """
    Process a CSV file and add grading columns
    """
    print(f"Processing {file_path}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize new columns
    prefix = output_column.split('_')[0]  # 'base' or 'dapt'
    tone_col = f"{prefix}_tone"
    accuracy_col = f"{prefix}_accuracy"
    structure_col = f"{prefix}_structure"
    
    df[tone_col] = None
    df[accuracy_col] = None
    df[structure_col] = None
    
    # Process each row
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
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Save the updated DataFrame
    df.to_csv(file_path, index=False)
    print(f"Completed processing {file_path}")
    
    return df

def main():
    """
    Main function to process both CSV files
    """
    # File paths
    base_file = "results_base_v3.csv"
    dapt_file = "results_dapt_v3.csv"
    
    # Check if files exist
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found")
        return
    
    if not os.path.exists(dapt_file):
        print(f"Error: {dapt_file} not found")
        return
    
    # Process base results
    print("=" * 50)
    print("Processing base results...")
    print("=" * 50)
    base_df = process_csv_file(base_file, "base_output")
    
    # Process DAPT results
    print("=" * 50)
    print("Processing DAPT results...")
    print("=" * 50)
    dapt_df = process_csv_file(dapt_file, "dapt_output")
    
    print("=" * 50)
    print("All processing completed!")
    print("=" * 50)
    
    # Print summary statistics
    print("\nBase Results Summary:")
    print(f"Tone: {base_df['base_tone'].mean():.2f} ± {base_df['base_tone'].std():.2f}")
    print(f"Accuracy: {base_df['base_accuracy'].mean():.2f} ± {base_df['base_accuracy'].std():.2f}")
    print(f"Structure: {base_df['base_structure'].mean():.2f} ± {base_df['base_structure'].std():.2f}")
    
    print("\nDAPT Results Summary:")
    print(f"Tone: {dapt_df['dapt_tone'].mean():.2f} ± {dapt_df['dapt_tone'].std():.2f}")
    print(f"Accuracy: {dapt_df['dapt_accuracy'].mean():.2f} ± {dapt_df['dapt_accuracy'].std():.2f}")
    print(f"Structure: {dapt_df['dapt_structure'].mean():.2f} ± {dapt_df['dapt_structure'].std():.2f}")

if __name__ == "__main__":
    main()
