import os
import csv
import time
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

industries = [
    "Aerospace & Defense",
    "Banking",
    "Technology",
    "Telecommunications",
    "Healthcare",
    "Insurance",
    "Real Estate",
    "Travel & Leisure",
    "Retail",
    "Media & Entertainment",
    "Manufacturing",
    "Utilities",
    "Chemicals & Materials",
    "Consumer Goods",
    "Energy",
]

def generate_synthetic_data():
    """
    Generate synthetic data for supervised fine-tuning using OpenAI API.
    Creates 100 input/output pairs (50 batches of 2 pairs each).
    """
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Prompt for generating synthetic data
    prompt = """Generate exactly ONE user/assistant pair for financial analysis training data.

You MUST generate BOTH the user prompt AND the assistant response in this exact format:

User:
[CFA level 1, 2, or 3 valuation question]

Assistant:
[In-depth answer to the given question]

REQUIREMENTS:
- Focus on {industry} industry
- The CFA question should be well thought out and non-ambiguous.
- There should be a 33%/33%/33% the question is CFA level 1, 2, or 3
- Use fake company names
- Do not overdo the calculations, do what is necessary to answer the question.
- Your answers MUST be in markdown format.
- Your assistant response MUST be less than 500 words. Make it concise and thorough.
- The user question should be short and concise, no more than 75 words.

You are free to make input/outputs as unique as you would like. This is being used to train an LLM, so the more varied
and unique the better. Do not be afraid to make user prompt a little outside the box, as long as it remains realistic.

Here are some question examples to use as inspiration:
1. "A telecom company with 3.5x net debt/EBITDA is trading at 6x EV/EBITDA. FCF yield is 8.5% but capex intensity is 26% of revenue. Analyze whether the leverage and capital intensity justify a valuation discount, or if the market is underestimating long-term infrastructure ROI."
2. "A specialty chemicals company has expanded EBITDA margins from 14% to 18% following a restructuring, yet ROIC remains below WACC. Evaluate whether improved operating metrics are creating shareholder value or merely masking inefficient capital deployment."
3. "An e-commerce platform shows consistent GMV growth but negative operating margins and rising fulfillment costs. Sales/Marketing expense is 42% of revenue. Evaluate whether scale economics are taking hold. Should valuation rely on EV/GMV multiples or adjusted EBITDA forecasts?"

Remember to be CREATIVE and UNIQUE for both the user prompt and the assistant response.

DO NOT mention anything about it being a CFA question, just ask the question.

Generate the complete pair now:"""
    
    # CSV file path
    csv_file_path = 'synthetic-data/synthetic_valuation.csv'
    backup_file_path = 'synthetic-data/synthetic_valuation_backup.txt'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Initialize empty list to store all pairs
    all_pairs = []
    
    # Generate 20 batches of 1 pairs each (20 total pairs)
    for batch in range(50):
        industry = random.choice(industries)
        print(f"Generating batch {batch + 1}/50...")
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a financial data generator creating synthetic training data."},
                    {"role": "user", "content": prompt.format(industry=industry)}
                ]
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content.strip()
            
            # Parse the generated text to extract user/assistant pairs
            pairs = parse_generated_pairs(generated_text)
            
            # Add pairs to the main list
            for user_text, assistant_text in pairs:
                all_pairs.append({
                    "user": user_text,
                    "assistant": assistant_text
                })
            
            print(f"Batch {batch + 1} completed. Generated {len(pairs)} pairs.")
            
            # SAVE INCREMENTALLY AFTER EACH BATCH TO PREVENT DATA LOSS
            # Save to backup text file first
            with open(backup_file_path, 'w', encoding='utf-8') as f:
                for i, pair in enumerate(all_pairs):
                    f.write(f"Pair {i+1}:\n")
                    f.write(f"USER: {pair['user']}\n")
                    f.write(f"ASSISTANT: {pair['assistant']}\n")
                    f.write("-" * 40 + "\n\n")
            
            # Also save to CSV incrementally
            df_temp = pd.DataFrame(all_pairs)
            df_temp.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            
            print(f"SAVED {len(all_pairs)} pairs to backup and CSV files")
            
            # Print sample every 10 batches for monitoring (including first batch)
            if (batch + 1) % 10 == 0 or batch == 0:
                print(f"\n{'='*60}")
                print(f"MONITORING CHECK - After {batch + 1} batches ({len(all_pairs)} total pairs)")
                print(f"{'='*60}")
                
                # Show the last 2 pairs as a sample
                if len(all_pairs) >= 2:
                    print("\nSAMPLE OF LATEST GENERATED PAIRS:")
                    print("-" * 40)
                    for i in range(max(0, len(all_pairs)-2), len(all_pairs)):
                        print(f"\nPair {i+1}:")
                        print(f"USER: {all_pairs[i]['user']}")
                        print(f"ASSISTANT: {all_pairs[i]['assistant']}")
                        print("-" * 40)
                
                print(f"\nTotal pairs collected so far: {len(all_pairs)}")
                print(f"Progress: {batch + 1}/50 batches completed")
                print(f"Data saved to: {csv_file_path}")
                print(f"Backup saved to: {backup_file_path}")
                print(f"{'='*60}\n")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in batch {batch + 1}: {e}")
            print(f"SAVING CURRENT DATA TO BACKUP...")
            # Save current data even if there's an error
            with open(backup_file_path, 'w', encoding='utf-8') as f:
                for i, pair in enumerate(all_pairs):
                    f.write(f"Pair {i+1}:\n")
                    f.write(f"USER: {pair['user']}\n")
                    f.write(f"ASSISTANT: {pair['assistant']}\n")
                    f.write("-" * 40 + "\n\n")
            continue
    
    # Final save
    df = pd.DataFrame(all_pairs)
    df.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    print(f"Synthetic data generation completed!")
    print(f"Total pairs generated: {len(df)}")
    print(f"Data saved to: {csv_file_path}")
    print(f"Backup saved to: {backup_file_path}")
    print(f"DataFrame shape: {df.shape}")

def parse_generated_pairs(text):
    """
    Parse the generated text to extract user/assistant pairs.
    """
    pairs = []
    lines = text.split('\n')
    
    current_user = ""
    current_assistant = ""
    in_user = False
    in_assistant = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('User:'):
            # Save previous pair if exists
            if current_user and current_assistant:
                pairs.append((current_user.strip(), current_assistant.strip()))
            
            # Start new user
            current_user = line[5:].strip()  # Remove "User:" prefix
            current_assistant = ""
            in_user = True
            in_assistant = False
            
        elif line.startswith('Assistant:'):
            # Start new assistant
            current_assistant = line[11:].strip()  # Remove "Assistant:" prefix
            in_user = False
            in_assistant = True
            
        elif line and in_user:
            # Continue building user text
            current_user += "\n" + line
            
        elif line and in_assistant:
            # Continue building assistant text
            current_assistant += "\n" + line
    
    # Add the last pair
    if current_user and current_assistant:
        pairs.append((current_user.strip(), current_assistant.strip()))
    
    return pairs

if __name__ == "__main__":
    generate_synthetic_data()
