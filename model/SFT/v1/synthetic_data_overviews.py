import os
import csv
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_synthetic_data():
    """
    Generate synthetic data for supervised fine-tuning using OpenAI API.
    Creates 100 input/output pairs (50 batches of 2 pairs each).
    """
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Prompt for generating synthetic data
    prompt = """You are generating synthetic supervised fine-tuning data for a financial analysis LLM.

Generate 2 unique and realistic user/assistant example pairs in the following format:

User:
<full user prompt>
Assistant:
<3–4 sentence creative and grounded analysis, ending with a bolded buy/hold/sell recommendation>

Each user prompt should include:
- A short company description (1–3 sentences)
- A labeled "Details" section with Sector and Industry
- A labeled "Financials & Earnings" section with:
  - Market Cap (B), EBITA (B), Revenue TTM (B), Gross Profit TTM (B)
  - EPS, Diluted EPS, Profit Margin, Operating Margin
  - Return on Assets, Return on Equity
  - Quarterly Earnings Growth YOY, Quarterly Revenue Growth YOY
- A labeled "Ratios" section with:
  - P/E Ratio, PEG Ratio, Book Value, Revenue Per Share
  - Trailing P/E, Forward P/E, P/S, P/B, EV/Revenue, EV/EBITA, Beta
- A labeled "Stock Information" section with:
  - Shares Outstanding, Shares Float (millions), Dividend Per Share, Dividend Yield
  - 52 Week High, 52 Week Low, 50-Day and 200-Day Moving Averages
- A final instruction:
  "You need to give an overall assessment of this company's financial health based on ONLY the numbers above and nothing else. Keep your answer to 3–4 sentences. Also include an investor recommendation (buy/hold/sell)."

Assistant responses must:
- Be grounded in the numbers only.
- Be nuanced and avoid boilerplate phrasing.
- Vary in tone (bullish, cautious, skeptical, etc.)
- Sound like a professional financial analyst
- DO NOT use real company names or tickers. Make sure the fake company you use
is the same for both the user and assistant.

Do not include explanations or headings. Just generate 2 user/assistant pairs in raw plain text."""
    
    # CSV file path
    csv_file_path = 'synthetic-data/synthetic_overview.csv'
    backup_file_path = 'synthetic-data/synthetic_overview_backup.txt'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Initialize empty list to store all pairs
    all_pairs = []
    
    # Generate 50 batches of 2 pairs each (100 total pairs)
    for batch in range(50):
        print(f"Generating batch {batch + 1}/50...")
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial data generator creating synthetic training data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
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
