import os
import csv
import time
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_synthetic_data():
    """
    Generate synthetic data for supervised fine-tuning using OpenAI API.
    Creates 63 input/output pairs (21 batches of 3 pairs each).
    """
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Target industries for random selection
    target_industries = [
        "Technology/SaaS",
        "Oil and gas",
        "Construction/industrial",
        "Defense",
        "Telecommunications",
        "Banking",
        "Insurance",
        "Real Estate",
        "Transportation",
        "Media & Entertainment",
        "Aerospace",
        "Pharmaceuticals"
    ]
    

    
    # CSV file path
    csv_file_path = 'synthetic-data/synthetic_balance-sheet.csv'
    backup_file_path = 'synthetic-data/synthetic_balance-sheet_backup.txt'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Initialize empty list to store all pairs
    all_pairs = []
    
    # Generate 21 batches of 3 pairs each (63 total pairs)
    for batch in range(21):
        print(f"Generating batch {batch + 1}/21...")
        
        # Select a new random industry for each batch to ensure diversity
        selected_industry = random.choice(target_industries)
        print(f"Selected industry for this batch: {selected_industry}")
        
        # Create a completely new prompt for each batch with the selected industry
        batch_prompt = f"""You are generating synthetic supervised fine-tuning data for a financial analysis LLM focused on interpreting balance sheets.

**CRITICAL: You MUST focus on the {selected_industry} industry. Do NOT use any other industry.**

Generate 3 unique and realistic user/assistant example pairs in the following format:

User:
<full user prompt>
Assistant:
<Detailed 5-7 sentence financial analysis with proper formatting and line breaks, including specific ratios, metrics, and insights, ending with a bolded buy/hold/sell recommendation>

Each user prompt should include:
- A short company description (1–3 sentences) about a {selected_industry} company
- A labeled **Balance Sheet** section with the following fields, properly formatted and scaled:

Balance Sheet:
(Note: The values below are examples only — your outputs must generate values consistent with the company profile.)

- Total Assets: 137.2 billion  
- Current Assets: 34.5 billion  
- Cash and Cash Equivalents: 13.9 billion  
- Inventory: 1.3 billion  
- Net Receivables: 7.2 billion  
- Non-Current Assets: 102.7 billion  
- Property, Plant, and Equipment (PP&E): 8.9 billion  
- Intangible Assets: 10.7 billion  
- Goodwill: 60.7 billion  
- Short-Term Investments: 644 million  
- Other Current Assets: 11.4 billion  
- Total Liabilities: 109.8 billion  
- Current Liabilities: 33.1 billion  
- Accounts Payable: 4.0 billion  
- Short-Term Debt: 5.9 billion  
- Non-Current Liabilities: 76.6 billion  
- Capital Lease Obligations: 3.4 billion  
- Long-Term Debt: 49.9 billion  
- Other Current Liabilities: 7.3 billion  
- Other Non-Current Liabilities: 981 million  
- Shareholder Equity: 27.3 billion  
- Common Stock: 61.4 billion  
- Retained Earnings: 151.2 billion  
- Shares Outstanding: 937.2 million

After the balance sheet, include the following instruction at the end of each user prompt:

"You need to provide a comprehensive financial analysis of this company's balance sheet position based on ONLY the numbers above and nothing else. Include specific ratios, metrics, and detailed insights. Also include an investor recommendation (buy/hold/sell)."

Additional instructions:
All values in the Balance Sheet section must:
- Be logically consistent with the company description and financial profile  
- Use appropriate scale and units (e.g., a high-growth SaaS firm should have high intangibles and receivables; a manufacturing company should show large PP&E)  
- Be normalized into billions or millions and clearly labeled (e.g., "4.5 billion", "760 million")

Assistant responses must:
- Be grounded only in the financial data provided above  
- Provide comprehensive, investment-grade analysis in 5-7 well-formatted sentences with proper line breaks
- Include specific financial ratios and calculations (current ratio, debt-to-equity, working capital, etc.)
- Analyze liquidity, solvency, and financial structure in detail
- Vary in tone and insight (bullish, cautious, skeptical, etc.)  
- **CRITICAL: Cite specific metrics from the balance sheet in your analysis (e.g., "With a current ratio of 1.04, the company has adequate short-term liquidity")**
- End with: **Recommendation: Buy/Hold/Sell.**
- DO NOT use real company names or tickers. Make sure the fake company you use
is the same for both the user and assistant.

Do not include explanations, system messages, or markdown formatting. Generate exactly 3 user/assistant pairs in raw plain text."""
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial data generator creating synthetic training data for balance sheet analysis."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.8,
                max_tokens=2048
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
            
            # Print sample every 5 batches for monitoring (including first batch)
            if (batch + 1) % 5 == 0 or batch == 0:
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
                print(f"Progress: {batch + 1}/21 batches completed")
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
