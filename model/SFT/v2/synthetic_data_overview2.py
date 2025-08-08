import os
import csv
import time
import random
import pandas as pd
import anthropic
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
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
    
    # Prompt for generating synthetic data
    prompt = """Generate exactly ONE user/assistant pair for financial analysis training data.

You MUST generate BOTH the user prompt AND the assistant response in this exact format:

User:
[Company description and financial data]

Assistant:
[Financial analysis with buy/hold/sell recommendation]

REQUIREMENTS:
- Focus on {industry} industry
- Generate a company with either healthy OR poor financials (choose one)
- Use fake company names
- Include detailed financial metrics
- End assistant response with **BUY/HOLD/SELL** recommendation
- Make sure to perform any necessary calculations to arrive at the recommendation.

Generate the complete pair now:"""
    
    # CSV file path
    csv_file_path = 'synthetic-data/synthetic_overview2.csv'
    backup_file_path = 'synthetic-data/synthetic_overview2_backup.txt'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Initialize empty list to store all pairs
    all_pairs = []
    
    # Generate 40 batches of 1 pair each (40 total pairs)
    for batch in range(25):
        industry = random.choice(industries)
        print(f"Generating batch {batch + 1}/25...")
        
        try:
            # Call OpenAI API
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                messages=[
                    {"role": "user", "content": prompt.format(industry=industry)}
                ],
                temperature=0.8,
                max_tokens=6000,
                top_p=0.9
            )
            
            # Extract the generated text
            generated_text = response.content[0].text
            
            # DEBUG: Print the raw response to see what Claude is generating
            print(f"\nDEBUG - Raw Claude response for batch {batch + 1}:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)
            
            # Parse the generated text to extract user/assistant pairs
            pairs = parse_generated_pairs(generated_text)
            
            # DEBUG: Print parsing results
            print(f"DEBUG - Parsing found {len(pairs)} pairs")
            for i, (user, assistant) in enumerate(pairs):
                print(f"DEBUG - Pair {i+1}:")
                print(f"  User length: {len(user)} chars")
                print(f"  Assistant length: {len(assistant)} chars")
                print(f"  User preview: {user[:100]}...")
                print(f"  Assistant preview: {assistant[:100]}...")
            
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
                print(f"Progress: {batch + 1}/40 batches completed")
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
    
    print(f"DEBUG - Parsing {len(lines)} lines of text")
    
    current_user = ""
    current_assistant = ""
    in_user = False
    in_assistant = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('User:'):
            print(f"DEBUG - Found User: at line {i+1}")
            # Save previous pair if exists
            if current_user and current_assistant:
                pairs.append((current_user.strip(), current_assistant.strip()))
                print(f"DEBUG - Saved pair {len(pairs)}")
            
            # Start new user
            current_user = line[5:].strip()  # Remove "User:" prefix
            current_assistant = ""
            in_user = True
            in_assistant = False
            
        elif line.startswith('Assistant:') or line.startswith('A:'):
            print(f"DEBUG - Found Assistant/A: at line {i+1}")
            # Start new assistant
            if line.startswith('Assistant:'):
                current_assistant = line[11:].strip()  # Remove "Assistant:" prefix
            else:
                current_assistant = line[2:].strip()  # Remove "A:" prefix
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
        print(f"DEBUG - Saved final pair {len(pairs)}")
    
    print(f"DEBUG - Total pairs found: {len(pairs)}")
    print(f"DEBUG - Final user text length: {len(current_user)}")
    print(f"DEBUG - Final assistant text length: {len(current_assistant)}")
    
    return pairs

if __name__ == "__main__":
    generate_synthetic_data()
