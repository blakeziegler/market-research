import os
import csv
import time
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_synthetic_valuation_data():
    """
    Generate synthetic data for supervised fine-tuning using OpenAI API.
    Creates 50 input/output pairs (25 batches of 2 pairs each) for valuation questions.
    Appends to existing files instead of overwriting.
    """
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Target industries for random selection
    target_industries = [
        "hospitals",
        "automotive",
        "Oil and gas",
        "Cybersecurity",
        "Construction/industrial",
        "Defense",
        "Industrial machinery"
    ]
    
    # Select a random industry for this batch
    selected_industry = random.choice(target_industries)
    
    # Prompt for generating synthetic valuation data
    prompt = f"""You are generating synthetic supervised fine-tuning data for a valuation-focused financial language model.

Generate 2 input/output (I/O) examples in the format below.

**CRITICAL: You MUST focus on the {selected_industry} industry. Do NOT use any other industry.**

Each **user input** should be a concise, high-level financial valuation question that could realistically be asked by an investment team member. The question MUST be about a {selected_industry} company and use realistic context such as:
- Strategic capital allocation (R&D, capex, dividends)
- Cash flow and capital efficiency
- Burn rates and time-to-runway
- Revenue concentration or margin pressure
- Industry-specific valuation tensions

**IMPORTANT: EXCLUDE these overused industries in your questions:**
- Biotech/pharmaceuticals
- SaaS/software companies
- Renewable energy/solar/wind
- Semiconductors
- Cloud computing companies

**EXAMPLES of the types of valuation questions to generate (but create your own unique questions):**

1. "A telecom company with 3.5x net debt/EBITDA is trading at 6x EV/EBITDA. FCF yield is 8.5% but capex intensity is 26% of revenue. Analyze whether the leverage and capital intensity justify a valuation discount, or if the market is underestimating long-term infrastructure ROI."

2. "A specialty chemicals company has expanded EBITDA margins from 14% to 18% following a restructuring, yet ROIC remains below WACC. Evaluate whether improved operating metrics are creating shareholder value or merely masking inefficient capital deployment."

3. "An e-commerce platform shows consistent GMV growth but negative operating margins and rising fulfillment costs. Sales/Marketing expense is 42% of revenue. Evaluate whether scale economics are taking hold. Should valuation rely on EV/GMV multiples or adjusted EBITDA forecasts?"

4. "An automotive manufacturer has 28% of revenue tied to electric vehicle production, with R&D spending at 4.2% of revenue and gross margins declining from 18% to 15% over the past year. The company trades at 8x EV/EBITDA vs. 12x for pure EV competitors. Analyze whether the market is correctly pricing the transition risk or undervaluing the legacy ICE business."

5. "An industrial machinery company has 45% of its backlog in emerging markets, with 60-day payment terms and 12% of receivables over 90 days past due. Operating margins are 16% but working capital has increased by 30% year-over-year. Evaluate whether the international expansion is creating value or masking underlying cash flow issues."

**MANDATORY REQUIREMENT:** Create your own unique, creative valuation questions focused SPECIFICALLY on the {selected_industry} industry. Do NOT use consulting firms, SaaS, biotech, or any other industry. Your questions must be about {selected_industry} companies.

Each **assistant output** should be a detailed financial analysis memo written in **Markdown** format. Use the following tone and structure:

Constraints:
- Use only the data in the prompt — do not make up or assume any extra numbers.
- No instructions or explanations — just generate the input and answer directly.
- Each output must be 3–6 paragraphs, and formatted using markdown headers, bullet points, or tables where appropriate.
- Avoid boilerplate ("P/E is high so it must be overvalued"). Instead, use nuanced reasoning based on real-world financial logic.
- Show intermediate calculations (if any) and explain what they mean.
- Write as if preparing an internal investment memo for a professional investment firm.

Output format:

User:
TASK:
""<Insert valuation question about {selected_industry} industry>""

Assistant:
### <Insert your memo title>

<Markdown-formatted in-depth answer goes here>

Generate exactly 2 user/assistant pairs in this format."""
    
    # CSV file path
    csv_file_path = 'synthetic-data/synthetic_valuation.csv'
    backup_file_path = 'synthetic-data/synthetic_valuation_backup.txt'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Initialize empty list to store all pairs
    all_pairs = []
    
    # Load existing data if CSV exists
    if os.path.exists(csv_file_path):
        try:
            existing_df = pd.read_csv(csv_file_path, quoting=csv.QUOTE_ALL)
            for _, row in existing_df.iterrows():
                all_pairs.append({
                    "user": row['user'],
                    "assistant": row['assistant']
                })
            print(f"Loaded {len(all_pairs)} existing pairs from CSV")
        except Exception as e:
            print(f"Error loading existing CSV: {e}")
    
    # Generate 9 batches of 2 pairs each (18 total pairs to add)
    for batch in range(9):  # Generate 9 batches (18 pairs)
        print(f"Generating batch {batch + 1}/9...")
        
        # Select a new random industry for each batch to ensure diversity
        selected_industry = random.choice(target_industries)
        print(f"Selected industry for this batch: {selected_industry}")
        
        # Create a completely new prompt for each batch with the selected industry
        batch_prompt = f"""You are generating synthetic supervised fine-tuning data for a valuation-focused financial language model.

Generate 2 input/output (I/O) examples in the format below.

**CRITICAL: You MUST focus on the {selected_industry} industry. Do NOT use any other industry.**

Each **user input** should be a concise, high-level financial valuation question that could realistically be asked by an investment team member. The question MUST be about a {selected_industry} company and use realistic context such as:
- Strategic capital allocation (R&D, capex, dividends)
- Cash flow and capital efficiency
- Burn rates and time-to-runway
- Revenue concentration or margin pressure
- Industry-specific valuation tensions

**IMPORTANT: EXCLUDE these overused industries in your questions:**
- Biotech/pharmaceuticals
- SaaS/software companies
- Renewable energy/solar/wind
- Semiconductors
- Cloud computing companies

**EXAMPLES of the types of valuation questions to generate (but create your own unique questions):**

1. "A telecom company with 3.5x net debt/EBITDA is trading at 6x EV/EBITDA. FCF yield is 8.5% but capex intensity is 26% of revenue. Analyze whether the leverage and capital intensity justify a valuation discount, or if the market is underestimating long-term infrastructure ROI."

2. "A specialty chemicals company has expanded EBITDA margins from 14% to 18% following a restructuring, yet ROIC remains below WACC. Evaluate whether improved operating metrics are creating shareholder value or merely masking inefficient capital deployment."

3. "An e-commerce platform shows consistent GMV growth but negative operating margins and rising fulfillment costs. Sales/Marketing expense is 42% of revenue. Evaluate whether scale economics are taking hold. Should valuation rely on EV/GMV multiples or adjusted EBITDA forecasts?"

4. "An automotive manufacturer has 28% of revenue tied to electric vehicle production, with R&D spending at 4.2% of revenue and gross margins declining from 18% to 15% over the past year. The company trades at 8x EV/EBITDA vs. 12x for pure EV competitors. Analyze whether the market is correctly pricing the transition risk or undervaluing the legacy ICE business."

5. "An industrial machinery company has 45% of its backlog in emerging markets, with 60-day payment terms and 12% of receivables over 90 days past due. Operating margins are 16% but working capital has increased by 30% year-over-year. Evaluate whether the international expansion is creating value or masking underlying cash flow issues."

**MANDATORY REQUIREMENT:** Create your own unique, creative valuation questions focused SPECIFICALLY on the {selected_industry} industry. Do NOT use consulting firms, SaaS, biotech, or any other industry. Your questions must be about {selected_industry} companies.

Each **assistant output** should be a detailed financial analysis memo written in **Markdown** format. Use the following tone and structure:

Constraints:
- Use only the data in the prompt — do not make up or assume any extra numbers.
- No instructions or explanations — just generate the input and answer directly.
- Each output must be 3–6 paragraphs, and formatted using markdown headers, bullet points, or tables where appropriate.
- Avoid boilerplate ("P/E is high so it must be overvalued"). Instead, use nuanced reasoning based on real-world financial logic.
- Show intermediate calculations (if any) and explain what they mean.
- Write as if preparing an internal investment memo for a professional investment firm.

Output format:

User:
TASK:
""<Insert valuation question about {selected_industry} industry>""

Assistant:
### <Insert your memo title>

<Markdown-formatted in-depth answer goes here>

Generate exactly 2 user/assistant pairs in this format."""
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial valuation expert creating synthetic training data for investment analysis."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.8,
                max_tokens=2048
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content.strip()
            
            # Parse the generated text to extract user/assistant pairs
            pairs = parse_valuation_pairs(generated_text)
            
            # Add pairs to the main list
            for user_text, assistant_text in pairs:
                all_pairs.append({
                    "user": user_text,
                    "assistant": assistant_text
                })
            
            print(f"Batch {batch + 1} completed. Generated {len(pairs)} pairs.")
            
            # SAVE INCREMENTALLY AFTER EACH BATCH TO PREVENT DATA LOSS
            # Save to backup text file first (append mode)
            with open(backup_file_path, 'a', encoding='utf-8') as f:
                for i, pair in enumerate(all_pairs[-len(pairs):]):  # Only save the new pairs
                    f.write(f"Pair {len(all_pairs)-len(pairs)+i+1}:\n")
                    f.write(f"USER: {pair['user']}\n")
                    f.write(f"ASSISTANT: {pair['assistant']}\n")
                    f.write("-" * 40 + "\n\n")
            
            # Also save to CSV incrementally (overwrite with all data)
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
                print(f"Progress: {batch + 1}/9 batches completed")
                print(f"Data saved to: {csv_file_path}")
                print(f"Backup saved to: {backup_file_path}")
                print(f"{'='*60}\n")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in batch {batch + 1}: {e}")
            print(f"SAVING CURRENT DATA TO BACKUP...")
            # Save current data even if there's an error
            with open(backup_file_path, 'a', encoding='utf-8') as f:
                for i, pair in enumerate(all_pairs[-len(pairs):]):  # Only save the new pairs
                    f.write(f"Pair {len(all_pairs)-len(pairs)+i+1}:\n")
                    f.write(f"USER: {pair['user']}\n")
                    f.write(f"ASSISTANT: {pair['assistant']}\n")
                    f.write("-" * 40 + "\n\n")
            continue
    
    # Final save
    df = pd.DataFrame(all_pairs)
    df.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    print(f"Synthetic valuation data generation completed!")
    print(f"Total pairs generated: {len(df)}")
    print(f"Data saved to: {csv_file_path}")
    print(f"Backup saved to: {backup_file_path}")
    print(f"DataFrame shape: {df.shape}")

def parse_valuation_pairs(text):
    """
    Parse the generated text to extract user/assistant pairs for valuation data.
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
    generate_synthetic_valuation_data()
