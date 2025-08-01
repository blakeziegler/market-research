import os
import glob
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../../../../.env')

hf_api_key = os.getenv('HUGGING_FACE_API_KEY')

# Load Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B", token=hf_api_key)

def count_tokens_in_file(file_path):
    """Count tokens in a single text file using Qwen tokenizer."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Tokenize the text
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return 0

def main():
    """Count tokens in all .txt files in the current directory."""
    # Get all .txt files in the current directory
    txt_files = glob.glob("*.txt")
    
    if not txt_files:
        print("NO .TXT FILES FOUND IN CURRENT DIRECTORY!")
        return
    
    total_tokens = 0
    file_counts = []
    
    # Process each .txt file
    for file_path in sorted(txt_files):
        token_count = count_tokens_in_file(file_path)
        file_counts.append((file_path, token_count))
        total_tokens += token_count
        
        print(f"{file_path}: {token_count:,} tokens")
    
    print("-" * 60)
    print(f"TOTAL TOKENS ACROSS ALL FILES: {total_tokens:,}")
    
    print(f"Files processed: {len(txt_files)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per file: {total_tokens // len(txt_files):,}")

if __name__ == "__main__":
    main()
