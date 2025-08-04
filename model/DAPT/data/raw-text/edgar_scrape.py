from edgar import Company, set_identity
import os
import re

USER_AGENT = "zieglerblake2@gmail.com"

# Set identity for SEC compliance
set_identity(USER_AGENT)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "V", "JNJ", "WMT", "PG",
    "NVDA", "TSLA", "DIS", "BAC", "XOM",
    "KO", "VZ", "PEP", "CSCO", "NFLX"
]

OUT_DIR = "10k_texts"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_10k_text(raw_text: str) -> str:
    """
    Clean 10-K text for optimal domain adaptive pre-training by removing
    useless sections, headers, footers, and metadata. Start from PART I.
    """
    text = raw_text
    
    # Find the start of PART I section (handles both "PART I" and "Part I")
    part_i_pattern = r"║\s*(?:PART I|Part I)\s*║"
    part_i_match = re.search(part_i_pattern, text)
    
    if part_i_match:
        # Start from the beginning of the PART I section
        start_pos = part_i_match.start()
        text = text[start_pos:]
        
        # Find the end of PART I (start of PART II or end of document)
        part_ii_pattern = r"║\s*(?:PART II|Part II)\s*║"
        part_ii_match = re.search(part_ii_pattern, text)
        
        if part_ii_match:
            # End at the start of PART II
            end_pos = part_ii_match.start()
            text = text[:end_pos]
    
    # Remove excessive whitespace and normalize
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    # Remove decorative boxes and borders around PART I
    text = re.sub(r"╔═ § ═══════════════════════════════════════════════════════════════════════════════════════════════╗", "", text)
    text = re.sub(r"║ ║", "", text)
    text = re.sub(r"╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝", "", text)
    text = re.sub(r"╭─  ───────────────────────────────────────────────────────────────────────────────────────────────╮", "", text)
    text = re.sub(r"│ │", "", text)
    text = re.sub(r"╰───────────────────────────────────────────────────────────────────────────────────────────────────╯", "", text)
    
    # Remove the PART I header itself (handles both variations)
    text = re.sub(r"║\s*(?:PART I|Part I)\s*║", "", text)
    
    # Remove any remaining decorative lines and borders
    text = re.sub(r"^[─_=]{10,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[─_=]{3,}.*[─_=]{3,}$", "", text, flags=re.MULTILINE)
    
    # Remove empty lines and excessive spacing
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
    
    # Remove lines that are just numbers or single characters
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[A-Za-z]\s*$", "", text, flags=re.MULTILINE)
    
    # Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    text = text.strip()
    
    return text

def fetch_10k_texts_for_ticker(ticker):
    print(f"Processing {ticker} …")
    c = Company(ticker)
    filings = c.get_filings(form="10-K")
    
    # Get the 5 most recent filings
    recent_filings = list(filings.head(5))
    
    for idx, filing in enumerate(recent_filings, start=1):
        try:
            raw_text = filing.text()
            clean_text = clean_10k_text(raw_text)
            
            # Skip if the cleaned text is too short (likely just metadata)
            if len(clean_text.strip()) < 1000:
                print(f"  Skipping filing {filing.accession_number}: cleaned text too short")
                continue
                
        except Exception as e:
            print(f"  Failed to get text for filing {filing.accession_number}: {e}")
            continue

        fname = os.path.join(OUT_DIR, f"{ticker}_10K_{idx}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"  Saved {fname} ({len(clean_text)} characters)")

def main():
    for ticker in TICKERS:
        fetch_10k_texts_for_ticker(ticker)

if __name__ == "__main__":
    main()