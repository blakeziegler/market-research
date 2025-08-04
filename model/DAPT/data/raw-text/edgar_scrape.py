from edgar import Company, set_identity
import os
import re

USER_AGENT = "zieglerblake2@gmail.com"

# Set identity for SEC compliance
set_identity(USER_AGENT)

TICKERS = [
    "MSFT", "NVDA", "COST", "BRK.B", "V", "JPM", "PG", "MCD", "T",
    "UBER", "IBM", "TXN", "PFE", "COF", "PANW", "GOOG", "SHW", "APO", "CMG"
]

OUT_DIR = "10k_texts"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_10k_sections(raw_text: str) -> dict:
    """
    Extract critical sections from 10-K text for domain adaptive pre-training.
    Returns a dictionary with Items 1, 1A, 7, 7A, and 8 as keys.
    """
    # Normalize text for consistent processing
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Define the sections we want to extract with more flexible patterns
    target_sections = {
        'item_1': [
            r'Item 1\.?\s*Business',
            r'ITEM 1\.?\s*Business',
            r'Item 1\.?\s*$',
            r'ITEM 1\.?\s*$',
            r'Item 1\s*[:\-—]',
            r'ITEM 1\s*[:\-—]'
        ],
        'item_1a': [
            r'Item 1A\.?\s*Risk Factors',
            r'ITEM 1A\.?\s*Risk Factors',
            r'Item 1A\.?\s*$',
            r'ITEM 1A\.?\s*$',
            r'Item 1A\s*[:\-—]',
            r'ITEM 1A\s*[:\-—]'
        ],
        'item_7': [
            r'Item 7\.?\s*Management',
            r'ITEM 7\.?\s*Management',
            r'Item 7\.?\s*$',
            r'ITEM 7\.?\s*$',
            r'Item 7\s*[:\-—]',
            r'ITEM 7\s*[:\-—]'
        ],
        'item_7a': [
            r'Item 7A\.?\s*Quantitative',
            r'ITEM 7A\.?\s*Quantitative',
            r'Item 7A\.?\s*$',
            r'ITEM 7A\.?\s*$',
            r'Item 7A\s*[:\-—]',
            r'ITEM 7A\s*[:\-—]'
        ],
        'item_8': [
            r'Item 8\.?\s*Financial',
            r'ITEM 8\.?\s*Financial',
            r'Item 8\.?\s*$',
            r'ITEM 8\.?\s*$',
            r'Item 8\s*[:\-—]',
            r'ITEM 8\s*[:\-—]'
        ]
    }
    
    # Define the next items that mark the end of each section
    section_end_markers = {
        'item_1': ['Item 1A', 'ITEM 1A', 'Item 2', 'ITEM 2'],
        'item_1a': ['Item 1B', 'ITEM 1B', 'Item 2', 'ITEM 2'],
        'item_7': ['Item 7A', 'ITEM 7A', 'Item 8', 'ITEM 8'],
        'item_7a': ['Item 8', 'ITEM 8'],
        'item_8': ['Item 9', 'ITEM 9', 'Item 9A', 'ITEM 9A']
    }
    
    extracted_sections = {}
    
    for section_key, patterns in target_sections.items():
        section_content = None
        
        # Try to find the section start with any of the patterns
        section_start = None
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                section_start = match.start()
                break
        
        if section_start is not None:
            # Find the end of this section by looking for the next item
            section_end = len(text)
            end_markers = section_end_markers[section_key]
            
            for end_marker in end_markers:
                # Look for the next item after the current section start
                end_pattern = rf"(?:^|\n)\s*{re.escape(end_marker)}\s*[:\-—\.]?\s*(?:[A-Za-z\s]+)?\s*(?:\n|$)"
                end_match = re.search(end_pattern, text[section_start + 100:], re.MULTILINE | re.IGNORECASE)
                if end_match:
                    potential_end = section_start + 100 + end_match.start()
                    if potential_end < section_end:
                        section_end = potential_end
                    break
            
            # Extract the section content
            section_content = text[section_start:section_end].strip()
            
            # Clean the extracted content
            if section_content:
                section_content = clean_section_content(section_content)
        
        extracted_sections[section_key] = section_content
    
    return extracted_sections

def clean_section_content(content: str) -> str:
    """
    Clean extracted section content by removing excessive whitespace and formatting artifacts.
    """
    # Remove excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'[ \t]{2,}', ' ', content)
    
    # Remove Table of Contents sections
    content = re.sub(r'(?i)table of contents.*?(?=\n\n|\n[A-Z]|\nItem|\nITEM)', '', content, flags=re.DOTALL | re.MULTILINE)
    content = re.sub(r'(?i)contents.*?(?=\n\n|\n[A-Z]|\nItem|\nITEM)', '', content, flags=re.DOTALL | re.MULTILINE)
    content = re.sub(r'(?i)index.*?(?=\n\n|\n[A-Z]|\nItem|\nITEM)', '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Remove decorative borders and boxes
    content = re.sub(r'╔═.*╗', '', content, flags=re.MULTILINE)
    content = re.sub(r'║.*║', '', content, flags=re.MULTILINE)
    content = re.sub(r'╚═.*╝', '', content, flags=re.MULTILINE)
    content = re.sub(r'╭─.*╮', '', content, flags=re.MULTILINE)
    content = re.sub(r'│.*│', '', content, flags=re.MULTILINE)
    content = re.sub(r'╰─.*╯', '', content, flags=re.MULTILINE)
    
    # Remove page headers/footers
    content = re.sub(r'(?i)page \d+ of \d+', '', content)
    content = re.sub(r'(?i)united states securities and exchange commission.*', '', content, flags=re.MULTILINE)
    content = re.sub(r'(?i)washington, d\.c\. 20549.*', '', content, flags=re.MULTILINE)
    
    # Remove company name headers that appear throughout
    content = re.sub(r'^\s*[A-Za-z\s&.,]+ \| \d{4} Form 10-K \| \d+\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*[A-Za-z\s&.,]+ \| \d{4} Form 10-K/A \| \d+\s*$', '', content, flags=re.MULTILINE)
    
    # Remove decorative lines
    content = re.sub(r'^[─_=]{10,}$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^[─_=]{3,}.*[─_=]{3,}$', '', content, flags=re.MULTILINE)
    
    # Remove empty lines and excessive spacing
    content = re.sub(r'\n\s*\n', '\n', content)
    content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)
    
    # Remove lines that are just numbers or single characters
    content = re.sub(r'^\d+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^[A-Za-z]\s*$', '', content, flags=re.MULTILINE)
    
    # Final cleanup
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    
    return content

def clean_10k_text(raw_text: str) -> str:
    """
    Extract and combine critical 10-K sections for domain adaptive pre-training.
    """
    # Extract the target sections
    sections = extract_10k_sections(raw_text)
    
    # Combine all found sections into a single text
    combined_text = ""
    
    # Define the order and labels for the sections
    section_order = [
        ('item_1', 'ITEM 1: BUSINESS'),
        ('item_1a', 'ITEM 1A: RISK FACTORS'),
        ('item_7', 'ITEM 7: MANAGEMENT\'S DISCUSSION AND ANALYSIS'),
        ('item_7a', 'ITEM 7A: QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK'),
        ('item_8', 'ITEM 8: FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA')
    ]
    
    for section_key, section_label in section_order:
        if sections[section_key]:
            combined_text += f"\n\n{section_label}\n"
            combined_text += "=" * len(section_label) + "\n\n"
            combined_text += sections[section_key]
            combined_text += "\n"
    
    return combined_text.strip()

def fetch_10k_texts_for_ticker(ticker):
    print(f"Processing {ticker} …")
    c = Company(ticker)
    filings = c.get_filings(form="10-K")
    
    # Get the 5 most recent filings
    recent_filings = list(filings.head(1))
    
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