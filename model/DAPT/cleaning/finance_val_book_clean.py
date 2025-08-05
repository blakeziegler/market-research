
import os
import sys
import re
from pathlib import Path
from typing import Optional
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from unidecode import unidecode
import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_directories() -> tuple[Path, Path, Path]:

    # script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    input_dir = project_root / "DAPT" / "data" / "valuation"
    chunking_input_dir = project_root / "DAPT" / "data" / "raw-text"
    output_dir = project_root / "DAPT" / "data" / "raw-text"
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, chunking_input_dir, output_dir

def get_pdf_file(input_dir: Path) -> Optional[Path]:

    pdf_file = input_dir / "finance_val_book.pdf"
    
    if pdf_file.exists():
        logger.info(f"FOUND PDF: {pdf_file}")
        return pdf_file
    else:
        logger.error(f"PDF NOT FOUND: {pdf_file}")
        return None

def clean_text(text: str) -> str:
    """
    Clean the PDF's into basic formatting
    - strip whitespace
    - normalize unicode
    - fix hyphenated line breaks
    - flatten tables
    - preserve casing for key terms
    - normalize whitespace while preserving structure
    - remove unwanted repeating strings
    """
    if not text:
        return ""
    
    # Remove unwanted repeating strings
    # 1. Remove Cengage Learning copyright notice
    copyright_pattern = r'Copyright 2012 Cengage Learning\. All Rights Reserved\. May not be copied, scanned, or duplicated, in whole or in part\. Due to electronic rights, some third party content may be suppressed from the eBook and/or eChapter\(s\)\.\s*Editorial review has deemed that any suppressed content does not materially affect the overall learning experience\. Cengage Learning reserves the right to remove additional content at any time if subsequent rights restrictions require it\.'
    text = re.sub(copyright_pattern, '', text, flags=re.MULTILINE | re.DOTALL)
    
    # 2. Remove (cid:129) patterns
    text = re.sub(r'\(cid:129\)', '', text)
    
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(lines)
    
    text = unidecode(text)
    
    # Pattern: word-\nword -> word-word
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    financial_terms = [
        'P/E', 'EPS', 'EBITDA', 'EBIT', 'ROE', 'ROA', 'ROI', 'PEG', 'P/B', 'P/S',
        'EV/EBITDA', 'EV/EBIT', 'DCF', 'D/E', 'WACC', 'CAPM', 'NPV', 'IRR',
        'GDP', 'CPI', 'PPI', 'FOMC', 'SEC', 'IRS', 'NASDAQ', 'NYSE', 'S&P',
        'DJIA', 'VIX', 'LIBOR', 'SOFR', 'TIPS', 'T-Bill', 'T-Bond', 'T-Note',
        'CDO', 'CDS', 'MBS', 'ABS', 'ETF', 'IPO', 'M&A', 'LBO', 'PE', 'B2C', 'CAGR', 'YoY', 'QoQ',
        'MoM', 'YTD', 'TTM', 'LTM', 'NTM', 'FY', 'Q1', 'Q2', 'Q3', 'Q4'
    ]
    
    term_mapping = {}
    for term in financial_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = pattern.findall(text)
        for match in matches:
            if match != term:
                term_mapping[match] = term
    
    # Apply the mapping to preserve correct casing
    for original, correct in term_mapping.items():
        text = text.replace(original, correct)
    
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(lines)
    
    return text

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract Text from pdf 
    """
    try:
        text = extract_text(str(pdf_path))
        
        if text:
            text = clean_text(text)
            
            return text
        else:
            logger.warning(f"NO TEXT EXTRACTED FROM {pdf_path.name}")
            return None
            
    except Exception as e:
        logger.error(f"ERROR EXTRACTING TEXT FROM {pdf_path.name}: {str(e)}")
        return None

def get_chunking_input_file(chunking_input_dir: Path) -> Optional[Path]:
    """
    Get the cleaned finance valuation book text file for chunking.
    """
    input_file = chunking_input_dir / "CLEAN_finance_val_book_raw.txt"
    
    if input_file.exists():
        logger.info(f"FOUND CHUNKING INPUT FILE: {input_file}")
        return input_file
    else:
        logger.error(f"CHUNKING INPUT FILE NOT FOUND: {input_file}")
        return None

def split_text_into_chunks(input_file: Path, output_dir: Path, base_filename: str) -> bool:
    try:
        # Read the existing cleaned text file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"READ {len(text)} CHARACTERS FROM INPUT FILE")
        
        # Split text by !BREAK! markers
        chunks = text.split('!BREAK!')
        
        # Remove empty chunks and strip whitespace
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        logger.info(f"FOUND {len(chunks)} TEXT CHUNKS")
        
        successful_saves = 0
        for i, chunk in enumerate(chunks, 1):
            # Generate chunk filename
            chunk_filename = f"{base_filename}_chunk_{i:03d}.txt"
            chunk_path = output_dir / chunk_filename
            
            try:
                # Write chunk to file
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                
                logger.info(f"SUCCESSFULLY SAVED: {chunk_filename}")
                successful_saves += 1
                
            except Exception as e:
                logger.error(f"ERROR SAVING {chunk_filename}: {str(e)}")
        
        logger.info(f"TOTAL CHUNKS SAVED: {successful_saves}/{len(chunks)}")
        return successful_saves > 0
        
    except Exception as e:
        logger.error(f"ERROR SPLITTING TEXT INTO CHUNKS: {str(e)}")
        return False

def convert_pdf_to_text(pdf_path: Path, output_dir: Path) -> bool:
    """
    Convert the finance valuation book PDF file to text and save it.
    """

    # Generate output filename
    pdf_name = pdf_path.stem
    output_filename = f"{pdf_name}_raw.txt"
    output_path = output_dir / output_filename
    
    text = extract_text_from_pdf(pdf_path)
    
    if text is None:
        return False
    
    try:
        # Write text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"SUCCESSFULLY SAVED: {output_filename}")
        return True
        
    except Exception as e:
        logger.error(f"ERROR SAVING {output_filename}: {str(e)}")
        return False

def chunk_existing_text(chunking_input_dir: Path, output_dir: Path) -> bool:
    """
    Chunk the existing cleaned text file into separate files.
    """
    input_file = get_chunking_input_file(chunking_input_dir)
    
    if input_file is None:
        return False
    
    logger.info("STARTING CHUNKING PROCESS")
    logger.info("=" * 50)
    
    base_filename = "finance_val_book"
    
    if split_text_into_chunks(input_file, output_dir, base_filename):
        logger.info("=" * 50)
        logger.info("CHUNKING COMPLETED SUCCESSFULLY")
        return True
    else:
        logger.info("=" * 50)
        logger.info("CHUNKING FAILED")
        return False

def main():
    """
    Main function to convert PDF to raw text and chunk the existing cleaned text.
    """
    # Directory setup
    input_dir, chunking_input_dir, output_dir = setup_directories()
    
    # First, convert PDF to raw text (without CLEAN_ prefix)
    if not input_dir.exists():
        logger.error(f"INPUT DIRECTORY NOT FOUND: {input_dir}")
        return
    
    pdf_file = get_pdf_file(input_dir)
    
    if pdf_file:
        logger.info("STARTING PDF CONVERSION PROCESS")
        logger.info("=" * 50)
        
        if convert_pdf_to_text(pdf_file, output_dir):
            logger.info("=" * 50)
            logger.info("SUCCESSFUL PDF CONVERSION")
        else:
            logger.info("=" * 50)
            logger.info("FAILED PDF CONVERSION")
    
    # Then, chunk the existing cleaned text file
    if not chunking_input_dir.exists():
        logger.error(f"CHUNKING INPUT DIRECTORY NOT FOUND: {chunking_input_dir}")
        return
    
    logger.info("STARTING CHUNKING PROCESS")
    logger.info("=" * 50)
    
    if chunk_existing_text(chunking_input_dir, output_dir):
        logger.info("=" * 50)
        logger.info("SUCCESSFUL CHUNKING")
    else:
        logger.info("=" * 50)
        logger.info("FAILED CHUNKING")

if __name__ == "__main__":
    main()
