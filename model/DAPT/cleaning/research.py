#!/usr/bin/env python3
"""
PDF to Text Converter for Analyst Reports
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from unidecode import unidecode
from tqdm import tqdm
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

def setup_directories() -> tuple[Path, Path]:
    """
    Setup input and output directories.
    """
    # script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # input and output directories
    input_dir = project_root / "DAPT" / "data" / "research-2"
    output_dir = project_root / "DAPT" / "data" / "raw-text"
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir

def get_pdf_files(input_dir: Path) -> List[Path]:
    """
    Get all PDF files from the input directory.
    """

    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"PDF's found {len(pdf_files)} in directory:{input_dir}")
    return pdf_files

def clean_text(text: str) -> str:
    """
    Clean the PDF's into basic formatting
    - strip whitespace
    - normalize unicode
    - fix hyphenated line breaks
    - flatten tables
    - preserve casing for key terms
    - normalize whitespace while preserving structure
    """
    if not text:
        return ""
    
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

def convert_pdf_to_text(pdf_path: Path, output_dir: Path) -> bool:
    """
    Convert a single PDF file to text and save it.
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
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR SAVING {output_filename}: {str(e)}")
        return False

def main():
    """
    Main function to convert all PDF files to text.
    """
    # Directory setup
    input_dir, output_dir = setup_directories()
    
    if not input_dir.exists():
        logger.error(f"INPUT DIRECTORY NOT FOUND: {input_dir}")
        return
    
    pdf_files = get_pdf_files(input_dir)
    
    if not pdf_files:
        return
    
    successful_conversions = 0
    failed_conversions = 0
    
    logger.info("STARTING CONVERSION PROCESS")
    logger.info("=" * 50)
    
    for pdf_file in tqdm(pdf_files, desc="Converting PDFs", unit="file"):
        if convert_pdf_to_text(pdf_file, output_dir):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"SUCCESSFUL CONVERSIONS: {successful_conversions}")
    logger.info(f"FAILED CONVERSIONS: {failed_conversions}")

if __name__ == "__main__":
    main()
