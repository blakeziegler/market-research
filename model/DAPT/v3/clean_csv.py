#!/usr/bin/env python3
"""
Script to clean CSV files with encoding issues
"""

import pandas as pd
import os

def clean_csv_file(input_file, output_file=None):
    """
    Clean a CSV file by handling encoding issues and problematic characters
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Cleaning {input_file}...")
    
    try:
        # Try to read with ISO-8859-1 encoding first
        df = pd.read_csv(input_file, encoding='iso-8859-1')
        print("Successfully read with ISO-8859-1 encoding")
    except Exception as e:
        print(f"Error reading with ISO-8859-1: {e}")
        try:
            # Fallback: read as binary and clean
            with open(input_file, 'rb') as f:
                content = f.read()
            
            # Replace common problematic characters
            replacements = {
                b'\xd5': b"'",  # Curly apostrophe
                b'\xd4': b"'",  # Curly single quote
                b'\xd3': b'"',  # Curly double quote
                b'\xd2': b'"',  # Curly double quote
                b'\x96': b'-',  # En dash
                b'\x97': b'-',  # Em dash
                b'\x85': b'...',  # Ellipsis
            }
            
            for old, new in replacements.items():
                content = content.replace(old, new)
            
            # Write cleaned content to temporary file
            temp_file = f"temp_{input_file}"
            with open(temp_file, 'wb') as f:
                f.write(content)
            
            # Read the cleaned file
            df = pd.read_csv(temp_file)
            
            # Clean up temporary file
            os.remove(temp_file)
            print("Successfully cleaned and read file")
            
        except Exception as e2:
            print(f"Error with fallback method: {e2}")
            return False
    
    # Save the cleaned DataFrame
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Cleaned file saved as {output_file}")
    return True

if __name__ == "__main__":
    # Clean the benchmark file
    clean_csv_file("benchmark_v2.csv") 