import os
import pandas as pd
import glob

def combine_csv_files():
    # Find all CSV files in the directory
    csv_files = glob.glob('*.csv')
    csv_files = [f for f in csv_files if 'combined' not in f.lower()]
    
    all_dataframes = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, quoting=1)
            if 'user' in df.columns and 'assistant' in df.columns:
                all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if not all_dataframes:
        print("No valid CSV files found!")
        return
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv('combined_sft-260.csv', index=False, quoting=1)
    
    print(f"Combined {len(csv_files)} files into combined_sft-260.csv ({len(combined_df)} total rows)")

if __name__ == "__main__":
    combine_csv_files()
