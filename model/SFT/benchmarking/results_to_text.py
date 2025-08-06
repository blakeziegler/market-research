import pandas as pd
import textwrap

def csv_to_text(csv_file, output_file, max_width=100):
    """Convert CSV results to human-readable text format"""
    
    print(f"Converting {csv_file} to {output_file}...")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"BENCHMARK RESULTS: {csv_file}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, row in df.iterrows():
            f.write(f"RESULT #{idx + 1}\n")
            f.write("-" * 40 + "\n")
            
            # Write prompt
            f.write("PROMPT:\n")
            prompt_text = str(row['prompt'])
            # Preserve original formatting, only wrap if line exceeds max_width
            lines = prompt_text.split('\n')
            for line in lines:
                if len(line) > max_width:
                    wrapped_lines = textwrap.wrap(line, width=max_width)
                    for wrapped_line in wrapped_lines:
                        f.write(f"  {wrapped_line}\n")
                else:
                    f.write(f"  {line}\n")
            f.write("\n")
            
            # Write model output
            output_col = None
            if 'base_output' in row:
                output_col = 'base_output'
                model_name = "BASE MODEL"
            elif 'sft_dapt_output' in row:
                output_col = 'sft_dapt_output'
                model_name = "SFT MODEL"
            
            if output_col and pd.notna(row[output_col]):
                f.write(f"{model_name} OUTPUT:\n")
                output_text = str(row[output_col])
                # Preserve original formatting, only wrap if line exceeds max_width
                lines = output_text.split('\n')
                for line in lines:
                    if len(line) > max_width:
                        wrapped_lines = textwrap.wrap(line, width=max_width)
                        for wrapped_line in wrapped_lines:
                            f.write(f"  {wrapped_line}\n")
                    else:
                        f.write(f"  {line}\n")
                f.write("\n")
            
            # Write evaluation scores if available
            if 'base_tone' in row and pd.notna(row['base_tone']):
                f.write("EVALUATION SCORES:\n")
                f.write(f"  Tone: {row['base_tone']:.1f}/10\n")
                f.write(f"  Accuracy: {row['base_accuracy']:.1f}/10\n")
                f.write(f"  Structure: {row['base_structure']:.1f}/10\n")
                f.write(f"  Hallucinated: {'Yes' if row['base_hallucinated'] else 'No'}\n")
                if pd.notna(row['base_justification']):
                    f.write("  Justification:\n")
                    justification_text = str(row['base_justification'])
                    # Preserve original formatting, only wrap if line exceeds max_width
                    lines = justification_text.split('\n')
                    for line in lines:
                        if len(line) > max_width - 4:  # Account for indentation
                            wrapped_lines = textwrap.wrap(line, width=max_width-4)
                            for wrapped_line in wrapped_lines:
                                f.write(f"    {wrapped_line}\n")
                        else:
                            f.write(f"    {line}\n")
                f.write("\n")
            elif 'sft_tone' in row and pd.notna(row['sft_tone']):
                f.write("EVALUATION SCORES:\n")
                f.write(f"  Tone: {row['sft_tone']:.1f}/10\n")
                f.write(f"  Accuracy: {row['sft_accuracy']:.1f}/10\n")
                f.write(f"  Structure: {row['sft_structure']:.1f}/10\n")
                f.write(f"  Hallucinated: {'Yes' if row['sft_hallucinated'] else 'No'}\n")
                if pd.notna(row['sft_justification']):
                    f.write("  Justification:\n")
                    justification_text = str(row['sft_justification'])
                    # Preserve original formatting, only wrap if line exceeds max_width
                    lines = justification_text.split('\n')
                    for line in lines:
                        if len(line) > max_width - 4:  # Account for indentation
                            wrapped_lines = textwrap.wrap(line, width=max_width-4)
                            for wrapped_line in wrapped_lines:
                                f.write(f"    {wrapped_line}\n")
                        else:
                            f.write(f"    {line}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"✓ Saved {output_file}")

def main():
    # Load both CSV files
    base_csv = "results_base.csv"
    sft_csv = "results_dapt_sft-v1.csv"
    
    if not os.path.exists(base_csv):
        print(f"Error: {base_csv} not found")
        return
    
    if not os.path.exists(sft_csv):
        print(f"Error: {sft_csv} not found")
        return
    
    # Read both CSV files
    base_df = pd.read_csv(base_csv)
    sft_df = pd.read_csv(sft_csv)
    
    # Create combined output
    output_file = "results_combined.txt"
    print(f"Creating combined results file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("BENCHMARK RESULTS: COMBINED COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        # Process each result pair
        for idx in range(len(base_df)):
            f.write(f"RESULT #{idx + 1}\n")
            f.write("=" * 60 + "\n\n")
            
            # Write prompt
            f.write("PROMPT:\n")
            prompt_text = str(base_df.iloc[idx]['prompt'])
            lines = prompt_text.split('\n')
            for line in lines:
                if len(line) > 100:
                    wrapped_lines = textwrap.wrap(line, width=100)
                    for wrapped_line in wrapped_lines:
                        f.write(f"  {wrapped_line}\n")
                else:
                    f.write(f"  {line}\n")
            f.write("\n")
            
            # Write base model output
            f.write("BASE MODEL OUTPUT:\n")
            base_output = str(base_df.iloc[idx]['base_output'])
            lines = base_output.split('\n')
            for line in lines:
                if len(line) > 100:
                    wrapped_lines = textwrap.wrap(line, width=100)
                    for wrapped_line in wrapped_lines:
                        f.write(f"  {wrapped_line}\n")
                else:
                    f.write(f"  {line}\n")
            f.write("\n")
            
            # Write base model evaluation
            if 'base_tone' in base_df.columns and pd.notna(base_df.iloc[idx]['base_tone']):
                f.write("BASE MODEL EVALUATION:\n")
                f.write(f"  Tone: {base_df.iloc[idx]['base_tone']:.1f}/10\n")
                f.write(f"  Accuracy: {base_df.iloc[idx]['base_accuracy']:.1f}/10\n")
                f.write(f"  Structure: {base_df.iloc[idx]['base_structure']:.1f}/10\n")
                f.write(f"  Hallucinated: {'Yes' if base_df.iloc[idx]['base_hallucinated'] else 'No'}\n")
                if pd.notna(base_df.iloc[idx]['base_justification']):
                    f.write("  Justification:\n")
                    justification_text = str(base_df.iloc[idx]['base_justification'])
                    lines = justification_text.split('\n')
                    for line in lines:
                        if len(line) > 96:  # Account for indentation
                            wrapped_lines = textwrap.wrap(line, width=96)
                            for wrapped_line in wrapped_lines:
                                f.write(f"    {wrapped_line}\n")
                        else:
                            f.write(f"    {line}\n")
                f.write("\n")
            
            # Write SFT model output
            f.write("SFT MODEL OUTPUT:\n")
            sft_output = str(sft_df.iloc[idx]['sft_dapt_output'])
            lines = sft_output.split('\n')
            for line in lines:
                if len(line) > 100:
                    wrapped_lines = textwrap.wrap(line, width=100)
                    for wrapped_line in wrapped_lines:
                        f.write(f"  {wrapped_line}\n")
                else:
                    f.write(f"  {line}\n")
            f.write("\n")
            
            # Write SFT model evaluation
            if 'sft_tone' in sft_df.columns and pd.notna(sft_df.iloc[idx]['sft_tone']):
                f.write("SFT MODEL EVALUATION:\n")
                f.write(f"  Tone: {sft_df.iloc[idx]['sft_tone']:.1f}/10\n")
                f.write(f"  Accuracy: {sft_df.iloc[idx]['sft_accuracy']:.1f}/10\n")
                f.write(f"  Structure: {sft_df.iloc[idx]['sft_structure']:.1f}/10\n")
                f.write(f"  Hallucinated: {'Yes' if sft_df.iloc[idx]['sft_hallucinated'] else 'No'}\n")
                if pd.notna(sft_df.iloc[idx]['sft_justification']):
                    f.write("  Justification:\n")
                    justification_text = str(sft_df.iloc[idx]['sft_justification'])
                    lines = justification_text.split('\n')
                    for line in lines:
                        if len(line) > 96:  # Account for indentation
                            wrapped_lines = textwrap.wrap(line, width=96)
                            for wrapped_line in wrapped_lines:
                                f.write(f"    {wrapped_line}\n")
                        else:
                            f.write(f"    {line}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"✓ Saved {output_file}")
    print("\nConversion completed!")

if __name__ == "__main__":
    import os
    main()
