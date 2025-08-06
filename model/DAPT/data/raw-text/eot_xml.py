import os
from pathlib import Path

# Set the working directory to the current script's directory
data_dir = Path(__file__).parent

# Control token to append
END_TOKEN = "<|endoftext|>"

# Iterate through all .txt files in the same directory
for file_path in data_dir.glob("*.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().rstrip()

    # Skip if the end token already exists
    if content.endswith(END_TOKEN):
        print(f"Already ends with token: {file_path.name}")
        continue

    # Append end token
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n" + END_TOKEN + "\n")
        print(f"➕ Added token to: {file_path.name}")