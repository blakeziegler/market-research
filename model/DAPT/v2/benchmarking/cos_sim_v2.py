import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pathlib import Path

# Load both CSV files
df_base = pd.read_csv("results_base_v2.csv")
df_dapt = pd.read_csv("results_dapt_v2.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

corpus_dir = Path("../data/raw-text")

corpus_text = ""
for f in corpus_dir.glob("*.txt"):
    with open(f, "r", encoding="utf-8") as file:
        corpus_text += file.read() + "\n"

print("Total text length:", len(corpus_text))

corpus_embeddings = model.encode(corpus_text, show_progress_bar=True, convert_to_tensor=True)

# Calculate similarity scores for base model
similarity_score_base = []

print("Calculating similarity scores for base model...")
for _, row in tqdm(df_base.iterrows(), total=len(df_base)):
    base_emb = model.encode(row["base_output"], convert_to_tensor=True)
    sim_base = util.cos_sim(base_emb, corpus_embeddings).item()
    similarity_score_base.append(sim_base)

df_base["similarity_score_base"] = similarity_score_base
df_base.to_csv("results_base_v2.csv", index=False)
print("Base model similarity scores saved to results_base_v2.csv")

# Calculate similarity scores for DAPT model
similarity_score_dapt = []

print("Calculating similarity scores for DAPT model...")
for _, row in tqdm(df_dapt.iterrows(), total=len(df_dapt)):
    dapt_emb = model.encode(row["dapt_output"], convert_to_tensor=True)
    sim_dapt = util.cos_sim(dapt_emb, corpus_embeddings).item()
    similarity_score_dapt.append(sim_dapt)

df_dapt["similarity_score_dapt"] = similarity_score_dapt
df_dapt.to_csv("results_dapt_v2.csv", index=False)
print("DAPT model similarity scores saved to results_dapt_v2.csv")