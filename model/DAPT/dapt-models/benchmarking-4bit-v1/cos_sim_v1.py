import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pathlib import Path

df = pd.read_csv("results.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

corpus_dir = Path("../data/raw-text")

corpus_text = ""
for f in corpus_dir.glob("*.txt"):
    with open(f, "r", encoding="utf-8") as file:
        corpus_text += file.read() + "\n"

print("Total text length:", len(corpus_text))

corpus_embeddings = model.encode(corpus_text, show_progress_bar=True, convert_to_tensor=True)

similarity_score_base = []
similarity_score_dapt = []

print("Calculating similarity scores...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    base_emb = model.encode(row["base_output"], convert_to_tensor=True)
    dapt_emb = model.encode(row["dapt_output"], convert_to_tensor=True)

    sim_base = util.cos_sim(base_emb, corpus_embeddings).item()
    sim_dapt = util.cos_sim(dapt_emb, corpus_embeddings).item()

    similarity_score_base.append(sim_base)
    similarity_score_dapt.append(sim_dapt)

df["similarity_score_base"] = similarity_score_base
df["similarity_score_dapt"] = similarity_score_dapt

df.to_csv("results.csv", index=False)
print("Similarity scores saved to results.csv")