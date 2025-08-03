import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_CSV_PATH = "results_base_v2.csv"
DAPT_CSV_PATH = "results_dapt_v2.csv"
CORPUS_DIR = Path("../data/raw-text")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BASE_COL = "base_output"
DAPT_COL = "dapt_output"

# ---------------- LOAD MODEL ----------------
print("🔄 Loading sentence transformer...")
model = SentenceTransformer(EMBED_MODEL_NAME)

# ---------------- LOAD CORPUS ----------------
print("📚 Loading corpus documents...")
corpus_text = ""
for file in CORPUS_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        corpus_text += f.read() + "\n"

print(f"✅ Corpus loaded (length={len(corpus_text):,} characters)")
corpus_embedding = model.encode(corpus_text, convert_to_tensor=True)

# ---------------- UTILS ----------------
def is_valid_text(val) -> bool:
    return isinstance(val, str) and val.strip() != ""

def safe_encode(text: str):
    return model.encode(text, convert_to_tensor=True)

def compute_similarity(row_text: str) -> float:
    try:
        if is_valid_text(row_text):
            emb = safe_encode(row_text)
            return util.cos_sim(emb, corpus_embedding).item()
    except Exception as e:
        print(f"[Error] Encoding failed: {e}")
    return float("nan")

# ---------------- PROCESS BASE MODEL ----------------
print("📊 Processing base model outputs...")
df_base = pd.read_csv(BASE_CSV_PATH)
df_base = df_base[df_base[BASE_COL].notna()]

similarity_scores_base = []
for _, row in tqdm(df_base.iterrows(), total=len(df_base)):
    sim_score = compute_similarity(str(row[BASE_COL]))
    similarity_scores_base.append(sim_score)

df_base["similarity_score_base"] = similarity_scores_base
df_base.to_csv(BASE_CSV_PATH, index=False)
print(f"✅ Saved updated base results to {BASE_CSV_PATH}")

# ---------------- PROCESS DAPT MODEL ----------------
print("📊 Processing DAPT model outputs...")
df_dapt = pd.read_csv(DAPT_CSV_PATH)
df_dapt = df_dapt[df_dapt[DAPT_COL].notna()]

similarity_scores_dapt = []
for _, row in tqdm(df_dapt.iterrows(), total=len(df_dapt)):
    sim_score = compute_similarity(str(row[DAPT_COL]))
    similarity_scores_dapt.append(sim_score)

df_dapt["similarity_score_dapt"] = similarity_scores_dapt
df_dapt.to_csv(DAPT_CSV_PATH, index=False)
print(f"✅ Saved updated DAPT results to {DAPT_CSV_PATH}")
