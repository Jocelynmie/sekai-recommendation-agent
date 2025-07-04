import pandas as pd, numpy as np, faiss, pathlib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
MODEL_NAME = "all-MiniLM-L6-v2"      # 40 MB，CPU 友好
EMB_PATH  = DATA_DIR / "contents_embeddings.npy"
IDX_PATH  = DATA_DIR / "faiss_index.bin"

print("\U0001F4E5  Loading contents.csv …")
df = pd.read_csv(RAW_DIR / "contents.csv")   # 至少含 title/intro 字段
texts = (df["title"].fillna("") + " " + df["intro"].fillna("")).tolist()

print(f"\U0001F9E0  Encoding {len(texts)} texts with {MODEL_NAME} …")
model = SentenceTransformer(MODEL_NAME)
emb = model.encode(
    texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
).astype("float32")

print("\U0001F4BE  Saving embeddings …")
np.save(EMB_PATH, emb)

print("\U0001F527  Building FAISS HNSW index …")
dim = emb.shape[1]
index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200
index.add(emb)
faiss.write_index(index, str(IDX_PATH))
print(f"\u2705  Done. Index size = {index.ntotal:,}") 