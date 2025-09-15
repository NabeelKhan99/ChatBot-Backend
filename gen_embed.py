import os
import torch
from sentence_transformers import SentenceTransformer
import pickle

knowledge_dir = "knowledge_base"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

knowledge_texts = []
knowledge_files = []

if os.path.exists(knowledge_dir):
    for fname in os.listdir(knowledge_dir):
        if fname.endswith(".txt"):
            text = open(os.path.join(knowledge_dir, fname), "r", encoding="utf-8").read().strip()
            if text:
                knowledge_texts.append(text)
                knowledge_files.append(fname)

# Compute embeddings
knowledge_embeddings = embedder.encode(knowledge_texts, convert_to_tensor=True)

# Save embeddings + filenames
with open("knowledge_embeddings.pkl", "wb") as f:
    pickle.dump({
        "texts": knowledge_texts,
        "files": knowledge_files,
        "embeddings": knowledge_embeddings.cpu()  # move to CPU for saving
    }, f)

print(f"✅ Saved {len(knowledge_texts)} embeddings to knowledge_embeddings.pkl")
