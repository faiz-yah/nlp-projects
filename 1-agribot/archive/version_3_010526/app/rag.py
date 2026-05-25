# app/rag.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = SentenceTransformer(EMBED_MODEL_ID)

def load_kb(index_path="faiss_index.bin", docs_path="kb_docs.pkl"):
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    return index, docs

def retrieve(question: str, index, docs, top_k=3, threshold=0.30):
    q_emb = embed_model.encode([question], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_emb, dtype="float32"), top_k)
    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold and idx < len(docs):
            chunks.append({**docs[idx], "score": float(score)})
    return chunks

def build_rag_prompt(question: str, chunks: list) -> str:
    context = "\n".join([c["text"] for c in chunks])
    return f"Context: {context}\n\nQuestion: {question}\nAnswer:"