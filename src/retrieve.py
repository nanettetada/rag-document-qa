"""Query the FAISS index for top-k relevant chunks."""
from __future__ import annotations

import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CHUNKS_FILE, EMBED_MODEL, INDEX_FILE, TOP_K


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _load_index():
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"No index at {INDEX_FILE}. Run `python -m src.ingest` first."
        )
    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(CHUNKS_FILE.read_text())
    return index, chunks


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    index, chunks = _load_index()
    model = _load_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results
