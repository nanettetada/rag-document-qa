"""Two-stage retrieval: FAISS embedding search + optional cross-encoder rerank."""
from __future__ import annotations

import json
from functools import lru_cache

from .config import (
    CHUNKS_FILE,
    EMBED_MODEL,
    INDEX_FILE,
    RERANKER_MODEL,
    TOP_K_FINAL,
    TOP_K_RETRIEVAL,
    USE_RERANKER,
)


@lru_cache(maxsize=1)
def _embedder():
    # Imported lazily — sentence-transformers pulls in torch and can take
    # tens of seconds to import, which would otherwise freeze app startup.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


@lru_cache(maxsize=1)
def _index_and_chunks():
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"No index at {INDEX_FILE}. Run `python -m src.ingest` first."
        )
    import faiss  # lazy — faiss import is a few seconds; keep it off the startup path
    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(CHUNKS_FILE.read_text())
    return index, chunks


def retrieve(query: str,
             k_final: int = TOP_K_FINAL,
             k_retrieval: int = TOP_K_RETRIEVAL,
             use_reranker: bool = USE_RERANKER) -> list[dict]:
    """Return top-k passages for `query`, with reranking if enabled."""
    index, chunks = _index_and_chunks()

    q_vec = _embedder().encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_vec, k_retrieval)

    candidates = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["embed_score"] = float(score)
        candidates.append(chunk)

    if use_reranker and candidates:
        try:
            ce = _reranker()
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = ce.predict(pairs).tolist()
            for c, s in zip(candidates, rerank_scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        except Exception:
            # Reranker is optional — if it fails, fall back to embedding order.
            for c in candidates:
                c.setdefault("rerank_score", c["embed_score"])

    final = candidates[:k_final]
    for c in final:
        c["score"] = c.get("rerank_score", c["embed_score"])
    return final
