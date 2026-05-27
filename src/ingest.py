"""Build the FAISS index from documents in data/docs/."""
from __future__ import annotations

import json

import faiss
from sentence_transformers import SentenceTransformer

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    DOCS_DIR,
    EMBED_MODEL,
    INDEX_DIR,
    INDEX_FILE,
)
from .loaders import load_documents


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window word chunking with overlap."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def build_index() -> int:
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(
            f"No documents found in {DOCS_DIR}. "
            "Add .txt / .md / .pdf files and re-run."
        )

    records = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            records.append({"source": doc["source"], "chunk_id": i, "text": chunk})

    print(f"Loaded {len(docs)} documents -> {len(records)} chunks")
    print(f"Embedding with {EMBED_MODEL} (first run downloads the model)...")
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(
        [r["text"] for r in records],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(records, indent=2))
    print(f"Index saved to {INDEX_FILE} ({index.ntotal} vectors)")
    return index.ntotal


if __name__ == "__main__":
    build_index()
