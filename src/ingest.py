"""Load documents, chunk them, embed with sentence-transformers, build a FAISS index."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    DOCS_DIR,
    EMBED_MODEL,
    INDEX_DIR,
    INDEX_FILE,
)


def read_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    docs = []
    for path in sorted(docs_dir.glob("**/*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


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
    docs = read_documents()
    if not docs:
        raise SystemExit(
            f"No documents found in {DOCS_DIR}. Add .txt or .md files and re-run."
        )

    records = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            records.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
            })

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
