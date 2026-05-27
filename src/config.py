"""Central configuration for the RAG pipeline."""
from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 220          # words
CHUNK_OVERLAP = 40        # words
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the provided "
    "context. If the answer is not in the context, say you don't know. "
    "Be concise and cite the source IDs in square brackets like [1], [2]."
)
