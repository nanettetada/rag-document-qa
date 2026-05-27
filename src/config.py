"""Central configuration for the RAG chatbot."""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# --- Embeddings + chunking -------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True
CHUNK_SIZE = 220           # words
CHUNK_OVERLAP = 40         # words
TOP_K_RETRIEVAL = 12       # over-fetch for reranker
TOP_K_FINAL = 5            # what the LLM sees

# --- Chat memory -----------------------------------------------------------
MAX_HISTORY_TURNS = 6      # how many prior user+assistant turns to keep in prompt
HISTORY_TOKEN_BUDGET = 1500

# --- LLM backend preference order -----------------------------------------
# First backend whose API key is present (or whose local runtime is reachable)
# is used. Set env vars to control which one runs.
BACKEND_PREFERENCE = ["groq", "anthropic", "openai", "ollama", "hf_local"]

MODEL_DEFAULTS = {
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
    "hf_local": os.getenv("HF_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
}

SYSTEM_PROMPT = (
    "You are a careful, friendly research assistant.\n"
    "\n"
    "RULES:\n"
    "1. Answer ONLY from the provided CONTEXT passages. Cite sources inline as [1], [2], etc.\n"
    "2. If the answer is not in the context, say so plainly and suggest what to look for.\n"
    "3. Use prior conversation turns for follow-up phrasing (\"that\", \"it\", \"how about...\") but ground each answer in the new context.\n"
    "4. Be concise. Bullet points for lists. Plain sentences otherwise.\n"
    "5. Never invent product names, numbers, or facts that are not in the context.\n"
)
