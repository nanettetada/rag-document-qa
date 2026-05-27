"""Top-level RAG orchestration + CLI.

Usage:
    python -m src.rag "What is the warranty period for the Aurora-X1?"
"""
from __future__ import annotations

import sys

from .generate import generate_answer
from .retrieve import retrieve


def ask(question: str, k: int = 4) -> dict:
    contexts = retrieve(question, k=k)
    result = generate_answer(question, contexts)
    return result


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if not argv:
        print('Usage: python -m src.rag "your question"')
        sys.exit(1)
    question = " ".join(argv)
    result = ask(question)
    print(f"\n[Backend: {result['backend']}]\n")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['id']}] {s['source']} (chunk {s['chunk_id']}, score {s['score']:.3f})")


if __name__ == "__main__":
    main()
