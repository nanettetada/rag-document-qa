"""High-level RAG chatbot orchestration + CLI.

Usage (single-turn):
    python -m src.rag "What's the warranty on the X1?"

Usage (multi-turn REPL):
    python -m src.rag

Set GROQ_API_KEY (free tier at console.groq.com) for the default backend.
Or any of: ANTHROPIC_API_KEY, OPENAI_API_KEY. Or install ollama / transformers
for fully local generation.
"""
from __future__ import annotations

import sys
from typing import Iterator

from .config import MAX_HISTORY_TURNS, TOP_K_FINAL
from .generate import available_backends, pick_backend, stream_answer
from .retrieve import retrieve


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Keep the last `max_turns` user+assistant turn pairs."""
    if not history:
        return []
    # Each turn = 1 user + 1 assistant message. Keep the most recent `max_turns` pairs.
    return history[-(max_turns * 2):]


def ask_stream(question: str,
               history: list[dict] | None = None,
               k: int = TOP_K_FINAL,
               backend: str | None = None) -> tuple[Iterator[str], list[dict], str]:
    """Stream an answer. Returns (token_iterator, contexts, backend_name)."""
    contexts = retrieve(question, k_final=k)
    history = trim_history(history or [])
    bk = pick_backend(backend)
    iterator = stream_answer(question, contexts, history, backend=bk)
    return iterator, contexts, bk


def ask(question: str,
        history: list[dict] | None = None,
        k: int = TOP_K_FINAL,
        backend: str | None = None) -> dict:
    """Non-streaming convenience version."""
    iterator, contexts, bk = ask_stream(question, history, k, backend)
    answer = "".join(iterator)
    return {
        "answer": answer,
        "backend": bk,
        "contexts": contexts,
        "sources": [
            {"id": i + 1, "source": c["source"], "chunk_id": c["chunk_id"],
             "score": c.get("score", 0.0)}
            for i, c in enumerate(contexts)
        ],
    }


def _print_sources(contexts: list[dict]) -> None:
    print("\nSources:")
    for i, c in enumerate(contexts, 1):
        print(f"  [{i}] {c['source']} (chunk {c['chunk_id']}, score {c.get('score', 0):.3f})")


def repl() -> None:
    avail = available_backends()
    if not avail:
        print("ERROR: no LLM backend available. Set GROQ_API_KEY (free tier) "
              "or ANTHROPIC_API_KEY / OPENAI_API_KEY, or install ollama / transformers.")
        sys.exit(1)
    print(f"RAG chatbot (backend: {avail[0]}). Ctrl-C to quit.\n")
    history: list[dict] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        iterator, contexts, bk = ask_stream(q, history)
        print("bot> ", end="", flush=True)
        answer_parts = []
        for token in iterator:
            print(token, end="", flush=True)
            answer_parts.append(token)
        print()
        _print_sources(contexts)
        history.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": "".join(answer_parts)},
        ])


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if not argv:
        repl()
        return
    question = " ".join(argv)
    iterator, contexts, bk = ask_stream(question)
    print(f"[Backend: {bk}]\n")
    for token in iterator:
        print(token, end="", flush=True)
    print()
    _print_sources(contexts)


if __name__ == "__main__":
    main()
