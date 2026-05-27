"""Generate an answer from a question + retrieved context.

Backends, in order of preference:
1. OpenAI (if OPENAI_API_KEY is set and `openai` is installed)
2. Anthropic (if ANTHROPIC_API_KEY is set and `anthropic` is installed)
3. Local extractive fallback (no API key needed) — returns the best-matching span
"""
from __future__ import annotations

import os
import textwrap
from typing import Any

from .config import SYSTEM_PROMPT


def _format_context(contexts: list[dict]) -> str:
    lines = []
    for i, c in enumerate(contexts, 1):
        lines.append(f"[{i}] (source: {c['source']})\n{c['text']}\n")
    return "\n".join(lines)


def _format_prompt(question: str, contexts: list[dict]) -> str:
    return textwrap.dedent(f"""\
        Use the following context to answer the question. Cite sources as [1], [2], etc.
        If the answer cannot be found in the context, say "I don't know based on the provided documents."

        CONTEXT:
        {_format_context(contexts)}

        QUESTION: {question}

        ANSWER:""")


def _try_openai(question: str, contexts: list[dict]) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_prompt(question, contexts)},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def _try_anthropic(question: str, contexts: list[dict]) -> str | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None
    client = Anthropic()
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _format_prompt(question, contexts)}],
    )
    return resp.content[0].text


def _extractive_fallback(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "I don't know based on the provided documents."
    best = contexts[0]
    snippet = best["text"]
    if len(snippet) > 600:
        snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
    return (
        f"(Extractive fallback — no LLM API key set.)\n\n"
        f"Most relevant passage from **{best['source']}**:\n\n{snippet}\n\n"
        f"[1] {best['source']}"
    )


def generate_answer(question: str, contexts: list[dict]) -> dict[str, Any]:
    answer = _try_openai(question, contexts) or _try_anthropic(question, contexts)
    backend = "llm" if answer else "extractive"
    if answer is None:
        answer = _extractive_fallback(question, contexts)
    return {
        "answer": answer,
        "backend": backend,
        "sources": [
            {"id": i + 1, "source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
            for i, c in enumerate(contexts)
        ],
    }
