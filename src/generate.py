"""Multi-backend chat generation with streaming.

Backends supported, in preference order (configurable via BACKEND_PREFERENCE):
1. Groq        — free tier, fast inference, default
2. Anthropic   — Claude
3. OpenAI      — GPT
4. Ollama      — local server (if running)
5. HF local    — small transformers model on CPU/GPU (offline fallback)

The first backend whose credentials/runtime is available wins.
"""
from __future__ import annotations

import os
from typing import Any, Iterator

from .config import BACKEND_PREFERENCE, MODEL_DEFAULTS, SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Context formatting + prompt construction
# --------------------------------------------------------------------------- #
def format_context(contexts: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(f"[{i}] (source: {c['source']}, chunk {c['chunk_id']})\n{c['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str,
                   contexts: list[dict],
                   history: list[dict] | None = None) -> list[dict]:
    """Build OpenAI-style messages with conversation history + retrieved context."""
    history = history or []
    user_with_context = (
        f"CONTEXT PASSAGES:\n{format_context(contexts)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer the question above using only the context. Cite sources inline as [1], [2]."
    )
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_with_context}]
    )


# --------------------------------------------------------------------------- #
# Backend availability checks
# --------------------------------------------------------------------------- #
def _has(env_var: str) -> bool:
    return bool(os.getenv(env_var))


def available_backends() -> list[str]:
    out = []
    for b in BACKEND_PREFERENCE:
        if b == "groq" and _has("GROQ_API_KEY"):
            try:
                import groq  # noqa: F401
                out.append("groq")
            except ImportError:
                pass
        elif b == "anthropic" and _has("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                out.append("anthropic")
            except ImportError:
                pass
        elif b == "openai" and _has("OPENAI_API_KEY"):
            try:
                import openai  # noqa: F401
                out.append("openai")
            except ImportError:
                pass
        elif b == "ollama":
            try:
                import ollama  # noqa: F401
                # If ollama is installed we assume a local server may be reachable.
                out.append("ollama")
            except ImportError:
                pass
        elif b == "hf_local":
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401
                out.append("hf_local")
            except ImportError:
                pass
    return out


# --------------------------------------------------------------------------- #
# Streaming generators
# --------------------------------------------------------------------------- #
def _stream_groq(messages: list[dict]) -> Iterator[str]:
    from groq import Groq
    client = Groq()
    stream = client.chat.completions.create(
        model=MODEL_DEFAULTS["groq"],
        messages=messages,
        temperature=0.2,
        max_tokens=800,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_anthropic(messages: list[dict]) -> Iterator[str]:
    from anthropic import Anthropic
    client = Anthropic()
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    convo = [m for m in messages if m["role"] != "system"]
    with client.messages.stream(
        model=MODEL_DEFAULTS["anthropic"],
        max_tokens=800,
        system=system,
        messages=convo,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _stream_openai(messages: list[dict]) -> Iterator[str]:
    from openai import OpenAI
    client = OpenAI()
    stream = client.chat.completions.create(
        model=MODEL_DEFAULTS["openai"],
        messages=messages,
        temperature=0.2,
        max_tokens=800,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_ollama(messages: list[dict]) -> Iterator[str]:
    import ollama
    stream = ollama.chat(
        model=MODEL_DEFAULTS["ollama"],
        messages=messages,
        stream=True,
        options={"temperature": 0.2, "num_predict": 800},
    )
    for chunk in stream:
        delta = chunk.get("message", {}).get("content")
        if delta:
            yield delta


# HuggingFace local: lazy global to avoid loading on import.
_hf_pipe = None


def _stream_hf_local(messages: list[dict]) -> Iterator[str]:
    global _hf_pipe
    if _hf_pipe is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        import torch
        name = MODEL_DEFAULTS["hf_local"]
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _hf_pipe = (tok, model, TextIteratorStreamer)
    from threading import Thread
    tok, model, TextIteratorStreamer = _hf_pipe
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs, max_new_tokens=600, do_sample=True,
        temperature=0.3, top_p=0.9, streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for token in streamer:
        yield token


_STREAMERS = {
    "groq": _stream_groq,
    "anthropic": _stream_anthropic,
    "openai": _stream_openai,
    "ollama": _stream_ollama,
    "hf_local": _stream_hf_local,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def pick_backend(prefer: str | None = None) -> str:
    """Choose a backend. Honour `prefer` if available; otherwise first available."""
    avail = available_backends()
    if not avail:
        raise RuntimeError(
            "No LLM backend available. Set one of: GROQ_API_KEY, "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY — or install ollama / transformers."
        )
    if prefer and prefer in avail:
        return prefer
    return avail[0]


def stream_answer(question: str,
                  contexts: list[dict],
                  history: list[dict] | None = None,
                  backend: str | None = None) -> Iterator[str]:
    """Yield answer tokens for `question` grounded in `contexts`, given chat `history`."""
    backend = pick_backend(backend)
    messages = build_messages(question, contexts, history)
    yield from _STREAMERS[backend](messages)


def generate_answer(question: str,
                    contexts: list[dict],
                    history: list[dict] | None = None,
                    backend: str | None = None) -> dict[str, Any]:
    """Non-streaming convenience wrapper. Returns answer + metadata."""
    backend = pick_backend(backend)
    pieces = list(stream_answer(question, contexts, history, backend=backend))
    return {
        "answer": "".join(pieces),
        "backend": backend,
        "model": MODEL_DEFAULTS[backend],
        "sources": [
            {"id": i + 1, "source": c["source"], "chunk_id": c["chunk_id"],
             "score": c.get("score", 0.0)}
            for i, c in enumerate(contexts)
        ],
    }
