"""Streamlit chat UI for the RAG chatbot.

Features:
* Streaming token-by-token responses
* Multi-turn conversation memory
* Drop-in document upload (.txt / .md / .pdf) with on-the-fly re-indexing
* Source citations under every answer
* Backend selector + status badge in the sidebar
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow running with `streamlit run app/streamlit_app.py` from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import DOCS_DIR, INDEX_FILE, MODEL_DEFAULTS  # noqa: E402
from src.generate import available_backends, pick_backend  # noqa: E402
from src.ingest import build_index  # noqa: E402
from src.rag import ask_stream, trim_history  # noqa: E402


st.set_page_config(
    page_title="Doc-Grounded Chatbot",
    page_icon=":speech_balloon:",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .hero {
        background: linear-gradient(135deg, #6B5BFF 0%, #9B59B6 50%, #FF6B9D 100%);
        padding: 28px 26px;
        border-radius: 16px;
        color: white;
        margin: -10px 0 18px 0;
        box-shadow: 0 12px 28px rgba(107, 91, 255, 0.25);
    }
    .hero h1 { margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.5px; }
    .hero p  { margin: 6px 0 0 0; font-size: 15px; opacity: 0.92; }
    .badge {
        display: inline-block;
        background: #ECF0F1; color: #2C3E50;
        padding: 4px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600;
        margin-right: 6px;
    }
    .badge.ok { background: #D5F5E3; color: #1E8449; }
    .badge.warn { background: #FDEBD0; color: #B9770E; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>:speech_balloon: Doc-Grounded Chatbot</h1>
      <p>Chat with your own documents. Answers are streamed from a real LLM and grounded in retrieved passages, with citations on every reply.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar — backend, settings, document management
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("LLM backend")
    avail = available_backends()
    if avail:
        chosen = st.selectbox(
            "Active backend",
            options=avail,
            format_func=lambda b: f"{b} ({MODEL_DEFAULTS[b]})",
            index=0,
        )
        st.markdown(
            f'<span class="badge ok">CONNECTED</span> <code>{chosen}</code>',
            unsafe_allow_html=True,
        )
    else:
        chosen = None
        st.markdown(
            '<span class="badge warn">NO BACKEND</span>',
            unsafe_allow_html=True,
        )
        st.warning(
            "No LLM backend reachable. Set one of these env vars and refresh:\n\n"
            "- `GROQ_API_KEY` — free tier at console.groq.com (recommended)\n"
            "- `ANTHROPIC_API_KEY` — Claude\n"
            "- `OPENAI_API_KEY` — GPT\n\n"
            "Or install `ollama` / `transformers` for offline use."
        )
        with st.expander("Why a real LLM?"):
            st.markdown(
                "Earlier versions of this app returned the most-relevant passage "
                "verbatim when no key was set. That's retrieval — not really a "
                "chatbot. The new version requires a generative model so answers "
                "are written, conversational, and aware of prior turns."
            )

    st.divider()
    st.subheader("Documents")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted([p.name for p in DOCS_DIR.glob("*") if p.is_file()])
    if existing:
        st.caption(f"{len(existing)} file(s) in `data/docs/`")
        with st.expander("Show indexed files"):
            for n in existing:
                st.markdown(f"- {n}")
    else:
        st.caption("No documents yet — upload some below.")

    uploads = st.file_uploader(
        "Add documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploads:
        for up in uploads:
            (DOCS_DIR / up.name).write_bytes(up.getbuffer())
        st.success(f"Saved {len(uploads)} file(s) to {DOCS_DIR}.")

    if st.button(":arrows_counterclockwise: Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            try:
                n = build_index()
                st.success(f"Indexed {n} chunks.")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Chunk** documents and embed them with MiniLM\n"
        "2. **Retrieve** the top candidates via FAISS\n"
        "3. **Rerank** with a cross-encoder for quality\n"
        "4. **Generate** a written answer from an LLM, grounded in the passages\n"
        "5. **Cite** sources on every reply\n"
        "6. **Remember** prior turns so follow-up questions feel natural"
    )

# --------------------------------------------------------------------------- #
# Main pane — chat
# --------------------------------------------------------------------------- #
if not INDEX_FILE.exists():
    st.info(
        "No search index yet. Drop some `.txt` / `.md` / `.pdf` files in the "
        "sidebar and click **Rebuild index**, or run `python -m src.ingest` "
        "from the project root."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Replay prior turns
for turn in st.session_state.history:
    avatar = ":bust_in_silhouette:" if turn["role"] == "user" else ":robot_face:"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander(f":books: {len(turn['sources'])} source(s)"):
                for s in turn["sources"]:
                    st.markdown(
                        f"- **[{s['id']}] `{s['source']}`** "
                        f"(chunk {s['chunk_id']}, score {s.get('score', 0):.3f})"
                    )

# New input
if not avail:
    st.chat_input("Set an API key in the sidebar to start chatting...", disabled=True)
elif question := st.chat_input("Ask a question..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":bust_in_silhouette:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":robot_face:"):
        msg_box = st.empty()
        src_box = st.container()
        with st.spinner("Searching documents..."):
            try:
                history = trim_history([
                    {"role": t["role"], "content": t["content"]}
                    for t in st.session_state.history[:-1]
                ])
                iterator, contexts, bk = ask_stream(
                    question, history=history, backend=chosen,
                )
            except Exception as e:
                msg_box.error(f"Retrieval failed: {e}")
                st.stop()

        # Stream tokens
        buf = ""
        try:
            for token in iterator:
                buf += token
                msg_box.markdown(buf + " ▌")
            msg_box.markdown(buf)
        except Exception as e:
            msg_box.error(f"Generation failed: {e}")
            st.stop()

        sources_meta = [
            {"id": i + 1, "source": c["source"], "chunk_id": c["chunk_id"],
             "score": c.get("score", 0.0)}
            for i, c in enumerate(contexts)
        ]
        with src_box.expander(f":books: {len(sources_meta)} source(s)"):
            for s in sources_meta:
                st.markdown(
                    f"- **[{s['id']}] `{s['source']}`** "
                    f"(chunk {s['chunk_id']}, score {s['score']:.3f})"
                )

        st.session_state.history.append({
            "role": "assistant",
            "content": buf,
            "sources": sources_meta,
        })
