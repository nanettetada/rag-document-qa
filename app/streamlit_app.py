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
    page_title="Document chat",
    page_icon="💬",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .badge {
        display: inline-block;
        background: #eef0f2; color: #3a434d;
        padding: 4px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600;
        margin-right: 6px;
    }
    .badge.ok { background: #e7f3ec; color: #2f7a52; }
    .badge.warn { background: #fbf1e2; color: #9a6a18; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin:-6px 0 4px 0;">
      <div style="font-size:13px; letter-spacing:.8px; color:#6b7280;
                  text-transform:uppercase;">Retrieval-augmented chat</div>
      <h1 style="margin:2px 0 4px 0; font-size:28px; font-weight:700; color:#1f2933;">
        Ask questions about your own documents</h1>
      <div style="font-size:15px; color:#6b7280;">
        Answers are written by a language model but grounded in the passages it
        retrieves, with a citation behind every reply.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

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

    if st.button("Rebuild index", use_container_width=True):
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
        st.session_state.is_demo = False
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

# Seed a demo conversation when no real backend is available, so the UI is
# meaningful for screenshots and first-time visitors. Cleared on first real
# question or when the user clicks "Clear chat".
DEMO_HISTORY = [
    {"role": "user", "content": "What's the warranty on the Aurora-X1?"},
    {
        "role": "assistant",
        "content": (
            "The Aurora-X1 ships with a **24-month manufacturer warranty** "
            "covering defects in materials and workmanship. Accidental damage "
            "is **not** covered by the base warranty — you'd need the optional "
            "Aurora Care+ plan for that. [1][2]"
        ),
        "sources": [
            {"id": 1, "source": "warranty_policy.txt", "chunk_id": 1, "score": 0.83},
            {"id": 2, "source": "product_handbook.md", "chunk_id": 4, "score": 0.71},
        ],
    },
    {"role": "user", "content": "And how do I extend it?"},
    {
        "role": "assistant",
        "content": (
            "**Aurora Care+** extends coverage to **36 months** and adds "
            "accidental damage protection (drops, water exposure, prop "
            "strikes). You can buy it within the first 30 days of receiving "
            "your unit, either at checkout or from the customer portal. [1]"
        ),
        "sources": [
            {"id": 1, "source": "warranty_policy.txt", "chunk_id": 2, "score": 0.79},
        ],
    },
]

if "history" not in st.session_state:
    # Seed a sample conversation on first load so the landing screen shows what
    # the bot does. It clears the moment the user asks their own question.
    st.session_state.history = list(DEMO_HISTORY)
    st.session_state.is_demo = True

if st.session_state.get("is_demo"):
    st.caption(
        "Sample conversation — ask your own question below, or use *Clear chat*, "
        "to start fresh."
    )

# Replay prior turns
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander(f"{len(turn['sources'])} source(s)"):
                for s in turn["sources"]:
                    st.markdown(
                        f"- **[{s['id']}] `{s['source']}`** "
                        f"(chunk {s['chunk_id']}, score {s.get('score', 0):.3f})"
                    )

# New input
if not avail:
    st.chat_input("Set an API key in the sidebar to start chatting...", disabled=True)
elif question := st.chat_input("Ask a question..."):
    # Drop the sample conversation the first time the user asks something real.
    if st.session_state.get("is_demo"):
        st.session_state.history = []
        st.session_state.is_demo = False
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
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
        with src_box.expander(f"{len(sources_meta)} source(s)"):
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
