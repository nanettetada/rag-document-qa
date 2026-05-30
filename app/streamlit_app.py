"""Streamlit chat UI for the RAG chatbot.

Features:
* Streaming token-by-token responses
* Multi-turn conversation memory
* Drop-in document upload (.txt / .md / .pdf) with on-the-fly re-indexing
* Source citations under every answer
* Backend selector in the main pane (sidebar is hidden on mobile)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import DOCS_DIR, INDEX_FILE, MODEL_DEFAULTS  # noqa: E402
from src.generate import available_backends  # noqa: E402
from src.ingest import build_index  # noqa: E402
from src.rag import ask_stream, trim_history  # noqa: E402


st.set_page_config(
    page_title="Document chat",
    page_icon="•",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Styling — editorial light theme, mobile-first
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, span, div, label, input, textarea, button {
        font-family: 'Inter', system-ui, sans-serif;
    }
    .stApp { background: #FBFAF7; }
    #MainMenu, footer, header[data-testid="stHeader"] { display: none; }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 4.5rem;
        max-width: 760px;
    }

    /* Wordmark header */
    .wordmark { font-family: 'Fraunces', serif; font-weight: 600; font-size: 30px;
                color: #1A1A17; letter-spacing: -0.4px; line-height: 1.1; margin: 0; }
    .wordmark .dot { color: #5B5BD6; }
    .tagline { color: #7A756A; font-size: 14.5px; margin: 6px 0 0 0;
               max-width: 560px; line-height: 1.55; }

    /* Backend strip */
    .backend-row { display: flex; align-items: center; gap: 10px;
                   flex-wrap: wrap; margin: 14px 0 4px 0; }
    .backend-row .label { font-size: 12px; font-weight: 600; color: #5B5BD6;
                          letter-spacing: 0.4px; text-transform: uppercase; }
    .pill { display: inline-flex; align-items: center; gap: 7px; background: #fff;
            border: 1px solid #E7E3DA; border-radius: 999px; padding: 5px 12px;
            font-size: 12.5px; color: #4A463E; }
    .pill.ok  { border-color: #BFD8C6; color: #16794C; }
    .pill.warn{ border-color: #E6CFA8; color: #8C5A0E; }
    .pill .dot{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

    .rule { height: 1px; background: #E7E3DA; border: 0; margin: 18px 0 14px 0; }

    /* Section heading */
    .sec { font-family: 'Fraunces', serif; font-weight: 500; font-size: 19px;
           color: #1A1A17; margin: 6px 0 2px 0; letter-spacing: -0.2px; }
    .sec-sub { color: #7A756A; font-size: 14px; margin: 0 0 10px 0; line-height: 1.5; }

    /* Note panels */
    .note { background: #F3F1EA; border-radius: 12px; padding: 12px 14px;
            color: #4A463E; font-size: 13.5px; line-height: 1.55; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: transparent; border: 0; padding: 0 0 6px 0;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 15px; line-height: 1.6; color: #2A2620;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #fff; border: 1px solid #E7E3DA; color: #1A1A17;
    }

    /* Chat input — make it readable on phones */
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important;  /* iOS won't auto-zoom at 16px+ */
        line-height: 1.5;
    }

    /* Expanders */
    details summary { font-size: 13.5px; color: #5B5BD6; font-weight: 500; }

    /* Buttons */
    .stButton > button {
        font-size: 14px; font-weight: 500; border-radius: 10px;
        border: 1px solid #E7E3DA; background: #fff; color: #1A1A17;
        padding: 8px 16px;
    }
    .stButton > button:hover { border-color: #5B5BD6; color: #5B5BD6; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: #fff; border: 1px dashed #D6CFC0; border-radius: 12px;
    }

    /* Mobile tweaks */
    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem;
                           padding-top: 1.1rem; max-width: 100%; }
        .wordmark { font-size: 25px; }
        .tagline { font-size: 14px; }
        .sec { font-size: 17px; }
        .pill { font-size: 12px; padding: 4px 10px; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            font-size: 14.5px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <h1 class="wordmark">Document chat<span class="dot">.</span></h1>
    <p class="tagline">Ask questions about your own documents. Every answer is
    grounded in retrieved passages and shows the sources behind it, so you can
    check the receipts.</p>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Backend picker — main pane, so phone users don't have to dig in the sidebar
# --------------------------------------------------------------------------- #
avail = available_backends()
chosen = None

st.markdown('<div class="backend-row"><span class="label">Model</span>',
            unsafe_allow_html=True)

if avail:
    chosen = st.selectbox(
        "Active backend",
        options=avail,
        format_func=lambda b: f"{b} · {MODEL_DEFAULTS[b]}",
        index=0,
        label_visibility="collapsed",
    )
    st.markdown(
        f'<span class="pill ok"><span class="dot"></span>Connected · '
        f'<code>{chosen}</code></span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="pill warn"><span class="dot"></span>'
        'No backend reachable</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander("How to connect a backend"):
        st.markdown(
            "Set one of these environment variables and refresh:\n\n"
            "- `GROQ_API_KEY` — free tier at console.groq.com (recommended)\n"
            "- `ANTHROPIC_API_KEY` — Claude\n"
            "- `OPENAI_API_KEY` — GPT\n\n"
            "Or install `ollama` / `transformers` for offline use."
        )

# --------------------------------------------------------------------------- #
# Documents — kept in main pane (mobile-friendly)
# --------------------------------------------------------------------------- #
DOCS_DIR.mkdir(parents=True, exist_ok=True)
with st.expander("Your documents", expanded=not INDEX_FILE.exists()):
    existing_main = sorted([p.name for p in DOCS_DIR.glob("*") if p.is_file()])
    if existing_main:
        st.caption(f"{len(existing_main)} file(s) indexed: " + ", ".join(existing_main))
    else:
        st.caption("No documents yet — upload a .txt, .md or .pdf to get started.")
    main_uploads = st.file_uploader(
        "Upload documents", type=["txt", "md", "pdf"],
        accept_multiple_files=True, key="main_uploader",
    )
    if main_uploads:
        for up in main_uploads:
            (DOCS_DIR / up.name).write_bytes(up.getbuffer())
        st.success(f"Saved {len(main_uploads)} file(s).")
    if st.button("Build / rebuild index", use_container_width=True, key="main_rebuild"):
        with st.spinner("Rebuilding index..."):
            try:
                n = build_index()
                st.success(f"Indexed {n} chunks.")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Ingest failed: {e}")

st.markdown('<hr class="rule" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
if not INDEX_FILE.exists():
    st.info(
        "No search index yet. Open **Your documents** above and drop in a "
        "`.txt`, `.md` or `.pdf`, then click **Build / rebuild index**."
    )
    st.stop()

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
    st.session_state.history = list(DEMO_HISTORY)
    st.session_state.is_demo = True

col_a, col_b = st.columns([4, 1])
with col_a:
    if st.session_state.get("is_demo"):
        st.caption("Sample conversation — ask your own question below.")
    else:
        st.caption(f"{len(st.session_state.history) // 2} turn(s) in this chat.")
with col_b:
    if st.button("Clear", use_container_width=True):
        st.session_state.history = []
        st.session_state.is_demo = False
        st.rerun()

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
    st.chat_input("Set an API key above to start chatting...", disabled=True)
elif question := st.chat_input("Ask a question..."):
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

# --------------------------------------------------------------------------- #
# Footer / how it works
# --------------------------------------------------------------------------- #
with st.expander("How this works"):
    st.markdown(
        "1. **Chunk** documents and embed them with MiniLM\n"
        "2. **Retrieve** the top candidates via FAISS\n"
        "3. **Rerank** with a cross-encoder for quality\n"
        "4. **Generate** a written answer from an LLM, grounded in the passages\n"
        "5. **Cite** sources on every reply\n"
        "6. **Remember** prior turns so follow-up questions feel natural"
    )
