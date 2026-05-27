"""Streamlit chat-style UI for the RAG pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running with `streamlit run app/streamlit_app.py` from project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import INDEX_FILE  # noqa: E402
from src.rag import ask  # noqa: E402


st.set_page_config(page_title="Document Q&A", page_icon=":books:", layout="centered")
st.title(":books: Document Q&A")
st.caption("Ask natural-language questions of your own document set. Built by Tadaishe Maumbe.")

if not INDEX_FILE.exists():
    st.error(
        "No index found. Run `python -m src.ingest` from the project root to build it, "
        "then refresh this page."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(
                        f"- **[{s['id']}] {s['source']}** "
                        f"(chunk {s['chunk_id']}, score {s['score']:.3f})"
                    )

if question := st.chat_input("Ask a question..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + reasoning..."):
            result = ask(question)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(
                    f"- **[{s['id']}] {s['source']}** "
                    f"(chunk {s['chunk_id']}, score {s['score']:.3f})"
                )
        st.session_state.history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. Documents are chunked and embedded with `all-MiniLM-L6-v2`\n"
        "2. A FAISS index does top-k semantic search\n"
        "3. An LLM (OpenAI / Anthropic) answers from the retrieved context\n"
        "4. Without an API key, the system shows the best matching passage\n"
    )
    st.divider()
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()
