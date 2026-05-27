<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6B9D,50:9B59B6,100:6B5BFF&height=220&section=header&text=RAG%20Document%20Q%26A&fontSize=58&fontColor=ffffff&fontAlignY=38&animation=fadeIn&desc=Talk+to+your+own+documents&descSize=18&descAlignY=64" />

<a href="https://github.com/nanettetada">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3500&pause=800&color=9B59B6&center=true&vCenter=true&width=700&lines=Stop+the+hallucinations;FAISS+%2B+sentence-transformers+%2B+Streamlit;Grounded+answers+with+citations" />
</a>

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FAISS-009688?style=for-the-badge" />
<img src="https://img.shields.io/badge/sentence--transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
<img src="https://img.shields.io/badge/Anthropic-D97757?style=for-the-badge&logo=anthropic&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<a href="https://huggingface.co/spaces/NanetteTada/doc-grounded-rag"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Live%20Demo-FFD21E?style=for-the-badge" /></a>

</div>

---

## :dart: Why I built this

Every general-purpose chatbot I tried hallucinated the moment I asked it anything niche. I built this to learn how Retrieval-Augmented Generation actually fixes that — by retrieving the right passages from *your* documents and forcing the model to answer only from what it finds.

## :sparkles: At a glance

|  |  |
|---|---|
| **What it does** | Ask natural-language questions of any folder of `.txt`/`.md` files |
| **Retrieval** | sentence-transformers (MiniLM-L6-v2) embeddings + FAISS top-k search |
| **Generation** | OpenAI / Anthropic, with an extractive fallback for offline demos |
| **UI** | Streamlit chat with source citations on every answer |
| **Stack** | sentence-transformers · FAISS · Streamlit · plain Python (no LangChain) |

## :building_construction: How it works

```
   User question
        |
        v
   src/ingest.py     (one-off)
   load .txt/.md
   -> chunk (200 tokens, sliding window)
   -> embed with MiniLM
   -> FAISS index
        |
        v
   src/retrieve.py
   top-k semantic search
        |
        v
   src/generate.py
   prompt LLM with context + question
   (OpenAI / Anthropic / local extractive fallback)
        |
        v
   Answer + citations
```

## :sparkles: What I built in

- **Multi-format ingestion** — `.txt` and `.md` out of the box (PDF is one small loader away).
- **Overlapping chunking** so context isn't sliced mid-sentence.
- **Local embeddings** with `all-MiniLM-L6-v2` — CPU-friendly, no API key needed.
- **FAISS vector store** with a persisted index.
- **Pluggable LLM backends** — OpenAI, Anthropic, or a local extractive fallback so the demo runs without keys.
- **Citations** on every answer.
- **Streamlit chat UI**.
- **CLI** — `python -m src.rag "your question"` for scripting and testing.
- **No LangChain dependency** — I wanted to understand each piece, so I wrote them.

## :computer: Quick start

```bash
pip install -r requirements.txt
python -m src.ingest                       # build the FAISS index from sample docs
streamlit run app/streamlit_app.py         # chat UI
```

Or query from the CLI:

```bash
python -m src.rag "What is the warranty period for the Aurora-X1?"
```

To use a real LLM, set one of:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, the system returns the most relevant passage — handy for offline demos and for sanity-checking retrieval.

## :open_file_folder: Use it with your own documents

Drop `.txt` or `.md` files into `data/docs/`, then re-run:

```bash
python -m src.ingest
```

## :speech_balloon: Example session

```
> What is the warranty period for the Aurora-X1 drone?

The Aurora-X1 drone comes with a 24-month manufacturer warranty
that covers defects in materials and workmanship. Accidental
damage requires the optional Aurora Care+ plan.

Sources:
  [1] product_handbook.md  (chunk 4)
  [2] warranty_policy.txt  (chunk 1)
```

## :file_folder: Project layout

```
doc-grounded-rag/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── retrieve.py
│   ├── generate.py
│   └── rag.py
├── app/
│   └── streamlit_app.py
└── data/
    ├── docs/
    │   ├── company_overview.md
    │   ├── product_handbook.md
    │   ├── warranty_policy.txt
    │   └── faq.md
    └── index/   (built on first ingest)
```

## :rocket: What I'd build next

- PDF support via `pypdf`.
- Swap FAISS for Qdrant or pgvector once there's more than one user.
- Add a reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for stronger top-k results.
- Stream LLM tokens to the Streamlit UI for a snappier feel.
- Evaluate properly with [RAGAS](https://github.com/explodinggradients/ragas) — faithfulness, answer relevancy, context precision.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6B9D,50:9B59B6,100:6B5BFF&height=100&section=footer" />

Built by <b>Tadaishe Maumbe</b> · <a href="https://github.com/nanettetada">@nanettetada</a> · <a href="mailto:maumbetadaishe@gmail.com">email</a>

</div>
