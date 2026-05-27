<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6B5BFF,50:9B59B6,100:FF6B9D&height=220&section=header&text=Doc-Grounded%20Chatbot&fontSize=52&fontColor=ffffff&fontAlignY=38&animation=fadeIn&desc=A+real+conversational+chatbot+over+your+own+documents&descSize=17&descAlignY=64" />

<a href="https://github.com/nanettetada">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=23&duration=3500&pause=800&color=9B59B6&center=true&vCenter=true&width=720&lines=Real+streaming+LLM+%E2%80%94+not+extractive+snippets;FAISS+%2B+cross-encoder+reranker+%2B+chat+memory;Groq+%C2%B7+Claude+%C2%B7+GPT+%C2%B7+Ollama+%C2%B7+local+HF" />
</a>

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FAISS-009688?style=for-the-badge" />
<img src="https://img.shields.io/badge/sentence--transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white" />
<img src="https://img.shields.io/badge/Anthropic-D97757?style=for-the-badge&logo=anthropic&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<a href="https://huggingface.co/spaces/NanetteTada/doc-grounded-rag"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Live%20Demo-FFD21E?style=for-the-badge" /></a>

</div>

---

## :dart: Why I rebuilt this

The first version of this project did **retrieval** — given a question, it found the closest passage in a FAISS index and printed it back. Without an API key the "fallback" was literally a copy-paste of the most relevant snippet.

That's not a chatbot. That's a search engine in disguise.

The rebuild is what RAG is *supposed* to feel like:

- A **real generative LLM** writes the answer — Groq's free tier by default, with Claude, GPT, local Ollama, and offline HuggingFace transformers as alternatives
- **Conversation memory** so follow-up questions ("how about for the X1?", "and the price?") make sense
- **Streaming responses** — tokens land in the UI as the model produces them
- **A cross-encoder reranker** sharpens the top-k passages before they reach the LLM
- **PDF + Markdown + TXT** loaders, with on-the-fly re-indexing from the Streamlit sidebar
- **Citations** under every reply so you can audit the source

## :sparkles: At a glance

|  |  |
|---|---|
| **What it does** | Multi-turn chat over a folder of your own documents |
| **Retrieval** | sentence-transformers (MiniLM-L6) embeddings → FAISS top-12 → cross-encoder rerank → top-5 |
| **Generation** | Streaming from Groq (default, free tier), Claude, GPT, Ollama, or local HF transformers |
| **Memory** | Last 6 user+assistant turn pairs are sent with each new question |
| **UI** | Streamlit chat with avatars, citation expander, drag-drop document upload |
| **Stack** | sentence-transformers · FAISS · cross-encoder · Streamlit · plain Python (no LangChain) |

## :building_construction: How it works

```
   You ask a question
          |
          v
   Retrieval ─── FAISS top-12 candidates ─── cross-encoder reranks ─── top-5 final
          |
          v
   Prompt assembly ─── system prompt + last 6 turns + retrieved passages + question
          |
          v
   LLM streams the answer ─── Groq / Claude / GPT / Ollama / HF local
          |
          v
   Answer + citations ─── rendered token-by-token into the Streamlit chat
```

## :computer: Quick start

```bash
pip install -r requirements.txt

# Pick ONE backend (Groq free tier is recommended):
export GROQ_API_KEY=gsk_...             # free tier at console.groq.com
# or
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

python -m src.ingest                    # build the FAISS index from sample docs
streamlit run app/streamlit_app.py      # chat UI
```

Prefer a terminal REPL?

```bash
python -m src.rag                       # interactive multi-turn chat
python -m src.rag "What is the warranty on the X1?"
```

## :open_file_folder: Use it with your own documents

Two ways:

**Sidebar upload** — drag `.txt`, `.md`, or `.pdf` into the Streamlit sidebar and hit **Rebuild index**.

**Folder drop** — copy files into `data/docs/` and run:

```bash
python -m src.ingest
```

## :rocket: Backends

The first backend whose key/runtime is available wins. Preference order:

| Backend | Setup | Why use it |
|---|---|---|
| **Groq** | `pip install groq` + `GROQ_API_KEY` | Free tier, fastest inference, llama-3.3-70b default |
| **Anthropic** | `pip install anthropic` + `ANTHROPIC_API_KEY` | Best reasoning quality (Claude 3.5 Sonnet default) |
| **OpenAI** | `pip install openai` + `OPENAI_API_KEY` | Familiar GPT-4o-mini default |
| **Ollama** | `pip install ollama` + `ollama serve` locally | Fully local, no cost, no rate limits |
| **HF local** | `pip install transformers torch` | Final offline fallback — runs a small instruct model on CPU |

Switch backends live from the sidebar.

## :speech_balloon: Example session

```
> What's the warranty period on the Aurora-X1?

The Aurora-X1 ships with a 24-month manufacturer warranty covering defects
in materials and workmanship. Accidental damage is not covered by the base
warranty — you'd need Aurora Care+ for that. [1][2]

> And how do I extend it?

Aurora Care+ extends coverage to 36 months and adds accidental damage
protection. You can buy it within the first 30 days of receiving the unit.
[1]

Sources:
  [1] warranty_policy.txt  (chunk 1)
  [2] product_handbook.md  (chunk 4)
```

Notice the second answer understood what "it" referred to — that's the chat memory at work.

## :file_folder: Project layout

```
doc-grounded-rag/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py        # paths, model names, backend preference, system prompt
│   ├── loaders.py       # txt / md / pdf
│   ├── ingest.py        # chunk → embed → FAISS
│   ├── retrieve.py      # top-k FAISS + cross-encoder rerank
│   ├── generate.py      # streaming multi-backend LLM
│   └── rag.py           # orchestration + CLI / REPL
├── app/
│   └── streamlit_app.py # chat UI
└── data/
    └── docs/            # drop your documents here
```

## :rocket: What I'd build next

- Hybrid retrieval (BM25 + dense) for better keyword recall.
- Per-document namespaces so one index can serve multiple users.
- RAGAS evaluation harness — faithfulness, answer relevancy, context precision.
- Tool-use loop so the bot can `search_web()` when documents don't have the answer.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6B5BFF,50:9B59B6,100:FF6B9D&height=100&section=footer" />

Built by <b>Tadaishe Maumbe</b> · <a href="https://github.com/nanettetada">@nanettetada</a> · <a href="mailto:maumbetadaishe@gmail.com">email</a>

</div>
