# Deploy this app

## Streamlit Community Cloud (free, ~3 min)

1. [streamlit.io/cloud](https://streamlit.io/cloud) → **Sign in with GitHub** (use `nanettetada`).
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `nanettetada/doc-grounded-rag`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. **Deploy**.

## Adding an LLM API key (optional)

Without a key the app falls back to extractive answers. To enable real generation:

1. In the deployed app's dashboard → **Settings** → **Secrets**.
2. Add either:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
   or
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
3. Save. The app auto-restarts with the key live.

## Notes

- First load downloads the sentence-transformer model (~80MB) and embeds the sample docs — takes a couple of minutes.
- After that, subsequent loads are instant.
- Alternative: Hugging Face Spaces with SDK = Streamlit.
