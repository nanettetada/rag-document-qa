"""Document loaders. Supports .txt, .md, .pdf."""
from __future__ import annotations

from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "pypdf is required to load PDF files. `pip install pypdf`."
        ) from e
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


SUPPORTED = {
    ".txt": _read_text,
    ".md": _read_text,
    ".pdf": _read_pdf,
}


def load_documents(docs_dir: Path) -> list[dict]:
    """Load all supported documents under `docs_dir` recursively."""
    docs = []
    for path in sorted(docs_dir.glob("**/*")):
        if not path.is_file():
            continue
        loader = SUPPORTED.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            text = loader(path)
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
            continue
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs
