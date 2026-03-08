"""
Document loader – reads .md and .txt files from ``data/raw_docs/`` and returns
a list of LangChain ``Document`` objects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Supported file extensions
SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(directory: Path | str) -> List[Document]:
    """
    Recursively scan *directory* for ``.md`` and ``.txt`` files and return a
    list of LangChain ``Document`` objects.

    Each document carries metadata with the ``filename`` and ``source`` path.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: List[Document] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                        },
                    )
                )
        except Exception as exc:
            print(f"⚠ Failed to load {file_path}: {exc}")

    return documents


# ── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve project root so imports & paths work from anywhere
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from utils.config import RAW_DOCS_DIR

    docs = load_documents(RAW_DOCS_DIR)
    print(f"Loaded {len(docs)} documents")
    for d in docs:
        print(f"  - {d.metadata['filename']}  ({len(d.page_content)} chars)")
