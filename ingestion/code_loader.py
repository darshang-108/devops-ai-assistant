"""
Code loader – recursively scans a local repository directory for source code
files and returns LangChain ``Document`` objects compatible with the ingestion
pipeline (chunk_documents → generate_embeddings → create_vector_store).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Supported source-code extensions
SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs"}

# Directories to skip during traversal
IGNORED_DIRS = {".git", "node_modules", "__pycache__", "tests"}

# Map file extension → language name
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}


def load_codebase(repo_path: str | Path) -> List[Document]:
    """
    Recursively scan *repo_path* for source code files and return a list of
    LangChain ``Document`` objects.

    Uses ``os.walk`` to correctly traverse Python packages and nested modules.

    Parameters
    ----------
    repo_path : str | Path
        Path to the root of the repository to scan.

    Returns
    -------
    List[Document]
        One document per source file, with metadata including
        ``source``, ``file_path``, and ``language``.
    """
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    documents: List[Document] = []

    for dirpath, dirnames, filenames in os.walk(repo_path):
        # Prune ignored directories in-place so os.walk skips them entirely
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORED_DIRS
        )

        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_path = Path(dirpath) / filename
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    relative_path = os.path.relpath(file_path, repo_path)
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": relative_path,
                                "file_path": relative_path,
                                "type": "code",
                            },
                        )
                    )
            except Exception as exc:
                print(f"Warning: Failed to load {file_path}: {exc}")

    print(f"Loaded {len(documents)} code files from repository.")
    return documents


# ── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import argparse

    parser = argparse.ArgumentParser(description="Load source code from a repository")
    parser.add_argument("path", help="Path to the repository root")
    args = parser.parse_args()

    docs = load_codebase(args.path)
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"{i}. [{meta['language']}] {meta['file_path']} ({len(doc.page_content)} chars)")
