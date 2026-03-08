"""
Chunking module – splits LangChain ``Document`` objects into smaller,
overlapping chunks suitable for embedding and vector storage.

Uses ``RecursiveCharacterTextSplitter`` with chunk_size=500 and
chunk_overlap=100.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Chunking parameters ─────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split each document into smaller chunks.

    Returns a flat list of LangChain ``Document`` objects.  Each chunk
    inherits the original document's metadata plus a ``chunk_index``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for doc in documents:
        texts = splitter.split_text(doc.page_content)
        for idx, text in enumerate(texts):
            chunks.append(
                Document(
                    page_content=text,
                    metadata={**doc.metadata, "chunk_index": idx},
                )
            )

    return chunks


# ── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from utils.config import RAW_DOCS_DIR
    from ingestion.load_documents import load_documents

    docs = load_documents(RAW_DOCS_DIR)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    if chunks:
        print("\n--- Example chunk ---")
        print(f"Content: {chunks[0].page_content[:200]}")
        print(f"Metadata: {chunks[0].metadata}")
