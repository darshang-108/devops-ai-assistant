"""
Vector store module – uses FAISS to store and search document embeddings.

The index is persisted to ``vector_store/faiss_index/`` so it survives
process restarts.
**IMPORTANT**: When new documents are added to ``data/raw_docs/``, the FAISS
index must be rebuilt by re-running the ingestion + embedding pipeline:

    python -m ingestion.load_documents
    python -m ingestion.chunk_documents
    python -m vector_store.vector_db"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from embeddings.embedder import generate_embeddings_with_cache

# ── Default save path ────────────────────────────────────────────────────────
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "index_metadata.json"


def create_vector_store(
    chunks: List[Document],
    index_dir: Path = INDEX_DIR,
) -> faiss.IndexFlatL2:
    """
    Generate embeddings for *chunks*, build a FAISS index, and persist it
    to disk along with a sidecar metadata file.

    Parameters
    ----------
    chunks : List[Document]
        LangChain Document objects to embed and store.
    index_dir : Path
        Directory where the FAISS index and metadata will be saved.

    Returns
    -------
    faiss.IndexFlatL2
        The populated FAISS index.
    """
    # 1. Generate embeddings (cache-aware)
    embeddings = generate_embeddings_with_cache(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks")

    # 2. Build FAISS index (L2 / Euclidean distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # 3. Persist index to disk
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / "index.faiss"
    meta_file = index_dir / "index_metadata.json"

    faiss.write_index(index, str(index_file))

    # 4. Save chunk texts + metadata as sidecar JSON
    metadata = [
        {
            "text": c.page_content,
            "source": c.metadata.get("source"),
            "file_path": c.metadata.get("file_path"),
        }
        for c in chunks
    ]
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Vector database created successfully")
    print(f"  Index saved to: {index_file}")
    print(f"  Metadata saved to: {meta_file}")
    print(f"  Total vectors: {index.ntotal}  |  Dimension: {dimension}")

    return index


def load_vector_store(
    index_dir: Path = INDEX_DIR,
) -> tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Load a previously saved FAISS index and its metadata from disk.

    Returns
    -------
    tuple
        ``(index, metadata_list)`` where *metadata_list* contains dicts with
        ``text``, ``source``, and ``file_path`` for every stored vector.
    """
    index_file = index_dir / "index.faiss"
    meta_file = index_dir / "index_metadata.json"

    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_file}")

    index = faiss.read_index(str(index_file))
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    return index, metadata


def search(query_embedding: np.ndarray, top_k: int = 5, index_dir: Path = INDEX_DIR) -> List[Dict]:
    """
    Search the FAISS index for the *top_k* nearest neighbours.

    Parameters
    ----------
    query_embedding : np.ndarray
        1-D vector for the query.
    top_k : int
        Number of results to return.

    Returns
    -------
    List[Dict]
        Each dict has ``text``, ``source``, ``file_path``, and ``score``.
    """
    index, metadata = load_vector_store(index_dir)
    query_vec = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vec, top_k)

    results: List[Dict] = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        doc_meta = metadata[idx]
        results.append({
            "text": doc_meta["text"],
            "source": doc_meta.get("source", "unknown"),
            "file_path": doc_meta.get("file_path", "unknown"),
            "score": float(distances[0][i]),
        })
    return results


# ── Run directly to test the full pipeline ───────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from utils.config import RAW_DOCS_DIR
    from ingestion.load_documents import load_documents
    from ingestion.chunk_documents import chunk_documents

    # Full pipeline: load → chunk → embed → store
    docs = load_documents(RAW_DOCS_DIR)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    index = create_vector_store(chunks)
