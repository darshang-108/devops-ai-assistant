"""
Embedding module – uses SentenceTransformers (all-MiniLM-L6-v2) to convert
document chunks into dense vector embeddings.

The model is loaded lazily on first use and cached for the process lifetime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# ── Model configuration ─────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache so the model is loaded only once.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load (and cache) the SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"Loading embedding model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embeddings(chunks: List[Document]) -> np.ndarray:
    """
    Generate vector embeddings for a list of document chunks.

    Parameters
    ----------
    chunks : List[Document]
        LangChain Document objects whose ``page_content`` will be embedded.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(len(chunks), embedding_dim)``.
    """
    model = _get_model()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def generate_embeddings_with_cache(chunks: List[Document]) -> np.ndarray:
    """
    Cache-aware wrapper around embedding generation.

    Delegates to :func:`embedding_cache.generate_embeddings_cached` so that
    only new or changed chunks are sent through the model.
    """
    from embeddings.embedding_cache import generate_embeddings_cached

    return generate_embeddings_cached(chunks)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string and return a 1-D vector."""
    model = _get_model()
    return model.encode(query, convert_to_numpy=True)


# ── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from utils.config import RAW_DOCS_DIR
    from ingestion.load_documents import load_documents
    from ingestion.chunk_documents import chunk_documents

    docs = load_documents(RAW_DOCS_DIR)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = generate_embeddings(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks")
    print(f"Embedding shape: {embeddings.shape}")
