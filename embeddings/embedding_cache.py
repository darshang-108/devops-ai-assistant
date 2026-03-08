"""
Embedding cache – avoids re-embedding unchanged chunks by storing a
``{chunk_hash: embedding_vector}`` mapping on disk as a pickle file.

Cache file: ``vector_store/faiss_index/embedding_cache.pkl``
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain_core.documents import Document

# ── Default cache location (next to the FAISS index) ─────────────────────────
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent
    / "vector_store"
    / "faiss_index"
    / "embedding_cache.pkl"
)


# ── Hash helper ──────────────────────────────────────────────────────────────
def get_chunk_hash(chunk: Document) -> str:
    """Return an MD5 hex-digest of the chunk's content + source metadata."""
    key = chunk.page_content + chunk.metadata.get("source", "")
    return hashlib.md5(key.encode("utf-8")).hexdigest()


# ── Load / save ──────────────────────────────────────────────────────────────
def load_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> Dict[str, np.ndarray]:
    """
    Load the embedding cache from *cache_path*.

    Returns an empty dict when the file is missing or corrupted.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as fh:
            cache: Dict[str, np.ndarray] = pickle.load(fh)
        return cache
    except Exception as exc:
        print(f"⚠️  Embedding cache corrupted — starting fresh ({exc})")
        return {}


def save_cache(
    cache: Dict[str, np.ndarray],
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> None:
    """
    Persist *cache* to disk atomically (write to ``.tmp`` then rename).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # Atomic rename (works on POSIX; on Windows replaces if exists)
        if sys.platform == "win32" and cache_path.exists():
            os.replace(tmp_path, cache_path)
        else:
            tmp_path.rename(cache_path)
        print(f"💾  Embedding cache saved — {len(cache)} entries")
    except Exception as exc:
        print(f"⚠️  Failed to save embedding cache: {exc}")


# ── Cache-aware embedding generator ─────────────────────────────────────────
def generate_embeddings_cached(
    chunks: List[Document],
    cache_path: Path = DEFAULT_CACHE_PATH,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate embeddings for *chunks*, re-using cached vectors for chunks
    whose content+source hash has not changed.

    Parameters
    ----------
    chunks : List[Document]
        LangChain Document objects to embed.
    cache_path : Path
        Location of the pickle cache file.
    batch_size : int
        Number of texts per ``model.encode()`` call.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(chunks), embedding_dim)``.
    """
    from embeddings.embedder import _get_model  # lazy import to avoid circular

    cache = load_cache(cache_path)

    # Classify chunks
    hashes = [get_chunk_hash(c) for c in chunks]
    uncached_indices = [i for i, h in enumerate(hashes) if h not in cache]
    cached_count = len(chunks) - len(uncached_indices)

    print(f"📦  Cache hit: {cached_count} chunks  |  To embed: {len(uncached_indices)} chunks")

    # Embed only the uncached chunks
    if uncached_indices:
        model = _get_model()
        uncached_texts = [chunks[i].page_content for i in uncached_indices]

        new_embeddings = model.encode(
            uncached_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size,
        )

        # Store new embeddings in cache
        for idx_in_batch, chunk_idx in enumerate(uncached_indices):
            cache[hashes[chunk_idx]] = new_embeddings[idx_in_batch]

        save_cache(cache, cache_path)

    # Assemble final array in original order
    embeddings = np.array([cache[h] for h in hashes], dtype=np.float32)
    return embeddings


# ── Cache invalidation ───────────────────────────────────────────────────────
def clear_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> None:
    """Delete the embedding cache file if it exists."""
    try:
        if cache_path.exists():
            cache_path.unlink()
            print("🗑️  Embedding cache cleared")
        else:
            print("ℹ️  No embedding cache to clear")
    except Exception as exc:
        print(f"⚠️  Could not clear cache: {exc}")
