"""
Retriever – given a natural-language query, embeds it, searches the FAISS
vector store, and returns the most relevant document chunks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from embeddings.embedder import embed_query
from retrieval.reranker import rerank_documents
from vector_store.vector_db import load_vector_store, search


def retrieve_documents(query: str, top_k: int = 8) -> List[Dict]:
    """
    Retrieve the *top_k* most relevant document chunks for *query*.

    Steps
    -----
    1. Convert the query to an embedding via ``embed_query()``.
    2. Load the FAISS index via ``load_vector_store()``.
    3. Perform vector similarity search.
    4. Deduplicate by content and return results sorted by relevance.

    Parameters
    ----------
    query : str
        Natural-language question.
    top_k : int
        Number of results to return (default 8).

    Returns
    -------
    List[Dict]
        Each dict has keys: ``text``, ``source``, ``score``.
    """
    # 1. Embed the query
    query_embedding = embed_query(query)

    # 2-3. Search FAISS index
    raw_results = search(query_embedding, top_k=top_k)

    # 4. Deduplicate by text content and format into clean result dicts
    seen_texts: set = set()
    results: List[Dict] = []
    for r in raw_results:
        text = r["text"]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        results.append({
            "text": text,
            "source": r.get("source", "unknown"),
            "file_path": r.get("file_path", "unknown"),
            "score": round(r["score"], 4),
        })

    # Rerank with CrossEncoder for better relevance ordering
    reranked = rerank_documents(query, results, top_k=min(8, len(results)))
    return reranked


def format_context(results: List[Dict]) -> str:
    """
    Render retrieved results into a single context string suitable for
    the LLM.  Each chunk is labelled with its source for better reasoning.
    """
    parts: List[str] = []
    for r in results:
        parts.append(f"Source: {r['source']}\nContent:\n{r['text']}")
    return "\n\n---\n\n".join(parts)


# ── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    while True:
        query = input("\nEnter search query (or 'exit'): ")

        if query.lower() == "exit":
            break

        results = retrieve_documents(query, top_k=5)

        print("\nTop results:\n")

        for r in results:
            source = r.get("file_path") or r.get("source") or "unknown"
            print(f"Source: {source}")
            print(r["text"][:300])
            print("-" * 50)
