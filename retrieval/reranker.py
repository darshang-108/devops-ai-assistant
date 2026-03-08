"""
Reranker – uses a CrossEncoder model to re-score retrieved document chunks
against the original query for more accurate relevance ranking.
"""

from __future__ import annotations

from typing import Dict, List

from sentence_transformers import CrossEncoder

# Lazy-loaded model singleton
_model: CrossEncoder | None = None
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank_documents(
    query: str, retrieved_docs: List[Dict], top_k: int = 8
) -> List[Dict]:
    """
    Re-score *retrieved_docs* against *query* using a CrossEncoder and
    return the *top_k* highest-scoring documents.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    retrieved_docs : List[Dict]
        Documents from the initial retrieval stage, each with keys
        ``text``, ``source``, ``score``.
    top_k : int
        Number of best documents to return after reranking.

    Returns
    -------
    List[Dict]
        Top-k documents with keys: ``text``, ``source``, ``score``
        (score is now the CrossEncoder relevance score).
    """
    if not retrieved_docs:
        return []

    model = _get_model()

    # Build (query, document_text) pairs for the CrossEncoder
    pairs = [(query, doc["text"]) for doc in retrieved_docs]

    # Score each pair
    scores = model.predict(pairs)

    # Attach scores and sort descending (higher = more relevant)
    scored_docs = []
    for doc, score in zip(retrieved_docs, scores):
        scored_docs.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": round(float(score), 4),
        })

    scored_docs.sort(key=lambda d: d["score"], reverse=True)

    return scored_docs[:top_k]
