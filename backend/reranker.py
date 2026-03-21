"""DocuMind — Cross-Encoder Re-Ranker
Owner: Ashwin (Embeddings + Retrieval Optimization)
Purpose: Second-pass neural re-ranking of FAISS+BM25 candidate chunks.
         Uses a lightweight cross-encoder (ms-marco-MiniLM-L-6-v2, ~66 MB) to
         score each (query, chunk) pair jointly — far more accurate than
         bi-encoder cosine similarity alone.

Architecture:
  Stage 1 — Bi-encoder (FAISS + BM25):  fast candidate retrieval (top k*EXPAND)
  Stage 2 — Cross-encoder (this module): accurate re-ranking (top k returned)

Connection: Called by rag.py -> retrieve_context() after hybrid search.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Expand factor: retrieve this many candidates before re-ranking
RERANK_EXPAND = int(os.getenv("RERANK_EXPAND", "3"))

# Cross-encoder model — lightweight, works without GPU
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (downloads ~66 MB on first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder: %s", CROSS_ENCODER_MODEL)
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
            logger.info("Cross-encoder loaded successfully")
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s) — skipping re-rank", exc)
            _cross_encoder = False   # sentinel: don't retry on every call
    return _cross_encoder if _cross_encoder is not False else None


def rerank(query: str, docs: list[dict], top_k: int) -> list[dict]:
    """Re-rank retrieved docs using a cross-encoder and return the top_k best.

    Args:
        query:  The user's question string.
        docs:   Candidate chunks from Stage 1 (FAISS + BM25).
        top_k:  Number of chunks to return after re-ranking.

    Returns:
        The top_k chunks, sorted by cross-encoder score (highest first).
        Falls back to the original order if the model is unavailable.
    """
    if not docs:
        return docs

    model = _get_cross_encoder()
    if model is None:
        # Graceful degradation: return top_k from Stage 1 ranking as-is
        return docs[:top_k]

    try:
        pairs = [(query, doc["text"]) for doc in docs]
        scores = model.predict(pairs)

        # Attach cross-encoder score and sort
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
        logger.debug(
            "Re-ranked %d -> %d chunks for query='%s...'",
            len(docs), top_k, query[:40],
        )
        return reranked[:top_k]

    except Exception as exc:
        logger.warning("Re-ranking failed (%s) — returning Stage 1 order", exc)
        return docs[:top_k]
