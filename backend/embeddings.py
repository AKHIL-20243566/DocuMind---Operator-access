"""DocuMind — Embeddings Module
Owner: Ashwin (Embeddings + Retrieval Optimization)
Purpose: Wraps sentence-transformers (all-MiniLM-L6-v2, 384-dim) with lazy loading
         and MD5-keyed in-memory cache to avoid re-embedding repeated texts.
Connection: Called by rag.py (ingest_file + retrieve_context) and vector_store.py
            (remove_by_source rebuild). Use embed_cached() for query embeddings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import hashlib

logger = logging.getLogger(__name__)

_model = None
_cache = {}


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts):
    """Embed one or more texts. Accepts a string or list of strings."""
    model = _get_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def embed_cached(texts):
    """Embed with in-memory caching for repeated texts."""
    if isinstance(texts, str):
        texts = [texts]

    results = [None] * len(texts)
    to_embed = []
    to_embed_indices = []

    for i, text in enumerate(texts):
        key = hashlib.md5(text.encode()).hexdigest()
        if key in _cache:
            results[i] = _cache[key]
        else:
            to_embed.append(text)
            to_embed_indices.append(i)

    if to_embed:
        model = _get_model()
        new_embeddings = model.encode(to_embed, convert_to_numpy=True, show_progress_bar=False)
        for idx, text, emb in zip(to_embed_indices, to_embed, new_embeddings):
            key = hashlib.md5(text.encode()).hexdigest()
            _cache[key] = emb
            results[idx] = emb

    return np.array(results, dtype="float32")


def get_embedding_dim():
    return 384