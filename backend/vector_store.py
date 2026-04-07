"""DocuMind — Vector Store (FAISS)
Owner: Ashwin (Embeddings + Retrieval Optimization)
Purpose: Manages the FAISS index — create, add, remove, search, persist
Connection: Called by rag.py for all vector DB operations;
            supports chat_id isolation so each chat only sees its own documents

Performance note:
  Embeddings are cached in `_embeddings` (numpy array) so that document deletion
  only requires a numpy row-filter + FAISS index rebuild — NOT a full re-embed.
  This reduces deletion from ~30 s to < 1 s.
"""

import faiss
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STORE_DIR  = os.path.join(BASE_DIR, "vector_data")
INDEX_PATH = os.path.join(STORE_DIR, "faiss.index")
DOCS_PATH  = os.path.join(STORE_DIR, "documents.json")
EMBS_PATH  = os.path.join(STORE_DIR, "embeddings.npy")   # cached embedding matrix

index       = None
documents   = []       # list[dict]: text, source, page, chunk_id, [chat_id]
_embeddings = None     # np.ndarray shape (N, dim) — mirrors documents 1-to-1

# Evaluation-query keywords for section boost (research paper domain)
_EVAL_KEYWORDS = frozenset([
    "accuracy", "performance", "result", "results", "score", "metric", "metrics",
    "bleu", "rouge", "f1", "precision", "recall", "benchmark", "evaluation",
    "experiment", "experiments", "baseline", "comparison", "improvement",
])


def _has_eval_keyword(query: str) -> bool:
    """Return True if the query contains research-paper evaluation keywords."""
    words = set(query.lower().split())
    return bool(words & _EVAL_KEYWORDS)


def _ensure_store_dir():
    os.makedirs(STORE_DIR, exist_ok=True)


def _rebuild_faiss(embs: np.ndarray):
    """Rebuild an HNSW index from a (N, dim) float32 array.

    HNSW (Hierarchical Navigable Small World) provides O(log N) approximate
    nearest-neighbour search — 10–100× faster than IndexFlatL2 at scale with
    less than 1% accuracy loss. M=32 neighbours per node is a good default for
    384-dim sentence embeddings.

    Note: existing saved indexes (FlatL2) are loaded as-is on startup;
    HNSW kicks in for all new rebuilds (on document delete or first create).
    """
    global index
    if embs is None or len(embs) == 0:
        index = None
        return
    idx = faiss.IndexHNSWFlat(embs.shape[1], 32)   # M=32 neighbours
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    idx.add(embs)
    index = idx


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------

def create_index(embeddings, docs):
    """Create a brand-new FAISS L2 index from embeddings + document metadata.
    Owner: Ashwin — called on first run or full rebuild."""
    global _embeddings, documents

    _ensure_store_dir()
    embs = np.atleast_2d(np.array(embeddings, dtype="float32"))
    _rebuild_faiss(embs)
    _embeddings = embs
    documents   = list(docs)

    save_index()
    logger.info("FAISS index created: %d docs, dim=%d", len(docs), embs.shape[1])


# ---------------------------------------------------------------------------
# Add documents
# ---------------------------------------------------------------------------

def add_to_index(embeddings, new_docs):
    """Append new embeddings + metadata to the existing index.
    Owner: Ashwin — called after every successful document upload."""
    global index, _embeddings, documents

    new_embs = np.atleast_2d(np.array(embeddings, dtype="float32"))

    if index is None or _embeddings is None:
        create_index(new_embs, new_docs)
        return

    _embeddings = np.vstack([_embeddings, new_embs])
    documents.extend(new_docs)
    index.add(new_embs)   # incremental add — fast
    save_index()
    logger.info("Added %d docs to index (total: %d)", len(new_docs), len(documents))


# ---------------------------------------------------------------------------
# Remove documents  (FAST — no re-embedding)
# ---------------------------------------------------------------------------

def remove_by_source(source_name: str, chat_id: str = None) -> int:
    """Remove all chunks for a given source, optionally scoped to a chat_id.
    Owner: Ashwin — O(N) filter + FAISS index rebuild from cached embeddings.
    No call to embed() — deletion is now near-instant regardless of corpus size."""
    global index, documents, _embeddings

    if not documents:
        return 0

    original_count = len(documents)

    # Compute which rows to keep
    if chat_id:
        keep_mask = [
            not (d.get("source") == source_name and d.get("chat_id") == chat_id)
            for d in documents
        ]
    else:
        keep_mask = [d.get("source") != source_name for d in documents]

    removed = original_count - sum(keep_mask)
    if removed == 0:
        return 0

    # Filter documents list
    documents = [d for d, keep in zip(documents, keep_mask) if keep]

    # Filter embedding matrix (numpy fancy indexing — no re-computation)
    if _embeddings is not None and len(_embeddings) == original_count:
        indices = np.array([i for i, keep in enumerate(keep_mask) if keep], dtype=np.int64)
        _embeddings = _embeddings[indices] if len(indices) > 0 else None
    else:
        # Safety: embeddings out of sync — will rebuild from scratch next add
        _embeddings = None

    # Rebuild FAISS index from cached embeddings (milliseconds, no GPU/model needed)
    _rebuild_faiss(_embeddings)

    save_index()
    logger.info(
        "Removed %d chunks for source='%s' chat_id='%s' (instant — no re-embed)",
        removed, source_name, chat_id,
    )
    return removed


# ---------------------------------------------------------------------------
# Search (with chat_id isolation)
# ---------------------------------------------------------------------------

def search(query_embedding, k: int = 5, chat_id: str = None,
           query_text: str = None, use_hybrid: bool = True,
           section_filter: str = None) -> list:
    """Hybrid search: FAISS semantic + BM25 keyword, merged via Reciprocal Rank Fusion.
    Owner: Ashwin — core retrieval function.

    chat_id isolation rules:
    - Docs WITH a chat_id are private: only returned when the same chat_id is queried.
    - Docs WITHOUT a chat_id (e.g. sample docs) are global: returned to everyone.
    - If chat_id is None, all docs are returned (backward-compat / admin use).
    """
    if index is None or not documents:
        return []

    # ── Collect documents visible to this chat ──────────────────────────
    visible: list[tuple[int, dict]] = []   # (original_index, doc)
    for i, doc in enumerate(documents):
        doc_chat_id = doc.get("chat_id")
        if chat_id and doc_chat_id and doc_chat_id != chat_id:
            continue
        visible.append((i, doc))

    # Metadata filter: restrict retrieval to one or more sections
    # Accepts str (single section) or list[str] (multiple sections from PageIndex)
    if section_filter:
        if isinstance(section_filter, str):
            section_filter = [section_filter]
        filter_set = set(section_filter)
        visible = [(i, doc) for i, doc in visible if doc.get("section") in filter_set]

    if not visible:
        return []

    # ── FAISS semantic search ────────────────────────────────────────────
    query_np  = np.atleast_2d(np.array(query_embedding, dtype="float32"))
    fetch_k   = min(k * 8, len(documents))
    D, I      = index.search(query_np, fetch_k)

    # Map original FAISS indices → position in visible list
    orig_to_vis: dict[int, int] = {orig: pos for pos, (orig, _) in enumerate(visible)}

    faiss_ranked: list[tuple[int, float]] = []   # (visible_pos, raw_score)
    for i, dist in zip(I[0], D[0]):
        if i in orig_to_vis:
            faiss_ranked.append((orig_to_vis[i], float(1 / (1 + dist))))

    # ── BM25 keyword search ──────────────────────────────────────────────
    bm25_ranked: list[tuple[int, float]] = []
    if use_hybrid and query_text and len(visible) > 0:
        try:
            from bm25 import BM25Index, reciprocal_rank_fusion
            corpus   = [doc["text"] for _, doc in visible]
            bm25_idx = BM25Index(corpus)
            bm25_raw = bm25_idx.search(query_text, k=min(k * 8, len(visible)))
            bm25_ranked = bm25_raw   # already (visible_pos, score)

            merged = reciprocal_rank_fusion(faiss_ranked, bm25_ranked)
            top_positions = [pos for pos, _ in merged[:k]]
        except Exception as e:
            logger.warning("BM25 hybrid search failed, falling back to FAISS: %s", e)
            top_positions = [pos for pos, _ in faiss_ranked[:k]]
    else:
        top_positions = [pos for pos, _ in faiss_ranked[:k]]

    # ── Build result list ────────────────────────────────────────────────
    faiss_score_map = {pos: sc for pos, sc in faiss_ranked}

    results = []
    for pos in top_positions:
        if pos >= len(visible):
            continue
        _, doc = visible[pos]
        out = dict(doc)
        out["score"] = faiss_score_map.get(pos, 0.0)
        results.append(out)

    # Section boost: up-weight abstract/results chunks for evaluation-type queries.
    # Improves precision when asking about paper results, metrics, or performance.
    if query_text and _has_eval_keyword(query_text):
        _BOOST_SECTIONS = {"abstract", "results", "experiments", "conclusion"}
        for r in results:
            if r.get("section") in _BOOST_SECTIONS:
                r["score"] = min(1.0, r.get("score", 0.0) * 1.2)
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return results


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def save_index():
    """Serialize FAISS index + metadata + embeddings matrix to disk."""
    _ensure_store_dir()

    # Save FAISS binary
    if index is not None:
        try:
            index_bytes = faiss.serialize_index(index)
            with open(INDEX_PATH, "wb") as f:
                f.write(index_bytes)
        except Exception as e:
            logger.error("Failed to write FAISS index: %s", e)
            raise

    # Save document metadata
    try:
        with open(DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to save document metadata: %s", e)
        raise

    # Save embeddings matrix (enables fast deletion on next run)
    if _embeddings is not None:
        try:
            np.save(EMBS_PATH, _embeddings)
        except Exception as e:
            logger.warning("Failed to save embeddings cache: %s", e)

    logger.info("Index saved: %d documents", len(documents))


def load_index() -> bool:
    """Deserialize FAISS index + metadata from disk.
    Also loads embeddings cache if available."""
    global index, documents, _embeddings

    _ensure_store_dir()

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                index_bytes = f.read()
            index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype="uint8"))

            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                documents = json.load(f)

            # Load embeddings cache if present
            if os.path.exists(EMBS_PATH):
                try:
                    _embeddings = np.load(EMBS_PATH)
                    logger.info(
                        "Embeddings cache loaded: shape=%s", _embeddings.shape
                    )
                except Exception as e:
                    logger.warning("Could not load embeddings cache: %s", e)
                    _embeddings = None

            logger.info("Index loaded: %d documents", len(documents))
            return True
        except Exception as e:
            logger.error("Failed to load index: %s", e)
            return False

    return False


# ---------------------------------------------------------------------------
# Document list (chat-scoped)
# ---------------------------------------------------------------------------

def get_document_list(chat_id: str = None) -> list:
    """Return a summary list of documents, grouped by source.
    Owner: Anirudh (Data Engineer) — metadata tagging."""
    sources: dict = {}

    for doc in documents:
        doc_chat_id = doc.get("chat_id")

        if chat_id and doc_chat_id and doc_chat_id != chat_id:
            continue

        src = doc.get("source", "Unknown")
        if src not in sources:
            sources[src] = {"name": src, "chunks": 0, "pages": set()}

        sources[src]["chunks"] += 1
        sources[src]["pages"].add(doc.get("page", 1))

    return [
        {"name": info["name"], "chunks": info["chunks"], "pages": len(info["pages"])}
        for info in sources.values()
    ]


def get_total_documents(chat_id: str = None) -> int:
    """Count documents, optionally scoped to a chat."""
    if not chat_id:
        return len(documents)
    return sum(
        1 for d in documents
        if not d.get("chat_id") or d.get("chat_id") == chat_id
    )
