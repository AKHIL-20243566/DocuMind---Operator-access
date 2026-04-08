"""DocuMind — RAG Pipeline Orchestrator
Owner: Akhil (Frontend Lead + Partial RAG Integration) + Ashwin (Embeddings + Retrieval)
Purpose: Coordinates document ingestion (parse → chunk → embed → index) and
         context retrieval (embed query → search index → return ranked chunks)
Connection: Imported by main.py; calls document_parser, embeddings, and vector_store
"""

import os
import logging
from embeddings import embed, embed_cached
from vector_store import (
    create_index, add_to_index, remove_by_source, search,
    load_index, get_document_list, get_total_documents,
)
from document_parser import parse_bytes_with_diagnostics, is_supported
from reranker import rerank, RERANK_EXPAND
from page_index import PageIndex

logger = logging.getLogger(__name__)

# When True, generates N query variants and merges retrieval results via best
# score. Increases recall at the cost of ~2–3× latency (one Ollama call + N
# embeddings). Enable with: ENABLE_MULTI_QUERY=true in .env
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "false").lower() == "true"

# When True (default), uses the PageIndex hierarchical tree to let the LLM
# select relevant sections before running FAISS+BM25 search.  Disable with:
# USE_PAGE_INDEX=false in .env (falls back to flat full-corpus search).
USE_PAGE_INDEX = os.getenv("USE_PAGE_INDEX", "false").lower() == "true"

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PAGE_INDEX_PATH = os.path.join(BASE_DIR, "vector_data", "page_index.json")

_page_index = PageIndex()

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _rebuild_page_index() -> None:
    """Rebuild the PageIndex tree from the current vector store contents and persist it."""
    import vector_store as vs
    _page_index.build_from_documents(vs.documents)
    _page_index.save(PAGE_INDEX_PATH)


def initialize():
    """Load the persisted index from disk. Starts empty if no index exists."""
    if load_index():
        logger.info("Loaded persisted vector index (%d chunks)", get_total_documents())
    else:
        logger.info("No persisted index found — starting with empty document store")
    # Load the page index (or silently skip if not yet built)
    _page_index.load(PAGE_INDEX_PATH)


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def ingest_file(content: bytes, filename: str, chat_id: str = None) -> dict:
    """Parse an uploaded file, embed its chunks, and add them to the vector index.

    Owner: Anirudh (Data Engineer) + Ashwin (Embeddings)
    chat_id: scopes the document to a specific chat session for isolation.

    Returns a status dict with success, chunks, loader_used, ocr_triggered, etc.
    """
    if not is_supported(filename):
        return {
            "success": False,
            "error": f"Unsupported file type: {filename}",
            "error_code": "UNSUPPORTED_FILE_TYPE",
            "status_messages": [],
        }

    # Multi-layer parsing: PyPDF → Unstructured → OCR fallback
    parse_result = parse_bytes_with_diagnostics(content, filename)
    chunks = parse_result.get("documents", [])

    if not chunks:
        error_message = (
            parse_result.get("error_message")
            or f"No readable text could be extracted from {filename}"
        )
        return {
            "success": False,
            "error": error_message,
            "error_code": parse_result.get("error_code") or "NO_READABLE_TEXT",
            "status_messages": parse_result.get("status_messages", []),
            "loader_used": parse_result.get("loader_used"),
            "ocr_triggered": parse_result.get("ocr_triggered", False),
        }

    # Tag every chunk with the chat_id for isolation
    # Owner: Anirudh — metadata tagging (document_name, page_number, section, chat_id)
    if chat_id:
        for chunk in chunks:
            chunk["chat_id"] = chat_id

    # Remove stale version of this file in this chat before re-indexing
    remove_by_source(filename, chat_id=chat_id)

    # Embed and add to FAISS
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed(texts)
    add_to_index(embeddings, chunks)

    # Rebuild hierarchical PageIndex so new sections are immediately queryable
    _rebuild_page_index()

    logger.info(
        "Ingested '%s': chunks=%d loader=%s ocr=%s chat_id=%s",
        filename, len(chunks),
        parse_result.get("loader_used"),
        parse_result.get("ocr_triggered", False),
        chat_id,
    )
    return {
        "success": True,
        "filename": filename,
        "chunks": len(chunks),
        "message": f"Successfully ingested {filename} ({len(chunks)} chunks)",
        "loader_used": parse_result.get("loader_used"),
        "ocr_triggered": parse_result.get("ocr_triggered", False),
        "status_messages": parse_result.get("status_messages", ["Extraction successful"]),
    }


# ---------------------------------------------------------------------------
# Document deletion
# ---------------------------------------------------------------------------

def delete_document(source_name: str, chat_id: str = None) -> dict:
    """Remove all chunks for a document, optionally scoped to a chat."""
    removed = remove_by_source(source_name, chat_id=chat_id)
    if removed > 0:
        # Rebuild hierarchical PageIndex to remove stale section nodes
        _rebuild_page_index()
        return {
            "success": True,
            "removed": removed,
            "message": f"Removed {removed} chunks from {source_name}",
        }
    return {"success": False, "message": f"Document '{source_name}' not found"}


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

def retrieve_context(question: str, k: int = 5, chat_id: str = None,
                     section_filter: str = None) -> list:
    """Embed the query and retrieve top-k relevant chunks via a three-stage pipeline:
      Stage 1 — PageIndex section selection: LLM reads section summaries and picks
                 the 1-3 most relevant sections, narrowing the search space before
                 any vector computation (vectorless first pass).
                 Controlled by USE_PAGE_INDEX env var (default: true).
      Stage 2 — Hybrid FAISS (semantic) + BM25 (keyword) retrieval of k * RERANK_EXPAND
                 candidates, restricted to PageIndex-selected sections when available.
                 Falls back to full corpus if targeted search returns < k/2 results.
                 When ENABLE_MULTI_QUERY=true, generates query variants and merges by
                 best score for higher recall on ambiguous or complex questions.
      Stage 3 — Cross-encoder neural re-ranking to select the best k chunks.

    Owner: Ashwin (Retrieval Optimization)
    - Uses embed_cached() so repeated queries skip re-computation.
    - chat_id: when provided, only returns docs belonging to that chat (+ global docs).
    - section_filter: explicit override — skips PageIndex selection when provided.
    - ENABLE_MULTI_QUERY env var: N query rephrases merged before re-ranking.
    - USE_PAGE_INDEX env var: disable to bypass the LLM section selection step.
    """
    if get_total_documents(chat_id=chat_id) == 0:
        return []

    # ── Stage 1: PageIndex — LLM-guided section selection ──────────────────
    effective_filter = section_filter   # honour explicit caller override

    if USE_PAGE_INDEX and not section_filter and len(_page_index) > 0:
        try:
            import vector_store as vs
            from query_understanding import select_relevant_sections
            summaries = _page_index.get_section_summaries(
                chat_id=chat_id, documents=vs.documents,
            )
            if summaries:
                selected = select_relevant_sections(question, summaries, max_sections=3)
                if selected:
                    effective_filter = selected   # list[str] accepted by search()
        except Exception as e:
            logger.warning("PageIndex section selection failed, using full search: %s", e)

    candidates_k = k * RERANK_EXPAND

    # ── Stage 2: Hybrid FAISS + BM25 retrieval ─────────────────────────────
    if ENABLE_MULTI_QUERY:
        try:
            from query_understanding import generate_multi_queries
            queries = generate_multi_queries(question, n=2)
        except Exception:
            queries = [question]

        seen: dict[str, dict] = {}
        for q in queries:
            q_emb = embed_cached(q)
            for chunk in search(
                q_emb, k=candidates_k, chat_id=chat_id,
                query_text=q, use_hybrid=True, section_filter=effective_filter,
            ):
                cid = chunk.get("chunk_id") or chunk.get("text", "")[:40]
                if cid not in seen or chunk.get("score", 0) > seen[cid].get("score", 0):
                    seen[cid] = chunk

        candidates = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)[:candidates_k]
    else:
        query_embedding = embed_cached(question)
        candidates = search(
            query_embedding, k=candidates_k, chat_id=chat_id,
            query_text=question, use_hybrid=True, section_filter=effective_filter,
        )

    # Fallback: if targeted section search returned too few candidates, widen to full corpus
    if effective_filter and len(candidates) < max(1, k // 2):
        logger.info(
            "PageIndex targeted search returned only %d candidates (threshold %d) — "
            "falling back to full corpus search",
            len(candidates), k // 2,
        )
        query_embedding = embed_cached(question)
        candidates = search(
            query_embedding, k=candidates_k, chat_id=chat_id,
            query_text=question, use_hybrid=True, section_filter=None,
        )

    # ── Stage 3: cross-encoder re-ranking — returns top k ──────────────────
    return rerank(question, candidates, top_k=k)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def list_documents(chat_id: str = None) -> list:
    """Return a grouped summary of indexed documents, scoped to a chat if provided."""
    return get_document_list(chat_id=chat_id)


def has_embeddings(chat_id: str = None) -> bool:
    """Check whether any documents exist for the given chat (or globally)."""
    return get_total_documents(chat_id=chat_id) > 0


# Initialize on import
initialize()
