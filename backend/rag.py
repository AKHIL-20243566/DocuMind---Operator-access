"""DocuMind — RAG Pipeline Orchestrator
Owner: Akhil (Frontend Lead + Partial RAG Integration) + Ashwin (Embeddings + Retrieval)
Purpose: Coordinates document ingestion (parse → chunk → embed → index) and
         context retrieval (embed query → search index → return ranked chunks)
Connection: Imported by main.py; calls document_parser, embeddings, and vector_store
"""

import logging
from embeddings import embed, embed_cached
from vector_store import (
    create_index, add_to_index, remove_by_source, search,
    load_index, get_document_list, get_total_documents,
)
from document_parser import parse_bytes_with_diagnostics, is_supported
from reranker import rerank, RERANK_EXPAND

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample documents — global (no chat_id), available to all chats on first run
# Owner: Anirudh (Data Engineer) — document metadata tagging
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {"text": "Employees are entitled to 20 days of annual leave per year.",
     "source": "employee_policy.pdf",   "page": 4,  "chunk_id": "sample_0"},
    {"text": "Annual leave must be approved by the department manager.",
     "source": "company_handbook.pdf",  "page": 12, "chunk_id": "sample_1"},
    {"text": "Remote work is allowed up to 3 days per week with manager approval.",
     "source": "remote_work_policy.pdf","page": 1,  "chunk_id": "sample_2"},
    {"text": "The company provides health insurance for all full-time employees.",
     "source": "benefits_guide.pdf",    "page": 7,  "chunk_id": "sample_3"},
    {"text": "Performance reviews are conducted every 6 months by the direct supervisor.",
     "source": "hr_handbook.pdf",       "page": 15, "chunk_id": "sample_4"},
    {"text": "New employees must complete onboarding training within the first 2 weeks.",
     "source": "onboarding_guide.pdf",  "page": 3,  "chunk_id": "sample_5"},
    {"text": "The IT department handles all software installation requests via the internal helpdesk portal.",
     "source": "it_guidelines.pdf",     "page": 8,  "chunk_id": "sample_6"},
    {"text": "Expense reimbursement requires submission within 30 days with valid receipts.",
     "source": "finance_policy.pdf",    "page": 22, "chunk_id": "sample_7"},
]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize():
    """Load the persisted index from disk, or seed with sample documents on first run."""
    if load_index():
        logger.info("Loaded persisted vector index (%d chunks)", get_total_documents())
        return

    logger.info("No persisted index found — initializing with sample documents")
    texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    embeddings = embed(texts)
    create_index(embeddings, SAMPLE_DOCUMENTS)
    logger.info("RAG initialized with %d sample documents", len(SAMPLE_DOCUMENTS))


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
        return {
            "success": True,
            "removed": removed,
            "message": f"Removed {removed} chunks from {source_name}",
        }
    return {"success": False, "message": f"Document '{source_name}' not found"}


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

def retrieve_context(question: str, k: int = 5, chat_id: str = None) -> list:
    """Embed the query and retrieve top-k relevant chunks via a two-stage pipeline:
      Stage 1 — Hybrid FAISS (semantic) + BM25 (keyword) retrieval of k * RERANK_EXPAND candidates.
      Stage 2 — Cross-encoder neural re-ranking to select the best k chunks.

    Owner: Ashwin (Retrieval Optimization)
    - Uses embed_cached() so repeated queries skip re-computation.
    - chat_id: when provided, only returns docs belonging to that chat (+ global docs).
    """
    if get_total_documents(chat_id=chat_id) == 0:
        return []

    # Stage 1: retrieve more candidates than needed so re-ranker has room to work
    candidates_k = k * RERANK_EXPAND
    query_embedding = embed_cached(question)
    candidates = search(
        query_embedding, k=candidates_k, chat_id=chat_id,
        query_text=question, use_hybrid=True,
    )

    # Stage 2: cross-encoder re-ranking — returns top k
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
