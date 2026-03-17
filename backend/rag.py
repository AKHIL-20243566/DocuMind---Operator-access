import os
import logging
from embeddings import embed
from vector_store import (
    create_index, add_to_index, remove_by_source, search,
    load_index, get_document_list, get_total_documents
)
from document_parser import parse_bytes, is_supported

logger = logging.getLogger(__name__)

# Sample documents loaded on first start
SAMPLE_DOCUMENTS = [
    {"text": "Employees are entitled to 20 days of annual leave per year.", "source": "employee_policy.pdf", "page": 4, "chunk_id": "sample_0"},
    {"text": "Annual leave must be approved by the department manager.", "source": "company_handbook.pdf", "page": 12, "chunk_id": "sample_1"},
    {"text": "Remote work is allowed up to 3 days per week with manager approval.", "source": "remote_work_policy.pdf", "page": 1, "chunk_id": "sample_2"},
    {"text": "The company provides health insurance for all full-time employees.", "source": "benefits_guide.pdf", "page": 7, "chunk_id": "sample_3"},
    {"text": "Performance reviews are conducted every 6 months by the direct supervisor.", "source": "hr_handbook.pdf", "page": 15, "chunk_id": "sample_4"},
    {"text": "New employees must complete onboarding training within the first 2 weeks.", "source": "onboarding_guide.pdf", "page": 3, "chunk_id": "sample_5"},
    {"text": "The IT department handles all software installation requests via the internal helpdesk portal.", "source": "it_guidelines.pdf", "page": 8, "chunk_id": "sample_6"},
    {"text": "Expense reimbursement requires submission within 30 days with valid receipts.", "source": "finance_policy.pdf", "page": 22, "chunk_id": "sample_7"},
]


def initialize():
    """Initialize the RAG system — load persisted index or build from samples."""
    if load_index():
        logger.info(f"Loaded persisted vector index ({get_total_documents()} chunks)")
        return

    # First run — index sample documents
    logger.info("No persisted index found, initializing with sample documents")
    texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    embeddings = embed(texts)
    create_index(embeddings, SAMPLE_DOCUMENTS)
    logger.info("RAG initialized with sample documents")


def ingest_file(content: bytes, filename: str) -> dict:
    """Ingest an uploaded file: parse → chunk → embed → index."""
    if not is_supported(filename):
        return {"success": False, "error": f"Unsupported file type: {filename}"}

    # Parse file into chunks
    chunks = parse_bytes(content, filename)
    if not chunks:
        return {"success": False, "error": f"No text could be extracted from {filename}"}

    # Remove old version if re-uploading
    remove_by_source(filename)

    # Embed and add to index
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed(texts)
    add_to_index(embeddings, chunks)

    logger.info(f"Ingested '{filename}': {len(chunks)} chunks")
    return {
        "success": True,
        "filename": filename,
        "chunks": len(chunks),
        "message": f"Successfully ingested {filename} ({len(chunks)} chunks)"
    }


def delete_document(source_name: str) -> dict:
    """Remove all chunks for a document from the index."""
    removed = remove_by_source(source_name)
    if removed > 0:
        return {"success": True, "removed": removed, "message": f"Removed {removed} chunks from {source_name}"}
    return {"success": False, "message": f"Document '{source_name}' not found"}


def retrieve_context(question, k=3):
    """Retrieve relevant documents for a question."""
    if get_total_documents() == 0:
        return []
    query_embedding = embed(question)
    results = search(query_embedding, k=k)
    return results


def list_documents():
    """Get a summary of all indexed documents."""
    return get_document_list()


# Initialize on import
initialize()