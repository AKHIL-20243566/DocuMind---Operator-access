"""Document ingestion module — loads documents from files."""
import os
import logging

logger = logging.getLogger(__name__)


def load_text_files(directory):
    """Load .txt files from a directory."""
    documents = []
    if not os.path.isdir(directory):
        logger.warning(f"Directory not found: {directory}")
        return documents
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".txt") and os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append({
                        "text": text,
                        "source": filename,
                        "page": 1
                    })
    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ingest_documents(raw_dir, chunk_size=500, overlap=50):
    """Load and chunk all documents from raw directory."""
    raw_docs = load_text_files(raw_dir)
    processed = []
    for doc in raw_docs:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            processed.append({
                "text": chunk,
                "source": doc["source"],
                "page": i + 1,
                "chunk_index": i
            })
    logger.info(f"Ingested {len(processed)} chunks from {len(raw_docs)} documents")
    return processed


# Default sample documents for demo
SAMPLE_DOCUMENTS = [
    {"text": "Employees are entitled to 20 days of annual leave per year.", "source": "employee_policy.pdf", "page": 4},
    {"text": "Annual leave must be approved by the department manager.", "source": "company_handbook.pdf", "page": 12},
    {"text": "Remote work is allowed up to 3 days per week with manager approval.", "source": "remote_work_policy.pdf", "page": 1},
    {"text": "The company provides health insurance for all full-time employees.", "source": "benefits_guide.pdf", "page": 7},
    {"text": "Performance reviews are conducted every 6 months by the direct supervisor.", "source": "hr_handbook.pdf", "page": 15},
    {"text": "New employees must complete onboarding training within the first 2 weeks.", "source": "onboarding_guide.pdf", "page": 3},
    {"text": "The IT department handles all software installation requests via the internal helpdesk portal.", "source": "it_guidelines.pdf", "page": 8},
    {"text": "Expense reimbursement requires submission within 30 days with valid receipts.", "source": "finance_policy.pdf", "page": 22},
]