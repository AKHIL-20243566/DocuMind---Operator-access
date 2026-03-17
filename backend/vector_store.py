import faiss
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# SAFE PATH INITIALIZATION
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE_DIR, "vector_data")

INDEX_PATH = os.path.join(STORE_DIR, "faiss.index")
DOCS_PATH = os.path.join(STORE_DIR, "documents.json")

index = None
documents = []


def _ensure_store_dir():
    """Ensure the vector storage directory exists."""
    os.makedirs(STORE_DIR, exist_ok=True)


# ------------------------------------------------------------------
# INDEX CREATION
# ------------------------------------------------------------------

def create_index(embeddings, docs):
    """Create a new FAISS index from embeddings and documents."""
    global index, documents

    _ensure_store_dir()

    embeddings_np = np.array(embeddings, dtype="float32")

    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)

    dim = embeddings_np.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    documents = docs

    save_index()

    logger.info(f"FAISS index created with {len(docs)} documents, dim={dim}")


# ------------------------------------------------------------------
# ADD DOCUMENTS
# ------------------------------------------------------------------

def add_to_index(embeddings, new_docs):
    """Add new documents to the existing index."""
    global index, documents

    embeddings_np = np.array(embeddings, dtype="float32")

    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)

    if index is None:
        create_index(embeddings_np, new_docs)
        return

    index.add(embeddings_np)

    documents.extend(new_docs)

    save_index()

    logger.info(f"Added {len(new_docs)} documents to index (total: {len(documents)})")


# ------------------------------------------------------------------
# REMOVE DOCUMENTS
# ------------------------------------------------------------------

def remove_by_source(source_name):
    """Remove all documents from a specific source and rebuild the index."""
    global index, documents

    if not documents:
        return 0

    from embeddings import embed

    original_count = len(documents)

    documents = [d for d in documents if d.get("source") != source_name]

    removed = original_count - len(documents)

    if removed > 0:

        if documents:

            texts = [d["text"] for d in documents]

            embeddings = embed(texts)

            embeddings_np = np.array(embeddings, dtype="float32")

            dim = embeddings_np.shape[1]

            index = faiss.IndexFlatL2(dim)

            index.add(embeddings_np)

        else:

            index = None

        save_index()

        logger.info(f"Removed {removed} chunks from source '{source_name}'")

    return removed


# ------------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------------

def search(query_embedding, k=3):
    """Search for the top-k most similar documents."""

    if index is None or len(documents) == 0:
        logger.warning("Search called on empty index")
        return []

    query_np = np.array(query_embedding, dtype="float32")

    if query_np.ndim == 1:
        query_np = query_np.reshape(1, -1)

    actual_k = min(k, len(documents))

    D, I = index.search(query_np, actual_k)

    results = []

    for i, dist in zip(I[0], D[0]):

        if 0 <= i < len(documents):

            doc = dict(documents[i])

            doc["score"] = float(1 / (1 + dist))

            results.append(doc)

    return results


# ------------------------------------------------------------------
# SAVE INDEX
# ------------------------------------------------------------------

def save_index():
    """Persist FAISS index and documents to disk."""

    _ensure_store_dir()

    if index is not None:
        try:
            # Use serialize + Python file I/O to handle Unicode paths on Windows
            index_bytes = faiss.serialize_index(index)
            with open(INDEX_PATH, "wb") as f:
                f.write(index_bytes)
        except Exception as e:
            logger.error(f"Failed to write FAISS index: {e}")
            raise

    try:
        with open(DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save documents metadata: {e}")
        raise

    logger.info(f"Index saved: {len(documents)} documents")


# ------------------------------------------------------------------
# LOAD INDEX
# ------------------------------------------------------------------

def load_index():
    """Load FAISS index and documents from disk."""

    global index, documents

    _ensure_store_dir()

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):

        try:
            # Use deserialize + Python file I/O to handle Unicode paths on Windows
            with open(INDEX_PATH, "rb") as f:
                index_bytes = f.read()
            index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype="uint8"))

            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                documents = json.load(f)

            logger.info(f"Index loaded: {len(documents)} documents")

            return True

        except Exception as e:

            logger.error(f"Failed to load index: {e}")

            return False

    return False


# ------------------------------------------------------------------
# DOCUMENT LIST
# ------------------------------------------------------------------

def get_document_list():
    """Get a summary of all indexed documents grouped by source."""

    sources = {}

    for doc in documents:

        src = doc.get("source", "Unknown")

        if src not in sources:
            sources[src] = {
                "name": src,
                "chunks": 0,
                "pages": set()
            }

        sources[src]["chunks"] += 1

        sources[src]["pages"].add(doc.get("page", 1))

    result = []

    for src, info in sources.items():

        result.append({
            "name": info["name"],
            "chunks": info["chunks"],
            "pages": len(info["pages"])
        })

    return result


# ------------------------------------------------------------------
# DOCUMENT COUNT
# ------------------------------------------------------------------

def get_total_documents():
    return len(documents)