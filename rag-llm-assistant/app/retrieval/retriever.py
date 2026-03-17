"""Retriever module — FAISS vector search."""
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class Retriever:
    """FAISS-based document retriever."""

    def __init__(self):
        self.index = None
        self.documents = []

    def build_index(self, embeddings, documents):
        """Build FAISS index from embeddings and documents."""
        embeddings_np = np.array(embeddings, dtype="float32")
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        dim = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings_np)
        self.documents = documents
        logger.info(f"Built FAISS index: {len(documents)} docs, dim={dim}")

    def search(self, query_embedding, k=3):
        """Search for the top-k most relevant documents."""
        if self.index is None:
            logger.warning("Index not built yet")
            return []
        query_np = np.array(query_embedding, dtype="float32")
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        D, I = self.index.search(query_np, k)
        results = []
        for i, dist in zip(I[0], D[0]):
            if 0 <= i < len(self.documents):
                doc = dict(self.documents[i])
                doc["score"] = float(1 / (1 + dist))
                results.append(doc)
        return results

    @property
    def is_ready(self):
        return self.index is not None and len(self.documents) > 0