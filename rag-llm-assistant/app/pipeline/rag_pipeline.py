"""RAG Pipeline — orchestrates the full retrieval-augmented generation flow."""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.embeddings.embedder import Embedder
from app.ingestion.ingest_docs import ingest_documents, SAMPLE_DOCUMENTS
from app.retrieval.retriever import Retriever
from app.llm.llama_loader import LLMLoader
from app.prompts.prompt_template import build_rag_prompt
from config.settings import (
    EMBEDDING_MODEL, OLLAMA_URL, OLLAMA_MODEL, LLM_TIMEOUT,
    DATA_RAW_DIR, RETRIEVAL_TOP_K
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline."""

    _instance = None

    def __init__(self):
        self.embedder = Embedder(model_name=EMBEDDING_MODEL)
        self.retriever = Retriever()
        self.llm = LLMLoader(ollama_url=OLLAMA_URL, model=OLLAMA_MODEL, timeout=LLM_TIMEOUT)
        self.documents = []
        self._initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, doc_dir=None):
        """Load documents, embed, and build index."""
        if self._initialized:
            return

        doc_dir = doc_dir or DATA_RAW_DIR

        # Try to load from directory first, fall back to samples
        self.documents = ingest_documents(doc_dir)
        if not self.documents:
            logger.info("No documents in data dir, using sample documents")
            self.documents = SAMPLE_DOCUMENTS

        # Embed and index
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.embedder.embed(texts)
        self.retriever.build_index(embeddings, self.documents)

        self._initialized = True
        logger.info(f"RAG pipeline initialized with {len(self.documents)} documents")

    def query(self, question, k=None):
        """Run a full RAG query: retrieve → prompt → generate."""
        if not self._initialized:
            self.initialize()

        k = k or RETRIEVAL_TOP_K

        # Retrieve
        query_embedding = self.embedder.embed(question)
        retrieved_docs = self.retriever.search(query_embedding, k=k)

        # Build prompt
        context_text = "\n\n".join([doc["text"] for doc in retrieved_docs])
        prompt = build_rag_prompt(question, retrieved_docs)

        # Generate
        answer = self.llm.generate_with_fallback(prompt, context=context_text)

        # Build sources
        sources = [
            {
                "doc": doc.get("source", "Unknown"),
                "page": doc.get("page", 0),
                "score": doc.get("score", 0)
            }
            for doc in retrieved_docs
        ]

        scores = [doc.get("score", 0) for doc in retrieved_docs]
        confidence = sum(scores) / len(scores) if scores else 0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 2),
            "context": [doc["text"] for doc in retrieved_docs]
        }