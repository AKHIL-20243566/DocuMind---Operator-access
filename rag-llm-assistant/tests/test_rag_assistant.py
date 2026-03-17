"""Tests for the rag-llm-assistant module."""
import sys
import os
import pytest

# Add rag-llm-assistant to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag-llm-assistant"))


class TestEmbedder:
    """Test the Embedder class."""

    def test_embed_single(self):
        from app.embeddings.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed("Hello world")
        assert result.shape == (1, 384)

    def test_embed_multiple(self):
        from app.embeddings.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed(["Hello", "World"])
        assert result.shape == (2, 384)

    def test_embed_cached(self):
        from app.embeddings.embedder import Embedder
        embedder = Embedder()
        result1 = embedder.embed_cached(["Hello"])
        result2 = embedder.embed_cached(["Hello"])
        assert (result1 == result2).all()
        assert "Hello" in embedder._cache


class TestIngestion:
    """Test document ingestion."""

    def test_chunk_text(self):
        from app.ingestion.ingest_docs import chunk_text
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        assert all(len(c) <= 500 for c in chunks)

    def test_chunk_short_text(self):
        from app.ingestion.ingest_docs import chunk_text
        text = "Short text"
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_sample_documents(self):
        from app.ingestion.ingest_docs import SAMPLE_DOCUMENTS
        assert len(SAMPLE_DOCUMENTS) > 0
        for doc in SAMPLE_DOCUMENTS:
            assert "text" in doc
            assert "source" in doc
            assert "page" in doc


class TestRetriever:
    """Test FAISS retriever."""

    def test_build_and_search(self):
        from app.embeddings.embedder import Embedder
        from app.retrieval.retriever import Retriever
        embedder = Embedder()
        retriever = Retriever()
        docs = [
            {"text": "Annual leave is 20 days", "source": "policy.pdf", "page": 1},
            {"text": "Remote work 3 days per week", "source": "remote.pdf", "page": 1},
        ]
        embeddings = embedder.embed([d["text"] for d in docs])
        retriever.build_index(embeddings, docs)
        assert retriever.is_ready
        query_emb = embedder.embed("leave days")
        results = retriever.search(query_emb, k=1)
        assert len(results) == 1
        assert "score" in results[0]

    def test_search_before_build(self):
        from app.retrieval.retriever import Retriever
        retriever = Retriever()
        assert not retriever.is_ready
        results = retriever.search([0.0] * 384, k=1)
        assert results == []


class TestPromptTemplate:
    """Test prompt construction."""

    def test_build_rag_prompt(self):
        from app.prompts.prompt_template import build_rag_prompt
        docs = [{"text": "Leave is 20 days"}, {"text": "Approved by manager"}]
        prompt = build_rag_prompt("How many leave days?", docs)
        assert "Leave is 20 days" in prompt
        assert "How many leave days?" in prompt
        assert "DocuMind" in prompt

    def test_build_standalone_prompt(self):
        from app.prompts.prompt_template import build_standalone_prompt
        prompt = build_standalone_prompt("What is RAG?")
        assert "What is RAG?" in prompt


class TestLLMLoader:
    """Test LLM loader."""

    def test_generate_with_fallback_no_ollama(self):
        from app.llm.llama_loader import LLMLoader
        llm = LLMLoader(ollama_url="http://localhost:99999/api/generate")
        result = llm.generate_with_fallback("test prompt", context="some context")
        assert "Fallback" in result or "context" in result.lower()


class TestRAGPipeline:
    """Test the full RAG pipeline."""

    def test_pipeline_query(self):
        from app.pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.initialize()
        result = pipeline.query("How many days of annual leave?")
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "context" in result
        assert len(result["sources"]) > 0


class TestEvaluation:
    """Test evaluation metrics."""

    def test_precision(self):
        from app.evaluation.evaluate import compute_retrieval_precision
        retrieved = [{"source": "a.pdf"}, {"source": "b.pdf"}]
        relevant = [{"source": "a.pdf"}, {"source": "c.pdf"}]
        assert compute_retrieval_precision(retrieved, relevant) == 0.5

    def test_recall(self):
        from app.evaluation.evaluate import compute_retrieval_recall
        retrieved = [{"source": "a.pdf"}, {"source": "b.pdf"}]
        relevant = [{"source": "a.pdf"}, {"source": "c.pdf"}]
        assert compute_retrieval_recall(retrieved, relevant) == 0.5

    def test_evaluate_response(self):
        from app.evaluation.evaluate import evaluate_response
        result = evaluate_response(
            "test?", "Annual leave is 20 days",
            [{"score": 0.8}],
            expected_keywords=["annual", "leave"]
        )
        assert result["has_answer"] is True
        assert result["keyword_coverage"] == 1.0
