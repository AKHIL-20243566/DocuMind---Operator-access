"""Tests for the backend RAG system."""
import sys
import os
import pytest
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))


class TestEmbeddings:
    """Test the embeddings module."""

    def test_embed_single_text(self):
        from embeddings import embed
        result = embed("Hello world")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert result.shape[1] == 384

    def test_embed_multiple_texts(self):
        from embeddings import embed
        texts = ["Hello world", "How are you?", "Test document"]
        result = embed(texts)
        assert result.shape == (3, 384)

    def test_embed_cached(self):
        from embeddings import embed_cached
        r1 = embed_cached(["Hello world"])
        r2 = embed_cached(["Hello world"])
        assert (r1 == r2).all()

    def test_embed_similar_texts_closer(self):
        from embeddings import embed
        texts = ["annual leave policy", "vacation days off", "python programming"]
        embeddings = embed(texts)
        from numpy.linalg import norm
        def cosine_sim(a, b):
            return np.dot(a, b) / (norm(a) * norm(b))
        sim_related = cosine_sim(embeddings[0], embeddings[1])
        sim_unrelated = cosine_sim(embeddings[0], embeddings[2])
        assert sim_related > sim_unrelated


class TestVectorStore:
    """Test FAISS vector store."""

    def teardown_method(self, method):
        """Reset the vector store to a clean empty state after each test so
        test data (e.g. test.pdf) never leaks into the persisted documents.json."""
        import vector_store as vs
        vs.index = None
        vs.documents = []
        vs._embeddings = None
        vs.save_index()

    def test_create_and_search(self):
        from vector_store import create_index, search
        from embeddings import embed
        docs = [
            {"text": "Annual leave is 20 days", "source": "policy.pdf", "page": 1},
            {"text": "Remote work allowed 3 days", "source": "remote.pdf", "page": 1},
        ]
        embeddings = embed([d["text"] for d in docs])
        create_index(embeddings, docs)
        results = search(embed("how many days of leave?"), k=1)
        assert len(results) == 1
        assert "leave" in results[0]["text"].lower()
        assert "score" in results[0]

    def test_add_to_index(self):
        from vector_store import create_index, add_to_index, search, get_total_documents
        from embeddings import embed
        docs1 = [{"text": "First doc", "source": "a.pdf", "page": 1}]
        create_index(embed([d["text"] for d in docs1]), docs1)
        docs2 = [{"text": "Second doc", "source": "b.pdf", "page": 1}]
        add_to_index(embed([d["text"] for d in docs2]), docs2)
        assert get_total_documents() == 2

    def test_remove_by_source(self):
        from vector_store import create_index, remove_by_source, get_total_documents
        from embeddings import embed
        docs = [
            {"text": "Doc A", "source": "a.pdf", "page": 1},
            {"text": "Doc B", "source": "b.pdf", "page": 1},
        ]
        create_index(embed([d["text"] for d in docs]), docs)
        removed = remove_by_source("a.pdf")
        assert removed == 1
        assert get_total_documents() == 1

    def test_document_list(self):
        from vector_store import create_index, get_document_list
        from embeddings import embed
        docs = [
            {"text": "Chunk 1", "source": "test.pdf", "page": 1},
            {"text": "Chunk 2", "source": "test.pdf", "page": 2},
        ]
        create_index(embed([d["text"] for d in docs]), docs)
        doc_list = get_document_list()
        assert len(doc_list) == 1
        assert doc_list[0]["name"] == "test.pdf"
        assert doc_list[0]["chunks"] == 2


class TestDocumentParser:
    """Test document parsing."""

    def test_parse_txt_bytes(self):
        from document_parser import parse_bytes
        content = b"This is a test document with some text content."
        chunks = parse_bytes(content, "test.txt")
        assert len(chunks) > 0
        assert chunks[0]["source"] == "test.txt"
        assert "text" in chunks[0]

    def test_parse_csv_bytes(self):
        from document_parser import parse_bytes
        content = b"Name,Role\nAlice,Engineer\nBob,Manager"
        chunks = parse_bytes(content, "data.csv")
        assert len(chunks) == 2

    def test_chunk_text(self):
        from document_parser import chunk_text
        text = "A" * 1200
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2

    def test_supported_formats(self):
        from document_parser import is_supported
        assert is_supported("test.pdf")
        assert is_supported("test.docx")
        assert is_supported("test.txt")
        assert is_supported("test.csv")
        assert is_supported("test.md")
        assert not is_supported("test.exe")

    def test_pdf_normal_extraction_success(self, monkeypatch):
        from document_parser import parse_bytes_with_diagnostics

        monkeypatch.setattr(
            "document_parser._parse_pdf_bytes",
            lambda content, filename: [{"text": "Normal PDF text", "source": filename, "page": 1, "chunk_id": "c1"}],
        )

        result = parse_bytes_with_diagnostics(b"%PDF-1.4", "normal.pdf")
        assert result["success"] is True
        assert result["loader_used"] == "PyPDFLoader"
        assert result["ocr_triggered"] is False
        assert len(result["documents"]) > 0

    def test_pdf_scanned_triggers_ocr(self, monkeypatch):
        from document_parser import parse_bytes_with_diagnostics

        monkeypatch.setattr("document_parser._parse_pdf_bytes", lambda content, filename: [])
        monkeypatch.setattr("document_parser._parse_pdf_with_unstructured_loader", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            "document_parser._parse_pdf_with_ocr",
            lambda content, filename: [{"text": "OCR extracted text", "source": filename, "page": 1, "chunk_id": "ocr1"}],
        )

        result = parse_bytes_with_diagnostics(b"%PDF-1.4", "scanned.pdf")
        assert result["success"] is True
        assert result["loader_used"] == "OCR"
        assert result["ocr_triggered"] is True
        assert "Scanned document detected" in result["status_messages"]
        assert "Applying OCR..." in result["status_messages"]

    def test_pdf_empty_after_ocr_returns_error(self, monkeypatch):
        from document_parser import parse_bytes_with_diagnostics

        monkeypatch.setattr("document_parser._parse_pdf_bytes", lambda content, filename: [])
        monkeypatch.setattr("document_parser._parse_pdf_with_unstructured_loader", lambda *args, **kwargs: [])
        monkeypatch.setattr("document_parser._parse_pdf_with_ocr", lambda content, filename: [])

        result = parse_bytes_with_diagnostics(b"%PDF-1.4", "empty.pdf")
        assert result["success"] is False
        assert result["error_code"] == "NO_READABLE_TEXT"
        assert "No readable text found" in result["status_messages"]

    def test_pdf_corrupted_handled_gracefully(self, monkeypatch):
        from document_parser import parse_bytes_with_diagnostics

        monkeypatch.setattr("document_parser._parse_pdf_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad pdf")))
        monkeypatch.setattr("document_parser._parse_pdf_with_unstructured_loader", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad unstructured")))
        monkeypatch.setattr("document_parser._parse_pdf_with_ocr", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad ocr")))

        result = parse_bytes_with_diagnostics(b"not-a-pdf", "corrupt.pdf")
        assert result["success"] is False
        assert result["error_code"] == "CORRUPTED_OR_UNREADABLE_PDF"

    def test_pdf_ocr_dependency_missing_reported(self, monkeypatch):
        from document_parser import parse_bytes_with_diagnostics

        monkeypatch.setattr("document_parser._parse_pdf_bytes", lambda content, filename: [])
        monkeypatch.setattr("document_parser._parse_pdf_with_unstructured_loader", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            "document_parser._parse_pdf_with_ocr",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("poppler not found")),
        )

        result = parse_bytes_with_diagnostics(b"%PDF-1.4", "scanned.pdf")
        assert result["success"] is False
        assert result["error_code"] == "OCR_DEPENDENCY_MISSING"
        assert "OCR engine unavailable" in result["status_messages"]


class TestHybridRAG:
    """Test hybrid RAG logic."""

    def test_should_use_context_high_score(self):
        from llm import should_use_context
        docs = [{"score": 0.8, "text": "test"}]
        assert should_use_context(docs) is True

    def test_should_use_context_low_score(self):
        from llm import should_use_context
        docs = [{"score": 0.1, "text": "test"}]
        assert should_use_context(docs) is False

    def test_should_use_context_empty(self):
        from llm import should_use_context
        assert should_use_context([]) is False

    def test_build_hybrid_prompt_with_context(self):
        from llm import build_hybrid_prompt
        docs = [{"text": "Leave is 20 days", "source": "policy.pdf", "page": 1}]
        prompt = build_hybrid_prompt("How many leave days?", docs, use_context=True)
        assert "Leave is 20 days" in prompt
        assert "DocuMind" in prompt

    def test_build_hybrid_prompt_without_context(self):
        from llm import build_hybrid_prompt
        prompt = build_hybrid_prompt("What is Python?", [], use_context=False)
        assert "general knowledge" in prompt.lower()


class TestRAG:
    """Test RAG pipeline."""

    def test_retrieve_context(self):
        from rag import retrieve_context
        results = retrieve_context("How many days of annual leave?")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "text" in results[0]
        assert "source" in results[0]
        assert "score" in results[0]

    def test_list_documents(self):
        from rag import list_documents
        docs = list_documents()
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_ingest_and_delete(self):
        from rag import ingest_file, delete_document, list_documents
        content = b"Test upload content for automated testing."
        result = ingest_file(content, "test_upload.txt")
        assert result["success"] is True
        assert result["chunks"] > 0

        # Verify it's listed
        docs = list_documents()
        names = [d["name"] for d in docs]
        assert "test_upload.txt" in names

        # Delete it
        del_result = delete_document("test_upload.txt")
        assert del_result["success"] is True


class TestAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ollama" in data

    def test_chat(self, client):
        resp = client.post("/chat", json={"question": "How many days of annual leave?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "context" in data
        assert "mode" in data
        assert data["mode"] in ("rag", "llm")

    def test_chat_empty_question(self, client):
        resp = client.post("/chat", json={"question": "   "})
        assert resp.status_code == 400

    def test_get_documents(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total" in data

    def test_upload_txt(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("test_api.txt", b"This is test content for API upload.", "text/plain")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["chunks"] > 0

        # Cleanup
        client.delete("/documents/test_api.txt")

    def test_upload_unsupported(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("test.exe", b"binary content", "application/octet-stream")}
        )
        assert resp.status_code == 400

    def test_delete_nonexistent(self, client):
        resp = client.delete("/documents/nonexistent_doc.pdf")
        assert resp.status_code == 404

    def test_chat_stream(self, client):
        resp = client.post("/chat/stream", json={"question": "What is remote work policy?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_chat_no_documents_guard(self, client, monkeypatch):
        monkeypatch.setattr("main.has_embeddings", lambda: False)
        resp = client.post("/chat", json={"question": "Any docs?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "No documents available for this chat"
