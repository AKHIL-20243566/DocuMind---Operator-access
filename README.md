# DocuMind

A Hybrid LLM + RAG document assistant. Upload documents and ask questions — the system retrieves relevant passages from your files and generates grounded answers using a local LLM.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![React](https://img.shields.io/badge/React-19-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-yellow)
![Ollama](https://img.shields.io/badge/Ollama-LLM-ff6600)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [RAG Pipeline](#rag-pipeline)
4. [Frontend](#frontend)
5. [Backend](#backend)
6. [Technologies](#technologies)
7. [Installation](#installation)
8. [Running Locally](#running-locally)
9. [API Endpoints](#api-endpoints)
10. [Security](#security)
11. [Team Members](#team-members)
12. [Issues Encountered](#issues-encountered)
13. [Resources](#resources)
14. [Future Improvements](#future-improvements)

---

## Overview

DocuMind lets users upload PDF, DOCX, TXT, CSV, and Markdown files. Documents are split into chunks, embedded, and indexed in a FAISS vector store. When a question is asked, the system retrieves the most relevant chunks and passes them as context to a local LLM (via Ollama) to produce an answer.

The system operates in **hybrid mode**: if retrieval scores are above a configurable threshold the answer is grounded in document context (RAG mode); otherwise the LLM answers from general knowledge (LLM mode).

---

## Architecture

```
User Query
  -> Embedding Model (sentence-transformers/all-MiniLM-L6-v2)
  -> FAISS Vector Retrieval (L2 similarity search)
  -> Relevance Threshold Check
  -> Context Construction (top-k chunks)
  -> Prompt Generation (hybrid template)
  -> LLM Response (Ollama, streamed via SSE)
```

**Components:**

| Layer      | Technology               |
|------------|--------------------------|
| Frontend   | React 19 + Vite 7       |
| Backend    | Python / FastAPI         |
| Vector DB  | FAISS (persistent)       |
| Embeddings | sentence-transformers    |
| LLM        | Ollama (llama3 default)  |

---

## RAG Pipeline

**Type:** Naive / Vanilla RAG with hybrid fallback.

1. **Document Ingestion** — Files are parsed (`document_parser.py`) and split into overlapping chunks (~500 tokens, 50-token overlap) with sentence-boundary detection.
2. **Embedding** — Each chunk is embedded using `all-MiniLM-L6-v2` (384 dimensions).
3. **Indexing** — Embeddings are stored in a FAISS `IndexFlatL2` index. The index and document metadata persist to disk (`vector_data/`).
4. **Retrieval** — At query time the question is embedded and the top-5 nearest chunks are returned with similarity scores computed as `1 / (1 + L2_distance)`.
5. **Threshold** — If the top score exceeds `RELEVANCE_THRESHOLD` (default 0.3), context is included in the prompt; otherwise the LLM answers without context.
6. **Generation** — The LLM receives a structured prompt containing retrieved passages and generates a response, streamed token-by-token via SSE.

---

## Frontend

Built with **React 19** and **Vite 7**. Key features:

- Dark theme with professional SaaS-style design
- Responsive layout — works on desktop, tablet, and mobile
- Collapsible sidebar with session management
- Document panel — upload, list, and delete documents
- RAG insights panel — sources, confidence bar, context preview
- Streaming responses — tokens appear in real time via SSE
- Lucide React icons throughout

**Key files:**

| File | Purpose |
|------|---------|
| `Dashboard.jsx` | Main page — layout, state, chat logic |
| `ChatBox.jsx` | Input field and send button |
| `Message.jsx` | Renders user/AI messages with Markdown |
| `api.js` | API client (fetch, streaming, upload) |
| `App.css` | Full application styles |

---

## Backend

Built with **FastAPI**. Modules:

| File | Purpose |
|------|---------|
| `main.py` | API server — endpoints, security middleware |
| `rag.py` | RAG orchestrator — ingestion, retrieval, deletion |
| `llm.py` | LLM integration — Ollama, prompts, streaming |
| `vector_store.py` | FAISS index — add, search, persist, load |
| `embeddings.py` | Sentence-transformer embedding model |
| `document_parser.py` | Multi-format parser with smart chunking |
| `security.py` | Prompt injection, rate limiting, API auth, logging |

---

## Project Structure

```
AI-PROJECT-LLM-RAG/
├── backend/                        # FastAPI REST API
│   ├── main.py                    # API endpoints (chat, upload, documents, health, stream)
│   ├── rag.py                     # RAG pipeline (ingest, retrieve, delete, hybrid logic)
│   ├── embeddings.py              # Sentence-transformers (all-MiniLM-L6-v2) + caching
│   ├── vector_store.py            # Persistent FAISS index (save/load/add/remove)
│   ├── llm.py                     # Ollama interface + hybrid prompts + streaming
│   ├── document_parser.py         # Multi-format parser (PDF, DOCX, TXT, CSV, MD)
│   ├── security.py                # Prompt injection, rate limiting, API auth, audit logging
│   ├── requirements.txt           # Python dependencies
│   ├── test_backend.py            # Automated tests
│   └── vector_data/               # Persistent FAISS index storage
│
├── ida-frontend/                  # React 19 + Vite frontend
│   ├── src/
│   │   ├── App.jsx               # Root component
│   │   ├── pages/Dashboard.jsx   # Main UI (chat + docs + RAG panel)
│   │   ├── components/           # ChatBox, Message
│   │   └── services/api.js       # API client (chat, stream, upload, documents)
│   └── package.json
│
├── rag-llm-assistant/            # Modular RAG library (standalone)
│   ├── app/                      # Embedder, retriever, pipeline, prompts, API
│   ├── config/settings.py        # Configuration
│   ├── tests/                    # Module tests
│   └── requirements.txt
│
├── docker-compose.yml            # Full stack (backend + frontend + Ollama)
├── Dockerfile.backend
├── Dockerfile.frontend
├── .env.example
└── README.md
```

---

## Technologies

- **Python 3.11+** — Backend runtime
- **FastAPI** — HTTP framework
- **FAISS** — Vector similarity search (Facebook AI)
- **sentence-transformers** — Text embeddings
- **Ollama** — Local LLM inference
- **React 19** — Frontend UI framework
- **Vite 7** — Frontend build tool
- **lucide-react** — Icon library
- **react-markdown** — Markdown rendering
- **Docker** — Containerization

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Ollama installed and running (`ollama serve`)
- A model pulled (e.g. `ollama pull llama3`)

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd ida-frontend
npm install
```

### Environment

```bash
cp .env.example .env
# Edit .env with your values
```

---

## Running Locally

### Start Ollama

```bash
ollama serve
```

### Start Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
cd ida-frontend
npm run dev
```

The frontend runs at `http://localhost:5173` and the API at `http://localhost:8000`.

Without Ollama, the system still works — it returns document-based answers in fallback mode.

### Run Tests

```bash
cd backend && pytest test_backend.py -v
```

### Docker

```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + Ollama status |
| `POST` | `/chat` | Ask a question (non-streaming) |
| `POST` | `/chat/stream` | Ask a question (SSE streaming) |
| `POST` | `/upload` | Upload a document |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{name}` | Delete a document |

### POST /chat

```json
// Request
{ "question": "What is the refund policy?" }

// Response
{
  "answer": "According to the document...",
  "sources": [{"doc": "policy.pdf", "page": 4, "score": 0.92}],
  "confidence": 0.87,
  "context": ["Relevant passage from the document..."],
  "mode": "rag",
  "ollama_connected": true
}
```

### POST /upload

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/upload
# {"success": true, "filename": "handbook.pdf", "chunks": 42}
```

### Streaming

The `/chat/stream` endpoint returns Server-Sent Events:
- `meta` event — sources, confidence, mode
- `token` events — individual response tokens
- `done` event — signals completion

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3` | LLM model name |
| `RELEVANCE_THRESHOLD` | `0.3` | Hybrid RAG threshold (0-1) |
| `DOCUMIND_API_KEY` | *(empty)* | API key for authentication (empty = disabled) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per minute per IP |

---

## Security

DocuMind includes several security protections defined in `security.py`:

**Prompt Injection Defense** — 13 compiled regex patterns detect common injection attempts (e.g. "ignore previous instructions", "reveal system prompt"). Blocked queries return HTTP 400.

**API Key Authentication** — Optional Bearer token authentication. Set `DOCUMIND_API_KEY` in `.env` to enable. When empty, auth is disabled.

**Rate Limiting** — In-memory per-IP rate limiting. Default: 30 requests per minute. Configurable via `RATE_LIMIT_PER_MINUTE`.

**Input Sanitization** — Control characters are stripped, input is truncated to 5000 characters.

**Query Audit Logging** — Every query is logged with: question text, response time, sources used, mode (RAG/LLM), and client IP.

---

## Team Members

| Name | Role |
|------|------|
| Akhil | Developer |

---

## Issues Encountered

1. **FAISS Unicode path crash** — `faiss.write_index()` uses C++ `fopen()` which fails on Windows paths containing non-ASCII characters (e.g. Cyrillic). Fixed by using `faiss.serialize_index()` with Python file I/O.

2. **FAISS 1D/2D array mismatch** — The embedding function returned a 1D array but FAISS expects 2D. Fixed with `np.atleast_2d()`.

3. **Misleading confidence scores** — Raw FAISS similarity scores (`1/(1+L2)`) cluster around 0.4-0.7 for typical queries, making them unintuitive. Implemented rescaling formula for more meaningful ranges.

4. **Vite default styles** — Default `index.css` from Vite template included `body { display: flex }` and large heading sizes that broke the application layout.

5. **SSE streaming on Windows** — Required careful handling of `StreamingResponse` and `ReadableStream` API on the frontend.

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [sentence-transformers](https://www.sbert.net/)
- [Ollama](https://ollama.ai/)
- [React Documentation](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [Lucide Icons](https://lucide.dev/)

---

## Future Improvements

- Chunk-level re-ranking for higher retrieval accuracy
- Multi-model support (switch between different Ollama models)
- Conversation memory across sessions (persistent chat history)
- User authentication with login/signup
- Cloud deployment with managed vector database
- PDF page-level navigation in source references
- Support for image-based documents (OCR)
- Advanced RAG strategies (HyDE, multi-query, iterative retrieval)

---

## License

MIT
