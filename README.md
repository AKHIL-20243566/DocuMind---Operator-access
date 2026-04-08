# DocuMind

A Hybrid LLM + RAG document assistant. Upload documents and ask questions — the system retrieves relevant passages from your files and generates grounded answers using a local LLM.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![React](https://img.shields.io/badge/React-19-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![FAISS](https://img.shields.io/badge/FAISS-HNSW_Vector_DB-yellow)
![Ollama](https://img.shields.io/badge/Ollama-LLM-ff6600)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-Deep_Learning_OCR-blue)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [RAG Pipeline](#rag-pipeline)
4. [OCR Pipeline](#ocr-pipeline)
5. [Vectorless Tree (PageIndex)](#vectorless-tree-pageindex)
6. [Frontend](#frontend)
7. [Backend](#backend)
8. [Technologies](#technologies)
9. [Installation](#installation)
10. [Running Locally](#running-locally)
11. [API Endpoints](#api-endpoints)
12. [Security](#security)
13. [Team Members](#team-members)
14. [Issues Encountered](#issues-encountered)
15. [Resources](#resources)
16. [Future Improvements](#future-improvements)

---

## Overview

DocuMind lets users upload PDF, DOCX, TXT, CSV, Markdown, and image files (PNG, JPG, TIFF, etc.). Documents are split into chunks, embedded, and indexed in a FAISS vector store. When a question is asked, the system uses a three-stage retrieval pipeline to find the most relevant passages and passes them as context to a local LLM (via Ollama) to produce an answer.

The system operates in **hybrid mode**: if retrieval scores are above a configurable threshold the answer is grounded in document context (RAG mode); otherwise the LLM answers from general knowledge (LLM mode).

**Scanned documents and images** are handled through a deep-learning OCR pipeline (PaddleOCR) that extracts text before indexing, making even non-digital documents fully searchable.

---

## Architecture

```
File Upload
  -> document_parser.py
       ├── Native text extraction (PyPDF / Unstructured)
       └── OCR fallback: PaddleOCR → EasyOCR → Tesseract
  -> Chunk + embed (all-MiniLM-L6-v2, 384-dim)
  -> FAISS HNSW index + BM25 index + PageIndex tree

User Query
  -> Stage 1: PageIndex (vectorless tree — LLM picks relevant sections)
  -> Stage 2: Hybrid retrieval (FAISS HNSW + BM25, fused via RRF)
  -> Stage 3: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  -> Context construction → Prompt → Ollama LLM → SSE stream
```

**Components:**

| Layer          | Technology                              |
|----------------|-----------------------------------------|
| Frontend       | React 19 + Vite 7                       |
| Backend        | Python / FastAPI                        |
| Vector DB      | FAISS HNSW (persistent)                 |
| Keyword search | BM25 (Okapi)                            |
| Embeddings     | sentence-transformers all-MiniLM-L6-v2  |
| Reranker       | cross-encoder ms-marco-MiniLM-L-6-v2   |
| LLM            | Ollama (llama3 default)                 |
| OCR            | PaddleOCR → EasyOCR → Tesseract         |

---

## RAG Pipeline

**Type:** Advanced hybrid RAG — three-stage retrieval with vectorless pre-filtering.

### Ingestion

1. **Parse** — `document_parser.py` applies a layered fallback strategy:
   - Native text extraction (PyPDFLoader, UnstructuredPDFLoader) for digital PDFs/DOCX.
   - Deep-learning OCR (PaddleOCR cascade) for scanned PDFs and image uploads.
2. **Chunk** — Structure-aware chunking preserves heading boundaries and table rows (~800 chars max, 100-char overlap). Raw/OCR text uses sliding-window chunking (~500 chars, 50 overlap).
3. **Embed** — Each chunk is embedded with `all-MiniLM-L6-v2` (384 dimensions, batch size 32). An MD5-keyed in-memory cache avoids re-embedding repeated texts.
4. **Index** — Embeddings are added to a FAISS `IndexHNSWFlat` (M=32, efConstruction=200). Metadata is stored in `documents.json`, the embedding matrix in `embeddings.npy`. A `PageIndex` hierarchical tree is rebuilt after each ingest/delete.

### Retrieval (three stages)

**Stage 1 — Vectorless section selection (PageIndex)**
The LLM reads lightweight section summaries from the PageIndex tree and selects 1–3 relevant sections. This narrows the FAISS search space without computing any embeddings — cheap and fast.

**Stage 2 — Hybrid FAISS + BM25 retrieval**
The query is embedded (with caching) and searched against the HNSW index. In parallel, BM25 keyword search runs over the same corpus. Results are fused via Reciprocal Rank Fusion (60 % FAISS + 40 % BM25). Multi-query generation expands recall when enabled.

**Stage 3 — Cross-encoder reranking**
A `cross-encoder/ms-marco-MiniLM-L-6-v2` model scores each candidate `(query, chunk)` pair and returns the final top-k results in order of relevance.

### Generation

The LLM receives a structured prompt with retrieved passages and generates a response streamed token-by-token via SSE. Answers are validated for grounding; ungrounded responses trigger one automatic retry with expanded context. Non-streaming answers are LRU-cached (100 entries, 300 s TTL).

---

## OCR Pipeline

DocuMind replaces legacy Tesseract OCR with a **deep-learning cascade**:

```
Scanned PDF / Image
  -> pdf2image (Poppler) converts PDF pages to PIL images
  -> _preprocess_image_for_ocr()  — grayscale + upscale + sharpen + contrast
  -> PaddleOCR  (primary)   — detection + recognition + angle classification
       ↓ if empty output
  -> EasyOCR   (secondary)  — alternative DL engine
       ↓ if empty output
  -> Tesseract (last resort) — legacy LSTM engine
  -> clean_ocr_text()  — NFC normalise, fix line-breaks, filter noise
  -> chunk_text()  — sliding-window chunking
  -> embed + FAISS index
```

**Key features:**

| Feature | Detail |
|---------|--------|
| Rotation handling | PaddleOCR `use_angle_cls=True` corrects rotated / skewed pages natively |
| Confidence filtering | Lines with PaddleOCR confidence < 0.5 are discarded |
| Parallel pages | `ThreadPoolExecutor(4)` processes PDF pages concurrently |
| Disk cache | SHA-256 hash of file bytes + page number → `vector_data/ocr_cache.json`; re-uploading the same file skips all OCR |
| Text cleaning | Hyphenated line-break fixing, Unicode normalisation, noise-line filtering (< 30 % alphanumeric), duplicate-line deduplication |
| Image formats | PNG, JPG, JPEG, TIFF, BMP, WEBP |

**Module:** `backend/ocr_engine.py` — self-contained, importable separately.

**Migration utility:** `backend/scripts/reindex_ocr.py` — clears stale Tesseract cache, identifies OCR-tagged documents in the index, and prints the re-upload commands needed to re-process them with PaddleOCR.

---

## Vectorless Tree (PageIndex)

The **PageIndex** is a lightweight hierarchical document tree built from the section and heading metadata already present on every chunk — no embeddings required.

```
PageIndex
  └── DocumentNode  (one per uploaded file)
        └── SectionNode  (one per unique section + heading within the file)
              ├── section   — classified type: "abstract", "methodology", "results", …
              ├── heading   — raw heading text
              ├── summary   — first 400 chars of combined chunk text
              └── chunk_ids — references back into the FAISS/BM25 index
```

**Why it matters:** Before running the expensive FAISS nearest-neighbour search, `retrieve_context()` asks the LLM to read the section summaries and identify which 1–3 sections are relevant to the query. The FAISS search is then restricted to chunks from those sections, reducing noise and improving precision — with no vector computation in the first pass.

**Persistence:** The tree is serialised to `vector_data/page_index.json` and survives restarts. It is rebuilt automatically after every document ingest or delete.

**Module:** `backend/page_index.py`

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
| `main.py` | API server — endpoints, security middleware, answer cache |
| `rag.py` | RAG orchestrator — ingestion, three-stage retrieval, deletion |
| `llm.py` | LLM integration — Ollama, prompts, streaming |
| `vector_store.py` | FAISS HNSW index — add, search, persist, load, hybrid RRF |
| `embeddings.py` | Sentence-transformer embedding model + MD5 cache |
| `document_parser.py` | Multi-format parser with structure-aware chunking |
| `ocr_engine.py` | PaddleOCR → EasyOCR → Tesseract cascade + SHA-256 disk cache |
| `page_index.py` | Vectorless hierarchical document tree (DocumentNode / SectionNode) |
| `bm25.py` | Okapi BM25 keyword index |
| `reranker.py` | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| `query_understanding.py` | Multi-query expansion + section selection via LLM |
| `security.py` | Prompt injection, rate limiting, API auth, audit logging |

---

## Project Structure

```
DocuMind/
├── backend/                        # FastAPI REST API
│   ├── main.py                    # API endpoints (chat, upload, documents, health, stream)
│   ├── rag.py                     # RAG pipeline (ingest, three-stage retrieve, delete)
│   ├── embeddings.py              # Sentence-transformers (all-MiniLM-L6-v2) + caching
│   ├── vector_store.py            # Persistent FAISS HNSW index + BM25 hybrid search
│   ├── llm.py                     # Ollama interface + hybrid prompts + streaming
│   ├── document_parser.py         # Multi-format parser (PDF, DOCX, TXT, CSV, MD, images)
│   ├── ocr_engine.py              # PaddleOCR cascade + disk OCR result cache
│   ├── page_index.py              # Vectorless hierarchical document tree
│   ├── bm25.py                    # Okapi BM25 keyword index
│   ├── reranker.py                # Cross-encoder reranking
│   ├── query_understanding.py     # Multi-query expansion + LLM section selection
│   ├── security.py                # Prompt injection, rate limiting, API auth, audit logging
│   ├── requirements.txt           # Python dependencies
│   ├── test_backend.py            # Automated tests
│   ├── scripts/
│   │   └── reindex_ocr.py         # OCR migration utility (Tesseract → PaddleOCR)
│   └── vector_data/               # Persistent storage
│       ├── faiss.index            # FAISS HNSW binary
│       ├── documents.json         # Chunk metadata
│       ├── embeddings.npy         # Cached embedding matrix
│       ├── page_index.json        # Vectorless tree
│       └── ocr_cache.json         # SHA-256 keyed OCR result cache
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
- **FAISS (HNSW)** — Approximate nearest-neighbour vector search (Facebook AI)
- **sentence-transformers** — Text embeddings (all-MiniLM-L6-v2)
- **PaddleOCR / PaddlePaddle** — Deep-learning OCR (primary engine)
- **EasyOCR** — Alternative DL OCR engine (fallback)
- **Tesseract / pytesseract** — Legacy OCR engine (last-resort fallback)
- **pdf2image / Poppler** — PDF to image conversion for OCR
- **BM25** — Keyword retrieval (Okapi BM25)
- **cross-encoder** — Result reranking (ms-marco-MiniLM-L-6-v2)
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
- Poppler (for PDF-to-image conversion in OCR path)
  - Windows: `winget install oschwartz10612.Poppler`
  - macOS: `brew install poppler`
  - Linux: `apt install poppler-utils`

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

> **Note:** `paddleocr` and `paddlepaddle` will download model weights (~50 MB) on first use. Ensure internet access on the initial run, or pre-download:
> ```bash
> python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
> ```

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

### OCR Migration (Tesseract → PaddleOCR)

If you have an existing index built with Tesseract, run the migration utility:

```bash
python backend/scripts/reindex_ocr.py          # report only
python backend/scripts/reindex_ocr.py --purge  # remove stale OCR chunks + rebuild FAISS
```

Then re-upload the affected files via the API or frontend.

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
# {"success": true, "filename": "handbook.pdf", "chunks": 42, "loader_used": "PyPDFLoader", "ocr_triggered": false}

curl -F "file=@scanned.pdf" http://localhost:8000/upload
# {"success": true, "filename": "scanned.pdf", "chunks": 18, "loader_used": "OCR", "ocr_triggered": true}
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
| `VECTOR_DATA_DIR` | `vector_data` | Directory for FAISS index, documents, OCR cache |
| `USE_PAGE_INDEX` | `true` | Enable vectorless section pre-filtering |
| `ENABLE_MULTI_QUERY` | `false` | Enable multi-query expansion for recall improvement |
| `RERANK_EXPAND` | `3` | Reranker candidate multiplier (k × RERANK_EXPAND) |

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

6. **PaddleOCR thread-safe initialisation** — PaddleOCR's model loader is not thread-safe. Fixed with a `threading.Lock()` guard around the singleton initialisation while keeping multi-threaded page inference intact.

7. **Tesseract → PaddleOCR args tuple mismatch** — The `_ocr_single_page` worker function was refactored to accept a three-element tuple `(page_num, image, file_hash)` instead of two; all call sites updated to pass the SHA-256 hash for cache keying.

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [sentence-transformers](https://www.sbert.net/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Ollama](https://ollama.ai/)
- [React Documentation](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [Lucide Icons](https://lucide.dev/)

---

## Future Improvements

- Multi-model support (switch between different Ollama models)
- Conversation memory across sessions (persistent chat history)
- User authentication with login/signup
- Cloud deployment with managed vector database
- PDF page-level navigation in source references
- GPU-accelerated OCR and embedding inference
- Table structure extraction from scanned documents
- Advanced RAG strategies (HyDE, iterative retrieval)

---

## License

MIT
