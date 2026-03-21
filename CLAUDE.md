# DocuMind — Claude Code Guidelines

## Project Overview
Hybrid LLM + RAG document assistant. Users upload documents, ask questions, and get answers grounded in document context with streaming SSE responses.

## Services & Ports
| Service | Path | Port | Stack |
|---|---|---|---|
| Backend API | `backend/` | 8000 | FastAPI + FAISS + Ollama |
| Frontend | `ida-frontend/` | 3000 | React 19 + Vite |
| RAG Library | `rag-llm-assistant/` | — | Modular, standalone |
| Ollama LLM | docker service | 11434 | ollama/ollama |

## Key File Map
```
backend/
  main.py           — FastAPI routes (chat, upload, documents, health)
  rag.py            — RAG pipeline orchestration
  llm.py            — Ollama integration & streaming
  vector_store.py   — FAISS index (save/load/add/remove)
  embeddings.py     — sentence-transformers wrapper
  document_parser.py — Multi-format parser (PDF, DOCX, TXT, CSV, MD)
  security.py       — Prompt injection, rate limiting, API auth, audit log

ida-frontend/src/
  pages/Dashboard.jsx     — Main UI (chat + docs + RAG insights)
  components/ChatBox.jsx  — Message input
  components/Message.jsx  — Markdown renderer
  services/api.js         — Fetch/streaming/file upload client

rag-llm-assistant/app/
  llm/llama_loader.py     — Ollama integration
  pipeline/rag_pipeline.py — RAG orchestrator
  retrieval/retriever.py  — Vector DB retrieval
```

---

## MANDATORY: Pre-Completion Checklist

**Before marking ANY task as done, always run through every step below. Do not skip.**

### Step 1 — Debug
- Re-read every file that was touched in this task
- Verify there are no syntax errors, missing imports, or broken references
- Trace the execution path end-to-end mentally (or via test run)
- Check for unhandled exceptions and edge cases (empty inputs, None values, missing keys)
- Verify any async/await usage is correct

### Step 2 — Recheck
- Compare the change against the original requirement — does it actually solve what was asked?
- Check for unintended side-effects on other parts of the codebase
- Confirm no existing behavior was silently broken
- Verify environment variables are used (not hardcoded values)
- For frontend: check that API calls handle loading, error, and success states

### Step 3 — Self-Heal
- If any issue is found in steps 1 or 2, fix it immediately before proceeding
- Do not hand back broken code and note it as "TODO" — fix it now
- Re-run steps 1 and 2 after each fix until everything passes
- For Python: check `requirements.txt` is updated if new packages were added
- For JS: check `package.json` is updated if new packages were added

### Step 4 — Secure
Apply these checks to every change, especially backend and API changes:

**Backend (FastAPI)**
- Input is sanitized via `sanitize_input()` from `backend/security.py` before processing
- Prompt injection check via `check_prompt_injection()` is applied to all user text
- Rate limiting via `check_rate_limit()` is applied to all mutating endpoints
- API key auth via `verify_api_key()` is applied to all protected endpoints
- CORS `allow_origins=["*"]` in `main.py` is development-only — flag if deploying to production
- No secrets, API keys, or credentials are hardcoded — use env vars via `.env` and `python-dotenv`
- File uploads: validate extension against `SUPPORTED_TYPES` and enforce 10 MB size limit
- Path traversal: never use user-supplied filenames directly in `os.path.join` without sanitization

**Frontend (React)**
- User input is never dangerously injected into HTML (`dangerouslySetInnerHTML` is banned unless sanitized)
- Sensitive data (API keys) is never stored in `localStorage` or logged to console
- API errors are caught and shown as user-friendly messages, not raw stack traces
- No hardcoded backend URLs — use environment variables or Vite's `import.meta.env`

**General**
- No new `print()` / `console.log()` statements left with sensitive data
- No `.env` files committed to git (check `.gitignore`)
- Audit log (`security.py:log_query`) is preserved for all chat queries

---

## Known Security Notes (Existing Issues to Keep in Mind)
1. **CORS is fully open** (`allow_origins=["*"]` in `backend/main.py:39`) — acceptable for dev, must be restricted for production deployment
2. **API key auth is opt-in** — `DOCUMIND_API_KEY` env var is empty by default, disabling auth. Always set it in production.
3. **Rate limiter is in-memory** — resets on restart, not suitable for multi-process/distributed deployments

## Running the Stack
```bash
# Full stack
docker-compose up

# Backend only (dev)
cd backend && uvicorn main:app --reload --port 8000

# Frontend only (dev)
cd ida-frontend && npm run dev
```

## Testing
```bash
# Backend tests
cd backend && pytest test_backend.py -v

# RAG assistant tests
cd rag-llm-assistant && pytest tests/ -v
```
