"""DocuMind API — Hybrid LLM + RAG Knowledge Assistant."""

import os
import time
import json
import logging
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag import retrieve_context, ingest_file, delete_document, list_documents, has_embeddings
from llm import (
    generate_answer, generate_answer_stream,
    should_use_context, check_ollama, is_ollama_available,
)
from security import (
    check_prompt_injection, sanitize_input,
    check_rate_limit, verify_api_key, log_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description="Hybrid LLM + RAG Knowledge Assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".csv", ".md", ".markdown"}


class Question(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _compute_confidence(scores: list[float]) -> float:
    """Normalize raw similarity scores into a 0-1 confidence value.

    Raw scores from FAISS use 1/(1+L2), which clusters around 0.4-0.7 for
    typical queries.  We rescale so that practical results map to a more
    intuitive range:
      - top_score >= 0.75  →  ~90-100 %
      - top_score ~  0.50  →  ~60 %
      - top_score <= 0.30  →  ~20-30 %
    """
    if not scores:
        return 0.0
    top = max(scores)
    # Scale with a slight curve so high-quality matches read as high confidence
    scaled = min(1.0, max(0.0, (top - 0.15) / 0.65))
    return round(scaled, 2)


def _build_sources(docs: list[dict], use_context: bool) -> list[dict]:
    if not use_context:
        return []
    return [
        {
            "doc": d.get("source", "Unknown"),
            "page": d.get("page", 0),
            "score": round(d.get("score", 0), 3),
        }
        for d in docs
    ]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "DocuMind API is running",
        "ollama": check_ollama(),
    }


# ---------------------------------------------------------------------------
# Chat (non-streaming)
# ---------------------------------------------------------------------------

@app.post("/chat")
def chat(q: Question, request: Request):
    verify_api_key(request)
    check_rate_limit(request)

    question = sanitize_input(q.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if check_prompt_injection(question):
        raise HTTPException(status_code=400, detail="Your query was blocked by the safety filter.")

    start = time.time()

    try:
        if not has_embeddings():
            return {
                "answer": "No documents available for this chat",
                "sources": [],
                "confidence": 0,
                "context": [],
                "mode": "llm",
                "ollama_connected": is_ollama_available(),
            }

        retrieved_docs = retrieve_context(question)
        use_context = should_use_context(retrieved_docs)
        answer = generate_answer(question, retrieved_docs, use_context)

        sources = _build_sources(retrieved_docs, use_context)
        scores = [d.get("score", 0) for d in retrieved_docs]
        confidence = _compute_confidence(scores)
        context_preview = [d["text"] for d in retrieved_docs] if use_context else []
        mode = "rag" if use_context else "llm"

        elapsed_ms = (time.time() - start) * 1000
        log_query(question, elapsed_ms, sources, mode, _get_client_ip(request))

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "context": context_preview,
            "mode": mode,
            "ollama_connected": is_ollama_available(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
def chat_stream(q: Question, request: Request):
    verify_api_key(request)
    check_rate_limit(request)

    question = sanitize_input(q.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if check_prompt_injection(question):
        raise HTTPException(status_code=400, detail="Your query was blocked by the safety filter.")

    if not has_embeddings():
        def empty_stream():
            meta = {
                "type": "meta",
                "sources": [],
                "confidence": 0,
                "context": [],
                "mode": "llm",
            }
            yield f"data: {json.dumps(meta)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': 'No documents available for this chat'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    retrieved_docs = retrieve_context(question)
    use_context = should_use_context(retrieved_docs)

    sources = _build_sources(retrieved_docs, use_context)
    scores = [d.get("score", 0) for d in retrieved_docs]
    confidence = _compute_confidence(scores)
    mode = "rag" if use_context else "llm"

    def event_stream():
        meta = {
            "type": "meta",
            "sources": sources,
            "confidence": confidence,
            "context": [d["text"] for d in retrieved_docs] if use_context else [],
            "mode": mode,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        for token in generate_answer_stream(question, retrieved_docs, use_context):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), request: Request = None):
    if request:
        verify_api_key(request)
        check_rate_limit(request)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_TYPES)}",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    result = ingest_file(content, file.filename)

    if result["success"]:
        logger.info(
            "Upload success | file=%s chunks=%s loader=%s ocr_triggered=%s",
            file.filename,
            result.get("chunks"),
            result.get("loader_used"),
            result.get("ocr_triggered", False),
        )
        return result
    logger.error(
        "Upload failed | file=%s error_code=%s loader=%s ocr_triggered=%s error=%s",
        file.filename,
        result.get("error_code"),
        result.get("loader_used"),
        result.get("ocr_triggered", False),
        result.get("error"),
    )
    raise HTTPException(status_code=422, detail=result)


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

@app.get("/documents")
def get_documents():
    docs = list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_name:path}")
def remove_document(doc_name: str, request: Request):
    verify_api_key(request)
    result = delete_document(doc_name)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result["message"])