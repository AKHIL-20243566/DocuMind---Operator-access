"""DocuMind API — Entry Point
Owner: Aaron (Backend RAG Engineer + Cybersecurity)
Purpose: FastAPI application with auth, chat (streaming + non-streaming),
         document management, and enhanced audit logging
Connection: Imports auth, rag, llm, security; served by Uvicorn on port 8000
"""

import os
import time
import json
import logging
import hashlib
import threading
from contextlib import asynccontextmanager
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
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
from auth import signup, login, verify_token_from_request
from evaluation import evaluate_retrieval, llm_judge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer cache — LRU, keyed by (normalized_question, chat_id)
# Avoids re-running the full RAG pipeline for repeated questions.
# ---------------------------------------------------------------------------

_CACHE_MAX   = int(os.getenv("ANSWER_CACHE_SIZE", "100"))
_CACHE_TTL   = float(os.getenv("ANSWER_CACHE_TTL", "300"))   # seconds
_answer_cache: "OrderedDict[str, tuple[dict, float]]" = OrderedDict()
_cache_lock  = threading.Lock()


def _cache_key(question: str, chat_id: str | None) -> str:
    norm = " ".join(question.lower().split())
    return hashlib.md5(f"{norm}||{chat_id}".encode()).hexdigest()


def _cache_get(key: str) -> "dict | None":
    with _cache_lock:
        if key not in _answer_cache:
            return None
        payload, ts = _answer_cache[key]
        if time.time() - ts > _CACHE_TTL:
            del _answer_cache[key]
            return None
        _answer_cache.move_to_end(key)   # LRU update
        return payload


def _cache_put(key: str, payload: dict) -> None:
    with _cache_lock:
        if key in _answer_cache:
            _answer_cache.move_to_end(key)
        _answer_cache[key] = (payload, time.time())
        while len(_answer_cache) > _CACHE_MAX:
            _answer_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Startup: pre-warm ML models so the first real query is fast
# ---------------------------------------------------------------------------

def _prewarm_models() -> None:
    """Load embedding model + cross-encoder into memory before any request arrives."""
    try:
        logger.info("Pre-warming embedding model…")
        from embeddings import embed
        embed("warmup")
        logger.info("Embedding model ready.")
    except Exception as e:
        logger.warning("Embedding pre-warm failed: %s", e)

    try:
        logger.info("Pre-warming cross-encoder…")
        from reranker import _get_cross_encoder
        _get_cross_encoder()
        logger.info("Cross-encoder ready.")
    except Exception as e:
        logger.warning("Cross-encoder pre-warm failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run model pre-warming in a background thread so startup is non-blocking
    t = threading.Thread(target=_prewarm_models, daemon=True, name="model-prewarm")
    t.start()
    yield


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DocuMind API",
    description="Hybrid LLM + RAG Knowledge Assistant — MNNIT Internal",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TYPES = {
    ".pdf", ".docx", ".txt", ".csv", ".md", ".markdown",
    # Image formats — text extracted via OCR (pytesseract + Pillow)
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Question(BaseModel):
    question: str
    chat_id: str | None = None   # Document isolation per chat session


class AuthRequest(BaseModel):
    email: str
    password: str


class EvalRequest(BaseModel):
    question: str
    chat_id: str | None = None
    relevant_docs: list[str] | None = None   # ground-truth doc names (optional)


class EvaluateRequest(BaseModel):
    question: str
    answer: str | None = None       # if provided, LLM-as-judge is run
    chat_id: str | None = None
    relevant_docs: list[str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _compute_confidence(scores: list[float]) -> float:
    """Rescale raw FAISS similarity scores to an intuitive 0–1 confidence value.
    Owner: Aditya (Evaluation & Metrics)"""
    if not scores:
        return 0.0
    top = max(scores)
    scaled = min(1.0, max(0.0, (top - 0.15) / 0.65))
    return round(scaled, 2)


def _build_sources(docs: list[dict], use_context: bool) -> list[dict]:
    """Group source chunks by document, deduplicate, and sort by score.
    Owner: Akhil (Frontend Lead) — fixes retrieval display to group by doc.
    Always shows retrieved docs (even below threshold) so users see what was found."""
    if not docs:
        return []

    # Group by document name, keep best score per doc
    doc_map: dict = {}
    for d in docs:
        src = d.get("source", "Unknown")
        score = round(d.get("score", 0), 3)
        page  = d.get("page", 0)

        if src not in doc_map or score > doc_map[src]["score"]:
            doc_map[src] = {"doc": src, "page": page, "score": score}

    # Sort by score descending
    return sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# Auth endpoints (public — no JWT required)
# ---------------------------------------------------------------------------

@app.post("/auth/signup")
def auth_signup(body: AuthRequest):
    """Register a new @mnnit.ac.in user and return a JWT token."""
    return signup(body.email, body.password)


@app.post("/auth/login")
def auth_login(body: AuthRequest):
    """Authenticate and return a JWT token."""
    return login(body.email, body.password)


# ---------------------------------------------------------------------------
# Health (public)
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "DocuMind API is running",
        "ollama": check_ollama(),
    }


# ---------------------------------------------------------------------------
# Chat — non-streaming
# ---------------------------------------------------------------------------

@app.post("/chat")
def chat(q: Question, request: Request):
    """Process a question and return a complete answer.
    Owner: Aaron — secured with JWT auth, rate limiting, injection defense."""
    user_email = verify_token_from_request(request)
    check_rate_limit(request)

    question = sanitize_input(q.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if check_prompt_injection(question):
        raise HTTPException(
            status_code=400,
            detail="Your query was blocked by the safety filter."
        )

    start   = time.time()
    chat_id = q.chat_id

    try:
        if not has_embeddings(chat_id=chat_id):
            return {
                "answer": "No documents available for this chat. Please upload documents first.",
                "sources": [], "confidence": 0, "context": [],
                "mode": "llm", "ollama_connected": is_ollama_available(),
            }

        # ── Cache check — skip entire pipeline for repeated questions ──────
        ckey   = _cache_key(question, chat_id)
        cached = _cache_get(ckey)
        if cached:
            logger.info("chat cache_hit user=%s chat_id=%s", user_email, chat_id)
            return cached

        retrieved_docs = retrieve_context(question, k=5, chat_id=chat_id)
        use_context    = should_use_context(retrieved_docs)
        answer         = generate_answer(question, retrieved_docs, use_context)

        scores     = [d.get("score", 0) for d in retrieved_docs]
        confidence = _compute_confidence(scores)

        # Auditor + Strategist: check grounding and retry once with a broader query
        if use_context:
            try:
                from validator import check_answer_grounded, should_retry_retrieval
                grounding = check_answer_grounded(answer, retrieved_docs)
                if not grounding["grounded"]:
                    should_retry, refined_q = should_retry_retrieval(
                        answer, retrieved_docs, question, confidence
                    )
                    if should_retry:
                        retry_docs = retrieve_context(refined_q, k=5, chat_id=chat_id)
                        if retry_docs:
                            retry_use    = should_use_context(retry_docs)
                            retry_answer = generate_answer(question, retry_docs, retry_use)
                            if "not found" not in retry_answer.lower():
                                retrieved_docs = retry_docs
                                answer         = retry_answer
                                use_context    = retry_use
                                scores         = [d.get("score", 0) for d in retrieved_docs]
                                confidence     = _compute_confidence(scores)
            except Exception as _ve:
                logger.warning("Validation/retry skipped: %s", _ve)

        sources         = _build_sources(retrieved_docs, use_context)
        context_preview = [d["text"] for d in retrieved_docs] if use_context else []
        mode            = "rag" if use_context else "llm"

        elapsed_ms = (time.time() - start) * 1000
        log_query(question, elapsed_ms, sources, mode, _get_client_ip(request))
        logger.info("chat user=%s chat_id=%s mode=%s conf=%.2f elapsed=%.0fms",
                    user_email, chat_id, mode, confidence, elapsed_ms)

        result = {
            "answer":          answer,
            "sources":         sources,
            "confidence":      confidence,
            "context":         context_preview,
            "mode":            mode,
            "ollama_connected": is_ollama_available(),
        }
        _cache_put(ckey, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Chat — streaming (SSE)
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
def chat_stream(q: Question, request: Request):
    """Stream an answer token-by-token via Server-Sent Events.
    Owner: Aaron + Akhil — SSE streaming with security guards."""
    user_email = verify_token_from_request(request)
    check_rate_limit(request)

    question = sanitize_input(q.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if check_prompt_injection(question):
        raise HTTPException(
            status_code=400,
            detail="Your query was blocked by the safety filter."
        )

    chat_id = q.chat_id

    if not has_embeddings(chat_id=chat_id):
        def empty_stream():
            meta = {
                "type": "meta", "sources": [], "confidence": 0,
                "context": [], "mode": "llm",
            }
            yield f"data: {json.dumps(meta)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': 'No documents available for this chat. Please upload documents first.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    retrieved_docs = retrieve_context(question, k=5, chat_id=chat_id)
    use_context    = should_use_context(retrieved_docs)

    sources    = _build_sources(retrieved_docs, use_context)
    scores     = [d.get("score", 0) for d in retrieved_docs]
    confidence = _compute_confidence(scores)
    mode       = "rag" if use_context else "llm"

    logger.info("stream user=%s chat_id=%s mode=%s conf=%.2f", user_email, chat_id, mode, confidence)

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
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str = Query(None),
    request: Request = None,
):
    """Upload and ingest a document, scoped to the given chat_id.
    Owner: Anirudh (Data Engineer) — multi-layer parsing pipeline."""
    user_email = verify_token_from_request(request)
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

    # Run the blocking ingest pipeline in a thread so the event loop stays free
    # for other requests during the (potentially slow) OCR + embed stage.
    result = await run_in_threadpool(
        lambda: ingest_file(content, file.filename, chat_id=chat_id)
    )

    if result["success"]:
        logger.info(
            "Upload success | user=%s file=%s chunks=%d chat_id=%s loader=%s ocr=%s",
            user_email, file.filename, result.get("chunks", 0),
            chat_id, result.get("loader_used"), result.get("ocr_triggered", False),
        )
        return result

    logger.error(
        "Upload failed | user=%s file=%s error_code=%s loader=%s ocr=%s error=%s",
        user_email, file.filename, result.get("error_code"),
        result.get("loader_used"), result.get("ocr_triggered", False), result.get("error"),
    )
    raise HTTPException(status_code=422, detail=result)


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

@app.get("/documents")
def get_documents(chat_id: str = Query(None), request: Request = None):
    """List all documents, scoped to the given chat_id.
    Owner: Akhil (Frontend) — drives the document panel."""
    verify_token_from_request(request)
    docs = list_documents(chat_id=chat_id)
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_name:path}")
def remove_document(doc_name: str, chat_id: str = Query(None), request: Request = None):
    """Delete a document by name, optionally scoped to a chat."""
    verify_token_from_request(request)
    result = delete_document(doc_name, chat_id=chat_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result["message"])


# ---------------------------------------------------------------------------
# Evaluation / Metrics
# ---------------------------------------------------------------------------

@app.post("/metrics")
def get_metrics(body: EvalRequest, request: Request):
    """Run retrieval evaluation and return Precision@K, Recall@K, MRR,
    avg/max confidence, and hallucination-risk rating.

    Owner: Aditya (Evaluation & Metrics)
    Optional: pass relevant_docs (ground-truth doc names) for supervised metrics.
    """
    verify_token_from_request(request)

    question = sanitize_input(body.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chat_id = body.chat_id
    k       = 5

    retrieved_docs = retrieve_context(question, k=k, chat_id=chat_id)
    use_context    = should_use_context(retrieved_docs)

    report = evaluate_retrieval(
        query              = question,
        retrieved_docs     = retrieved_docs,
        relevant_doc_names = body.relevant_docs,
        k                  = k,
        use_context        = use_context,
    )

    logger.info(
        "eval user=%s query='%s' risk=%s avg_score=%.3f",
        request.headers.get("authorization", "")[:20],
        question[:60], report.hallucination_risk, report.avg_score,
    )

    return {
        "query":             report.query,
        "k":                 report.k,
        "num_retrieved":     report.num_retrieved,
        "precision_at_k":    report.precision_at_k,
        "recall_at_k":       report.recall_at_k,
        "mrr":               report.mrr,
        "avg_score":         report.avg_score,
        "max_score":         report.max_score,
        "min_score":         report.min_score,
        "hallucination_risk":report.hallucination_risk,
        "sources":           report.sources,
        "note": "precision/recall/mrr are null when no ground-truth relevant_docs provided",
    }


# ---------------------------------------------------------------------------
# Full evaluation with LLM-as-judge
# ---------------------------------------------------------------------------

@app.post("/evaluate")
def evaluate_answer(body: EvaluateRequest, request: Request):
    """Full evaluation: quantitative retrieval metrics + LLM-as-judge scoring.

    Quantitative (always):
      Precision@K, Recall@K, MRR, hallucination_risk, avg/max/min score.

    LLM-as-judge (when body.answer is provided):
      llm_relevance_score    — does the answer address the question? (0.0–1.0)
      llm_faithfulness_score — is the answer grounded in the context? (0.0–1.0)
      llm_judge_reasoning    — one-sentence explanation from the judge.

    Note: LLM judge adds ~2–5 s latency. Run offline / on-demand, not inline.
    Owner: Aditya (Evaluation & Metrics)
    """
    verify_token_from_request(request)

    question = sanitize_input(body.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chat_id = body.chat_id
    k = 5

    retrieved_docs = retrieve_context(question, k=k, chat_id=chat_id)
    use_context    = should_use_context(retrieved_docs)

    report = evaluate_retrieval(
        query              = question,
        retrieved_docs     = retrieved_docs,
        relevant_doc_names = body.relevant_docs,
        k                  = k,
        use_context        = use_context,
    )

    # LLM-as-judge (only when an answer string is supplied)
    judge_result = {"llm_relevance_score": None, "llm_faithfulness_score": None, "llm_judge_reasoning": None}
    if body.answer and retrieved_docs:
        judge_result = llm_judge(question, body.answer, retrieved_docs)

    logger.info(
        "evaluate user=%s query='%s' risk=%s llm_faith=%s",
        request.headers.get("authorization", "")[:20],
        question[:60], report.hallucination_risk,
        judge_result.get("llm_faithfulness_score"),
    )

    return {
        "query":                  report.query,
        "k":                      report.k,
        "num_retrieved":          report.num_retrieved,
        "precision_at_k":         report.precision_at_k,
        "recall_at_k":            report.recall_at_k,
        "mrr":                    report.mrr,
        "avg_score":              report.avg_score,
        "max_score":              report.max_score,
        "min_score":              report.min_score,
        "hallucination_risk":     report.hallucination_risk,
        "sources":                report.sources,
        "llm_relevance_score":    judge_result["llm_relevance_score"],
        "llm_faithfulness_score": judge_result["llm_faithfulness_score"],
        "llm_judge_reasoning":    judge_result["llm_judge_reasoning"],
        "note": "precision/recall/mrr are null when no ground-truth relevant_docs provided",
    }
