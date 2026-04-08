"""DocuMind — LLM Integration (Ollama)
Owner: Aaron (Backend RAG Engineer)
Purpose: Builds prompts, calls Ollama for streaming/non-streaming generation,
         enforces STRICT RAG mode (answers only from document context),
         and provides fallback when Ollama is unavailable
Connection: Called by main.py for /chat and /chat/stream endpoints
"""

import requests
import json
import os
import logging
import time

logger = logging.getLogger(__name__)

# Ollama inference options — cap token budget and context window to reduce CPU time.
# num_predict: max tokens to generate (short factual answers rarely exceed 300).
# num_ctx:     KV-cache context window; 2048 covers all our prompts and is 4× faster
#              than the llama3 default (8192) on CPU.
# num_thread:  use all available CPU cores for token generation.
# temperature: 0 = deterministic, skips sampling overhead.
_OLLAMA_OPTIONS = {
    "temperature":  0,
    "num_predict":  int(os.getenv("OLLAMA_NUM_PREDICT", "400")),
    "num_ctx":      int(os.getenv("OLLAMA_NUM_CTX",     "2048")),
    "num_thread":   int(os.getenv("OLLAMA_NUM_THREAD",  "8")),
}

OLLAMA_BASE     = os.getenv("OLLAMA_URL",   "http://127.0.0.1:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS     = f"{OLLAMA_BASE}/api/tags"

# Relevance threshold: docs with score below this are not used as context
# Lowered to 0.2 so scanned/OCR PDFs (which score lower) still get used
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))

_ollama_status = None
_ollama_check_ts: float = 0.0
_OLLAMA_TTL: float = 30.0   # seconds between live health checks

# ---------------------------------------------------------------------------
# Research-paper domain instructions
# Activated when retrieved chunks carry doc_type="research_paper"
# ---------------------------------------------------------------------------

_RESEARCH_PAPER_INSTRUCTIONS = (
    "- Reference the specific section when citing information "
    "(e.g., 'According to the Methodology section...').\n"
    "- Quote specific metric values and numbers when discussing results.\n"
    "- Distinguish between the paper's claims and your interpretation.\n"
    "- When describing methods, use the exact technique name from the paper."
)


def _detect_doc_type(context_docs: list) -> str:
    """Return the primary doc_type from context chunks."""
    for doc in context_docs:
        if doc.get("doc_type") == "research_paper":
            return "research_paper"
    return "general"


# ---------------------------------------------------------------------------
# Ollama health
# ---------------------------------------------------------------------------

def check_ollama() -> dict:
    """Ping Ollama and return {available, models}.
    Tries OLLAMA_TAGS first; if that fails and the URL uses 'localhost',
    retries with '127.0.0.1' to work around Windows DNS resolution quirks."""
    global _ollama_status

    def _try_url(url: str):
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except Exception:
            pass
        return None

    models_raw = _try_url(OLLAMA_TAGS)

    if models_raw is not None:
        models = [m["name"] for m in models_raw]
        _ollama_status = {"available": True, "models": models}
        return _ollama_status

    _ollama_status = {"available": False, "models": []}
    return _ollama_status


def is_ollama_available() -> bool:
    """Cached health check — re-pings Ollama at most once every 30 seconds."""
    global _ollama_check_ts, _ollama_status
    now = time.time()
    if _ollama_status is not None and (now - _ollama_check_ts) < _OLLAMA_TTL:
        return _ollama_status.get("available", False)
    _ollama_check_ts = now
    return check_ollama().get("available", False)


# ---------------------------------------------------------------------------
# Prompt builder — STRICT RAG mode
# Owner: Aaron + Aditya (Evaluation & Metrics — ensures answer grounding)
# ---------------------------------------------------------------------------

def build_strict_rag_prompt(question: str, context_docs: list, use_context: bool) -> str:
    """Build the LLM prompt.

    STRICT RAG mode: when context is available, the model MUST answer only
    from the retrieved documents. If the answer is not in the docs, it must
    say so explicitly — no hallucination allowed.
    """
    # Max chars of each chunk text sent to Ollama.
    # Shorter context = faster prefill without losing key facts.
    _MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "600"))

    if use_context and context_docs:
        # Build context lines — truncate each chunk to cap prompt length
        context_parts = []
        for doc in context_docs:
            source_label = f"[Source: {doc.get('source', 'Unknown')}, Page {doc.get('page', '?')}"
            if doc.get("section"):
                source_label += f", Section: {doc['section'].replace('_', ' ').title()}"
            source_label += "]"
            text = doc["text"][:_MAX_CHUNK_CHARS]
            context_parts.append(f"{source_label}\n{text}")
        context_text = "\n\n".join(context_parts)

        # Add domain-specific instructions for research papers
        doc_type = _detect_doc_type(context_docs)
        domain_block = ""
        if doc_type == "research_paper":
            domain_block = f"\nDOMAIN INSTRUCTIONS (research paper):\n{_RESEARCH_PAPER_INSTRUCTIONS}\n"

        return f"""You are a precise enterprise documentation assistant called DocuMind.

STRICT RULES:
1. Answer ONLY from the retrieved context below. Do NOT use outside knowledge.
2. If the answer is not found in the context, respond with exactly:
   "This information is not found in the available documents."
3. Keep answers concise and professional (2-4 sentences unless detail is required).
4. Always cite the source document and page number when possible.
{domain_block}
---
Retrieved Context:
{context_text}
---

Question: {question}

Answer:"""

    else:
        # No relevant context found — inform user
        return f"""You are DocuMind, an enterprise document assistant.

No relevant documents were found for this query.
Respond with: "This information is not found in the available documents. Please upload relevant documentation."

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Non-streaming generation
# ---------------------------------------------------------------------------

def generate_answer(question: str, context_docs: list, use_context: bool = True) -> str:
    """Call Ollama synchronously and return the full answer string."""
    prompt = build_strict_rag_prompt(question, context_docs, use_context)

    try:
        response = requests.post(
            OLLAMA_GENERATE,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": _OLLAMA_OPTIONS},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.ConnectionError:
        logger.warning("Ollama unavailable — using fallback response")
        return _fallback_answer(question, context_docs, use_context)
    except Exception as e:
        logger.error("LLM error: %s", e)
        return _fallback_answer(question, context_docs, use_context)


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

def generate_answer_stream(question: str, context_docs: list, use_context: bool = True):
    """Call Ollama with streaming=True. Yields text tokens one at a time."""
    prompt = build_strict_rag_prompt(question, context_docs, use_context)

    try:
        response = requests.post(
            OLLAMA_GENERATE,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True,
                  "options": _OLLAMA_OPTIONS},
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    break

    except requests.ConnectionError:
        logger.warning("Ollama unavailable for streaming — using fallback")
        yield _fallback_answer(question, context_docs, use_context)
    except Exception as e:
        logger.error("LLM streaming error: %s", e)
        yield _fallback_answer(question, context_docs, use_context)


# ---------------------------------------------------------------------------
# Relevance check
# Owner: Aditya (Evaluation & Metrics) — thresholding for hallucination control
# ---------------------------------------------------------------------------

def should_use_context(retrieved_docs: list) -> bool:
    """Return True only if at least one retrieved doc clears the relevance threshold.
    Prevents the model from using low-quality, irrelevant context."""
    if not retrieved_docs:
        return False
    top_score = max(doc.get("score", 0) for doc in retrieved_docs)
    return top_score >= RELEVANCE_THRESHOLD


# ---------------------------------------------------------------------------
# Fallback (Ollama offline)
# ---------------------------------------------------------------------------

def _fallback_answer(question: str, context_docs: list, use_context: bool) -> str:
    """Return a formatted fallback when Ollama is not running."""
    if use_context and context_docs:
        context_text = "\n\n".join(
            f"- **{doc.get('source', 'Unknown')}** (p.{doc.get('page', '?')}): {doc['text']}"
            for doc in context_docs
        )
        return (
            f"Based on the retrieved documents:\n\n{context_text}\n\n"
            "---\n*Ollama LLM is offline. Showing raw retrieved context.*"
        )
    return (
        "This information is not found in the available documents.\n\n"
        "---\n*Ollama LLM is offline. Upload relevant documents to get context-based answers.*"
    )
