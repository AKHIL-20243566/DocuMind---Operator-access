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

logger = logging.getLogger(__name__)

OLLAMA_BASE     = os.getenv("OLLAMA_URL",   "http://127.0.0.1:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS     = f"{OLLAMA_BASE}/api/tags"

# Relevance threshold: docs with score below this are not used as context
# Lowered to 0.2 so scanned/OCR PDFs (which score lower) still get used
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))

_ollama_status = None


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
    """Always do a fresh check so the backend picks up Ollama if it starts later."""
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
    if use_context and context_docs:
        context_text = "\n\n".join(
            f"[Source: {doc.get('source', 'Unknown')}, Page {doc.get('page', '?')}]\n{doc['text']}"
            for doc in context_docs
        )
        return f"""You are a precise enterprise documentation assistant called DocuMind.

STRICT RULES:
1. Answer ONLY from the retrieved context below. Do NOT use outside knowledge.
2. If the answer is not found in the context, respond with exactly:
   "This information is not found in the available documents."
3. Keep answers concise and professional (2-4 sentences unless detail is required).
4. Always cite the source document and page number when possible.

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
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
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
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
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
