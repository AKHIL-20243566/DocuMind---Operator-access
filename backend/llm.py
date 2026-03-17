import requests
import json
import os
import logging

logger = logging.getLogger(__name__)

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_BASE}/api/tags"

# Relevance threshold for hybrid logic
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))

_ollama_status = None


def check_ollama():
    """Check if Ollama is running and return status info."""
    global _ollama_status
    try:
        resp = requests.get(OLLAMA_TAGS, timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            _ollama_status = {"available": True, "models": models}
            return _ollama_status
    except Exception:
        pass
    _ollama_status = {"available": False, "models": []}
    return _ollama_status


def is_ollama_available():
    if _ollama_status is None:
        check_ollama()
    return _ollama_status.get("available", False)


def build_hybrid_prompt(question, context_docs, use_context=True):
    """Build a prompt with hybrid RAG logic."""
    if use_context and context_docs:
        context_text = "\n\n".join(
            f"[Source: {doc.get('source', 'Unknown')}, Page {doc.get('page', '?')}]\n{doc['text']}"
            for doc in context_docs
        )
        return f"""You are DocuMind, an intelligent AI knowledge assistant.

You have access to the following retrieved documents. Use them to answer the question when relevant.
If the documents contain the answer, cite the source.
If the documents do NOT contain the answer, use your own knowledge to provide a helpful response.
Always be accurate, concise, and professional.

---
Retrieved Documents:
{context_text}
---

Question: {question}

Answer:"""
    else:
        return f"""You are DocuMind, an intelligent AI knowledge assistant.

No relevant documents were found for this question. Answer using your general knowledge.
Be helpful, accurate, and concise.

Question: {question}

Answer:"""


def generate_answer(question, context_docs, use_context=True):
    """Generate an answer using Ollama with hybrid RAG logic."""
    prompt = build_hybrid_prompt(question, context_docs, use_context)

    try:
        response = requests.post(
            OLLAMA_GENERATE,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.ConnectionError:
        logger.warning("Ollama not available, using fallback")
        return _fallback_answer(question, context_docs, use_context)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return _fallback_answer(question, context_docs, use_context)


def generate_answer_stream(question, context_docs, use_context=True):
    """Generate a streaming answer from Ollama. Yields text chunks."""
    prompt = build_hybrid_prompt(question, context_docs, use_context)

    try:
        response = requests.post(
            OLLAMA_GENERATE,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True
            },
            timeout=120,
            stream=True
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
        logger.warning("Ollama not available for streaming, using fallback")
        yield _fallback_answer(question, context_docs, use_context)
    except Exception as e:
        logger.error(f"LLM streaming error: {e}")
        yield _fallback_answer(question, context_docs, use_context)


def should_use_context(retrieved_docs):
    """Determine if retrieved documents are relevant enough to use."""
    if not retrieved_docs:
        return False
    top_score = max(doc.get("score", 0) for doc in retrieved_docs)
    return top_score >= RELEVANCE_THRESHOLD


def _fallback_answer(question, context_docs, use_context):
    """Fallback when Ollama is not available."""
    if use_context and context_docs:
        context_text = "\n\n".join(
            f"- **{doc.get('source', 'Unknown')}** (p.{doc.get('page', '?')}): {doc['text']}"
            for doc in context_docs
        )
        return (
            f"Based on the retrieved documents:\n\n{context_text}\n\n"
            "---\n*Running in fallback mode. Connect Ollama for AI-generated answers.*"
        )
    return (
        "I can search your uploaded documents for answers. "
        "Please upload relevant documents or connect Ollama for general AI responses.\n\n"
        "---\n*Ollama LLM is not connected.*"
    )