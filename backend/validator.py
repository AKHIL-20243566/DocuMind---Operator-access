"""DocuMind — Output Validation Pipeline
Purpose: Three-layer validation to catch unsafe queries, hallucinated answers,
         and low-confidence retrievals before they reach the user.

  1. Gatekeeper  — blocks trivially short or clearly unanswerable queries
  2. Auditor     — checks if the generated answer is grounded in retrieved context
  3. Strategist  — decides whether to retry retrieval with a step-back query
"""

import re
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Minimum cosine similarity for a sentence to be considered "supported" by context
GROUNDING_THRESHOLD = 0.30
# Minimum fraction of sentences that must be supported for the answer to be "grounded"
GROUNDING_COVERAGE  = 0.75


# ---------------------------------------------------------------------------
# 1. Gatekeeper
# ---------------------------------------------------------------------------

def is_query_answerable(question: str, has_documents: bool) -> tuple[bool, str]:
    """Return (allowed, reason).

    Rejects queries that are trivially short.
    (Prompt injection / rate limiting are handled upstream in security.py.)
    """
    q = question.strip()
    if len(q) < 3:
        return False, "Question is too short."
    return True, "ok"


# ---------------------------------------------------------------------------
# 2. Auditor — grounding check
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_answer_grounded(answer: str, context_chunks: list[dict]) -> dict:
    """Check whether the generated answer is supported by the retrieved context.

    Algorithm:
    1. Split the answer into sentences.
    2. Embed each sentence and each context chunk.
    3. For each sentence, find its max cosine similarity against all chunks.
    4. Flag sentences below GROUNDING_THRESHOLD as "unsupported".
    5. grounded = True if >= GROUNDING_COVERAGE fraction are supported.

    Returns:
        {
            "grounded": bool,
            "coverage": float,             # fraction of supported sentences
            "unsupported_claims": list[str],
        }

    Why this matters: it catches answers that contain plausible-sounding but
    unchunked information — i.e. hallucinations — by verifying each claim
    against the actual retrieved context via embedding similarity.
    """
    if not context_chunks or not answer.strip():
        return {"grounded": False, "coverage": 0.0, "unsupported_claims": []}

    # "Not found" answers are inherently grounded (the model is admitting ignorance)
    not_found_phrases = [
        "not found in the available documents",
        "please upload relevant documentation",
        "no relevant documents",
        "i don't know",
        "cannot find",
        "is not in the",
    ]
    if any(phrase in answer.lower() for phrase in not_found_phrases):
        return {"grounded": True, "coverage": 1.0, "unsupported_claims": []}

    try:
        from embeddings import embed_cached

        # Split answer into meaningful sentences (skip very short fragments)
        sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
            if len(s.strip()) > 15
        ]
        if not sentences:
            return {"grounded": True, "coverage": 1.0, "unsupported_claims": []}

        # Embed context chunks (hits cache — already embedded during retrieval)
        # and answer sentences (new text, not cached)
        context_texts = [c.get("text", "") for c in context_chunks if c.get("text")]
        if not context_texts:
            return {"grounded": False, "coverage": 0.0, "unsupported_claims": sentences}

        context_embs  = embed_cached(context_texts)   # shape (M, dim) — cache hits
        sentence_embs = embed_cached(sentences)        # shape (N, dim)

        supported   = []
        unsupported = []

        for i, sent_emb in enumerate(sentence_embs):
            sims    = [_cosine_sim(sent_emb, ctx_emb) for ctx_emb in context_embs]
            max_sim = max(sims) if sims else 0.0
            if max_sim >= GROUNDING_THRESHOLD:
                supported.append(sentences[i])
            else:
                unsupported.append(sentences[i])

        coverage = len(supported) / len(sentences) if sentences else 1.0
        grounded = coverage >= GROUNDING_COVERAGE

        return {
            "grounded":           grounded,
            "coverage":           round(coverage, 3),
            "unsupported_claims": unsupported,
        }

    except Exception as e:
        logger.warning("Auditor: grounding check failed (skipping): %s", e)
        # Fail open — don't block the answer on auditor error
        return {"grounded": True, "coverage": 1.0, "unsupported_claims": []}


# ---------------------------------------------------------------------------
# 3. Strategist — retry logic
# ---------------------------------------------------------------------------

def should_retry_retrieval(
    answer: str,
    context_chunks: list[dict],
    question: str,
    confidence: float,
) -> tuple[bool, str]:
    """Return (should_retry, refined_question).

    Triggers a single retry with a step-back (broader) query when:
    - The answer explicitly says "not found" but documents are present, OR
    - Confidence is very low (< 0.3) despite having documents.

    The refined question is generated by stepback_query() from query_understanding.py.
    Max one retry — this function never loops.
    """
    answer_lower = answer.lower()

    not_found = (
        "not found in the available documents" in answer_lower
        or "cannot find" in answer_lower
        or "no relevant" in answer_lower
    )
    low_confidence = confidence < 0.3 and len(context_chunks) > 0

    if not (not_found or low_confidence):
        return False, question

    try:
        from query_understanding import stepback_query
        broader = stepback_query(question)
        if broader and broader != question:
            logger.info("Strategist: retrying with broader query: '%s'", broader[:80])
            return True, broader
    except Exception as e:
        logger.warning("Strategist: stepback_query failed: %s", e)

    return False, question
