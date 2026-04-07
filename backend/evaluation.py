"""DocuMind — Evaluation & Metrics
Owner: Aditya (Evaluation & Metrics)
Purpose: Computes Precision@K, Recall@K, MRR, and hallucination-risk score
         over retrieved document sets. Exposed via /metrics endpoint in main.py.
Connection: Called by main.py /metrics endpoint; operates on retrieval results
            from rag.py (retrieve_context).
"""

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Represents one retrieval result for evaluation purposes."""
    doc: str
    page: int
    score: float
    text: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation metrics for a single query."""
    query: str
    k: int
    num_retrieved: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    avg_score: float
    max_score: float
    min_score: float
    hallucination_risk: str   # "low" | "medium" | "high"
    sources: list[str] = field(default_factory=list)
    # LLM-as-judge fields (populated by llm_judge(), None if not run)
    llm_relevance_score:    float | None = None
    llm_faithfulness_score: float | None = None
    llm_judge_reasoning:    str | None = None


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Precision@K = |retrieved[:k] ∩ relevant| / k

    Owner: Aditya — standard IR metric.
    Parameters
    ----------
    retrieved : list of doc names in ranked order
    relevant  : list of ground-truth relevant doc names
    """
    if k == 0:
        return 0.0
    top_k   = retrieved[:k]
    rel_set = set(relevant)
    hits    = sum(1 for r in top_k if r in rel_set)
    return round(hits / k, 4)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Recall@K = |retrieved[:k] ∩ relevant| / |relevant|

    Owner: Aditya — standard IR metric.
    """
    if not relevant:
        return 0.0
    top_k   = retrieved[:k]
    rel_set = set(relevant)
    hits    = sum(1 for r in top_k if r in rel_set)
    return round(hits / len(rel_set), 4)


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """MRR = 1 / rank_of_first_relevant_result

    Owner: Aditya — measures how early the first correct result appears.
    """
    rel_set = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in rel_set:
            return round(1.0 / rank, 4)
    return 0.0


def hallucination_risk(top_score: float, use_context: bool, num_retrieved: int) -> str:
    """Estimate hallucination risk based on retrieval quality signals.

    Owner: Aditya — tracks hallucination potential per query.

    Risk levels:
    - "low"    : good context retrieved, model has strong grounding
    - "medium" : borderline context or fell back to LLM-only
    - "high"   : no context / very low similarity scores
    """
    if not use_context or num_retrieved == 0:
        return "high"
    if top_score >= 0.6:
        return "low"
    if top_score >= 0.35:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Full evaluation for a single query
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    query: str,
    retrieved_docs: list[dict],
    relevant_doc_names: list[str] | None = None,
    k: int = 5,
    use_context: bool = True,
) -> EvalReport:
    """Compute all evaluation metrics for a single retrieval result.

    Owner: Aditya
    Parameters
    ----------
    query             : the user's question
    retrieved_docs    : output from rag.retrieve_context()
    relevant_doc_names: ground truth (optional; if None, skips precision/recall/MRR)
    k                 : evaluation cutoff
    """
    retrieved_names = [d.get("source", "Unknown") for d in retrieved_docs]
    scores          = [d.get("score", 0.0) for d in retrieved_docs]

    top_score = max(scores, default=0.0)
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Precision / Recall / MRR (require ground truth)
    rel = relevant_doc_names or []
    prec = precision_at_k(retrieved_names, rel, k) if rel else None
    rec  = recall_at_k(retrieved_names,    rel, k) if rel else None
    mrr  = mean_reciprocal_rank(retrieved_names, rel) if rel else None

    risk = hallucination_risk(top_score, use_context, len(retrieved_docs))

    return EvalReport(
        query           = query,
        k               = k,
        num_retrieved   = len(retrieved_docs),
        precision_at_k  = prec,
        recall_at_k     = rec,
        mrr             = mrr,
        avg_score       = avg_score,
        max_score       = round(top_score, 4),
        min_score       = round(min(scores, default=0.0), 4),
        hallucination_risk = risk,
        sources         = list(dict.fromkeys(retrieved_names)),   # deduped, ordered
    )


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------

def llm_judge(question: str, answer: str, context_chunks: list[dict]) -> dict:
    """Call Ollama to rate the answer on relevance and faithfulness.

    Why a separate endpoint (not inline with /chat):
    - LLM judge adds 2–5 s latency — should not slow down the chat flow.
    - Called explicitly via POST /evaluate for offline quality assessment.

    Returns:
        {
            "llm_relevance_score":    float 0.0–1.0  (or None on failure),
            "llm_faithfulness_score": float 0.0–1.0  (or None on failure),
            "llm_judge_reasoning":    str             (or None on failure),
        }
    """
    import requests as _requests

    OLLAMA_BASE  = os.getenv("OLLAMA_URL",   "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

    context_text = "\n\n".join(
        f"[Source: {c.get('source', '?')}, Section: {c.get('section', 'body')}]\n"
        f"{c.get('text', '')[:500]}"
        for c in context_chunks[:5]
    )

    prompt = (
        "You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.\n"
        "Rate the following answer on two dimensions (scale 0–10):\n"
        "  Relevance:    Does the answer address the question asked?\n"
        "  Faithfulness: Is the answer grounded in and supported by the provided context?\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Output ONLY a JSON object with these exact keys:\n"
        '{"relevance": <0-10>, "faithfulness": <0-10>, "reasoning": "<one sentence>"}'
    )

    try:
        resp = _requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(raw[start:end])
            return {
                "llm_relevance_score":    round(min(10.0, max(0.0, float(data.get("relevance",    5)))) / 10, 2),
                "llm_faithfulness_score": round(min(10.0, max(0.0, float(data.get("faithfulness", 5)))) / 10, 2),
                "llm_judge_reasoning":    str(data.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning("LLM judge call failed: %s", e)

    return {
        "llm_relevance_score":    None,
        "llm_faithfulness_score": None,
        "llm_judge_reasoning":    None,
    }
