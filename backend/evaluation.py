"""DocuMind — Evaluation & Metrics
Owner: Aditya (Evaluation & Metrics)
Purpose: Computes Precision@K, Recall@K, MRR, and hallucination-risk score
         over retrieved document sets. Exposed via /metrics endpoint in main.py.
Connection: Called by main.py /metrics endpoint; operates on retrieval results
            from rag.py (retrieve_context).
"""

import logging
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
