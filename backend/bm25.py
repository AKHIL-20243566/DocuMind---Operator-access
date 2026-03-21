"""DocuMind — BM25 Keyword Retrieval
Owner: Ashwin (Embeddings + Retrieval Optimization)
Purpose: Provides BM25 keyword-based ranking to complement FAISS semantic search.
         Combined with FAISS in hybrid_search() for improved retrieval accuracy.
Connection: Imported by vector_store.py; used in hybrid search alongside FAISS L2 index.

Why BM25 + FAISS?
- FAISS (semantic): great for meaning-based queries ("what is the leave policy")
- BM25 (keyword):  great for exact-term queries ("section 4.2", "form HR-21")
- Hybrid: normalise both scores, weight-average → beats either alone (Reciprocal Rank Fusion)
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Tokeniser (simple, no external deps)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """Okapi BM25 index over a list of raw text strings.

    Parameters
    ----------
    k1 : float
        Term-frequency saturation. Typical range 1.2–2.0.
    b  : float
        Length normalisation. 0 = no normalisation, 1 = full.
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.N  = len(corpus)

        # Per-document token lists and term-frequency maps
        self._tok: list[list[str]] = [_tokenise(doc) for doc in corpus]
        self._tf:  list[Counter]   = [Counter(t) for t in self._tok]

        # Average document length
        self._avgdl = sum(len(t) for t in self._tok) / max(self.N, 1)

        # Document frequency: how many docs contain each term
        df: dict[str, int] = {}
        for tokens in self._tok:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        # IDF — smoothed BM25 variant
        self._idf: dict[str, float] = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    # ------------------------------------------------------------------
    # Score a single document
    # ------------------------------------------------------------------

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        tf  = self._tf[doc_idx]
        dl  = len(self._tok[doc_idx])
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf   = self._idf.get(token, 0.0)
            tf_d  = tf[token]
            denom = tf_d + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * (tf_d * (self.k1 + 1)) / denom
        return score

    # ------------------------------------------------------------------
    # Ranked retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_index, bm25_score) pairs sorted best-first."""
        tokens = _tokenise(query)
        if not tokens:
            return []
        scores = [
            (i, self._score_doc(tokens, i))
            for i in range(self.N)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in scores[:k] if sc > 0.0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF) — merges FAISS + BM25 ranked lists
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    faiss_results: list[tuple[int, float]],
    bm25_results:  list[tuple[int, float]],
    k: int = 60,
    faiss_weight: float = 0.6,
    bm25_weight:  float = 0.4,
) -> list[tuple[int, float]]:
    """Merge two ranked lists using weighted Reciprocal Rank Fusion.

    RRF score = w1 / (k + rank_in_list1) + w2 / (k + rank_in_list2)
    Higher RRF = better combined rank.

    Parameters
    ----------
    k : int
        RRF smoothing constant (typically 60).
    faiss_weight / bm25_weight : float
        Relative importance of each retrieval method.
    """
    rrf: dict[int, float] = {}

    for rank, (idx, _) in enumerate(faiss_results, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + faiss_weight / (k + rank)

    for rank, (idx, _) in enumerate(bm25_results, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + bm25_weight / (k + rank)

    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)
