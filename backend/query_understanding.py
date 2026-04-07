"""DocuMind — Query Understanding
Purpose: Multi-query generation, query decomposition, and step-back prompting
         to improve retrieval coverage for complex or ambiguous questions.

Why each technique improves accuracy:
  - Multi-query:    Different phrasings surface different chunks via BM25/FAISS.
  - Decomposition:  Breaks multi-part questions so each sub-question retrieves
                    the right focused context.
  - Step-back:      A broader reformulation reaches chunks that specific
                    terminology would miss (used by the Strategist for retries).
"""

import os
import json
import logging

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE     = os.getenv("OLLAMA_URL",   "http://127.0.0.1:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"


def _call_ollama(prompt: str, timeout: int = 30) -> str:
    """Non-streaming Ollama call. Returns empty string on failure."""
    try:
        resp = requests.post(
            OLLAMA_GENERATE,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("query_understanding: Ollama call failed: %s", e)
        return ""


def generate_multi_queries(question: str, n: int = 2) -> list[str]:
    """Return n rephrased versions of the question plus the original.

    Used to improve retrieval recall: each phrasing may surface different
    chunks via BM25 (keyword) or FAISS (semantic). Results are merged by
    best score in rag.retrieve_context() when ENABLE_MULTI_QUERY=true.
    """
    prompt = (
        f"Generate {n} different phrasings of the following question for document retrieval. "
        "Each phrasing should use different vocabulary while preserving the same meaning. "
        "Output ONLY a JSON array of strings with no explanation.\n\n"
        f"Question: {question}\n\nJSON array:"
    )
    raw = _call_ollama(prompt)
    queries = [question]   # always include the original
    if not raw:
        return queries

    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list):
                queries.extend(
                    str(q) for q in parsed
                    if isinstance(q, str) and q.strip() and q.strip() != question
                )
    except Exception:
        pass

    return queries[:n + 1]   # cap at n variants + original


def decompose_query(question: str) -> list[str]:
    """Break a complex multi-part question into atomic sub-questions.

    Example:
      "What method was used and what accuracy did it achieve?"
      → ["What method was used?", "What accuracy did the model achieve?"]

    If the question is already atomic, returns it unchanged.
    """
    prompt = (
        "If the following question contains multiple sub-questions, split it into "
        "individual atomic questions. If it is already a single focused question, "
        "return it unchanged. Output ONLY a JSON array of question strings.\n\n"
        f"Question: {question}\n\nJSON array:"
    )
    raw = _call_ollama(prompt)
    if not raw:
        return [question]

    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list) and parsed:
                return [str(q) for q in parsed if isinstance(q, str) and q.strip()]
    except Exception:
        pass

    return [question]


def stepback_query(question: str) -> str:
    """Generate a broader, more general version of the question.

    Example:
      "What BLEU score did the transformer achieve on WMT14?"
      → "What evaluation metrics and results are reported in this paper?"

    Used by the Strategist in validator.py as a retry strategy when
    the first retrieval returns no grounded answer.
    """
    prompt = (
        "Rewrite the following specific question as a broader, more general question "
        "that would help retrieve relevant background context from a document. "
        "Output ONLY the broader question, nothing else.\n\n"
        f"Specific question: {question}\n\nBroader question:"
    )
    raw = _call_ollama(prompt)
    if not raw or len(raw) < 5:
        return question
    # Take only the first line to avoid extra commentary
    broader = raw.split("\n")[0].strip().strip('"\'')
    return broader if broader else question


def select_relevant_sections(
    question: str,
    section_summaries: list[dict],
    max_sections: int = 3,
) -> list[str]:
    """LLM reads section summaries and returns section type names relevant to the question.

    Primary path: Ollama is asked to choose from the available sections.
    Fallback: keyword overlap between question tokens and section name/heading tokens.

    Args:
        question: The user's query.
        section_summaries: List of dicts from PageIndex.get_section_summaries():
            [{source, section, heading, summary}, ...]
        max_sections: Maximum sections to return.

    Returns:
        List of section type strings, e.g. ["methodology", "results"].
        Returns [] if nothing could be selected (caller should use full search).
    """
    if not section_summaries:
        return []

    # Build the unique section types present (deduplicate by section name)
    seen: dict[str, dict] = {}
    for s in section_summaries:
        sec = s.get("section", "general")
        if sec not in seen:
            seen[sec] = s

    if not seen:
        return []

    # ── Primary path: ask the LLM ──────────────────────────────────────────
    section_list = "\n".join(
        f"- {sec}: {info.get('heading', sec)} — {info.get('summary', '')[:120]}"
        for sec, info in seen.items()
    )
    valid_names = list(seen.keys())

    prompt = (
        "You are helping a retrieval system pick which document sections to search.\n\n"
        f"Available sections:\n{section_list}\n\n"
        f'Question: "{question}"\n\n'
        f"Which sections (up to {max_sections}) are most relevant to answer this question?\n"
        f"Choose ONLY from these exact names: {json.dumps(valid_names)}\n"
        "Respond with ONLY a JSON array of section name strings. "
        'Example: ["methodology", "results"]'
    )

    raw = _call_ollama(prompt, timeout=20)
    if raw:
        try:
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start != -1 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list):
                    selected = [
                        s for s in parsed
                        if isinstance(s, str) and s.strip() in seen
                    ][:max_sections]
                    if selected:
                        logger.info(
                            "PageIndex: LLM selected sections %s for question: %.60s",
                            selected, question,
                        )
                        return selected
        except Exception:
            pass

    # ── Fallback: keyword overlap ──────────────────────────────────────────
    q_tokens = set(question.lower().split())
    scored: list[tuple[int, str]] = []
    for sec, info in seen.items():
        sec_tokens = set(
            (sec + " " + info.get("heading", "")).lower().split()
        )
        overlap = len(q_tokens & sec_tokens)
        if overlap:
            scored.append((overlap, sec))

    if scored:
        scored.sort(reverse=True)
        fallback = [sec for _, sec in scored[:max_sections]]
        logger.info(
            "PageIndex: keyword fallback selected sections %s for question: %.60s",
            fallback, question,
        )
        return fallback

    return []
