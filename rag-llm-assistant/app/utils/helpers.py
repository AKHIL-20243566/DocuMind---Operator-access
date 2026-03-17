"""Utility helpers for the RAG assistant."""
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def truncate_text(text, max_length=200):
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_sources(docs):
    """Format retrieved documents as a readable string."""
    lines = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "Unknown")
        page = doc.get("page", "?")
        score = doc.get("score", 0)
        lines.append(f"  [{i}] {source} (page {page}, score: {score:.2f})")
    return "\n".join(lines)