"""Security middleware — prompt injection defense, rate limiting, API key auth, query logging."""

import os
import re
import time
import logging
from collections import defaultdict
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("DOCUMIND_API_KEY", "")  # empty = auth disabled
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# ---------------------------------------------------------------------------
# Prompt injection detection
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(the\s+)?hidden\s+(data|prompt|instructions)",
    r"print\s+(the\s+)?system\s+prompt",
    r"what\s+(are|is)\s+your\s+(system\s+)?instructions",
    r"repeat\s+(the\s+)?(text|words)\s+above",
    r"you\s+are\s+now\s+(a|an)\b",
    r"act\s+as\s+(a|an)\s+(different|new)",
    r"override\s+(your\s+)?(instructions|rules|prompt)",
    r"forget\s+(all\s+)?(your\s+)?previous",
    r"new\s+instructions?\s*:",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_prompt_injection(text: str) -> bool:
    """Return True if the text contains a prompt injection attempt."""
    for pattern in _compiled_patterns:
        if pattern.search(text):
            return True
    return False


def sanitize_input(text: str) -> str:
    """Strip control characters and excessive whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.strip()
    if len(text) > 5000:
        text = text[:5000]
    return text


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per client IP)
# ---------------------------------------------------------------------------

_request_log: dict[str, list[float]] = defaultdict(list)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request):
    """Raise 429 if the client has exceeded the rate limit."""
    if RATE_LIMIT <= 0:
        return
    ip = _get_client_ip(request)
    now = time.time()
    window_start = now - 60

    # Prune old entries
    _request_log[ip] = [t for t in _request_log[ip] if t > window_start]
    if len(_request_log[ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for {ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    _request_log[ip].append(now)


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

def verify_api_key(request: Request):
    """Verify the API key if one is configured."""
    if not API_KEY:
        return  # auth disabled
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token == API_KEY:
            return
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Query audit logger
# ---------------------------------------------------------------------------

_audit_logger = logging.getLogger("documind.audit")


def log_query(question: str, response_time_ms: float, sources: list, mode: str, client_ip: str):
    """Write a structured audit log entry for every query."""
    source_names = [s.get("doc", "?") for s in sources] if sources else []
    _audit_logger.info(
        f"query=\"{question[:80]}\" "
        f"response_ms={response_time_ms:.0f} "
        f"mode={mode} "
        f"sources={source_names} "
        f"ip={client_ip}"
    )
