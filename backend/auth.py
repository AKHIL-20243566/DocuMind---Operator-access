"""DocuMind — Authentication Module
Owner: Aaron (Backend RAG Engineer + Cybersecurity)
Purpose: JWT-based auth restricted to @mnnit.ac.in domain only
Connection: Imported by main.py for /auth/signup and /auth/login endpoints;
            verify_token_from_request() guards all protected routes
"""

import os
import hashlib
import time
import logging
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "documind-mnnit-jwt-secret-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = int(os.getenv("TOKEN_EXPIRY_HOURS", "24"))
ALLOWED_DOMAIN = "@mnnit.ac.in"

# In-memory user store: email → hashed_password
# Replace with a real database (PostgreSQL/SQLite) in production
_users: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_mnnit_email(email: str) -> bool:
    """Validate that the email belongs to the MNNIT domain."""
    return email.strip().lower().endswith(ALLOWED_DOMAIN)


def _hash_password(password: str) -> str:
    """SHA-256 hash. Use bcrypt/argon2 in production."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _create_token(email: str) -> str:
    """Sign and return a JWT for the given email."""
    try:
        import jwt
    except ImportError:
        raise RuntimeError("PyJWT is required. Run: pip install PyJWT>=2.9.0")
    exp = int(time.time()) + TOKEN_EXPIRY_HOURS * 3600
    payload = {"sub": email, "exp": exp, "iat": int(time.time())}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# Public auth API
# ---------------------------------------------------------------------------

def signup(email: str, password: str) -> dict:
    """Register a new user. Only @mnnit.ac.in emails are accepted.

    Returns: {"token": str, "email": str}
    """
    email = email.strip().lower()

    if not _is_mnnit_email(email):
        raise HTTPException(
            status_code=403,
            detail="Access restricted to @mnnit.ac.in email addresses only."
        )
    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters."
        )
    if email in _users:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    _users[email] = _hash_password(password)
    logger.info("New user registered: %s", email)
    return {"token": _create_token(email), "email": email}


def login(email: str, password: str) -> dict:
    """Authenticate a registered @mnnit.ac.in user.

    Returns: {"token": str, "email": str}
    """
    email = email.strip().lower()

    if not _is_mnnit_email(email):
        raise HTTPException(
            status_code=403,
            detail="Access restricted to @mnnit.ac.in email addresses only."
        )

    stored = _users.get(email)
    if not stored or stored != _hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    logger.info("Login success: %s", email)
    return {"token": _create_token(email), "email": email}


def verify_token_from_request(request: Request) -> str:
    """Extract and verify the JWT from the Authorization: Bearer <token> header.

    Returns the authenticated user's email on success.
    Raises HTTP 401 on failure.
    """
    try:
        import jwt
        from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
    except ImportError:
        raise RuntimeError("PyJWT is required.")

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please log in."
        )

    token = auth_header[7:]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return str(payload["sub"])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please log in again."
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token."
        )


def get_masked_email(email: str) -> str:
    """Return a masked version of the email for display e.g. 'a***@mnnit.ac.in'."""
    parts = email.split("@")
    if len(parts) != 2:
        return "***@mnnit.ac.in"
    local = parts[0]
    masked = (local[0] + "***") if len(local) > 1 else "***"
    return f"{masked}@{parts[1]}"
