"""Security primitives for the RAG AI API.

Provides:
- ``verify_api_key`` — FastAPI Security dependency enforcing Bearer-token auth.
- ``verify_teams_signature`` — async dependency for Teams HMAC-SHA256 validation.
- ``RequestSizeLimitMiddleware`` — Starlette middleware capping raw body size.
- ``redact_for_log`` — strips credential-shaped strings before logging or LLM dispatch.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
from typing import Annotated

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from .config import API_KEY, MAX_REQUEST_BODY_BYTES, TEAMS_WEBHOOK_SECRET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential-pattern redaction
# ---------------------------------------------------------------------------

# Patterns that look like credentials or PII — used in logging and LLM pre-send.
_REDACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b[0-9a-f]{40,}\b", re.IGNORECASE),
]
_REDACTION_PLACEHOLDER = "[REDACTED]"


def redact_for_log(text: str, max_chars: int = 400) -> str:
    """Redact credential-shaped patterns and truncate *text* for safe logging."""
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(_REDACTION_PLACEHOLDER, text)
    if len(text) > max_chars:
        text = text[:max_chars] + "…[truncated]"
    return text


def redact_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a shallow copy of *messages* with credential patterns removed from content."""
    redacted: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content", "")
        clean = content
        for pattern in _REDACT_PATTERNS:
            clean = pattern.sub(_REDACTION_PLACEHOLDER, clean)
        redacted.append({**message, "content": clean})
    return redacted


# ---------------------------------------------------------------------------
# Bearer-token authentication
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(_bearer_scheme),
    ],
) -> None:
    """FastAPI Security dependency — enforces Bearer token when API_KEY is set.

    When ``API_KEY`` is empty (dev mode) this dependency is a no-op so existing
    behaviour is preserved without any configuration change.
    """
    if not API_KEY:
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(credentials.credentials, API_KEY):
        logger.warning("API key rejected — invalid token presented")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Microsoft Teams HMAC-SHA256 signature verification
# ---------------------------------------------------------------------------


async def verify_teams_signature(request: Request) -> bytes:
    """Validate the HMAC-SHA256 signature on a Teams outgoing-webhook request.

    Returns the raw request body so the route handler can deserialize it.
    When ``TEAMS_WEBHOOK_SECRET`` is empty the check is skipped (dev only).
    """
    body = await request.body()
    request.state.raw_body = body

    if not TEAMS_WEBHOOK_SECRET:
        logger.debug("Teams signature verification skipped — TEAMS_WEBHOOK_SECRET not set")
        return body

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("hmac "):
        logger.warning("Teams request missing HMAC Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Teams HMAC signature missing.",
        )

    presented_b64 = auth_header[5:].strip()
    try:
        key_bytes = base64.b64decode(TEAMS_WEBHOOK_SECRET)
    except Exception as exc:
        logger.error("TEAMS_WEBHOOK_SECRET is not valid base64 — check your configuration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error.",
        ) from exc

    expected = base64.b64encode(hmac.new(key_bytes, body, hashlib.sha256).digest()).decode()

    if not secrets.compare_digest(presented_b64, expected):
        logger.warning("Teams HMAC signature mismatch — request rejected")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Teams HMAC signature invalid.",
        )

    return body


# ---------------------------------------------------------------------------
# Request body size cap middleware
# ---------------------------------------------------------------------------


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds *max_bytes*.

    Also catches streaming bodies that omit Content-Length by aborting the
    upstream call and returning HTTP 413 before the body is fully buffered.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                return Response(
                    content='{"detail":"Invalid Content-Length header."}',
                    status_code=400,
                    media_type="application/json",
                )
            if length > self.max_bytes:
                logger.warning(
                    "Request body too large",
                    extra={"content_length": length, "limit": self.max_bytes},
                )
                return Response(
                    content=(
                        f'{{"detail":"Request body too large. Maximum is {self.max_bytes} bytes."}}'
                    ),
                    status_code=413,
                    media_type="application/json",
                )
        return await call_next(request)
