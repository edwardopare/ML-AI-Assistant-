from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / ".chromadb"
CACHE_DIR = ROOT_DIR / "data" / ".cache"
CONVERSATION_DB_PATH = CACHE_DIR / "conversations.sqlite3"
INDEX_MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"
COLLECTION_NAME = "rag_documents"
INDEX_SCHEMA_VERSION = 2
PROMPT_VERSION = 2


# Read and validate an integer environment variable.
def _int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


# Read and validate a floating-point environment variable.
def _float_env(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "all-MiniLM-L6-v2",
).strip()
ENABLE_OCR = os.getenv("ENABLE_OCR", "false").strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = (
    os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    .strip()
    .rstrip("/")
)
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini",
).strip()
OPENROUTER_TIMEOUT_SECONDS = _int_env("OPENROUTER_TIMEOUT_SECONDS", 120, 1)
OPENROUTER_MAX_RETRIES = _int_env("OPENROUTER_MAX_RETRIES", 3, 0)
OPENROUTER_MAX_TOKENS = _int_env("OPENROUTER_MAX_TOKENS", 800, 1)
OPENROUTER_TEMPERATURE = _float_env(
    "OPENROUTER_TEMPERATURE",
    0.1,
    minimum=0.0,
    maximum=2.0,
)

TEXT_CHUNK_SIZE = _int_env("TEXT_CHUNK_SIZE", 1200, 100)
TEXT_CHUNK_OVERLAP = _int_env("TEXT_CHUNK_OVERLAP", 200, 0)
if TEXT_CHUNK_OVERLAP >= TEXT_CHUNK_SIZE:
    raise ValueError("TEXT_CHUNK_OVERLAP must be smaller than TEXT_CHUNK_SIZE")

RETRIEVAL_TOP_K = _int_env("RETRIEVAL_TOP_K", 4, 1)
RETRIEVAL_CANDIDATE_K = _int_env("RETRIEVAL_CANDIDATE_K", 12, 1)
RETRIEVAL_MIN_SCORE = _float_env(
    "RETRIEVAL_MIN_SCORE",
    0.25,
    minimum=0.0,
    maximum=1.0,
)
RETRIEVAL_DENSE_WEIGHT = _float_env(
    "RETRIEVAL_DENSE_WEIGHT",
    0.7,
    minimum=0.0,
    maximum=1.0,
)
RETRIEVAL_MMR_LAMBDA = _float_env(
    "RETRIEVAL_MMR_LAMBDA",
    0.75,
    minimum=0.0,
    maximum=1.0,
)
MAX_CONTEXT_CHARS = _int_env("MAX_CONTEXT_CHARS", 12000, 500)
CONVERSATION_HISTORY_MESSAGES = _int_env(
    "CONVERSATION_HISTORY_MESSAGES",
    10,
    0,
)

# ---------------------------------------------------------------------------
# Security & deployment configuration
# ---------------------------------------------------------------------------

# Deployment profile: "dev" disables auth and uses human-readable logs.
# "prod" requires API_KEY and emits JSON-structured logs.
APP_ENV: str = os.getenv("APP_ENV", "dev").strip().lower()

# Static bearer token guarding the web channel endpoint.
# Leave empty in dev to disable authentication entirely.
API_KEY: str = os.getenv("API_KEY", "").strip()

# Secret for verifying Microsoft Teams outgoing-webhook HMAC-SHA256 signatures.
# Leave empty to skip Teams signature verification (development only).
TEAMS_WEBHOOK_SECRET: str = os.getenv("TEAMS_WEBHOOK_SECRET", "").strip()

# Comma-separated list of origins allowed by CORS middleware.
# Example: "https://myapp.example.com,https://localhost:3000"
CORS_ORIGINS: list[str] = [
    origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()
]

# Comma-separated list of Host header values that are accepted.
# Leave empty to allow any host (acceptable for local dev).
API_ALLOWED_HOSTS: list[str] = [
    host.strip() for host in os.getenv("API_ALLOWED_HOSTS", "").split(",") if host.strip()
]

# Hard cap on inbound HTTP request body size in bytes (default 64 KB).
MAX_REQUEST_BODY_BYTES: int = _int_env("MAX_REQUEST_BODY_BYTES", 65_536, 1024)

# Maximum requests per minute per IP address for channel endpoints.
RATE_LIMIT_PER_MINUTE: int = _int_env("RATE_LIMIT_PER_MINUTE", 30, 1)

# Maximum age (in days) of conversation rows before the cleanup job deletes them.
CONVERSATION_MAX_AGE_DAYS: int = _int_env("CONVERSATION_MAX_AGE_DAYS", 30, 1)

# Optional separate metrics bearer token.  When set, /metrics requires it.
METRICS_TOKEN: str = os.getenv("METRICS_TOKEN", "").strip()
