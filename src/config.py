from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / ".chromadb"
CACHE_DIR = ROOT_DIR / "data" / ".cache"
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
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).strip().rstrip("/")
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
