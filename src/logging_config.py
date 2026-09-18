"""Structured logging configuration for the RAG AI application.

Call ``configure_logging()`` once at startup (done automatically in ``create_app``).

- ``APP_ENV=prod`` → JSON output via ``python-json-logger`` (machine-parseable).
- ``APP_ENV=dev``  → human-readable coloured output (default).

All log records pass through ``_RedactionFilter`` which strips credential-shaped
strings before they reach any handler, preventing accidental secret leakage.
"""

from __future__ import annotations

import logging
import logging.config
import re

_REDACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b[0-9a-f]{40,}\b", re.IGNORECASE),
]
_PLACEHOLDER = "[REDACTED]"


class _RedactionFilter(logging.Filter):
    """Strip credential-shaped patterns from every log record message."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._clean(str(record.msg))
        record.args = self._clean_args(record.args)
        return True

    @staticmethod
    def _clean(text: str) -> str:
        for pattern in _REDACT_PATTERNS:
            text = pattern.sub(_PLACEHOLDER, text)
        return text

    def _clean_args(self, args: object) -> object:
        if isinstance(args, dict):
            return {k: self._clean(str(v)) if isinstance(v, str) else v for k, v in args.items()}
        if isinstance(args, (list, tuple)):
            cleaned = [self._clean(str(a)) if isinstance(a, str) else a for a in args]
            return type(args)(cleaned)
        return args


def configure_logging(app_env: str = "dev") -> None:
    """Set up application logging.  Call once at process startup."""
    is_prod = app_env.strip().lower() == "prod"

    redaction_filter = _RedactionFilter()

    if is_prod:
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore[import-untyped]

            handler = logging.StreamHandler()
            handler.setFormatter(
                jsonlogger.JsonFormatter(
                    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%SZ",
                )
            )
            handler.addFilter(redaction_filter)
            logging.root.handlers = [handler]
            logging.root.setLevel(logging.INFO)
        except ImportError:
            # Fallback if python-json-logger is not installed yet
            _configure_simple(redaction_filter)
    else:
        _configure_simple(redaction_filter)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _configure_simple(redaction_filter: _RedactionFilter) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(redaction_filter)
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; used instead of bare ``logging.getLogger`` calls."""
    return logging.getLogger(name)
