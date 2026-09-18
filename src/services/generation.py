"""Generation service boundary.

Allowed callers: API layer (channels/api) only.
Do NOT import from ingest or conversation services directly.
"""
from __future__ import annotations
from collections.abc import Generator
from typing import Any
from langchain_core.documents import Document
from ..agent import OpenRouterClient, OpenRouterError


class GenerationService:
    """Facade over the OpenRouter LLM client."""

    def __init__(self, model_name: str | None = None) -> None:
        self._client = OpenRouterClient(model_name=model_name)

    @property
    def is_available(self) -> bool:
        return self._client.is_available()

    def answer(
        self,
        question: str,
        documents: list[Document],
        history: list[dict[str, str]] | None = None,
        stream: bool = False,
    ) -> tuple[str | Generator[str, None, None], dict[str, Any]]:
        """Generate a grounded answer from *documents* for *question*."""
        return self._client.answer_question(question, documents, stream=stream, history=history)


__all__ = ["GenerationService", "OpenRouterError"]
