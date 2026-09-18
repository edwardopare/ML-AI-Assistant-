"""Retrieval service boundary.

Allowed callers: generation service only.
Do NOT import from ingest or conversation services.
"""

from __future__ import annotations

from langchain_core.documents import Document

from ..config import RETRIEVAL_TOP_K
from ..retrieval import Retriever


class RetrievalService:
    """Facade over the hybrid dense+BM25+MMR retriever."""

    def __init__(self, top_k: int = RETRIEVAL_TOP_K) -> None:
        self._retriever = Retriever(top_k=top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Return relevant, diverse evidence chunks for *query*."""
        return self._retriever.retrieve(query, top_k=top_k)

    def close(self) -> None:
        """Release the underlying ChromaDB client."""
        self._retriever.close()

    def __enter__(self) -> RetrievalService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
