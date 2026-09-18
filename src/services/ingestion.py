"""Ingestion service boundary.

Allowed callers: CLI (main.py) only.
Do NOT import from generation or retrieval services.
"""
from __future__ import annotations
from pathlib import Path
from ..ingest import IngestionReport, build_document_chunks_with_report
from ..config import PDF_DIR, TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE


class IngestService:
    """Facade over PDF loading, chunking, embedding and persistence."""

    def __init__(
        self,
        pdf_dir: Path = PDF_DIR,
        chunk_size: int = TEXT_CHUNK_SIZE,
        chunk_overlap: int = TEXT_CHUNK_OVERLAP,
    ) -> None:
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_report(self) -> IngestionReport:
        """Extract and chunk all PDFs; return a full ingestion report."""
        return build_document_chunks_with_report(
            self.pdf_dir, self.chunk_size, self.chunk_overlap
        )
