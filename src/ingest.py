from __future__ import annotations
from pathlib import Path
from typing import List
from pypdf import PdfReader
from src.config import PDF_DIR, TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE


def load_pdf_paths(pdf_dir: Path = PDF_DIR) -> List[Path]:
    """Load all PDF file paths from the specified directory."""
    return sorted(pdf_dir.glob("*.pdf"))


def extract_pdf_text(pdf_path: Path) -> List[tuple[int, str]]:
    """Extract text from all pages of a PDF file."""
    reader = PdfReader(pdf_path)
    pages: List[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text.strip()))

    return pages


def chunk_text(text: str, chunk_size: int = TEXT_CHUNK_SIZE, overlap: int = TEXT_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for better context retention."""
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_document_chunks(
    pdf_dir: Path = PDF_DIR,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
) -> List[dict]:
    """
    Extract and chunk all PDF documents in a directory.
    Returns list of document chunks with metadata (source, page, chunk index).
    """
    documents: List[dict] = []

    for pdf_path in load_pdf_paths(pdf_dir):
        pages = extract_pdf_text(pdf_path)
        for page_number, page_text in pages:
            page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            for chunk_index, chunk in enumerate(page_chunks, start=1):
                documents.append(
                    {
                        "id": f"{pdf_path.stem}-p{page_number}-c{chunk_index}",
                        "text": chunk,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk": chunk_index,
                            "path": str(pdf_path.resolve()),
                        },
                    }
                )

    return documents
