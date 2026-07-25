"""PDF extraction and sentence-aware text chunking."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from src.config import ENABLE_OCR, PDF_DIR, TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE

_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class IngestionReport:
    documents: list[dict] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_pdf_paths(pdf_dir: Path = PDF_DIR) -> list[Path]:
    """Discover PDFs recursively so document folders can be preserved."""
    return sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ocr_pages(pdf_path: Path) -> dict[int, str]:
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR is enabled but OCR dependencies are missing. "
            "Install the project with `pip install -e .[ocr]`."
        ) from exc

    results: dict[int, str] = {}
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            results[page_index + 1] = pytesseract.image_to_string(image).strip()
    return results


def extract_pdf_text(
    pdf_path: Path,
    *,
    enable_ocr: bool = ENABLE_OCR,
) -> list[tuple[int, str]]:
    """Extract text from every PDF page, preserving one-based page numbers."""
    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("PDF is encrypted and cannot be opened") from exc

    pages: list[tuple[int, str]] = []
    extracted: dict[int, str] = {}
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        extracted[page_number] = re.sub(r"[ \t]+", " ", text).strip()
    if enable_ocr and any(not text for text in extracted.values()):
        ocr_text = _ocr_pages(pdf_path)
        for page_number, text in extracted.items():
            pages.append((page_number, text or ocr_text.get(page_number, "")))
    else:
        pages.extend(extracted.items())
    return pages


def _split_oversized_unit(unit: str, chunk_size: int) -> list[str]:
    words = unit.split()
    if not words:
        return []
    parts: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and current_length + added > chunk_size:
            parts.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += added
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_text(
    text: str,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
) -> list[str]:
    """Build chunks on paragraph/sentence boundaries with bounded overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if not text.strip():
        return []

    raw_units = [unit.strip() for unit in _BOUNDARY_RE.split(text) if unit.strip()]
    units: list[str] = []
    for unit in raw_units:
        units.extend(_split_oversized_unit(unit, chunk_size))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        proposed = " ".join([*current, unit])
        if current and len(proposed) > chunk_size:
            chunk = " ".join(current).strip()
            chunks.append(chunk)
            overlap_text = chunk[-overlap:].lstrip() if overlap else ""
            current = [overlap_text, unit] if overlap_text else [unit]
            while len(" ".join(current)) > chunk_size and len(current) > 1:
                current.pop(0)
        else:
            current.append(unit)
    if current:
        final_chunk = " ".join(current).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)
    return chunks


def build_document_chunks_with_report(
    pdf_dir: Path = PDF_DIR,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
) -> IngestionReport:
    report = IngestionReport()
    pdf_dir = pdf_dir.resolve()

    for pdf_path in load_pdf_paths(pdf_dir):
        try:
            relative_path = pdf_path.resolve().relative_to(pdf_dir).as_posix()
            content_hash = file_sha256(pdf_path)
            pages = extract_pdf_text(pdf_path)
            extracted_any_text = False
            for page_number, page_text in pages:
                if not page_text:
                    report.warnings.append(
                        f"{relative_path} page {page_number} contains no extractable text; "
                        "OCR may be required."
                    )
                    continue
                extracted_any_text = True
                for chunk_index, chunk in enumerate(
                    chunk_text(page_text, chunk_size=chunk_size, overlap=overlap),
                    start=1,
                ):
                    chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    stable_id = hashlib.sha256(
                        f"{relative_path}:{page_number}:{chunk_index}:{chunk_hash}".encode("utf-8")
                    ).hexdigest()
                    report.documents.append(
                        {
                            "id": stable_id,
                            "text": chunk,
                            "metadata": {
                                "source": pdf_path.name,
                                "relative_path": relative_path,
                                "page": page_number,
                                "chunk": chunk_index,
                                "file_sha256": content_hash,
                                "chunk_sha256": chunk_hash,
                            },
                        }
                    )
            if extracted_any_text:
                report.processed_files.append(relative_path)
            else:
                report.warnings.append(
                    f"{relative_path} produced no text. Run OCR on the PDF before ingestion."
                )
        except Exception as exc:
            report.errors.append(f"{pdf_path.name}: {type(exc).__name__}: {exc}")
    return report


def build_document_chunks(
    pdf_dir: Path = PDF_DIR,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
) -> list[dict]:
    """Backward-compatible convenience wrapper."""
    return build_document_chunks_with_report(pdf_dir, chunk_size, overlap).documents
