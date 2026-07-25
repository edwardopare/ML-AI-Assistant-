from pathlib import Path

import pytest

from src.ingest import chunk_text, load_pdf_paths


def test_chunk_text_prefers_sentence_boundaries_and_preserves_content():
    text = "First sentence. Second sentence is longer. Third sentence finishes."
    chunks = chunk_text(text, chunk_size=35, overlap=0)

    assert chunks == [
        "First sentence.",
        "Second sentence is longer.",
        "Third sentence finishes.",
    ]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("content", chunk_size=100, overlap=100)


def test_pdf_discovery_is_recursive_and_case_insensitive_on_windows(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.pdf"
    second = nested / "b.pdf"
    first.touch()
    second.touch()

    assert load_pdf_paths(tmp_path) == [first, second]
