from __future__ import annotations
from pathlib import Path
import tempfile
from typing import List
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR


def store_exists(persist_directory: Path = CHROMA_DIR) -> bool:
    """Check if vector store has been populated (chromadb or fallback files)."""
    if persist_directory.exists():
        # Check chromadb directory
        if (persist_directory / 'metadata').exists():
            return True
        # Check fallback files
        if (persist_directory / 'embeddings.npy').exists() and (persist_directory / 'documents.json').exists():
            return True
    return False


def create_client(persist_directory: Path = CHROMA_DIR) -> chromadb.PersistentClient:
    """Create a ChromaDB client, with fallback for Windows path issues."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    # Prefer a project-relative POSIX path to avoid Windows drive-letter syntax issues
    try:
        rel_path = persist_directory.resolve().relative_to(Path.cwd())
        persist_dir_str = rel_path.as_posix()
    except Exception:
        persist_dir_str = persist_directory.resolve().as_posix()

    # If the path contains spaces or other problematic characters on Windows,
    # fall back to a temp directory without spaces to avoid Rust binding errors.
    if ' ' in persist_dir_str:
        fallback = Path(tempfile.gettempdir()) / 'rag_ai_chromadb'
        fallback.mkdir(parents=True, exist_ok=True)
        persist_dir_str = fallback.as_posix()

    try:
        return chromadb.PersistentClient(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir_str)
        )
    except Exception:
        # Fall back to a file-based store when chromadb bindings are unavailable.
        return None


def get_collection(client: chromadb.PersistentClient, name: str = "rag_documents"):
    """Get or create a ChromaDB collection."""
    if client is None:
        return None
    try:
        return client.get_collection(name=name)
    except ValueError:
        return client.create_collection(name=name)


def persist_documents(
    documents: list[dict],
    embeddings: list[list[float]],
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = "rag_documents",
):
    """
    Save documents and embeddings to ChromaDB.
    Falls back to numpy/JSON file storage if ChromaDB is unavailable.
    """
    client = create_client(persist_directory)
    if client is not None:
        collection = get_collection(client, collection_name)
        collection.upsert(
            ids=[document["id"] for document in documents],
            documents=[document["text"] for document in documents],
            metadatas=[document["metadata"] for document in documents],
            embeddings=embeddings,
        )
        client.persist()
        return collection

    # Fallback: save embeddings and metadata to files for a simple file-based retriever
    persist_directory.mkdir(parents=True, exist_ok=True)
    emb_path = persist_directory / "embeddings.npy"
    meta_path = persist_directory / "documents.json"
    np.save(emb_path, np.array(embeddings, dtype=np.float32))
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(documents, fh, ensure_ascii=False, indent=2)
    return None
