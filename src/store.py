"""ChromaDB persistence and index-manifest management."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    INDEX_MANIFEST_PATH,
    INDEX_SCHEMA_VERSION,
    TEXT_CHUNK_OVERLAP,
    TEXT_CHUNK_SIZE,
)


class IndexCompatibilityError(RuntimeError):
    """Raised when the persisted index does not match current configuration."""


def create_client(persist_directory: Path = CHROMA_DIR):
    """Create the single authoritative persistent Chroma client."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_directory.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(
    client,
    name: str = COLLECTION_NAME,
    *,
    create: bool = False,
):
    try:
        return client.get_collection(name=name)
    except NotFoundError:
        if not create:
            return None
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def corpus_fingerprint(documents: list[dict]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item["id"]):
        digest.update(document["id"].encode("utf-8"))
        digest.update(document["metadata"]["file_sha256"].encode("utf-8"))
    return digest.hexdigest()


def load_manifest(path: Path = INDEX_MANIFEST_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return manifest if isinstance(manifest, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def validate_manifest(
    manifest: dict[str, Any] | None,
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    embedding_dimension: int | None = None,
) -> None:
    if not manifest:
        raise IndexCompatibilityError("The index manifest is missing or invalid; re-ingest documents.")
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise IndexCompatibilityError("The index schema changed; re-ingest documents.")
    if manifest.get("embedding_model") != embedding_model:
        raise IndexCompatibilityError(
            "The configured embedding model differs from the indexed model; re-ingest documents."
        )
    if (
        embedding_dimension is not None
        and manifest.get("embedding_dimension") != embedding_dimension
    ):
        raise IndexCompatibilityError(
            "The embedding dimension differs from the stored index; re-ingest documents."
        )
    if manifest.get("chunk_size") != TEXT_CHUNK_SIZE or manifest.get(
        "chunk_overlap"
    ) != TEXT_CHUNK_OVERLAP:
        raise IndexCompatibilityError(
            "Chunking configuration changed; re-ingest documents."
        )


def store_exists(
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    manifest_path = persist_directory / INDEX_MANIFEST_PATH.name
    if not persist_directory.exists() or not load_manifest(manifest_path):
        return False
    client = None
    try:
        client = create_client(persist_directory)
        collection = get_collection(client, collection_name, create=False)
        return collection is not None and collection.count() > 0
    except Exception:
        return False
    finally:
        if client is not None:
            client.close()


def persist_documents(
    documents: list[dict],
    embeddings: list[list[float]],
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> dict[str, Any]:
    """Replace the collection so removed or shortened documents cannot remain stale."""
    if not documents:
        raise ValueError("Cannot persist an empty document collection")
    if len(documents) != len(embeddings):
        raise ValueError("Document and embedding counts do not match")
    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1 or not next(iter(dimensions)):
        raise ValueError("All embeddings must have one consistent non-zero dimension")

    client = create_client(persist_directory)
    try:
        try:
            client.delete_collection(collection_name)
        except NotFoundError:
            pass
        collection = get_collection(client, collection_name, create=True)
        if collection is None:
            raise RuntimeError("Failed to create the Chroma collection")

        batch_size = 500
        for start in range(0, len(documents), batch_size):
            batch_documents = documents[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            collection.add(
                ids=[document["id"] for document in batch_documents],
                documents=[document["text"] for document in batch_documents],
                metadatas=[document["metadata"] for document in batch_documents],
                embeddings=batch_embeddings,
            )

        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "collection_name": collection_name,
            "corpus_fingerprint": corpus_fingerprint(documents),
            "embedding_model": embedding_model,
            "embedding_dimension": next(iter(dimensions)),
            "chunk_size": TEXT_CHUNK_SIZE,
            "chunk_overlap": TEXT_CHUNK_OVERLAP,
            "document_count": len(documents),
            "source_count": len(
                {document["metadata"]["relative_path"] for document in documents}
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = persist_directory / INDEX_MANIFEST_PATH.name
        _atomic_write_json(manifest_path, manifest)
        return manifest
    finally:
        client.close()


def reset_store(persist_directory: Path = CHROMA_DIR) -> bool:
    if not persist_directory.exists():
        return False
    resolved = persist_directory.resolve()
    if resolved == resolved.anchor or resolved == Path.home().resolve():
        raise ValueError(f"Refusing to delete unsafe path: {resolved}")
    shutil.rmtree(resolved)
    return True
