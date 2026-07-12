from __future__ import annotations
from pathlib import Path
from typing import List
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR


def create_client(persist_directory: Path = CHROMA_DIR) -> chromadb.PersistentClient:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_directory))
    )

def get_collection(client: chromadb.PersistentClient, name: str = "rag_documents"):
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
    client = create_client(persist_directory)
    collection = get_collection(client, collection_name)
    collection.upsert(
        ids=[document["id"] for document in documents],
        documents=[document["text"] for document in documents],
        metadatas=[document["metadata"] for document in documents],
        embeddings=embeddings,
    )
    client.persist()
    return collection
