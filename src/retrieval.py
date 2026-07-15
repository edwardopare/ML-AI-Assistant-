from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from src.embeddings import LocalEmbedder
from src.store import create_client, get_collection, CHROMA_DIR
import numpy as np
import json
from pathlib import Path


class Retriever:
    """
    Semantic search retriever for finding relevant document chunks.
    Uses ChromaDB for vector similarity search or falls back to numpy-based search.
    """

    def __init__(self, top_k: int = 2) -> None:
        self.client = create_client()
        self.collection = get_collection(self.client)
        self.embedder = LocalEmbedder()
        self.top_k = top_k

        # If chromadb client is unavailable, load fallback embeddings and metadata
        self.fallback_embeddings = None
        self.fallback_documents = None
        if self.client is None:
            emb_path = Path(CHROMA_DIR) / "embeddings.npy"
            meta_path = Path(CHROMA_DIR) / "documents.json"
            if emb_path.exists() and meta_path.exists():
                try:
                    self.fallback_embeddings = np.load(emb_path)
                    with open(meta_path, 'r', encoding='utf-8') as fh:
                        self.fallback_documents = json.load(fh)
                except Exception:
                    self.fallback_embeddings = None
                    self.fallback_documents = None

    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Retrieve top-k most similar documents for a given query.
        Uses ChromaDB if available, otherwise falls back to numpy cosine similarity.
        """
        if not query.strip():
            return []
        query_embedding = self.embedder.embed_texts([query])[0]

        # Use chromadb when available
        if self.collection is not None:
            result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k or self.top_k)
            documents = []
            docs = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            for doc_text, metadata in zip(docs, metadatas):
                documents.append(Document(page_content=doc_text, metadata=metadata))
            return documents

        # Otherwise use the fallback numpy-based nearest neighbor search
        if self.fallback_embeddings is None or self.fallback_documents is None:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        embs = np.array(self.fallback_embeddings, dtype=np.float32)
        # Compute cosine similarity between query and all documents
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sims = (embs_norm @ q_norm).astype(np.float32)
        topk = int(top_k or self.top_k)
        idxs = np.argsort(-sims)[:topk]
        documents = []
        for i in idxs:
            doc = self.fallback_documents[int(i)]
            documents.append(Document(page_content=doc["text"], metadata=doc["metadata"]))
        return documents
