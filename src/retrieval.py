from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from src.embeddings import LocalEmbedder
from src.store import create_client, get_collection


class Retriever:
    def __init__(self, top_k: int = 5) -> None:
        self.client = create_client()
        self.collection = get_collection(self.client)
        self.embedder = LocalEmbedder()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        if not query.strip():
            return []

        query_embedding = self.embedder.embed_texts([query])[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k or self.top_k)
        documents = []

        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        for doc_text, metadata in zip(docs, metadatas):
            documents.append(Document(page_content=doc_text, metadata=metadata))

        return documents
