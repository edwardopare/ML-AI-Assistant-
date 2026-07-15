from __future__ import annotations
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME


class LocalEmbedder:
    """
    Local embedding generator using sentence-transformers.
    Converts text into dense vector representations for semantic search.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Convert a list of text strings into embedding vectors."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [list(vector) for vector in embeddings]
