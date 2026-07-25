"""Local embedding generation."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


class LocalEmbedder:
    """Generate normalized embeddings with Sentence Transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("The embedding model did not report its vector dimension")
        return int(dimension)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in embeddings]
