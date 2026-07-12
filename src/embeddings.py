from __future__ import annotations
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

class LocalEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [list(vector) for vector in embeddings]
