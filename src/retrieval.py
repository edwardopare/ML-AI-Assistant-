from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass
import numpy as np
from langchain_core.documents import Document

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_MMR_LAMBDA,
    RETRIEVAL_TOP_K,
)
from src.embeddings import LocalEmbedder
from src.store import (
    IndexCompatibilityError,
    create_client,
    get_collection,
    load_manifest,
    validate_manifest,
)

_TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


@dataclass
class _Candidate:
    id: str
    text: str
    metadata: dict
    embedding: np.ndarray
    dense_score: float = 0.0
    lexical_score: float = 0.0
    combined_score: float = 0.0


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    query_terms = _tokens(query)
    if not query_terms or not documents:
        return [0.0] * len(documents)
    tokenized = [_tokens(document) for document in documents]
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    scores: list[float] = []
    k1, b = 1.5, 0.75
    document_count = len(tokenized)
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * length / max(average_length, 1.0)
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


class Retriever:

    def __init__(
        self,
        top_k: int = RETRIEVAL_TOP_K,
        *,
        candidate_k: int = RETRIEVAL_CANDIDATE_K,
        min_score: float = RETRIEVAL_MIN_SCORE,
        dense_weight: float = RETRIEVAL_DENSE_WEIGHT,
        mmr_lambda: float = RETRIEVAL_MMR_LAMBDA,
        embedder: LocalEmbedder | None = None,
        collection=None,
    ) -> None:
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)
        self.min_score = min_score
        self.dense_weight = dense_weight
        self.mmr_lambda = mmr_lambda
        self.embedder = embedder or LocalEmbedder()

        self.client = None
        if collection is None:
            manifest = load_manifest()
            validate_manifest(
                manifest,
                embedding_model=getattr(
                    self.embedder,
                    "model_name",
                    EMBEDDING_MODEL_NAME,
                ),
                embedding_dimension=getattr(self.embedder, "dimension", None),
            )
            self.client = create_client()
            collection = get_collection(self.client, COLLECTION_NAME, create=False)
        if collection is None or collection.count() == 0:
            raise IndexCompatibilityError(
                "No populated document index exists. Run `python main.py ingest` first."
            )
        self.collection = collection

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        if not query.strip():
            return []
        limit = int(top_k or self.top_k)
        query_embedding = np.asarray(
            self.embedder.embed_texts([query])[0],
            dtype=np.float32,
        )
        dense_count = min(self.candidate_k, self.collection.count())
        dense_result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=dense_count,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        candidates: dict[str, _Candidate] = {}
        dense_ids = dense_result.get("ids", [[]])[0]
        dense_documents = dense_result.get("documents", [[]])[0]
        dense_metadatas = dense_result.get("metadatas", [[]])[0]
        dense_distances = dense_result.get("distances", [[]])[0]
        dense_embeddings = dense_result.get("embeddings")
        dense_vectors = dense_embeddings[0] if dense_embeddings is not None else []
        for identifier, text, metadata, distance, embedding in zip(
            dense_ids,
            dense_documents,
            dense_metadatas,
            dense_distances,
            dense_vectors,
        ):
            candidates[identifier] = _Candidate(
                id=identifier,
                text=text,
                metadata=metadata or {},
                embedding=np.asarray(embedding, dtype=np.float32),
                dense_score=max(0.0, min(1.0, 1.0 - float(distance) / 2.0)),
            )

        all_records = self.collection.get(
            include=["documents", "metadatas", "embeddings"],
        )
        all_ids = all_records.get("ids", [])
        all_documents = all_records.get("documents") or []
        all_metadatas = all_records.get("metadatas") or []
        all_embeddings = all_records.get("embeddings")
        if all_embeddings is None:
            all_embeddings = []
        lexical_scores = _bm25_scores(query, all_documents)
        lexical_order = np.argsort(-np.asarray(lexical_scores))[: self.candidate_k]
        for index in lexical_order:
            index = int(index)
            if lexical_scores[index] <= 0:
                continue
            identifier = all_ids[index]
            candidate = candidates.get(identifier)
            if candidate is None:
                candidate = _Candidate(
                    id=identifier,
                    text=all_documents[index],
                    metadata=all_metadatas[index] or {},
                    embedding=np.asarray(all_embeddings[index], dtype=np.float32),
                )
                candidates[identifier] = candidate
            candidate.lexical_score = lexical_scores[index]

        for candidate in candidates.values():
            candidate.combined_score = (
                self.dense_weight * candidate.dense_score
                + (1.0 - self.dense_weight) * candidate.lexical_score
            )

        eligible = [
            candidate
            for candidate in candidates.values()
            if candidate.combined_score >= self.min_score
        ]
        selected: list[_Candidate] = []
        while eligible and len(selected) < limit:
            best = max(
                eligible,
                key=lambda candidate: self.mmr_lambda * candidate.combined_score
                - (1.0 - self.mmr_lambda)
                * max(
                    (
                        _cosine_similarity(candidate.embedding, chosen.embedding)
                        for chosen in selected
                    ),
                    default=0.0,
                ),
            )
            selected.append(best)
            eligible.remove(best)

        documents: list[Document] = []
        for rank, candidate in enumerate(selected, start=1):
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "retrieval_rank": rank,
                    "retrieval_score": round(candidate.combined_score, 4),
                    "dense_score": round(candidate.dense_score, 4),
                    "lexical_score": round(candidate.lexical_score, 4),
                }
            )
            documents.append(Document(page_content=candidate.text, metadata=metadata))
        return documents
