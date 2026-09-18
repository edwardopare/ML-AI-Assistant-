from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document

from .config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_MMR_LAMBDA,
    RETRIEVAL_TOP_K,
)
from .embeddings import LocalEmbedder
from .store import (
    IndexCompatibilityError,
    close_client,
    create_client,
    get_collection,
    load_manifest,
    validate_manifest,
)

_TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


# Represent one scored retrieval candidate.
@dataclass
class _Candidate:
    id: str
    text: str
    metadata: dict
    embedding: np.ndarray
    dense_score: float = 0.0
    lexical_score: float = 0.0
    combined_score: float = 0.0


# Normalize text into lexical retrieval tokens.
def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


# Calculate normalized BM25 scores for candidate documents.
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
            denominator = frequency + k1 * (1 - b + b * length / max(average_length, 1.0))
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


# Calculate cosine similarity between two vectors.
def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


# Retrieve evidence using dense search, BM25, and MMR.
class Retriever:
    # Initialize retrieval settings and open the indexed collection.
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

    # Close a ChromaDB client owned by this retriever.
    def close(self) -> None:
        if self.client is not None:
            close_client(self.client)
            self.client = None

    # Return relevant and diverse evidence for a query.
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
            strict=False,
        ):
            candidates[identifier] = _Candidate(
                id=identifier,
                text=text,
                metadata=metadata or {},
                embedding=np.asarray(embedding, dtype=np.float32),
                dense_score=max(0.0, min(1.0, 1.0 - float(distance) / 2.0)),
            )

        # Phase 2: BM25 lexical scoring over ids + documents only (no embeddings fetched yet).
        # This keeps memory cost O(corpus text) rather than O(corpus * embedding_dim).
        all_records_light = self.collection.get(
            include=["documents", "metadatas"],
        )
        all_ids = all_records_light.get("ids") or []
        all_docs_raw = all_records_light.get("documents")
        all_documents = all_docs_raw if all_docs_raw is not None else []
        all_metas_raw = all_records_light.get("metadatas")
        all_metadatas = all_metas_raw if all_metas_raw is not None else []

        lexical_scores = _bm25_scores(query, all_documents)
        # Find indices of the top-candidate_k documents by lexical score.
        lexical_order = [
            int(i)
            for i in sorted(
                range(len(lexical_scores)),
                key=lambda idx: -lexical_scores[idx],
            )[: self.candidate_k]
            if lexical_scores[int(i)] > 0
        ]

        # Collect the lexical candidates that are NOT already in the dense set.
        new_ids_needed = [all_ids[idx] for idx in lexical_order if all_ids[idx] not in candidates]

        # Targeted embedding fetch for only the new lexical-only candidates.
        if new_ids_needed:
            targeted = self.collection.get(
                ids=new_ids_needed,
                include=["documents", "metadatas", "embeddings"],
            )
            t_ids = targeted.get("ids") or []
            t_docs_raw = targeted.get("documents")
            t_docs = t_docs_raw if t_docs_raw is not None else []
            t_metas_raw = targeted.get("metadatas")
            t_metas = t_metas_raw if t_metas_raw is not None else []
            t_embs_raw = targeted.get("embeddings")
            t_embs = t_embs_raw if t_embs_raw is not None else []
            for identifier, text, metadata, embedding in zip(
                t_ids, t_docs, t_metas, t_embs, strict=False
            ):
                candidates[identifier] = _Candidate(
                    id=identifier,
                    text=text,
                    metadata=metadata or {},
                    embedding=np.asarray(embedding, dtype=np.float32),
                )

        # Apply lexical scores to all candidates in the lexical top-k.
        for idx in lexical_order:
            identifier = all_ids[idx]
            if identifier in candidates:
                candidates[identifier].lexical_score = lexical_scores[idx]
                # Supplement metadata if we fetched it via the light query.
                if not candidates[identifier].metadata and len(all_metadatas) > 0:
                    candidates[identifier].metadata = all_metadatas[idx] or {}

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
                key=lambda candidate: (
                    self.mmr_lambda * candidate.combined_score
                    - (1.0 - self.mmr_lambda)
                    * max(
                        (
                            _cosine_similarity(candidate.embedding, chosen.embedding)
                            for chosen in selected
                        ),
                        default=0.0,
                    )
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
