from pathlib import Path

from src.retrieval import Retriever
from src.store import create_client, get_collection, persist_documents, reset_store, store_exists


class FakeEmbedder:
    model_name = "fake-model"
    dimension = 2

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "banana" in lowered:
                vectors.append([1.0, 0.0])
            elif "engine" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.7071, 0.7071])
        return vectors


def _documents():
    return [
        {
            "id": "fruit",
            "text": "Bananas are yellow fruit rich in potassium.",
            "metadata": {
                "source": "food.pdf",
                "relative_path": "food.pdf",
                "page": 1,
                "chunk": 1,
                "file_sha256": "a" * 64,
                "chunk_sha256": "b" * 64,
            },
        },
        {
            "id": "engine",
            "text": "A combustion engine converts fuel into motion.",
            "metadata": {
                "source": "cars.pdf",
                "relative_path": "cars.pdf",
                "page": 2,
                "chunk": 1,
                "file_sha256": "c" * 64,
                "chunk_sha256": "d" * 64,
            },
        },
    ]


def test_chroma_round_trip_and_hybrid_retrieval(tmp_path: Path):
    index_dir = tmp_path / "index with spaces"
    documents = _documents()
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    persist_documents(
        documents,
        embeddings,
        persist_directory=index_dir,
        embedding_model="fake-model",
    )

    assert store_exists(index_dir)
    client = create_client(index_dir)
    collection = get_collection(client, create=False)
    retriever = Retriever(
        top_k=1,
        candidate_k=2,
        min_score=0.1,
        embedder=FakeEmbedder(),
        collection=collection,
    )
    results = retriever.retrieve("Which fruit is banana?", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "food.pdf"
    assert results[0].metadata["retrieval_score"] >= 0.1
    client.close()


def test_reingestion_removes_stale_documents(tmp_path: Path):
    index_dir = tmp_path / "index"
    documents = _documents()
    persist_documents(
        documents,
        [[1.0, 0.0], [0.0, 1.0]],
        persist_directory=index_dir,
        embedding_model="fake-model",
    )
    persist_documents(
        documents[:1],
        [[1.0, 0.0]],
        persist_directory=index_dir,
        embedding_model="fake-model",
    )

    client = create_client(index_dir)
    collection = get_collection(client, create=False)
    assert collection.count() == 1
    assert collection.get()["ids"] == ["fruit"]
    client.close()
    assert reset_store(index_dir)
    assert not index_dir.exists()
