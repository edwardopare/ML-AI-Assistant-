"""Command-line interface for the PDF RAG application."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from src.agent import OpenRouterError, RAGAgent, ResponseCache
from src.config import CHROMA_DIR, RETRIEVAL_TOP_K
from src.embeddings import LocalEmbedder
from src.ingest import build_document_chunks_with_report
from src.store import (
    IndexCompatibilityError,
    persist_documents,
    reset_store,
    store_exists,
)


def ingest() -> bool:
    report = build_document_chunks_with_report()
    for warning in report.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"Error: {error}", file=sys.stderr)
    if not report.documents:
        print("No extractable PDF text was found in data/.", file=sys.stderr)
        return False

    print(
        f"Embedding {len(report.documents)} chunks from "
        f"{len(report.processed_files)} PDF file(s)..."
    )
    embedder = LocalEmbedder()
    embeddings = embedder.embed_texts(
        [document["text"] for document in report.documents]
    )
    manifest = persist_documents(
        report.documents,
        embeddings,
        embedding_model=embedder.model_name,
    )
    print(
        f"Indexed {manifest['document_count']} chunks from "
        f"{manifest['source_count']} source(s) in {CHROMA_DIR}"
    )
    print(f"Corpus fingerprint: {manifest['corpus_fingerprint'][:12]}")
    return True


def query(
    question: str,
    *,
    auto_ingest: bool = True,
    force_ingest: bool = False,
    stream: bool = False,
    use_cache: bool = True,
    top_k: int = RETRIEVAL_TOP_K,
) -> bool:
    if not question.strip():
        print("Please provide a question.", file=sys.stderr)
        return False

    if force_ingest or (auto_ingest and not store_exists()):
        print("Ingesting documents...")
        if not ingest():
            return False
    if not store_exists():
        print(
            "No usable index exists. Run `python main.py ingest` first.",
            file=sys.stderr,
        )
        return False

    try:
        agent = RAGAgent(top_k=top_k, stream=stream, use_cache=use_cache)
        result = agent.answer(question)
    except (IndexCompatibilityError, OpenRouterError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False

    print("\n=== Answer ===")
    print(result["answer"])
    if result["citations"]:
        print("\n=== Evidence ===")
        for citation_id, citation in result["citations"].items():
            print(
                f"[{citation_id}] {citation.get('source')} page "
                f"{citation.get('page')} chunk {citation.get('chunk')}"
            )
    else:
        print("\nNo sufficiently relevant sources were retrieved.")

    metrics = result["metrics"]
    print("\n=== Performance ===")
    print(
        f"Retrieval: {metrics.get('retrieval_time_s', 0)}s | "
        f"Generation: {metrics.get('generation_time_s', 0)}s | "
        f"Total: {metrics.get('total_time_s', 0)}s | "
        f"Cache hit: {metrics.get('cache_hit', False)}"
    )
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounded PDF question answering powered by OpenRouter"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Ingest PDFs and rebuild the vector index")

    query_parser = subparsers.add_parser(
        "query",
        help="Ask a question over the indexed documents",
    )
    query_parser.add_argument("question", nargs="+", help="Question text")
    query_parser.add_argument(
        "--no-auto-ingest",
        action="store_true",
        help="Do not build a missing index automatically",
    )
    query_parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Rebuild the index before querying",
    )
    query_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the generated response",
    )
    query_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the response cache",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"Number of evidence chunks to use (default: {RETRIEVAL_TOP_K})",
    )

    subparsers.add_parser("reset", help="Delete the vector index")
    subparsers.add_parser("cache-clear", help="Clear generated-answer cache")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "ingest":
        return 0 if ingest() else 1
    if args.command == "query":
        if args.top_k < 1:
            print("Error: --top-k must be at least 1.", file=sys.stderr)
            return 2
        succeeded = query(
            " ".join(args.question),
            auto_ingest=not args.no_auto_ingest,
            force_ingest=args.force_ingest,
            stream=args.stream,
            use_cache=not args.no_cache,
            top_k=args.top_k,
        )
        return 0 if succeeded else 1
    if args.command == "reset":
        if reset_store():
            print(f"Removed vector index at {CHROMA_DIR}")
        else:
            print(f"No vector index found at {CHROMA_DIR}")
        return 0
    if args.command == "cache-clear":
        ResponseCache().clear()
        print("Response cache cleared.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
