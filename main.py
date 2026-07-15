from __future__ import annotations
import argparse
import shutil
from typing import Sequence
from src.agent import RAGAgent
from src.config import CHROMA_DIR
from src.embeddings import LocalEmbedder
from src.ingest import build_document_chunks
from src.store import persist_documents


def ingest() -> None:
    documents = build_document_chunks()
    if not documents:
        print('No PDF documents found in data/')
        return

    embedder = LocalEmbedder()
    embeddings = embedder.embed_texts([document['text'] for document in documents])
    persist_documents(documents, embeddings)
    print(f'Persisted {len(documents)} chunks to {CHROMA_DIR}')


def query(question: str, auto_ingest: bool = True, force_ingest: bool = False, stream: bool = False, use_cache: bool = True) -> None:
    if not question.strip():
        print('Please provide a question.')
        return
    
    # Auto-ingest: if store missing or --force-ingest, run ingestion first
    from src.store import store_exists
    if force_ingest or (auto_ingest and not store_exists()):
        print('Ingesting documents...')
        ingest()
    
    agent = RAGAgent(stream=stream, use_cache=use_cache)
    result = agent.answer(question)
    print('\n=== Answer ===')
    print(result['answer'])
    if result['sources']:
        print('\n=== Sources ===')
        for source in result['sources']:
            print(f"- {source.get('source')} page {source.get('page')} chunk {source.get('chunk')}")
    else:
        print('No sources were retrieved.')
    
    # Display metrics
    if 'metrics' in result:
        print('\n=== Performance ===')
        metrics = result['metrics']
        print(f"Retrieval: {metrics['retrieval_time_s']}s | Generation: {metrics['generation_time_s']}s | Total: {metrics['total_time_s']}s")


def reset() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f'Removed vector store at {CHROMA_DIR}')
    else:
        print(f'No vector store found at {CHROMA_DIR}')


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RAG AI local QA agent')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('ingest', help='Ingest PDFs and build the vector store')

    query_parser = sub.add_parser('query', help='Ask a question over the knowledge base')
    query_parser.add_argument('question', nargs='+', help='Question text')
    query_parser.add_argument('--no-auto-ingest', action='store_true', help='Disable automatic ingestion if store missing')
    query_parser.add_argument('--force-ingest', action='store_true', help='Force re-ingest before querying')
    query_parser.add_argument('--stream', action='store_true', help='Stream response in real-time')
    query_parser.add_argument('--no-cache', action='store_true', help='Disable response caching')

    sub.add_parser('reset', help='Remove persisted vector store and start fresh')
    sub.add_parser('cache-clear', help='Clear response cache')

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.command == 'ingest':
        ingest()
    elif args.command == 'query':
        auto_ingest = not args.no_auto_ingest if hasattr(args, 'no_auto_ingest') else True
        force_ingest = args.force_ingest if hasattr(args, 'force_ingest') else False
        stream = args.stream if hasattr(args, 'stream') else False
        use_cache = not args.no_cache if hasattr(args, 'no_cache') else True
        query(' '.join(args.question), auto_ingest=auto_ingest, force_ingest=force_ingest, stream=stream)
    elif args.command == 'reset':
        reset()
    elif args.command == 'cache-clear':
        from src.agent import ResponseCache
        cache = ResponseCache()
        cache.clear()
        print('Response cache cleared.')


if __name__ == '__main__':
    main()
