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


def query(question: str) -> None:
    if not question.strip():
        print('Please provide a question.')
        return
    agent = RAGAgent()
    result = agent.answer(question)
    print('\n=== Answer ===')
    print(result['answer'])
    if result['sources']:
        print('\n=== Sources ===')
        for source in result['sources']:
            print(f"- {source.get('source')} page {source.get('page')} chunk {source.get('chunk')}")
    else:
        print('No sources were retrieved.')


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

    sub.add_parser('reset', help='Remove persisted vector store and start fresh')

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.command == 'ingest':
        ingest()
    elif args.command == 'query':
        query(' '.join(args.question))
    elif args.command == 'reset':
        reset()


if __name__ == '__main__':
    main()
