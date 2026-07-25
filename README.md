# RAG AI - Retrieval-Augmented Generation

A RAG system for PDF documents. Retrieval and embeddings run locally; answer generation uses OpenRouter.

## Architecture

**Document Processing:**
- PDFs are extracted with `pypdf`
- Text is split into overlapping chunks
- Embeddings are generated with `sentence-transformers` (all-MiniLM-L6-v2)
- ChromaDB stores text chunks, vectors, and metadata for fast semantic search

**Query Processing:**
- Questions are used directly to search ChromaDB (no query rewriting for speed)
- Top-2 relevant document chunks are retrieved via cosine similarity
- A configurable OpenRouter model synthesizes answers from retrieved context
- Responses are cached for instant retrieval on repeat queries
- Performance metrics tracked for visibility into latency

## Key Features

- **Local Retrieval** - PDF processing, embeddings, and vector search stay on your machine
- **OpenRouter Generation** - Use any compatible model available to your OpenRouter account
- **Response Caching** - Identical queries answered instantly from cache
- **Real-time Streaming** - Watch answers generate live with `--stream` flag
- **Performance Metrics** - See retrieval/generation/total times
- **Low Latency** - ~2-3 min for first query, <1s for cached queries

## Prerequisites

- Python 3.9+
- An OpenRouter API key

## Setup

1. Activate the virtual environment:

```powershell
& .\env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy the environment template and add your API key:

```powershell
Copy-Item .env.example .env
# Edit .env and replace the placeholder with your rotated OpenRouter key.
```

The default model is `openai/gpt-4o-mini`. Change `OPENROUTER_MODEL` in `.env`
to another OpenRouter model identifier if desired.

## Usage

**Ingest PDFs** from `data/` folder into vector store:

```powershell
python main.py ingest
```

**Ask a question:**

```powershell
python main.py query "What is machine learning?"
```

**Stream responses in real-time:**

```powershell
python main.py query "What is Python?" --stream
```

**Disable caching** for a query:

```powershell
python main.py query "What is NumPy?" --no-cache
```

**Force re-ingestion:**

```powershell
python main.py query "What is scikit-learn?" --force-ingest
```

**Clear response cache:**

```powershell
python main.py cache-clear
```

**Reset the vector store:**

```powershell
python main.py reset
```

## Performance

| Metric | Time |
|--------|------|
| First Query | ~2-3 minutes |
| Cached Query | <1 second |
| Retrieval | ~1-2 seconds |
| Generation | Depends on the selected OpenRouter model |

## Troubleshooting

**`OPENROUTER_API_KEY is not set`** - Create `.env` from `.env.example` and add
your rotated key.

**OpenRouter 401 error** - Check that the key is current and has not been revoked.

**OpenRouter model error** - Set `OPENROUTER_MODEL` to a model identifier your
account can access.
