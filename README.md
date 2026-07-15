# RAG AI - Local Retrieval-Augmented Generation

A fully-local, privacy-focused Retrieval-Augmented Generation (RAG) system for PDF documents. No API keys. No quotas. No internet required.

## Architecture

**Document Processing:**
- PDFs are extracted with `pypdf`
- Text is split into overlapping chunks
- Embeddings are generated with `sentence-transformers` (all-MiniLM-L6-v2)
- ChromaDB stores text chunks, vectors, and metadata for fast semantic search

**Query Processing:**
- Questions are used directly to search ChromaDB (no query rewriting for speed)
- Top-2 relevant document chunks are retrieved via cosine similarity
- Local Ollama models synthesize answers from retrieved context
- Responses are cached for instant retrieval on repeat queries
- Performance metrics tracked for visibility into latency

## Key Features

- **100% Local & Private** - Runs entirely on your machine with Ollama
- **No API Keys Required** - Zero cloud dependencies
- **Auto Model Detection** - Automatically selects fastest available Ollama model
- **Response Caching** - Identical queries answered instantly from cache
- **Real-time Streaming** - Watch answers generate live with `--stream` flag
- **Performance Metrics** - See retrieval/generation/total times
- **Low Latency** - ~2-3 min for first query, <1s for cached queries

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- At least one model installed in Ollama (phi3, llama3, mistral, etc.)

## Setup

1. Activate the virtual environment:

```powershell
& .\env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure Ollama is running:

```bash
ollama serve
```

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

## Supported Ollama Models

Auto-detects and prefers the fastest available model:

1. **phi3** (2.2 GB) - Fastest ⚡
2. **orca-mini** (3.3 GB) - Fast
3. **neural-chat** (4.7 GB) - Balanced
4. **llama3** (4.7 GB) - High quality
5. **mistral** (4.4 GB) - Balanced

Install models:
```bash
ollama pull phi3
ollama pull llama3
```

## Performance

| Metric | Time |
|--------|------|
| First Query | ~2-3 minutes |
| Cached Query | <1 second |
| Retrieval | ~1-2 seconds |
| Generation | ~1-2 minutes |

## Troubleshooting

**"Ollama is unavailable"** - Ensure Ollama is running:
```bash
ollama serve
```

**"Model not found"** - Pull the model:
```bash
ollama pull phi3
```

**Slow responses** - Try a faster model (phi3 is recommended for speed)
