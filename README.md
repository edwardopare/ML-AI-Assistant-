# RAG AI

A grounded command-line Retrieval-Augmented Generation (RAG) application for
asking questions about PDF documents.

Document parsing, embeddings, indexing, and retrieval run locally. Only the
question and selected evidence chunks are sent to an OpenRouter model for
answer generation.

## Capabilities

- Recursive PDF discovery and per-file ingestion error reporting
- Sentence- and paragraph-aware overlapping chunks
- Normalized local embeddings with Sentence Transformers
- Persistent cosine-vector storage with ChromaDB
- Hybrid dense and BM25 retrieval
- Relevance filtering and MMR diversity reranking
- Grounded prompting with validated evidence identifiers
- Optional streamed OpenRouter responses
- Retries for rate limits and transient provider errors
- Corpus- and model-aware response caching
- Index manifests that detect incompatible configuration changes
- Optional OCR for scanned pages
- Retrieval and generation timing

## Architecture

```text
PDFs in data/
      |
      v
PDF text extraction ---- optional OCR
      |
      v
Sentence-aware chunking + source metadata + content hashes
      |
      v
Normalized local embeddings
      |
      v
Persistent ChromaDB index + versioned manifest
      |
      v
Dense search + BM25 lexical search
      |
      v
Score threshold + MMR reranking
      |
      v
Complete evidence chunks with [S1], [S2] identifiers
      |
      v
OpenRouter generation + citation validation
```

When retrieval finds no evidence above the configured threshold, the
application does not call the LLM. It reports that the indexed documents do
not contain sufficiently relevant information.

## Requirements

- Python 3.10 or later
- An OpenRouter API key
- Internet access for OpenRouter and the initial embedding-model download

Ollama and Gemini are not used.

## Installation

Create and activate a virtual environment.

PowerShell:

```powershell
python -m venv env
& .\env\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv env
source env/bin/activate
```

Install the application dependencies:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## OpenRouter Configuration

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENROUTER_API_KEY=your_rotated_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

The `.env` file is ignored by Git. Never commit an API key. Revoke and replace
any key that has been pasted into source code, chat, logs, or screenshots.

## Configuration Reference

### Generation

| Variable | Default | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | None | Required OpenRouter credential |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | OpenRouter model identifier |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `OPENROUTER_TIMEOUT_SECONDS` | `120` | HTTP timeout |
| `OPENROUTER_MAX_RETRIES` | `3` | Retries for network, `429`, and `5xx` errors |
| `OPENROUTER_MAX_TOKENS` | `800` | Maximum generated tokens |
| `OPENROUTER_TEMPERATURE` | `0.1` | Generation randomness |
| `MAX_CONTEXT_CHARS` | `12000` | Evidence budget sent to the LLM |

### Ingestion and retrieval

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local embedding model |
| `TEXT_CHUNK_SIZE` | `1200` | Approximate maximum chunk characters |
| `TEXT_CHUNK_OVERLAP` | `200` | Context shared by adjacent chunks |
| `ENABLE_OCR` | `false` | OCR pages without extractable text |
| `RETRIEVAL_TOP_K` | `4` | Final evidence chunks |
| `RETRIEVAL_CANDIDATE_K` | `12` | Candidates considered by each retriever |
| `RETRIEVAL_MIN_SCORE` | `0.25` | Minimum combined relevance score |
| `RETRIEVAL_DENSE_WEIGHT` | `0.7` | Dense share of the dense/BM25 score |
| `RETRIEVAL_MMR_LAMBDA` | `0.75` | Relevance-versus-diversity balance |

Changing the embedding model or chunk settings makes an existing index
incompatible. The CLI will request re-ingestion instead of querying mismatched
vectors.

## Usage

Place PDFs anywhere under `data/`. Nested folders are supported.

Build or completely rebuild the index:

```bash
python main.py ingest
```

Ask a question:

```bash
python main.py query "What are the report's main conclusions?"
```

The query command builds a missing index automatically. Other examples:

```bash
python main.py query "Compare the proposed approaches" --stream
python main.py query "Summarize the risks" --top-k 6
python main.py query "What changed?" --force-ingest
python main.py query "Question" --no-auto-ingest
python main.py query "Question" --no-cache
```

Maintenance commands:

```bash
python main.py cache-clear
python main.py reset
```

`reset` deletes only the configured `.chromadb/` index. Source PDFs are not
removed.

## Optional OCR

Install the OCR dependencies:

```bash
pip install -e ".[ocr]"
```

Install the Tesseract executable separately for your operating system, ensure
it is available on `PATH`, and set:

```dotenv
ENABLE_OCR=true
```

OCR is attempted only for pages where normal PDF extraction returns no text.

## Evidence and Citations

Retrieved chunks are assigned evidence IDs such as `[S1]`. The model is
instructed to cite factual claims using these IDs. Unknown IDs in a generated
answer are marked invalid, and the CLI prints the authoritative source, page,
and chunk mapping separately.

Retrieval metadata also includes:

- Combined retrieval score
- Dense similarity score
- Lexical BM25 score
- Final retrieval rank

## Index and Cache Lifecycle

Each ingestion replaces the collection, which prevents chunks from removed or
shortened PDFs from remaining searchable.

`.chromadb/index_manifest.json` records:

- Corpus fingerprint
- Index schema version
- Embedding model and dimension
- Chunk settings
- Source and chunk counts
- Ingestion timestamp

Answer-cache keys include the corpus fingerprint, question, OpenRouter model,
retrieval count, prompt version, temperature, and output limit. Changing the
corpus or generation configuration therefore does not return an old answer.

## Project Structure

```text
RAG AI/
|-- data/                   PDF inputs and generated answer cache
|-- src/
|   |-- agent.py            OpenRouter, grounding, citations, and cache
|   |-- config.py           Validated environment configuration
|   |-- embeddings.py       Normalized local embeddings
|   |-- ingest.py           PDF extraction, OCR, hashing, and chunking
|   |-- retrieval.py        Dense/BM25 retrieval and MMR
|   `-- store.py            Chroma lifecycle and index manifest
|-- tests/                  Offline unit and integration tests
|-- main.py                 CLI entry point
|-- pyproject.toml          Package and optional dependency metadata
`-- requirements.txt        Runtime dependencies
```

Generated state:

- `.chromadb/` contains the vector index and manifest.
- `data/.cache/query_cache.json` contains generated-answer cache entries.

Both are excluded from Git.

## Privacy and Security

PDF extraction, embeddings, and retrieval are local. For each non-empty
retrieval, the question and selected document chunks are transmitted to
OpenRouter and the selected downstream model provider.

Retrieved documents are treated as untrusted data in the prompt. Instructions
inside a PDF are explicitly excluded from the model's instruction hierarchy.
This reduces prompt-injection risk but cannot provide an absolute security
guarantee.

Review OpenRouter and model-provider data policies before processing sensitive
documents.

## Testing

Run the offline suite:

```bash
pytest
```

The tests cover:

- Chunking and configuration validation
- Recursive PDF discovery
- Real Chroma persistence on a path containing spaces
- Dense/BM25 retrieval
- Stale-document removal during re-ingestion
- Safe empty-evidence behavior
- Citation handling
- OpenRouter HTTP and streaming errors
- Versioned atomic caching
- CLI parsing

OpenRouter is mocked; tests do not consume API credits.

## Troubleshooting

### The index is incompatible

The embedding model, vector dimension, chunk settings, or schema changed:

```bash
python main.py ingest
```

### No sufficiently relevant information was found

The result is below `RETRIEVAL_MIN_SCORE`. Confirm that the correct PDFs were
indexed. If evaluation shows valid evidence is regularly rejected, lower the
threshold carefully.

### OpenRouter authentication failed

Confirm that `OPENROUTER_API_KEY` is current and has not been revoked.

### OpenRouter returns rate-limit or server errors

Transient `429` and `5xx` responses are retried automatically. Persistent
failures are returned as errors and are not cached.

### A scanned PDF produces no text

Enable optional OCR, or OCR the document before placing it in `data/`.

### Answers do not reflect changed PDFs

Re-run ingestion. The new corpus fingerprint automatically separates new
answers from old cache entries:

```bash
python main.py ingest
```

### First execution is slow

Sentence Transformers downloads the embedding model on first use. Later runs
reuse the local model files.

## Remaining Trade-offs

- BM25 currently scans stored chunk text and is intended for small or
  medium-sized local collections.
- OCR quality depends on scan quality, language data, and Tesseract.
- Table structure from complex PDFs may not be preserved perfectly.
- Citation correctness still depends partly on model behavior, although
  evidence IDs are validated.
- Retrieval thresholds should be tuned against a representative evaluation
  set for the target documents.
