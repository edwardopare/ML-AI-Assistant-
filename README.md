# RAG AI

RAG AI is a grounded Retrieval-Augmented Generation system for asking questions
about PDF documents. It can be used through:

- A command-line interface
- A JSON API for a web application
- A Microsoft Teams-compatible channel adapter
- Durable per-conversation chat memory backed by SQLite

PDF parsing, embeddings, indexing, and retrieval run locally. For questions
with relevant evidence, only the question and selected document chunks are sent
to OpenRouter for answer generation.

## What Is Built

### RAG pipeline

- Recursive PDF discovery
- Sentence- and paragraph-aware overlapping chunks
- Local Sentence Transformers embeddings
- Persistent ChromaDB vector storage
- Hybrid dense and BM25 retrieval
- Relevance filtering and MMR diversity reranking
- Grounded OpenRouter generation
- Validated evidence identifiers such as `[S1]`
- Corpus- and model-aware answer caching
- Optional OCR for scanned PDF pages
- Retrieval and generation performance metrics

### Application channels

| Channel | Endpoint or command | Current behavior |
| --- | --- | --- |
| CLI | `python main.py query "..."` | Prints an answer, evidence, and performance metrics |
| Web app | `POST /api/v1/channels/web/messages` | Returns a structured JSON RAG response |
| Microsoft Teams | `POST /api/v1/channels/teams/messages` | Accepts a Bot Framework-style activity and returns a reply activity |

Both API channels use the same RAG pipeline. Channel-specific code only
translates incoming messages and formats outgoing responses.

When `conversation_id` is supplied, recent messages are loaded before
retrieval and generation, and the new user/assistant exchange is stored after a
successful answer. History is isolated by channel, conversation ID, and user
ID. Requests without a conversation ID remain stateless.

## How It Works

```text
PDF files in data/
        |
        v
Text extraction + optional OCR
        |
        v
Chunking + source metadata + content hashes
        |
        v
Local normalized embeddings
        |
        v
ChromaDB index + versioned manifest
        |
        v
Dense retrieval + BM25 retrieval
        |
        v
Score threshold + MMR reranking
        |
        v
Evidence chunks labeled [S1], [S2], ...
        |
        v
OpenRouter generation + citation validation
        |
        v
CLI, web JSON, or Teams reply
```

If retrieval finds no evidence above the configured threshold, the LLM is not
called. The application reports that the indexed documents do not contain
sufficiently relevant information.

## Requirements

- Python 3.10 or later
- An OpenRouter API key
- Internet access for OpenRouter and the initial embedding-model download

Ollama and Gemini are not used.

## Quick Start

### 1. Create a virtual environment

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

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -e ".[dev]"
```

### 3. Configure OpenRouter

Create a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

The `.env` file is ignored by Git. Never commit credentials. Revoke and replace
any key exposed in source code, logs, screenshots, or chat.

### 4. Add and index documents

Place PDF files anywhere under `data/`. Nested folders are supported.

```bash
python main.py ingest
```

### 5. Choose how to use the RAG

Use the CLI:

```bash
python main.py query "What are the main conclusions?"
```

Or start the API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

While Uvicorn is running:

- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Readiness check: `http://localhost:8000/ready`

`0.0.0.0` makes the server listen on all local network interfaces. Use
`localhost` when opening it on the same computer. Press `Ctrl+C` to stop it.

## API Reference

### Operations

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Confirms that the FastAPI process is running |
| `GET` | `/ready` | Confirms that a document index exists |
| `GET` | `/docs` | Interactive OpenAPI documentation |

`/health` may return `200` before the RAG is usable. `/ready` returns `503`
until the document index has been created.

### Web application channel

```http
POST /api/v1/channels/web/messages
Content-Type: application/json
```

Request:

```json
{
  "message": "What are the main conclusions?",
  "conversation_id": "web-1",
  "user_id": "user-1"
}
```

Only `message` is required. Supply a stable `conversation_id` and `user_id` to
continue a stored conversation. Omitting `conversation_id` makes the request
stateless.

Example response:

```json
{
  "channel": "web",
  "conversation_id": "web-1",
  "user_id": "user-1",
  "history_messages_used": 0,
  "query": "What are the main conclusions?",
  "answer": "The grounded answer includes evidence [S1].",
  "citations": {
    "S1": {
      "source": "document.pdf",
      "relative_path": "document.pdf",
      "page": 2,
      "chunk": 1,
      "retrieval_score": 0.87
    }
  },
  "metrics": {
    "retrieval_time_s": 0.12,
    "generation_time_s": 1.43,
    "total_time_s": 1.55,
    "cache_hit": false
  }
}
```

PowerShell example:

```powershell
$body = @{
    message = "What are the main conclusions?"
    conversation_id = "web-1"
    user_id = "local-user"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/api/v1/channels/web/messages" `
    -ContentType "application/json" `
    -Body $body
```

### Microsoft Teams channel

```http
POST /api/v1/channels/teams/messages
Content-Type: application/json
```

The endpoint currently:

- Accepts a Bot Framework-style `message` activity
- Removes the bot `<at>...</at>` mention from the question
- Sends the question through the shared RAG service
- Formats citations as Teams-friendly Markdown
- Returns a Bot Framework-style reply activity
- Ignores non-message activities with a small status response

Example request:

```json
{
  "type": "message",
  "id": "activity-1",
  "text": "<at>RAG bot</at> What are the main conclusions?",
  "conversation": {"id": "teams-conversation-1"},
  "from": {"id": "teams-user-1", "name": "User"},
  "recipient": {"id": "rag-bot-1", "name": "RAG bot"}
}
```

Important: this is the channel payload adapter, not a complete production Teams
bot deployment. Microsoft Teams cannot call a local `localhost` URL. A
production integration still needs:

- A public HTTPS deployment
- Microsoft Entra ID and bot registration
- Bot Framework token/JWT validation
- The appropriate Bot Framework or Teams transport for delivering replies

Do not expose the Teams endpoint publicly without authentication.

## CLI Reference

The query command automatically builds a missing index unless disabled:

```bash
python main.py query "Compare the proposed approaches"
python main.py query "Summarize the risks" --top-k 6
python main.py query "What changed?" --force-ingest
python main.py query "Question" --stream
python main.py query "Question" --no-auto-ingest
python main.py query "Question" --no-cache
```

Maintenance commands:

```bash
python main.py cache-clear
python main.py reset
```

`reset` deletes only the configured `.chromadb/` index. It does not delete
source PDFs.

## Configuration

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
| `MAX_CONTEXT_CHARS` | `12000` | Evidence budget sent to the model |
| `CONVERSATION_HISTORY_MESSAGES` | `10` | Recent messages loaded for a conversation |

### Ingestion and retrieval

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local embedding model |
| `TEXT_CHUNK_SIZE` | `1200` | Approximate maximum chunk size |
| `TEXT_CHUNK_OVERLAP` | `200` | Context shared by adjacent chunks |
| `ENABLE_OCR` | `false` | OCR pages without extractable text |
| `RETRIEVAL_TOP_K` | `4` | Final evidence chunks |
| `RETRIEVAL_CANDIDATE_K` | `12` | Candidates considered by each retriever |
| `RETRIEVAL_MIN_SCORE` | `0.25` | Minimum combined relevance score |
| `RETRIEVAL_DENSE_WEIGHT` | `0.7` | Dense share of the hybrid score |
| `RETRIEVAL_MMR_LAMBDA` | `0.75` | Relevance-versus-diversity balance |

Changing the embedding model or chunk settings makes an existing index
incompatible. Re-run ingestion after changing those values.

## Optional OCR

Install the optional Python dependencies:

```bash
pip install -e ".[ocr]"
```

Install the Tesseract executable separately, make sure it is available on
`PATH`, and set:

```dotenv
ENABLE_OCR=true
```

OCR is attempted only when normal PDF extraction returns no text for a page.

## Evidence, Index, and Cache Behavior

Retrieved chunks receive evidence IDs such as `[S1]`. The model is instructed
to cite factual claims using these IDs. Unknown citation IDs in generated text
are marked invalid.

Each ingestion replaces the stored collection so chunks from removed or
shortened PDFs do not remain searchable. The index manifest records:

- Corpus fingerprint
- Index schema version
- Embedding model and dimension
- Chunk settings
- Source and chunk counts
- Ingestion timestamp

Answer-cache keys include the corpus fingerprint, normalized question,
OpenRouter model, retrieval count, prompt version, temperature, and output
limit. Changes to the corpus or generation configuration therefore do not
return an answer cached under the old configuration.

## Project Structure

```text
RAG AI/
|-- data/                   PDF inputs and generated answer cache
|-- src/
|   |-- agent.py            RAG orchestration, OpenRouter, citations, cache
|   |-- api.py              FastAPI application and channel endpoints
|   |-- channels.py         Web/Teams-neutral channel service and Teams adapter
|   |-- conversations.py    SQLite conversation history
|   |-- config.py           Validated environment configuration
|   |-- embeddings.py       Local normalized embeddings
|   |-- ingest.py           PDF extraction, OCR, hashing, and chunking
|   |-- retrieval.py        Dense/BM25 retrieval and MMR reranking
|   `-- store.py            ChromaDB lifecycle and index manifest
|-- tests/                  Offline unit and integration tests
|-- main.py                 CLI entry point
|-- pyproject.toml          Package and optional dependency metadata
|-- requirements.txt        Runtime dependencies
`-- uv.lock                 Reproducible dependency lock
```

Generated state:

- `.chromadb/` contains the vector index and manifest.
- `data/.cache/query_cache.json` contains generated answers.
- `data/.cache/conversations.sqlite3` contains channel conversation history.

Both paths are excluded from Git.

## Privacy and Security

PDF extraction, embeddings, and retrieval are local. When relevant evidence is
found, the question and selected chunks are transmitted to OpenRouter and the
selected downstream model provider.

Retrieved documents are treated as untrusted reference data. Instructions
inside PDFs are excluded from the model's instruction hierarchy. This reduces
prompt-injection risk but cannot guarantee complete protection.

Review OpenRouter and downstream-provider data policies before processing
sensitive documents. Add authentication, authorization, rate limiting, request
size limits, HTTPS, and appropriate logging controls before deploying the API
to an untrusted network.

## Testing

Run the offline suite:

```bash
pytest
```

The tests cover ingestion, chunking, storage, hybrid retrieval, stale-document
removal, citation handling, OpenRouter failures, answer caching, CLI parsing,
the web API channel, and the Teams activity adapter. OpenRouter is mocked, so
tests do not consume API credits.

## Troubleshooting

### `/ready` returns `503`

Create the document index:

```bash
python main.py ingest
```

### The index is incompatible

The embedding model, vector dimension, chunk settings, or schema changed.
Rebuild the index:

```bash
python main.py ingest
```

### No sufficiently relevant information was found

Confirm that the correct PDFs were indexed. The best results may be below
`RETRIEVAL_MIN_SCORE`; tune that threshold carefully against representative
questions.

### OpenRouter authentication failed

Confirm that `OPENROUTER_API_KEY` is current and has not been revoked.

### OpenRouter returns rate-limit or server errors

Transient `429` and `5xx` responses are retried. Persistent failures are
returned as service errors and are not cached.

### A scanned PDF produces no text

Enable optional OCR or OCR the document before placing it in `data/`.

### Answers do not reflect changed PDFs

Re-run ingestion. The new corpus fingerprint separates answers for the updated
documents from old cached answers.

### The first request is slow

Sentence Transformers downloads the embedding model on first use. Later runs
reuse the locally cached model.

## Current Limitations

- Conversation memory is local SQLite storage and is not yet equipped with
  retention jobs or user-facing deletion controls.
- The Teams adapter does not implement bot registration, JWT validation, or
  outbound Bot Framework transport.
- BM25 scans stored chunk text and is intended for small or medium collections.
- OCR quality depends on scan quality, language data, and Tesseract.
- Complex PDF table structure may not be preserved.
- Citation correctness still partly depends on model behavior, although
  evidence IDs are validated.
- Retrieval thresholds should be tuned against a representative evaluation
  set.
