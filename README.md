# RAG AI

A command-line Retrieval-Augmented Generation (RAG) application for asking
questions about PDF documents.

PDF parsing, embeddings, and semantic retrieval run locally. The retrieved
context is sent to an OpenRouter model to generate the final answer.

## Features

- Extracts and chunks text from PDF files
- Generates local embeddings with Sentence Transformers
- Stores and searches vectors with ChromaDB
- Uses LLM for answer generation
- Includes source file and page metadata in results
- Supports streamed responses
- Caches repeated questions
- Reports retrieval, generation, and total response times

## How It Works

```text
PDF files
   |
   v
Text extraction and chunking
   |
   v
Local embeddings
   |
   v
ChromaDB vector store
   |
   v
Semantic retrieval -- top relevant chunks
   |
   v
LLM -- generated answer with sources
```

The application has two main workflows:

1. **Ingestion:** PDFs in `data/` are converted into chunks, embedded locally,
   and saved in `.chromadb/`.
2. **Querying:** A question is embedded locally, matched against the stored
   chunks, and sent to OpenRouter with the retrieved context.

## Requirements

- Python 3.14 or later, as currently specified in `pyproject.toml`
- An OpenRouter account and API key
- Internet access for OpenRouter and the initial embedding-model download

Ollama and Gemini are not required.

## Installation

### 1. Create and activate a virtual environment

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

### 3. Configure Model

```powershell
Create a .env file
```

On macOS or Linux:

```bash
cp .env.example .env
```

```dotenv
Your_API_KEY=your_key_here
Your_MODEL=********
```

The `.env` file is ignored by Git. Never commit or share an API key. If a key
has been exposed, revoke it and create a new one.

## Configuration

All settings are optional except `OPENROUTER_API_KEY`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `Your_API_KEY` | None | Authenticates requests to OpenRouter |
| `Your_MODEL` | `openai/gpt-4o-mini` | OpenRouter model identifier |
| `BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local Sentence Transformers model |
| `TEXT_CHUNK_SIZE` | `1000` | Maximum characters in each document chunk |
| `TEXT_CHUNK_OVERLAP` | `200` | Characters shared between adjacent chunks |

## Usage

### 1. Add documents

Place one or more `.pdf` files directly inside the `data/` directory.

### 2. Build the vector store

```bash
python main.py ingest
```

### 3. Ask a question

```bash
python main.py query "What are the main conclusions?"
```

If the vector store does not exist, the query command automatically ingests
the PDFs before answering.

### Stream the answer

```bash
python main.py query "Summarize the document" --stream
```

### Force document re-ingestion

Use this after adding, removing, or changing PDFs:

```bash
python main.py query "What changed?" --force-ingest
```

### Disable automatic ingestion

```bash
python main.py query "Your question" --no-auto-ingest
```

### Bypass the response cache

```bash
python main.py query "Your question" --no-cache
```

## Command Reference

| Command | Description |
| --- | --- |
| `python main.py ingest` | Process PDFs and build the vector store |
| `python main.py query "..."` | Retrieve context and generate an answer |
| `python main.py cache-clear` | Clear cached answers |
| `python main.py reset` | Delete the persisted vector store |

Query options:

| Option | Description |
| --- | --- |
| `--stream` | Print generated text as it arrives |
| `--force-ingest` | Rebuild the document index before querying |
| `--no-auto-ingest` | Do not ingest automatically when the index is missing |
| `--no-cache` | Do not read or write a cached answer |

## Project Structure

```text
RAG AI/
|-- data/                  PDF documents and response cache
|-- src/
|   |-- agent.py           OpenRouter generation and response caching
|   |-- config.py          Environment and path configuration
|   |-- embeddings.py      Local embedding model
|   |-- ingest.py          PDF extraction and text chunking
|   |-- retrieval.py       Semantic similarity search
|   `-- store.py           ChromaDB persistence and file fallback
|-- .env.example           Environment-variable template
|-- main.py                Command-line entry point
|-- requirements.txt       Python dependencies
`-- pyproject.toml         Project metadata
```

Generated data is stored in:

- `.chromadb/` for the vector index
- `data/.cache/query_cache.json` for cached answers

## Privacy and Security

Document ingestion and similarity search happen locally. During a query, the
question and retrieved document excerpts are sent to OpenRouter and the
selected model provider. Do not use sensitive documents unless that data flow
meets your privacy requirements.

Keep `.env` private and rotate any API key that is accidentally exposed.

## Troubleshooting

### `OPENROUTER_API_KEY is not set`

Create `.env` from `.env.example`, add a valid key, and run the command from
the project root.

### OpenRouter returns `401 Unauthorized`

Confirm that the API key is correct, active, and has not been revoked.

### OpenRouter reports a model error

Set `OPENROUTER_MODEL` to a valid model identifier available to your account.

### No PDF documents found

Add `.pdf` files directly to `data/`. PDFs inside nested folders are not
currently discovered.

### Answers do not reflect updated documents

Rebuild the index and bypass old cached responses:

```bash
python main.py ingest
python main.py cache-clear
```

### First run is slow

Sentence Transformers may download the embedding model on first use. Later
runs reuse the local model files.

## Current Limitations

- Only PDF documents are supported
- PDFs are discovered only at the top level of `data/`
- Scanned PDFs require OCR before ingestion
- Retrieval currently uses the two most relevant chunks
- Citations depend on metadata and model compliance
