# RAG AI

A local Retrieval-Augmented Generation system for PDF documents.

## Architecture

- PDFs are extracted with `pypdf`
- Text is split into overlapping chunks
- Embeddings are generated with `sentence-transformers`
- ChromaDB stores text chunks, vectors, and metadata
- Gemini is used to rewrite questions and finalize answers
- LangGraph manages the agent workflow and decisions

## Setup

1. Activate the virtual environment:

```powershell
& .\\env\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file if you want to provide Gemini config:

```ini
GEMINI_API_KEY=your_api_key
GEMINI_API_ENDPOINT=
GEMINI_MODEL=gemini-1.5-pro
```

## Usage

Ingest PDFs from `data/` into ChromaDB:

```powershell
python main.py ingest
```

Ask a question:

```powershell
python main.py query "What does the document say about vaping prevention?"
```

Reset the local vector store:

```powershell
python main.py reset
```
