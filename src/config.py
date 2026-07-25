"""
Configuration module for RAG AI system.
Loads environment variables and sets up paths, models, and parameters.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directory configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / 'data'
CHROMA_DIR = ROOT_DIR / '.chromadb'

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2').strip()

# OpenRouter generation configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '').strip()
OPENROUTER_BASE_URL = os.getenv(
	'OPENROUTER_BASE_URL',
	'https://openrouter.ai/api/v1',
).strip().rstrip('/')
OPENROUTER_MODEL = os.getenv(
	'OPENROUTER_MODEL',
	'openai/gpt-4o-mini',
).strip()

# Text chunking configuration
TEXT_CHUNK_SIZE = int(os.getenv('TEXT_CHUNK_SIZE', '1000'))
TEXT_CHUNK_OVERLAP = int(os.getenv('TEXT_CHUNK_OVERLAP', '200'))
