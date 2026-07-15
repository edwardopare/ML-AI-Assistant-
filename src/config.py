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

# Legacy Gemini configuration (kept for backward compatibility)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
_raw_endpoint = os.getenv('GEMINI_API_ENDPOINT', '')
if _raw_endpoint is None:
	GEMINI_API_ENDPOINT = ''
else:
	_raw_endpoint = _raw_endpoint.strip()
	if _raw_endpoint.startswith('http://') or _raw_endpoint.startswith('https://'):
		GEMINI_API_ENDPOINT = _raw_endpoint
	else:
		# Try to find an embedded URL 
		import re
		m = re.search(r"https?://\S+", _raw_endpoint)
		GEMINI_API_ENDPOINT = m.group(0) if m else ''

GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro').strip()

# Text chunking configuration
TEXT_CHUNK_SIZE = int(os.getenv('TEXT_CHUNK_SIZE', '1000'))
TEXT_CHUNK_OVERLAP = int(os.getenv('TEXT_CHUNK_OVERLAP', '200'))
