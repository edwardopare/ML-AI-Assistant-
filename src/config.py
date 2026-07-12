from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / 'data'
CHROMA_DIR = ROOT_DIR / '.chromadb'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2').strip()

# Load Gemini/GenAI configuration and sanitize values (trim whitespace and
# attempt to extract an http(s) URL if the endpoint line contains extra text).
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
_raw_endpoint = os.getenv('GEMINI_API_ENDPOINT', '')
if _raw_endpoint is None:
	GEMINI_API_ENDPOINT = ''
else:
	_raw_endpoint = _raw_endpoint.strip()
	if _raw_endpoint.startswith('http://') or _raw_endpoint.startswith('https://'):
		GEMINI_API_ENDPOINT = _raw_endpoint
	else:
		# Try to find an embedded URL (e.g. lines like "POST https://...")
		import re
		m = re.search(r"https?://\S+", _raw_endpoint)
		GEMINI_API_ENDPOINT = m.group(0) if m else ''

GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro').strip()
TEXT_CHUNK_SIZE = int(os.getenv('TEXT_CHUNK_SIZE', '1000'))
TEXT_CHUNK_OVERLAP = int(os.getenv('TEXT_CHUNK_OVERLAP', '200'))
