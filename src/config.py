from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / 'data'
CHROMA_DIR = ROOT_DIR / '.chromadb'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_API_ENDPOINT = os.getenv('GEMINI_API_ENDPOINT', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
TEXT_CHUNK_SIZE = int(os.getenv('TEXT_CHUNK_SIZE', '1000'))
TEXT_CHUNK_OVERLAP = int(os.getenv('TEXT_CHUNK_OVERLAP', '200'))
