"""
Centralized configuration for the AI Engineering Knowledge Base.
All tuneable parameters live here so every other module stays clean.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
PROCESSED_DOCS_DIR = BASE_DIR / "data" / "processed_docs"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens / characters per chunk
CHUNK_OVERLAP = 64        # overlap between consecutive chunks

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # SentenceTransformers model
EMBEDDING_DIMENSION = 384                     # output dim for the model above

# ── Vector DB ────────────────────────────────────────────────────────────────
VECTOR_DB_BACKEND = "chroma"  # "chroma" or "faiss"
CHROMA_COLLECTION_NAME = "engineering_kb"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 5  # number of chunks to retrieve per query

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
