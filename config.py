"""
config.py
Loads all settings from .env and exposes them as typed constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────────
USE_OLLAMA         = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "openai")   # openai | groq | ollama
LLM_MODEL          = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE    = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── Embeddings ──────────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL    = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
)

# ── Vector store ────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")

# ── RAG ─────────────────────────────────────────────────────────────
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "64"))
RETRIEVER_K        = int(os.getenv("RETRIEVER_K", "5"))
