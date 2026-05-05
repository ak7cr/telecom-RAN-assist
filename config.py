"""
config.py
Loads all settings from .env and exposes them as typed constants.

LLM_PROVIDER options:
  openai      — ChatOpenAI (GPT-4o-mini default)
  groq        — ChatGroq   (free tier, fast; recommended for eval)
  ollama      — OllamaLLM  (local)
  huggingface — HuggingFacePipeline (Tele-LLMs and other HF models)

EMBEDDING_PROVIDER options:
  local  — sentence-transformers (CPU, free)
  openai — OpenAIEmbeddings
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────────
USE_OLLAMA         = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "openai")   # openai | groq | ollama | huggingface
LLM_MODEL          = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE    = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── HuggingFace pipeline (Tele-LLMs) ────────────────────────────────
# Set LLM_PROVIDER=huggingface and LLM_MODEL=AliMaatouk/Llama-3-8B-Instruct-Tele
HF_MAX_NEW_TOKENS  = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
HF_DEVICE_MAP      = os.getenv("HF_DEVICE_MAP", "auto")   # auto | cpu | cuda:0

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
