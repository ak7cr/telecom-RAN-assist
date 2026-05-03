"""
llm_factory.py
Returns the correct LLM and embeddings object based on .env settings.
Supports OpenAI, Groq (free), and local Ollama — swap without touching other code.
"""
from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL,
    USE_OLLAMA, OLLAMA_BASE_URL,
)


def get_llm():
    """Return a LangChain LLM based on LLM_PROVIDER in .env"""
    if USE_OLLAMA or LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(base_url=OLLAMA_BASE_URL, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq  # pip install langchain-groq
        return ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def get_embeddings():
    """Return a LangChain embeddings object based on EMBEDDING_PROVIDER in .env"""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Default: local sentence-transformers (free, CPU-friendly)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
