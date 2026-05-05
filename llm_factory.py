"""
llm_factory.py
Returns the correct LLM and embeddings object based on .env settings.

LLM providers:
  openai       — GPT-4o-mini (default, paid)
  groq         — Llama-3.1-8B-Instant via Groq cloud (free tier, recommended)
  ollama       — any local Ollama model (llama3.2, mistral, …)
  huggingface  — HuggingFace pipeline; use for Tele-LLMs:
                   LLM_MODEL=AliMaatouk/Llama-3-8B-Instruct-Tele
                   (fine-tuned on 2.5B telecom tokens from 3GPP, arXiv, Wikipedia)

Tele-LLMs available on HuggingFace (AliMaatouk collection):
  AliMaatouk/TinyLlama-1.1B-Instruct-Tele   (smallest, fastest)
  AliMaatouk/Phi-1.5-Tele
  AliMaatouk/Gemma-2B-Instruct-Tele
  AliMaatouk/Llama-3-8B-Instruct-Tele       (best accuracy)
  AliMaatouk/Llama-3.2-3B-Instruct-Tele

For Ollama with a Tele-LLM GGUF (if you have a GGUF version pulled):
  LLM_PROVIDER=ollama
  LLM_MODEL=hf.co/AliMaatouk/Llama-3.2-3B-Instruct-Tele-GGUF
"""
from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL,
    USE_OLLAMA, OLLAMA_BASE_URL,
    HF_MAX_NEW_TOKENS, HF_DEVICE_MAP,
)


def get_llm():
    """Return a LangChain-compatible LLM based on LLM_PROVIDER in .env"""
    if USE_OLLAMA or LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    if LLM_PROVIDER == "huggingface":
        return _get_hf_pipeline_llm()

    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def _get_hf_pipeline_llm():
    """
    Load a HuggingFace text-generation model as a LangChain LLM.
    Used for Tele-LLMs (domain-fine-tuned on 2.5B telecom tokens).
    Requires: pip install transformers accelerate
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        from langchain_huggingface import HuggingFacePipeline

        print(f"[llm_factory] Loading HuggingFace model: {LLM_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16 if HF_DEVICE_MAP != "cpu" else torch.float32,
            device_map=HF_DEVICE_MAP,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE if LLM_TEMPERATURE > 0 else None,
            do_sample=LLM_TEMPERATURE > 0,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except ImportError as e:
        raise ImportError(
            "HuggingFace provider requires: pip install transformers accelerate\n"
            f"Original error: {e}"
        )


def get_embeddings():
    """Return a LangChain embeddings object based on EMBEDDING_PROVIDER in .env"""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Default: local sentence-transformers (free, CPU-friendly, no API key)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
