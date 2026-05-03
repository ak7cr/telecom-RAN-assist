"""
ingest.py
Builds (or refreshes) the ChromaDB vector store from TeleQnA + optional PDFs.

Usage:
    python -m src.ingest
    python -m src.ingest --pdf_dir ./data/3gpp_specs
    python -m src.ingest --reset       # wipe and rebuild from scratch
"""
import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from data_loader import load_all_documents
from llm_factory import get_embeddings


def build_vectorstore(
    pdf_dir: str = None,
    reset: bool = False,
    collection_name: str = "telecom_rag",
) -> Chroma:
    """
    Load documents, embed them, and persist to ChromaDB.
    Returns the ready-to-query Chroma vectorstore.
    """
    persist_dir = Path(CHROMA_PERSIST_DIR)

    if reset and persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)
        print("[ingest] Cleared existing vector store.")

    embeddings = get_embeddings()

    # If store already exists and we're not resetting, just load it
    if persist_dir.exists() and not reset:
        print(f"[ingest] Loading existing vector store from {persist_dir}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    # ── Load & split ─────────────────────────────────────────────────
    docs = load_all_documents(pdf_dir=pdf_dir)

    # TeleQnA docs are already short Q&A blocks; PDFs need splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Only split long docs (PDFs). TeleQnA entries are fine as-is.
    short_docs = [d for d in docs if len(d.page_content) <= CHUNK_SIZE * 1.5]
    long_docs  = [d for d in docs if len(d.page_content) >  CHUNK_SIZE * 1.5]
    split_long = splitter.split_documents(long_docs)
    all_chunks = short_docs + split_long

    print(f"[ingest] Total chunks to embed: {len(all_chunks)}")
    print("[ingest] Embedding … (this may take a few minutes on first run)")

    # ── Embed in batches to avoid OOM ────────────────────────────────
    BATCH = 500
    vectorstore = None
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(persist_dir),
            )
        else:
            vectorstore.add_documents(batch)
        print(f"[ingest]   embedded {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}")

    print(f"[ingest] ✓ Vector store saved to {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TelecomRAG vector store")
    parser.add_argument("--pdf_dir", default=None, help="Dir with 3GPP PDF specs")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild")
    args = parser.parse_args()

    build_vectorstore(pdf_dir=args.pdf_dir, reset=args.reset)
