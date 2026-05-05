"""
ingest.py
Builds (or refreshes) the ChromaDB vector store from all configured sources.

Usage:
    python ingest.py                          # TeleQnA only (baseline)
    python ingest.py --all                    # every HuggingFace dataset + TeleQnA
    python ingest.py --telelogs --oran-bench  # pick specific sources
    python ingest.py --tele-eval 3000         # sample 3k rows from Tele-Eval
    python ingest.py --pdf_dir ./data/pdfs    # add local 3GPP PDF specs
    python ingest.py --reset                  # wipe and rebuild from scratch

Sources:
    TeleQnA     — always included (10k 3GPP MCQ)
    --telelogs  — netop/TeleLogs  (5G RCA scenarios)
    --teletables — netop/TeleTables (3GPP spec-table MCQ)
    --tele-eval N — AliMaatouk/Tele-Eval (stream N open-ended Q&A pairs)
    --oran-bench  — GSMA/ot-full oranbench (1,500 O-RAN Q&A)
    --pdf_dir   — local directory of 3GPP PDF files
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
    include_telelogs: bool = False,
    include_teletables: bool = False,
    tele_eval_samples: int = 0,
    include_oran_bench: bool = False,
) -> Chroma:
    """
    Load documents from all requested sources, embed, and persist to ChromaDB.
    """
    persist_dir = Path(CHROMA_PERSIST_DIR)

    if reset and persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)
        print("[ingest] Cleared existing vector store.")

    embeddings = get_embeddings()

    if persist_dir.exists() and not reset:
        print(f"[ingest] Loading existing vector store from {persist_dir}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    # ── Load all sources ─────────────────────────────────────────────
    docs = load_all_documents(
        pdf_dir=pdf_dir,
        include_telelogs=include_telelogs,
        include_teletables=include_teletables,
        tele_eval_samples=tele_eval_samples,
        include_oran_bench=include_oran_bench,
    )

    # HuggingFace Q&A docs and TeleQnA entries are already short;
    # only long PDF chunks need re-splitting.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    short_docs = [d for d in docs if len(d.page_content) <= CHUNK_SIZE * 1.5]
    long_docs  = [d for d in docs if len(d.page_content) >  CHUNK_SIZE * 1.5]
    all_chunks = short_docs + splitter.split_documents(long_docs)

    print(f"[ingest] Total chunks to embed: {len(all_chunks)}")
    print("[ingest] Embedding … (this may take several minutes on first run)")

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
    parser.add_argument("--pdf_dir",     default=None, help="Directory of 3GPP PDF specs")
    parser.add_argument("--reset",       action="store_true", help="Wipe and rebuild from scratch")
    parser.add_argument("--all",         action="store_true", help="Include all HuggingFace datasets")
    parser.add_argument("--telelogs",    action="store_true", help="Add TeleLogs RCA scenarios")
    parser.add_argument("--teletables",  action="store_true", help="Add TeleTables 3GPP table Q&A")
    parser.add_argument("--tele-eval",   type=int, default=0,  dest="tele_eval",
                        metavar="N",     help="Stream N samples from Tele-Eval (default 0 = skip)")
    parser.add_argument("--oran-bench",  action="store_true", dest="oran_bench",
                        help="Add O-RAN Bench (1,500 O-RAN Q&A from GSMA/ot-full)")
    args = parser.parse_args()

    if args.all:
        args.telelogs   = True
        args.teletables = True
        args.tele_eval  = args.tele_eval or 3000
        args.oran_bench = True

    build_vectorstore(
        pdf_dir=args.pdf_dir,
        reset=args.reset,
        include_telelogs=args.telelogs,
        include_teletables=args.teletables,
        tele_eval_samples=args.tele_eval,
        include_oran_bench=args.oran_bench,
    )
