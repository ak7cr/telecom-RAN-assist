"""
data_loader.py
Downloads TeleQnA from GitHub and converts it into LangChain Documents.
Also has a helper to load any local PDF (e.g. 3GPP spec PDFs).
"""
import json
import os
import zipfile
import requests
import pyzipper
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

TELEQNA_ZIP_URL = (
    "https://github.com/netop-team/TeleQnA/raw/main/TeleQnA.zip"
)
DATA_DIR = Path("./data")


# ── Download helpers ─────────────────────────────────────────────────

def download_teleqna(dest_dir: Path = DATA_DIR) -> Path:
    """Download TeleQnA.zip if not already present. Returns path to JSON file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "TeleQnA.zip"
    json_path = dest_dir / "TeleQnA.json"

    if json_path.exists():
        print(f"[data_loader] TeleQnA already downloaded → {json_path}")
        return json_path

    print("[data_loader] Downloading TeleQnA dataset …")
    r = requests.get(TELEQNA_ZIP_URL, timeout=60)
    r.raise_for_status()
    zip_path.write_bytes(r.content)

    with pyzipper.AESZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir, pwd=b"teleqnadataset")

    # The zip extracts as TeleQnA.txt (JSON content) or TeleQnA.json
    candidates = list(dest_dir.rglob("*.json")) + list(dest_dir.rglob("TeleQnA.txt"))
    if not candidates:
        raise FileNotFoundError("No data file found after extracting TeleQnA.zip")

    extracted = candidates[0]
    if extracted != json_path:
        extracted.rename(json_path)

    zip_path.unlink(missing_ok=True)
    print(f"[data_loader] TeleQnA saved → {json_path}")
    return json_path


# ── Parsers ──────────────────────────────────────────────────────────

def teleqna_to_documents(json_path: Path) -> List[Document]:
    """
    Convert TeleQnA JSON into LangChain Documents.

    TeleQnA format per entry:
    {
        "question": "What does ...",
        "option 1": "...",
        "option 2": "...",
        "option 3": "...",
        "option 4": "...",
        "option 5": "...",   # optional
        "answer": "option 1",
        "explanation": "...",
        "category": "Standards specifications"
    }
    We build a document from question + options + explanation.
    This is the knowledge the RAG retrieves — not the Q&A pairs as-is.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for qid, entry in data.items():
        question    = entry.get("question", "")
        explanation = entry.get("explanation", "")
        category    = entry.get("category", "general")
        answer      = entry.get("answer", "")

        # Collect options
        options = []
        for k in ["option 1", "option 2", "option 3", "option 4", "option 5"]:
            if k in entry:
                options.append(f"{k}: {entry[k]}")

        # Build rich text block the retriever will embed
        text = (
            f"Question: {question}\n"
            + "\n".join(options)
            + f"\nAnswer: {answer}\n"
            + (f"Explanation: {explanation}" if explanation else "")
        )

        docs.append(Document(
            page_content=text,
            metadata={
                "source": "TeleQnA",
                "question_id": qid,
                "category": category,
                "answer": answer,
            }
        ))

    print(f"[data_loader] Loaded {len(docs)} TeleQnA documents")
    return docs


def load_pdf_documents(pdf_dir: str) -> List[Document]:
    """
    Load all PDFs from a directory (e.g., 3GPP spec PDFs you downloaded manually).
    Split them into chunks.
    """
    pdf_dir = Path(pdf_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_docs = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        print(f"[data_loader] Loading PDF: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages  = loader.load()
        chunks = splitter.split_documents(pages)
        # Tag source
        for chunk in chunks:
            chunk.metadata["source"] = pdf_file.name
        all_docs.extend(chunks)
    print(f"[data_loader] Loaded {len(all_docs)} chunks from PDFs")
    return all_docs


# ── Main entry ───────────────────────────────────────────────────────

def load_all_documents(pdf_dir: str = None) -> List[Document]:
    """
    Load TeleQnA + optional PDFs and return a combined list of Documents.
    Call this from ingest.py.
    """
    json_path = download_teleqna()
    docs = teleqna_to_documents(json_path)

    if pdf_dir and Path(pdf_dir).exists():
        pdf_docs = load_pdf_documents(pdf_dir)
        docs.extend(pdf_docs)

    return docs
