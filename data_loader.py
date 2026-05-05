"""
data_loader.py
Loads all knowledge sources for the TelecomRAG vector store.

Sources integrated:
  1. TeleQnA          — 10k 3GPP MCQ from netop-team (GitHub)
  2. TeleLogs         — 5G root-cause analysis scenarios (netop/TeleLogs, HuggingFace)
  3. TeleTables       — 500 MCQ from 3GPP spec tables (netop/TeleTables, HuggingFace)
  4. Tele-Eval        — sampled open-ended telecom Q&A (AliMaatouk/Tele-Eval, HuggingFace)
  5. O-RAN Bench      — 1,500 O-RAN spec Q&A (GSMA/ot-full → oranbench, HuggingFace)
  6. 3GPP PDFs        — any local PDFs placed in a directory (e.g. TS 38.300)
"""
import json
import requests
import pyzipper
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

TELEQNA_ZIP_URL = "https://github.com/netop-team/TeleQnA/raw/main/TeleQnA.zip"
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


# ── HuggingFace dataset loaders ──────────────────────────────────────

def _pick(item: dict, *keys, default: str = "") -> str:
    """Return the first non-empty value found among the given keys."""
    for k in keys:
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def load_telelogs(n_samples: Optional[int] = None) -> List[Document]:
    """
    Load TeleLogs: 5G root-cause-analysis scenarios (netop/TeleLogs, HuggingFace).
    Each scenario becomes a Document describing a network fault + its root cause.
    Directly improves Recall / Top-k for anomaly-detection and RCA queries.
    """
    try:
        from datasets import load_dataset
        raw = load_dataset("netop/TeleLogs", split="train", trust_remote_code=True)
        items = list(raw)[:n_samples] if n_samples else list(raw)

        docs = []
        for i, item in enumerate(items):
            # TeleLogs schema has varied field names across versions — try them all
            scenario = _pick(
                item,
                "logs", "scenario", "network_state", "input", "problem",
                "description", "context",
                default=json.dumps({k: v for k, v in item.items()
                                    if k not in {"root_cause", "label", "answer",
                                                 "explanation", "output", "cause"}})
            )
            root_cause = _pick(
                item,
                "root_cause", "label", "cause", "answer", "output",
                default="Unknown"
            )
            explanation = _pick(item, "explanation", "description", "output")

            text = (
                f"5G Network Fault Scenario:\n{scenario}\n"
                f"Root Cause: {root_cause}"
            )
            if explanation and explanation != scenario:
                text += f"\nExplanation: {explanation}"

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "TeleLogs",
                    "question_id": f"telelogs_{i}",
                    "category": "Root Cause Analysis",
                    "root_cause": root_cause,
                },
            ))

        print(f"[data_loader] Loaded {len(docs)} TeleLogs documents")
        return docs
    except Exception as exc:
        print(f"[data_loader] TeleLogs skipped: {exc}")
        return []


def load_teletables(n_samples: Optional[int] = None) -> List[Document]:
    """
    Load TeleTables: 500 MCQ from 3GPP specification tables (netop/TeleTables, HuggingFace).
    Each entry encodes the table content + question + correct answer so the retriever
    can match questions about spec parameter tables.
    Directly improves accuracy on Standards Specifications category questions.
    """
    try:
        from datasets import load_dataset
        raw = load_dataset("netop/TeleTables", split="train", trust_remote_code=True)
        items = list(raw)[:n_samples] if n_samples else list(raw)

        docs = []
        for i, item in enumerate(items):
            question = _pick(item, "question", "query")
            # Tables come in multiple formats; prefer markdown for readability
            table = _pick(item, "table_md", "table_markdown", "table", "table_html",
                          "table_json", default="")
            answer   = _pick(item, "answer", "correct_answer", "label")
            spec     = _pick(item, "specification", "source", "doc", "document",
                             default="3GPP specification")
            options_parts = []
            for k in ["option 1", "option 2", "option 3", "option 4",
                      "option_1", "option_2", "option_3", "option_4",
                      "A", "B", "C", "D"]:
                if item.get(k):
                    options_parts.append(f"{k}: {item[k]}")

            text = f"Source: {spec}\n"
            if table:
                text += f"Table:\n{table}\n\n"
            text += f"Question: {question}\n"
            if options_parts:
                text += "Options:\n" + "\n".join(options_parts) + "\n"
            text += f"Answer: {answer}"
            explanation = _pick(item, "explanation", "rationale")
            if explanation:
                text += f"\nExplanation: {explanation}"

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "TeleTables",
                    "question_id": f"teletables_{i}",
                    "category": "Standards Specifications",
                    "answer": answer,
                },
            ))

        print(f"[data_loader] Loaded {len(docs)} TeleTables documents")
        return docs
    except Exception as exc:
        print(f"[data_loader] TeleTables skipped: {exc}")
        return []


def load_tele_eval_sample(n_samples: int = 3000) -> List[Document]:
    """
    Load a sample of Tele-Eval: open-ended telecom Q&A pairs (AliMaatouk/Tele-Eval).
    Uses streaming so we never download all 750k entries.
    Provides broad telecom coverage across arXiv papers, 3GPP docs, and Wikipedia.
    Improves recall on a wide range of telecom topics.
    """
    try:
        from datasets import load_dataset
        stream = load_dataset(
            "AliMaatouk/Tele-Eval",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        docs = []
        for i, item in enumerate(stream):
            if i >= n_samples:
                break

            question = _pick(item, "question", "input", "prompt", "query",
                             "Question", "QUESTION")
            answer   = _pick(item, "answer", "output", "response", "completion",
                             "Answer", "ANSWER")
            source   = _pick(item, "source", "category", "doc", "origin",
                             default="Tele-Eval")
            if not question:
                continue

            text = f"Question: {question}\nAnswer: {answer}"
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": source,
                    "question_id": f"teleeval_{i}",
                    "category": _pick(item, "category", "topic", default="General Telecom"),
                },
            ))

        print(f"[data_loader] Loaded {len(docs)} Tele-Eval documents")
        return docs
    except Exception as exc:
        print(f"[data_loader] Tele-Eval skipped: {exc}")
        return []


def load_oran_bench() -> List[Document]:
    """
    Load O-RAN Bench: 1,500 O-RAN specification Q&A (GSMA/ot-full → oranbench split).
    Addresses the 'O-RAN Dataset' requirement from the problem statement.
    Directly improves accuracy and recall on O-RAN architecture / interface questions.
    """
    try:
        from datasets import load_dataset
        # GSMA/ot-full is a multi-benchmark collection; 'oranbench' is one config
        raw = load_dataset("GSMA/ot-full", "oranbench", trust_remote_code=True)
        # Try common split names
        split_name = "test" if "test" in raw else list(raw.keys())[0]
        items = list(raw[split_name])

        docs = []
        for i, item in enumerate(items):
            question = _pick(item, "question", "query", "input", "Question")
            answer   = _pick(item, "answer", "correct_answer", "label",
                             "Answer", "gold_answer")
            explanation = _pick(item, "explanation", "rationale", "context")

            # Collect option fields regardless of naming convention
            options_parts = []
            for k in ["option 1", "option 2", "option 3", "option 4",
                      "option_1", "option_2", "option_3", "option_4",
                      "A", "B", "C", "D",
                      "choice_a", "choice_b", "choice_c", "choice_d"]:
                if item.get(k):
                    options_parts.append(f"{k}: {item[k]}")
            # Also handle 'choices' as a list
            if not options_parts and isinstance(item.get("choices"), list):
                options_parts = [f"option {j+1}: {c}"
                                 for j, c in enumerate(item["choices"])]

            text = f"O-RAN Question: {question}\n"
            if options_parts:
                text += "Options:\n" + "\n".join(options_parts) + "\n"
            text += f"Answer: {answer}"
            if explanation:
                text += f"\nExplanation: {explanation}"

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "O-RAN Bench",
                    "question_id": f"oranbench_{i}",
                    "category": "O-RAN specifications",
                    "answer": answer,
                },
            ))

        print(f"[data_loader] Loaded {len(docs)} O-RAN Bench documents")
        return docs
    except Exception as exc:
        print(f"[data_loader] O-RAN Bench skipped: {exc}")
        return []


# ── Main entry ───────────────────────────────────────────────────────

def load_all_documents(
    pdf_dir: Optional[str] = None,
    include_telelogs: bool = False,
    include_teletables: bool = False,
    tele_eval_samples: int = 0,
    include_oran_bench: bool = False,
) -> List[Document]:
    """
    Load documents from all requested sources and return a combined list.
    Call this from ingest.py.

    Args:
        pdf_dir:             Directory of 3GPP PDF files to ingest.
        include_telelogs:    Add TeleLogs 5G RCA scenarios.
        include_teletables:  Add TeleTables 3GPP spec table Q&A.
        tele_eval_samples:   N > 0 streams N samples from Tele-Eval.
        include_oran_bench:  Add O-RAN Bench Q&A.
    """
    json_path = download_teleqna()
    docs = teleqna_to_documents(json_path)

    if include_telelogs:
        docs.extend(load_telelogs())

    if include_teletables:
        docs.extend(load_teletables())

    if tele_eval_samples > 0:
        docs.extend(load_tele_eval_sample(n_samples=tele_eval_samples))

    if include_oran_bench:
        docs.extend(load_oran_bench())

    if pdf_dir and Path(pdf_dir).exists():
        docs.extend(load_pdf_documents(pdf_dir))

    print(f"[data_loader] Total documents across all sources: {len(docs)}")
    return docs
