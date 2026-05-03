"""
tests/test_rag.py
Unit tests for data loading and RAG chain components.
Run with: pytest tests/ -v
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Data loader tests ─────────────────────────────────────────────────

def test_teleqna_document_format():
    """Each TeleQnA doc must have required metadata fields."""
    from data_loader import teleqna_to_documents

    # Minimal synthetic data
    fake_data = {
        "q1": {
            "question": "What does 5G stand for?",
            "option 1": "Fifth Generation",
            "option 2": "Fast Generation",
            "answer": "option 1",
            "explanation": "5G stands for Fifth Generation wireless technology.",
            "category": "Lexicon"
        }
    }

    # Write to a temp file
    tmp = Path("/tmp/test_teleqna.json")
    tmp.write_text(json.dumps(fake_data))

    docs = teleqna_to_documents(tmp)
    assert len(docs) == 1
    doc = docs[0]
    assert "5G stand" in doc.page_content
    assert doc.metadata["category"] == "Lexicon"
    assert doc.metadata["answer"] == "option 1"
    assert doc.metadata["source"] == "TeleQnA"


def test_document_content_has_all_fields():
    """Document page_content should include question, options, answer, explanation."""
    from data_loader import teleqna_to_documents

    fake_data = {
        "q1": {
            "question": "What is beamforming?",
            "option 1": "A signal processing technique",
            "option 2": "A type of modulation",
            "answer": "option 1",
            "explanation": "Beamforming focuses a signal in a specific direction.",
            "category": "Research overview"
        }
    }
    tmp = Path("/tmp/test_teleqna2.json")
    tmp.write_text(json.dumps(fake_data))

    docs = teleqna_to_documents(tmp)
    content = docs[0].page_content
    assert "beamforming" in content.lower()
    assert "option 1" in content
    assert "Beamforming focuses" in content


# ── RAG chain tests ───────────────────────────────────────────────────

def test_format_docs():
    """format_docs should number sources and include content."""
    from langchain_core.documents import Document
    from rag_chain import format_docs

    docs = [
        Document(page_content="5G uses mmWave.", metadata={"source": "TeleQnA", "question_id": "q1"}),
        Document(page_content="LTE is 4G.",     metadata={"source": "3gpp_spec.pdf"}),
    ]
    formatted = format_docs(docs)
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "5G uses mmWave" in formatted
    assert "QID: q1" in formatted


def test_parse_predicted_answer():
    """Should correctly parse 'option N' from LLM free-text response."""
    from rag_chain import TelecomRAG

    # Bypass actual init
    with patch.object(TelecomRAG, '__init__', lambda *a, **k: None):
        rag = TelecomRAG.__new__(TelecomRAG)

    rag_method = TelecomRAG._parse_predicted_answer

    options = ["mmWave", "Sub-6GHz", "2.4GHz band"]
    assert rag_method(rag, "The correct answer is option 1 because...", options) == "option 1"
    assert rag_method(rag, "Option 3 is correct.", options) == "option 3"
    assert rag_method(rag, "No match here.", options) == "unknown"


# ── Evaluator tests ───────────────────────────────────────────────────

def test_compute_reciprocal_rank():
    """RR should be 1/rank when correct answer appears in retrieved docs."""
    from langchain_core.documents import Document
    from evaluator import TelecomRAGEvaluator

    with patch.object(TelecomRAGEvaluator, '__init__', lambda *a, **k: None):
        ev = TelecomRAGEvaluator.__new__(TelecomRAGEvaluator)

    docs = [
        Document(page_content="The answer is option 2: OFDM"),
        Document(page_content="Something else"),
        Document(page_content="option 2 appears here too"),
    ]

    rr = ev._compute_reciprocal_rank(docs, "option 2")
    assert rr == 1.0  # found at rank 1

    docs_no_match = [Document(page_content="unrelated content")]
    rr2 = ev._compute_reciprocal_rank(docs_no_match, "option 5")
    assert rr2 == 0.0


def test_evaluator_normalizes_teleqna_answer():
    """TeleQnA answers include option text; evaluator should compare labels."""
    from evaluator import TelecomRAGEvaluator

    with patch.object(TelecomRAGEvaluator, '__init__', lambda *a, **k: None):
        ev = TelecomRAGEvaluator.__new__(TelecomRAGEvaluator)

    assert ev._parse_correct_answer("option 4: the full answer text") == "option 4"
    assert ev._parse_correct_answer("Option 2") == "option 2"
