#  Telecom RAN Assistant — RAG System

A **Retrieval-Augmented Generation (RAG)** system for answering telecom RAN questions,
built with LangChain, ChromaDB, and the TeleQnA benchmark dataset.

---

## Architecture

```
User Query
    │
    ▼
[Retriever] ──── ChromaDB (TeleQnA + 3GPP PDFs)
    │              MMR Search (top-k=5)
    ▼
[Reranker] ──── Cross-encoder (optional, boosts faithfulness)
    │
    ▼
[Prompt] ──── System prompt with retrieved context
    │
    ▼
[LLM] ──── GPT-4o-mini / Groq / Ollama
    │
    ▼
Answer + Source Citations
```

---

## Quick Start

### 1. Install

```bash
git clone <your-repo>
cd telecom-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your API key (or set USE_OLLAMA=true for free local inference)
```

### 3. Ingest the dataset

```bash
# Downloads TeleQnA automatically, embeds ~10k documents
python -m src.ingest

# Also load 3GPP PDF specs (optional but improves accuracy)
python -m src.ingest --pdf_dir ./data/3gpp_specs
```

### 4. Chat

```bash
python app.py                # interactive CLI
python app.py --demo         # runs 5 pre-set questions
```

### 5. Evaluate

```bash
python -m src.evaluator --n 100            # 100 random questions
python -m src.evaluator --n 200 --category "Standards specifications"
python -m src.evaluator --full             # full 10k (slow, ~2hrs with GPT)
```

---

## LLM Options (in `.env`)

| Provider | Model | Cost | Speed | Quality |
|---|---|---|---|---|
| OpenAI | `gpt-4o-mini` | ~$0.001/q | Fast | ★★★★★ |
| Groq | `llama-3.1-8b-instant` | Free | Very Fast | ★★★★☆ |
| Ollama | `mistral` | Free | Slow (CPU) | ★★★☆☆ |

**Recommendation for hackathon:** Start with **Groq** (free, fast, good quality).
Sign up at https://console.groq.com — free tier is generous.

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=gsk_...
```

---

## Dataset

**TeleQnA** — 10,000 multiple-choice questions across 5 categories:

| Category | Questions | Description |
|---|---|---|
| Lexicon | 500 | Telecom terminology & definitions |
| Research overview | 2,000 | Broad telecom research topics |
| Research publications | 4,500 | Deep technical questions from papers |
| Standards specifications | 2,000 | 3GPP / ORAN spec questions |
| Standards overview | 1,000 | High-level standards knowledge |

### Adding 3GPP Specs (highly recommended)

Download PDFs from https://www.3gpp.org/specifications and place in `./data/3gpp_specs/`:
```
data/3gpp_specs/
  TS_38.300_NR_Overall_Description.pdf
  TS_38.401_NG-RAN_Architecture.pdf
  O-RAN_WG1_Architecture.pdf
  ...
```
Then re-run `python -m src.ingest --pdf_dir ./data/3gpp_specs`.

---

## KPI Targets

| Metric | Target | How measured |
|---|---|---|
| Accuracy | ≥ 80% | MCQ correct on TeleQnA |
| Top-5 Accuracy | ≥ 85% | Correct answer in retrieved docs |
| MRR | ≥ 75% | Mean Reciprocal Rank |
| Faithfulness | ≥ 90% | RAGAS evaluation |

Run `python -m src.evaluator --n 500` to check your scores.

---

## Project Structure

```
telecom-rag/
├── app.py                   # Interactive CLI chatbot
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py            # All settings from .env
│   ├── llm_factory.py       # LLM + embedding provider switcher
│   ├── data_loader.py       # Download & parse TeleQnA + PDFs
│   ├── ingest.py            # Build ChromaDB vector store
│   ├── rag_chain.py         # LangChain RAG chain (main logic)
│   └── evaluator.py         # Evaluate against TeleQnA benchmark
├── data/
│   ├── TeleQnA.json         # Auto-downloaded
│   └── 3gpp_specs/          # Add your PDF specs here
├── vectorstore/
│   └── chroma_db/           # Auto-created by ingest.py
├── notebooks/
│   └── 01_explore_dataset.ipynb
└── tests/
    └── test_rag.py
```

---

## Improving Performance (Iteration Ideas)

1. **Better chunking** — Try `SemanticChunker` instead of `RecursiveCharacterTextSplitter`
2. **HyDE** — Generate a hypothetical answer, embed that instead of the raw question
3. **Query expansion** — Rewrite the query 3 ways, retrieve for all, deduplicate
4. **Reranker** — Set `use_reranker=True` in `TelecomRAG()` (needs sentence-transformers)
5. **Fine-tuned embeddings** — Fine-tune `all-mpnet-base-v2` on TeleQnA pairs
6. **More data** — Add O-RAN WG specs, ATIS standards, or Simu5G docs

---

## Tests

```bash
pytest tests/ -v
```
