#  Telecom RAN Assistant — RAG System

A **Retrieval-Augmented Generation (RAG)** system for telecom RAN Q&A, anomaly detection, and root-cause analysis. Built with LangChain, ChromaDB, and multiple telecom-specific datasets.

---

## Architecture

```
User Query
    │
    ▼
[HyDE]  ←── generate hypothetical answer, embed it instead of raw query (optional)
    │
    ▼
[Retriever] ──── ChromaDB MMR search (top-k=5, fetch_k=20)
    │              Sources: TeleQnA · TeleLogs · TeleTables · Tele-Eval · O-RAN Bench · 3GPP PDFs
    ▼
[Cross-encoder Reranker] ──── re-scores by joint query-doc relevance (optional)
    │
    ▼
[Prompt] ──── MCQ prompt forces "option N:" prefix for reliable parsing
    │
    ▼
[LLM] ──── OpenAI / Groq / Ollama / Tele-LLMs (HuggingFace)
    │
    ▼
Answer + Source Citations
```

---

## Quick Start

### 1. Install

```bash
git clone <your-repo>
cd telecom-ran-assist
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp rename_this.env .env
# Edit .env — set your LLM provider and API key (see options below)
```

### 3. Build the vector store

```bash
# TeleQnA only (10k Q&As, baseline)
python ingest.py

# Full knowledge base — all HuggingFace datasets (recommended)
python ingest.py --all --reset

# Pick specific sources
python ingest.py --oran-bench --teletables --tele-eval 3000 --reset

# Add local 3GPP PDF specs on top of any of the above
python ingest.py --all --pdf_dir ./data/3gpp_specs --reset
```

### 4. Run the web UI

```bash
streamlit run streamlit_app.py
# Opens in browser at http://localhost:8501
```

### 5. Run the CLI chatbot

```bash
python app.py          # interactive chat
python app.py --demo   # 5 pre-set demo questions
```

### 6. Evaluate against TeleQnA benchmark

```bash
# Baseline (100 questions, no extras)
python evaluator.py --n 100

# With reranker + HyDE + faithfulness score
python evaluator.py --n 200 --reranker --hyde --faithfulness

# Filter to one category
python evaluator.py --n 200 --category "Standards specifications"

# Full 10k run (slow — ~2 hrs with GPT, ~30 min with Groq)
python evaluator.py --full --reranker --hyde
```

### 7. Run tests

```bash
pytest test_rag.py -v
```

---

## LLM Options (set in `.env`)

| Provider | `.env` settings | Cost | Speed | Domain quality |
|---|---|---|---|---|
| **Groq** ← recommended | `LLM_PROVIDER=groq`<br>`LLM_MODEL=llama-3.1-8b-instant`<br>`GROQ_API_KEY=gsk_...` | Free tier | Very fast | ★★★★☆ |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`LLM_MODEL=gpt-4o-mini`<br>`OPENAI_API_KEY=sk-...` | ~$0.001/q | Fast | ★★★★★ |
| **Tele-LLM** (best for telecom) | `LLM_PROVIDER=huggingface`<br>`LLM_MODEL=AliMaatouk/Llama-3-8B-Instruct-Tele`<br>`HF_DEVICE_MAP=auto` | Free | Slow (local GPU) | ★★★★★ |
| **Ollama** (local) | `LLM_PROVIDER=ollama`<br>`LLM_MODEL=llama3.2:latest`<br>`USE_OLLAMA=true` | Free | Slow (CPU) | ★★★☆☆ |

**Tele-LLMs** are fine-tuned on 2.5B telecom tokens (3GPP standards + arXiv papers + Wikipedia). Use the smallest that fits your hardware:

| Model | Size | GPU RAM |
|---|---|---|
| `AliMaatouk/TinyLlama-1.1B-Instruct-Tele` | 1.1B | ~3 GB |
| `AliMaatouk/Llama-3.2-3B-Instruct-Tele` | 3B | ~7 GB |
| `AliMaatouk/Llama-3-8B-Instruct-Tele` | 8B | ~16 GB |

---

## Knowledge Base Sources

| Source | Documents | Dataset | What it covers |
|---|---|---|---|
| **TeleQnA** | 10,000 | `netop/TeleQnA` (GitHub) | 3GPP MCQ — 5 categories |
| **TeleLogs** | ~1k | `netop/TeleLogs` (HuggingFace) | 5G RCA scenarios, 8 fault types |
| **TeleTables** | 500 | `netop/TeleTables` (HuggingFace) | MCQ from 13 3GPP spec tables |
| **Tele-Eval** | 750k (stream) | `AliMaatouk/Tele-Eval` (HuggingFace) | Open-ended Q&A from 3GPP + arXiv |
| **O-RAN Bench** | 1,500 | `GSMA/ot-full` → oranbench | O-RAN spec Q&A |
| **3GPP PDFs** | user-supplied | `--pdf_dir` | Raw spec documents |

### Ingest command reference

```bash
python ingest.py                        # TeleQnA only
python ingest.py --telelogs             # + 5G RCA scenarios
python ingest.py --teletables           # + 3GPP table Q&A
python ingest.py --tele-eval 3000       # + 3k sampled Tele-Eval pairs
python ingest.py --oran-bench           # + 1,500 O-RAN Q&A
python ingest.py --pdf_dir ./data/pdfs  # + local PDF files
python ingest.py --all                  # everything above (3k Tele-Eval)
python ingest.py --all --reset          # wipe and rebuild from scratch
```

### Adding 3GPP PDF specs manually

Download from [3gpp.org/specifications](https://www.3gpp.org/specifications) and place in `./data/3gpp_specs/`:
```
data/3gpp_specs/
  TS_38.300_NR_Overall_Description.pdf
  TS_38.401_NG-RAN_Architecture.pdf
  TS_38.213_NR_PHY_Layer.pdf
  O-RAN_WG1_Architecture.pdf
  ...
```
Then run `python ingest.py --pdf_dir ./data/3gpp_specs --reset`.

---

## KPI Targets

| Metric | Target | How measured |
|---|---|---|
| Accuracy | ≥ 80% | MCQ correct answer rate on TeleQnA |
| Top-5 Accuracy | ≥ 85% | Correct answer in top-5 retrieved docs |
| Recall | ≥ 85% | Ground-truth text present in any retrieved doc |
| MRR | ≥ 75% | Mean Reciprocal Rank |
| Faithfulness | ≥ 90% | RAGAS — answer grounded in retrieved context |

```bash
# Check all KPIs at once
python evaluator.py --n 200 --reranker --hyde --faithfulness
```

---

## Evaluator flags

```bash
python evaluator.py [options]

  --n N              Number of questions to evaluate (default: 100)
  --full             Evaluate all 10,000 questions
  --category NAME    Filter to one TeleQnA category:
                       "Lexicon"
                       "Research overview"
                       "Research publications"
                       "Standards overview"
                       "Standards specifications"
  --reranker         Enable cross-encoder reranker during eval
  --hyde             Enable HyDE retrieval during eval
  --faithfulness     Compute RAGAS faithfulness score (extra LLM calls)
```

---

## Project Structure

```
telecom-ran-assist/
├── app.py              # Interactive CLI chatbot
├── streamlit_app.py    # Web UI (streamlit run streamlit_app.py)
├── config.py           # All settings loaded from .env
├── llm_factory.py      # LLM + embedding provider switcher
├── data_loader.py      # Loaders for all datasets (TeleQnA, TeleLogs, etc.)
├── ingest.py           # Build / update ChromaDB vector store
├── rag_chain.py        # RAG chain: retriever → reranker → prompt → LLM
├── evaluator.py        # Benchmark evaluation (all KPIs)
├── test_rag.py         # Unit tests (pytest)
├── requirements.txt
├── rename_this.env     # .env template — copy to .env
├── data/
│   ├── TeleQnA.json    # Auto-downloaded on first ingest
│   ├── eval_results.json
│   └── 3gpp_specs/     # Drop 3GPP PDFs here (optional)
├── vectorstore/
│   └── chroma_db/      # Auto-created by ingest.py (~79 MB baseline)
└── notebooks/
    └── 01_explore_dataset.ipynb
```

---

## Retrieval options

| Feature | Default | How to enable |
|---|---|---|
| MMR search | ✅ on | `TelecomRAG(use_mmr=True)` |
| HyDE retrieval | off | `TelecomRAG(use_hyde=True)` or `--hyde` flag |
| Cross-encoder reranker | off | `TelecomRAG(use_reranker=True)` or `--reranker` flag |

**HyDE** — generates a hypothetical expert answer before retrieval; embeds the answer instead of the question. Answers are syntactically closer to stored documents → better precision.

**Reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the MMR candidates by joint query-document relevance → better top-1 accuracy.
