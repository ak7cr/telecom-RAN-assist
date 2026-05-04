"""
streamlit_app.py
Web UI for the Telecom RAN Assistant.

Run:
    streamlit run streamlit_app.py
"""
import streamlit as st
from rag_chain import TelecomRAG

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom RAN Assistant",
    page_icon="📡",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    use_hyde = st.toggle(
        "HyDE retrieval",
        value=False,
        help="Hypothetical Document Embeddings: generates a hypothetical answer "
             "and uses it for retrieval. Improves accuracy, adds ~1 extra LLM call.",
    )
    use_reranker = st.toggle(
        "Cross-encoder reranker",
        value=False,
        help="Re-ranks retrieved docs by relevance using a cross-encoder model. "
             "Improves precision at the cost of slightly higher latency.",
    )
    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()
    st.caption(
        "**Retrieval settings**  \n"
        f"Top-k: 5 docs  \n"
        "Embeddings: sentence-transformers/all-mpnet-base-v2  \n"
        "Vector store: ChromaDB (TeleQnA — 10k Q&As)"
    )

    st.divider()
    st.caption(
        "**KPI targets**  \n"
        "Accuracy ≥ 80% | Top-5 ≥ 85%  \n"
        "Recall ≥ 85% | MRR ≥ 75%  \n"
        "Faithfulness ≥ 90%"
    )


# ── RAG model (cached so it isn't rebuilt on every interaction) ──────
@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_rag(hyde: bool, reranker: bool) -> TelecomRAG:
    return TelecomRAG(use_mmr=True, use_reranker=reranker, use_hyde=hyde)


# ── Main UI ──────────────────────────────────────────────────────────
st.title("📡 Telecom RAN Assistant")
st.caption(
    "RAG-powered Q&A over TeleQnA (10k 3GPP / ORAN questions). "
    "Supports free-form and multiple-choice questions."
)

tab_chat, tab_mcq, tab_about = st.tabs(["💬 Chat", "📋 MCQ Mode", "ℹ️ About"])

# ── Chat tab ─────────────────────────────────────────────────────────
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} sources retrieved"):
                    for i, src in enumerate(msg["sources"], 1):
                        meta = src.metadata
                        qid  = meta.get("question_id", "")
                        cat  = meta.get("category", "")
                        label = f"[{i}] {meta.get('source', 'unknown')}"
                        if qid:
                            label += f" · QID {qid}"
                        if cat:
                            label += f" · {cat}"
                        st.markdown(f"**{label}**")
                        st.text(src.page_content[:400])
                        st.divider()

    if prompt := st.chat_input("Ask a telecom / 3GPP / ORAN question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                rag    = load_rag(use_hyde, use_reranker)
                result = rag.ask(prompt)

            st.markdown(result["answer"])

            sources = result.get("sources", [])
            if show_sources and sources:
                with st.expander(f"📄 {len(sources)} sources retrieved"):
                    for i, src in enumerate(sources, 1):
                        meta  = src.metadata
                        qid   = meta.get("question_id", "")
                        cat   = meta.get("category", "")
                        label = f"[{i}] {meta.get('source', 'unknown')}"
                        if qid:
                            label += f" · QID {qid}"
                        if cat:
                            label += f" · {cat}"
                        st.markdown(f"**{label}**")
                        st.text(src.page_content[:400])
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })

    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ── MCQ tab ──────────────────────────────────────────────────────────
with tab_mcq:
    st.subheader("Multiple-Choice Question")
    st.caption(
        "Mirrors the TeleQnA evaluation format. "
        "The model must select one of the options and begin its answer with 'option N:'."
    )

    mcq_question = st.text_area(
        "Question",
        placeholder="e.g. What is the functional split between CU and DU in O-RAN?",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        opt1 = st.text_input("Option 1", placeholder="First option")
        opt2 = st.text_input("Option 2", placeholder="Second option")
        opt3 = st.text_input("Option 3", placeholder="Third option")
    with col2:
        opt4 = st.text_input("Option 4", placeholder="Fourth option")
        opt5 = st.text_input("Option 5 (optional)", placeholder="Fifth option")

    correct_option = st.selectbox(
        "Ground-truth answer (optional — for accuracy check)",
        ["(none)", "option 1", "option 2", "option 3", "option 4", "option 5"],
    )

    if st.button("Submit MCQ", type="primary"):
        options = [o for o in [opt1, opt2, opt3, opt4, opt5] if o.strip()]
        if not mcq_question.strip() or len(options) < 2:
            st.error("Please enter a question and at least 2 options.")
        else:
            with st.spinner("Thinking…"):
                rag    = load_rag(use_hyde, use_reranker)
                result = rag.ask_mcq(mcq_question, options)

            answer = result["answer"]
            st.markdown("### Answer")
            st.markdown(answer)

            # Check if correct
            import re
            match = re.search(r"\boption\s*([1-5])\b", answer.lower())
            predicted = f"option {match.group(1)}" if match else "unknown"

            if correct_option != "(none)":
                if predicted == correct_option:
                    st.success(f"Correct! Predicted: **{predicted}**")
                else:
                    st.error(
                        f"Incorrect. Predicted: **{predicted}** | "
                        f"Ground truth: **{correct_option}**"
                    )
            else:
                st.info(f"Model selected: **{predicted}**")

            if show_sources:
                with st.expander(f"📄 {len(result['sources'])} sources retrieved"):
                    for i, src in enumerate(result["sources"], 1):
                        meta  = src.metadata
                        qid   = meta.get("question_id", "")
                        label = f"[{i}] {meta.get('source', 'unknown')}"
                        if qid:
                            label += f" · QID {qid}"
                        st.markdown(f"**{label}**")
                        st.text(src.page_content[:400])
                        st.divider()


# ── About tab ─────────────────────────────────────────────────────────
with tab_about:
    st.subheader("System Architecture")
    st.markdown("""
```
User Query
    │
    ▼
[HyDE] Generate hypothetical answer (optional)
    │
    ▼
ChromaDB MMR Retrieval  ──  sentence-transformers/all-mpnet-base-v2
    │
    ▼
[Cross-encoder Reranker] (optional)
    │
    ▼
Prompt Template  ←── Retrieved context (top-5 docs)
    │
    ▼
LLM  (OpenAI / Groq / Ollama)
    │
    ▼
Answer + Source citations
```
""")

    st.subheader("Knowledge Base")
    st.markdown("""
| Dataset | Documents | Description |
|---------|-----------|-------------|
| TeleQnA | 10,000 Q&As | 3GPP standard MCQ from 5 categories |
| 3GPP PDFs | _(add via `ingest.py --pdf_dir`)_ | Raw specification documents |
""")

    st.subheader("Key Techniques")
    st.markdown("""
- **MMR retrieval** — balances similarity and diversity (λ=0.7) to avoid redundant context
- **HyDE** — embeds a hypothetical answer instead of the raw question; answers are syntactically closer to documents, boosting retrieval precision
- **Cross-encoder reranker** — re-scores candidate docs by joint query-document relevance
- **Explicit MCQ prompt** — instructs the LLM to start with `option N:`, making answer extraction reliable
""")
