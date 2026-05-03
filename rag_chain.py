"""
rag_chain.py
Builds the end-to-end LangChain RAG chain for the Telecom RAN Assistant.

Architecture:
    User Query
        │
        ▼
    Retriever (ChromaDB MMR search, top-k docs)
        │
        ▼
    Reranker (optional cross-encoder, improves faithfulness)
        │
        ▼
    Prompt Template  ←── retrieved context
        │
        ▼
    LLM  (GPT-4o-mini / Groq / Ollama)
        │
        ▼
    Answer + Sources
"""
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import CHROMA_PERSIST_DIR, RETRIEVER_K
from llm_factory import get_llm, get_embeddings


# ── Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a specialized Telecom RAN (Radio Access Network) assistant \
with deep knowledge of 3GPP standards, ORAN specifications, and wireless communications.

Use ONLY the provided context to answer. If the answer is not in the context, say:
"I don't have enough information in my knowledge base to answer this confidently."

Always:
- Cite which source (TeleQnA question ID, document name) your answer comes from.
- Use precise telecom terminology.
- For multi-choice questions, clearly state which option is correct and why.

Context:
{context}
"""

HUMAN_PROMPT = "{question}"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Helpers ──────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        qid    = doc.metadata.get("question_id", "")
        header = f"[{i}] Source: {source}" + (f" | QID: {qid}" if qid else "")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def load_vectorstore(collection_name: str = "telecom_rag") -> Chroma:
    """Load the persisted Chroma vector store."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ── Main chain builder ────────────────────────────────────────────────

class TelecomRAG:
    """
    High-level wrapper around the LangChain RAG chain.

    Usage:
        rag = TelecomRAG()
        result = rag.ask("What is the role of the CU-DU split in ORAN?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(self, use_mmr: bool = True, use_reranker: bool = False):
        self.vectorstore = load_vectorstore()
        self.llm         = get_llm()

        # ── Retriever: MMR reduces redundant results ──────────────────
        if use_mmr:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RETRIEVER_K,
                    "fetch_k": RETRIEVER_K * 4,   # MMR candidate pool
                    "lambda_mult": 0.7,            # 1=similarity, 0=diversity
                },
            )
        else:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": RETRIEVER_K}
            )

        # ── Optional cross-encoder reranker ───────────────────────────
        if use_reranker:
            self._add_reranker()

        # ── Chain ─────────────────────────────────────────────────────
        self._chain = self._build_chain()

    def _add_reranker(self):
        """
        Wrap the retriever with a cross-encoder reranker.
        Requires: pip install sentence-transformers
        """
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            model = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            compressor    = CrossEncoderReranker(model=model, top_n=RETRIEVER_K)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.retriever,
            )
            print("[rag_chain] Cross-encoder reranker enabled")
        except ImportError:
            print("[rag_chain] Reranker skipped (sentence-transformers not installed)")

    def _build_chain(self):
        """Build the LCEL (LangChain Expression Language) RAG chain."""
        return (
            {
                "context":  self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and return the answer + source documents.

        Returns:
            {
                "answer":  str,
                "sources": List[Document],
                "question": str,
            }
        """
        # Retrieve sources separately so we can return them
        sources = self.retriever.invoke(question)
        answer  = self._chain.invoke(question)

        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
        }

    def ask_mcq(self, question: str, options: List[str]) -> Dict[str, Any]:
        """
        Answer a multiple-choice question (like TeleQnA format).
        Formats the options into the query for better retrieval.
        """
        options_text = "\n".join(
            [f"  option {i+1}: {opt}" for i, opt in enumerate(options)]
        )
        formatted_q = (
            f"{question}\n\nOptions:\n{options_text}\n\n"
            "Which option is correct? Explain your reasoning step by step."
        )
        return self.ask(formatted_q)


# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    rag = TelecomRAG()

    test_questions = [
        "What is the CU-DU functional split in O-RAN architecture?",
        "What does handover mean in LTE/5G networks?",
        "Explain the difference between FDD and TDD in 5G NR.",
    ]

    for q in test_questions:
        result = rag.ask(q)
        console.print(Panel(f"[bold cyan]Q:[/bold cyan] {result['question']}"))
        console.print(Panel(f"[bold green]A:[/bold green] {result['answer']}"))
        console.print(f"[dim]Sources ({len(result['sources'])} docs retrieved)[/dim]\n")
