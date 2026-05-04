"""
rag_chain.py
Builds the end-to-end LangChain RAG chain for the Telecom RAN Assistant.

Architecture:
    User Query
        │
        ▼
    [HyDE: generate hypothetical answer → embed it]  (optional)
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
import re
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import CHROMA_PERSIST_DIR, RETRIEVER_K
from llm_factory import get_llm, get_embeddings


# ── Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a specialized Telecom RAN (Radio Access Network) assistant \
with deep knowledge of 3GPP standards, ORAN specifications, and wireless communications.

Use ONLY the provided context to answer. If the answer is not in the context, say:
"I don't have enough information in my knowledge base to answer this confidently."

Always:
- Cite which source (TeleQnA question ID, document name) your answer comes from.
- Use precise telecom terminology.

Context:
{context}
"""

HUMAN_PROMPT = "{question}"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])

# MCQ prompt forces a parseable "option N" prefix so the evaluator can extract answers reliably.
MCQ_SYSTEM_PROMPT = """You are a specialized Telecom RAN (Radio Access Network) assistant \
with deep knowledge of 3GPP standards, ORAN specifications, and wireless communications.

Use ONLY the provided context to answer.

CRITICAL FORMATTING RULE: Your response MUST begin with exactly "option N:" where N is \
the number (1–5) of the correct answer. Then provide a concise explanation citing the source.

Example of a correctly formatted response:
  option 3: <explanation citing the source document>

Context:
{context}
"""

MCQ_HUMAN_PROMPT = """{question}

Select the single best option. Begin your answer with "option N:" (replace N with the correct number)."""

MCQ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MCQ_SYSTEM_PROMPT),
    ("human",  MCQ_HUMAN_PROMPT),
])

# HyDE prompt: generates a hypothetical expert answer used only for retrieval (not shown to user).
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a telecom expert. Generate a concise technical answer (2-3 sentences) "
               "as if writing from a 3GPP specification or research paper. "
               "This answer is used internally for document retrieval only."),
    ("human", "{question}"),
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

        # HyDE: embed a hypothetical answer instead of the raw question for better retrieval
        rag_hyde = TelecomRAG(use_hyde=True)
    """

    def __init__(
        self,
        use_mmr: bool = True,
        use_reranker: bool = False,
        use_hyde: bool = False,
    ):
        self.vectorstore = load_vectorstore()
        self.llm         = get_llm()
        self.embeddings  = get_embeddings()
        self.use_hyde    = use_hyde

        # ── Retriever: MMR reduces redundant results ──────────────────
        if use_mmr:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RETRIEVER_K,
                    "fetch_k": RETRIEVER_K * 4,
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

    def _add_reranker(self):
        """Wrap the retriever with a cross-encoder reranker."""
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            model = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            compressor     = CrossEncoderReranker(model=model, top_n=RETRIEVER_K)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.retriever,
            )
            print("[rag_chain] Cross-encoder reranker enabled")
        except ImportError:
            print("[rag_chain] Reranker skipped (sentence-transformers not installed)")

    def _retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents for a query, using HyDE if enabled.

        HyDE (Hypothetical Document Embeddings): instead of embedding the raw
        question, generate a short hypothetical expert answer and embed that.
        Answers are syntactically closer to the stored documents than questions,
        so retrieval precision improves significantly.
        """
        if not self.use_hyde:
            return self.retriever.invoke(query)

        hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
        hypothetical_answer = hyde_chain.invoke({"question": query})
        return self.vectorstore.similarity_search(hypothetical_answer, k=RETRIEVER_K)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a free-form question and return the answer + source documents.

        Returns:
            {"answer": str, "sources": List[Document], "question": str}
        """
        sources = self._retrieve(question)
        context = format_docs(sources)
        answer  = (
            PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke({"context": context, "question": question})

        return {"question": question, "answer": answer, "sources": sources}

    def ask_mcq(self, question: str, options: List[str]) -> Dict[str, Any]:
        """
        Answer a multiple-choice question (TeleQnA format).

        Uses the MCQ prompt that explicitly instructs the LLM to begin
        its response with 'option N:' for reliable answer parsing.
        """
        options_text = "\n".join(
            [f"  option {i+1}: {opt}" for i, opt in enumerate(options)]
        )
        formatted_q = f"{question}\n\nOptions:\n{options_text}"

        sources = self._retrieve(formatted_q)
        context = format_docs(sources)
        answer  = (
            MCQ_PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke({"context": context, "question": formatted_q})

        return {"question": formatted_q, "answer": answer, "sources": sources}

    def _parse_predicted_answer(self, llm_answer: str, options: List[str]) -> str:
        """Extract an 'option N' label from a multiple-choice LLM answer."""
        answer_lower = llm_answer.lower()
        match = re.search(r"\boption\s*([1-5])\b", answer_lower)
        if match:
            return f"option {match.group(1)}"

        for i, opt in enumerate(options, 1):
            option_prefix = opt.strip().lower()[:30]
            if option_prefix and option_prefix in answer_lower:
                return f"option {i}"

        return "unknown"


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
