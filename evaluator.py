"""
evaluator.py
Evaluates the RAG system against the TeleQnA benchmark.

Metrics (aligned with project KPIs):
    - Accuracy          : correct answer rate on MCQ questions
    - Top-k Accuracy    : correct answer appears in top-k retrieved docs
    - Recall            : retrieved docs contain ground-truth answer text
    - MRR               : Mean Reciprocal Rank of correct answer in sources
    - Faithfulness      : via RAGAS (answer grounded in context) [--faithfulness flag]

Usage:
    python evaluator.py --n 100
    python evaluator.py --n 200 --category "Standards specifications"
    python evaluator.py --full
    python evaluator.py --n 50 --reranker --hyde --faithfulness
"""
import json
import argparse
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from data_loader import download_teleqna
from rag_chain import TelecomRAG
from config import LLM_PROVIDER, LLM_MODEL, OLLAMA_BASE_URL, USE_OLLAMA

console = Console()


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class EvalResult:
    question_id: str
    question: str
    options: List[str]
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    reciprocal_rank: float         # 1/rank if found in sources, 0 otherwise
    recall_hit: bool = False       # True if ground-truth answer text in any retrieved doc
    retrieved_sources: List[str] = field(default_factory=list)
    generated_answer: str = ""     # raw LLM answer (for RAGAS faithfulness)
    contexts: List[str] = field(default_factory=list)  # raw page_content for RAGAS
    category: str = ""


# ── Core evaluation ───────────────────────────────────────────────────

class TelecomRAGEvaluator:
    def __init__(
        self,
        n_samples: int = 100,
        category: Optional[str] = None,
        use_reranker: bool = False,
        use_hyde: bool = False,
    ):
        self.rag      = TelecomRAG(use_mmr=True, use_reranker=use_reranker, use_hyde=use_hyde)
        self.n        = n_samples
        self.category = category
        self.data     = self._load_data()

    def _load_data(self) -> Dict:
        json_path = download_teleqna()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Filter by category if specified
        if self.category:
            data = {
                k: v for k, v in data.items()
                if v.get("category", "").lower() == self.category.lower()
            }
            console.print(f"[yellow]Filtered to category '{self.category}': {len(data)} questions[/yellow]")

        # Sample
        keys = list(data.keys())
        random.shuffle(keys)
        sampled_keys = keys[:self.n]
        return {k: data[k] for k in sampled_keys}

    def _extract_options(self, entry: Dict) -> List[str]:
        options = []
        for k in ["option 1", "option 2", "option 3", "option 4", "option 5"]:
            if k in entry:
                options.append(entry[k])
        return options

    def _parse_predicted_answer(self, llm_answer: str, options: List[str]) -> str:
        """
        Extract which option the LLM chose from its free-text response.
        Priority:
          1. 'Final answer: option N'   (the explicit format we asked for)
          2. The LAST 'option N' mention (the model may discuss others while reasoning)
          3. Match against option text
        """
        answer_lower = llm_answer.lower()

        final = re.search(r"final\s*answer\s*[:\-]?\s*option\s*([1-5])", answer_lower)
        if final:
            return f"option {final.group(1)}"

        all_mentions = re.findall(r"\boption\s*([1-5])\b", answer_lower)
        if all_mentions:
            return f"option {all_mentions[-1]}"

        # Try matching option text
        for i, opt in enumerate(options, 1):
            option_prefix = opt.strip().lower()[:30]
            if option_prefix and option_prefix in answer_lower:
                return f"option {i}"
        return "unknown"

    def _parse_correct_answer(self, answer: str) -> str:
        """Normalize TeleQnA answers like 'option 4: text' to 'option 4'."""
        match = re.search(r"\boption\s*([1-5])\b", answer.lower())
        if match:
            return f"option {match.group(1)}"
        return answer.strip().lower()

    def _answer_text(self, entry: Dict, options: List[str]) -> str:
        """
        Extract the *text* of the correct option (e.g. the actual content, not 'option 4').
        Used to check whether retrieved PDF chunks contain the answer's substance.
        """
        raw = entry.get("answer", "")
        match = re.search(r"\boption\s*([1-5])\b", raw.lower())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(options):
                return options[idx]
        # Fallback: text after the colon, e.g. "option 4: <text>"
        if ":" in raw:
            return raw.split(":", 1)[1].strip()
        return raw

    def _significant_terms(self, text: str) -> List[str]:
        """Tokenise to lowercase content terms (>=4 chars), drop stopwords."""
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "are",
            "was", "were", "have", "has", "been", "will", "would", "shall",
            "can", "may", "such", "any", "all", "not", "but", "which", "when",
            "what", "where", "option",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", text.lower())
        return [w for w in words if w not in stop]

    def _compute_reciprocal_rank(self, sources: List, answer_text: str) -> float:
        """1/rank of the first retrieved doc that contains the answer's key terms."""
        terms = self._significant_terms(answer_text)
        if not terms:
            return 0.0
        # Need at least half of the answer's content terms (or 3, whichever smaller) to count.
        threshold = max(1, min(3, len(terms) // 2))
        for rank, doc in enumerate(sources, 1):
            page = doc.page_content.lower()
            hits = sum(1 for t in terms if t in page)
            if hits >= threshold:
                return 1.0 / rank
        return 0.0

    def _compute_recall_hit(self, sources: List, answer_text: str) -> bool:
        """True if ANY retrieved source overlaps with the answer's key terms."""
        terms = self._significant_terms(answer_text)
        if not terms:
            return False
        threshold = max(1, min(3, len(terms) // 2))
        for doc in sources:
            page = doc.page_content.lower()
            if sum(1 for t in terms if t in page) >= threshold:
                return True
        return False

    def _provider_error_message(self, error: Exception) -> Optional[str]:
        """Return a setup hint for LLM provider errors that should stop evaluation."""
        message = str(error).lower()
        using_ollama = USE_OLLAMA or LLM_PROVIDER == "ollama"

        if using_ollama and (
            "connection refused" in message
            or "failed to connect" in message
            or "[errno 61]" in message
            or "server disconnected" in message
        ):
            return (
                "Could not connect to Ollama while evaluating.\n\n"
                f"Current config: LLM_PROVIDER=ollama, LLM_MODEL={LLM_MODEL}, "
                f"OLLAMA_BASE_URL={OLLAMA_BASE_URL}\n\n"
                "Start Ollama and make sure the model is available:\n"
                "  ollama serve\n"
                f"  ollama pull {LLM_MODEL}\n\n"
                "Or switch .env to another provider such as Groq/OpenAI and set the "
                "matching API key."
            )

        return None

    def run(self) -> List[EvalResult]:
        results = []

        console.print(f"\n[bold]Running evaluation on {len(self.data)} questions…[/bold]\n")

        for qid, entry in tqdm(self.data.items(), desc="Evaluating"):
            question       = entry.get("question", "")
            options        = self._extract_options(entry)
            raw_answer     = entry.get("answer", "")
            correct_answer = self._parse_correct_answer(raw_answer)
            answer_text    = self._answer_text(entry, options)
            category       = entry.get("category", "")

            # Ask the RAG system
            try:
                result      = self.rag.ask_mcq(question, options)
                predicted   = self._parse_predicted_answer(result["answer"], options)
                is_correct  = predicted.lower() == correct_answer.lower()
                rr          = self._compute_reciprocal_rank(result["sources"], answer_text)
                recall_hit  = self._compute_recall_hit(result["sources"], answer_text)
                source_ids  = [
                    d.metadata.get("question_id", d.metadata.get("source", ""))
                    for d in result["sources"]
                ]
                contexts    = [d.page_content for d in result["sources"]]
            except Exception as e:
                provider_error = self._provider_error_message(e)
                if provider_error:
                    raise RuntimeError(provider_error) from e

                console.print(f"[red]Error on {qid}: {e}[/red]")
                predicted, is_correct, rr = "error", False, 0.0
                recall_hit, source_ids, contexts = False, [], []

            results.append(EvalResult(
                question_id=qid,
                question=question,
                options=options,
                correct_answer=correct_answer,
                predicted_answer=predicted,
                is_correct=is_correct,
                reciprocal_rank=rr,
                recall_hit=recall_hit,
                retrieved_sources=source_ids,
                generated_answer=result.get("answer", "") if isinstance(result, dict) else "",
                contexts=contexts,
                category=category,
            ))

        return results

    def _compute_faithfulness(self, results: List[EvalResult]) -> Optional[float]:
        """
        Compute faithfulness using RAGAS — measures how well the answer
        is grounded in the retrieved context (no hallucination).
        Requires an LLM (uses current config provider).
        """
        try:
            from datasets import Dataset as HFDataset
            from ragas import evaluate
            from ragas.metrics import faithfulness

            data = {
                "question": [r.question for r in results],
                "answer":   [r.generated_answer for r in results],
                "contexts": [r.contexts for r in results],
            }
            dataset = HFDataset.from_dict(data)
            score   = evaluate(dataset, metrics=[faithfulness])
            return float(score["faithfulness"])
        except Exception as exc:
            console.print(f"[yellow]RAGAS faithfulness skipped: {exc}[/yellow]")
            return None

    def report(self, results: List[EvalResult], run_faithfulness: bool = False):
        """Print a rich summary table of evaluation metrics."""
        n = len(results)
        if n == 0:
            console.print("[yellow]No questions matched the evaluation filters.[/yellow]")
            return

        accuracy  = sum(r.is_correct for r in results) / n
        mrr       = sum(r.reciprocal_rank for r in results) / n
        top5_acc  = sum(r.reciprocal_rank > 0 for r in results) / n
        recall    = sum(r.recall_hit for r in results) / n

        faithfulness_score: Optional[float] = None
        if run_faithfulness:
            console.print("\n[dim]Computing RAGAS faithfulness (this may take a moment)…[/dim]")
            faithfulness_score = self._compute_faithfulness(results)

        # Per-category breakdown
        categories: Dict[str, List] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r.is_correct)

        # ── Summary table ─────────────────────────────────────────────
        table = Table(title="TelecomRAG Evaluation Results", show_header=True)
        table.add_column("Metric",     style="cyan",  justify="left")
        table.add_column("Score",      style="green", justify="right")
        table.add_column("KPI Target", style="yellow", justify="right")
        table.add_column("Status",     justify="center")

        def status(val, target):
            return "✅" if val >= target else "❌"

        table.add_row("Accuracy",       f"{accuracy*100:.1f}%",  "80%", status(accuracy, 0.80))
        table.add_row("Top-5 Accuracy", f"{top5_acc*100:.1f}%",  "85%", status(top5_acc, 0.85))
        table.add_row("Recall",         f"{recall*100:.1f}%",    "85%", status(recall, 0.85))
        table.add_row("MRR",            f"{mrr*100:.1f}%",       "75%", status(mrr, 0.75))
        if faithfulness_score is not None:
            table.add_row(
                "Faithfulness",
                f"{faithfulness_score*100:.1f}%",
                "90%",
                status(faithfulness_score, 0.90),
            )
        table.add_row("N questions", str(n), "-", "-")

        console.print(table)

        # Per-category
        cat_table = Table(title="Accuracy by Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("N",        style="dim",  justify="right")
        cat_table.add_column("Accuracy", style="green", justify="right")
        cat_table.add_column("Recall",   style="blue",  justify="right")
        for cat, correct_list in sorted(categories.items()):
            cat_results = [r for r in results if r.category == cat]
            cat_acc    = sum(correct_list) / len(correct_list)
            cat_recall = sum(r.recall_hit for r in cat_results) / len(cat_results)
            cat_table.add_row(
                cat,
                str(len(correct_list)),
                f"{cat_acc*100:.1f}%",
                f"{cat_recall*100:.1f}%",
            )
        console.print(cat_table)

        # Save results to JSON
        out_path = Path("./data/eval_results.json")
        with open(out_path, "w") as f:
            json.dump(
                [
                    {
                        "qid":       r.question_id,
                        "question":  r.question,
                        "correct":   r.correct_answer,
                        "predicted": r.predicted_answer,
                        "is_correct":  r.is_correct,
                        "recall_hit":  r.recall_hit,
                        "mrr":         r.reciprocal_rank,
                        "category":    r.category,
                        "faithfulness": faithfulness_score,
                    }
                    for r in results
                ],
                f, indent=2,
            )
        console.print(f"\n[dim]Full results saved → {out_path}[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TelecomRAG on TeleQnA")
    parser.add_argument("--n",           type=int,  default=100,  help="Number of questions to evaluate")
    parser.add_argument("--category",    type=str,  default=None, help="Filter by TeleQnA category")
    parser.add_argument("--full",        action="store_true",     help="Run all 10k questions (slow)")
    parser.add_argument("--reranker",    action="store_true",     help="Enable cross-encoder reranker")
    parser.add_argument("--hyde",        action="store_true",     help="Enable HyDE retrieval")
    parser.add_argument("--faithfulness",action="store_true",     help="Compute RAGAS faithfulness score")
    args = parser.parse_args()

    if args.full:
        args.n = 10_000

    evaluator = TelecomRAGEvaluator(
        n_samples=args.n,
        category=args.category,
        use_reranker=args.reranker,
        use_hyde=args.hyde,
    )
    results = evaluator.run()
    evaluator.report(results, run_faithfulness=args.faithfulness)
