"""
evaluator.py
Evaluates the RAG system against the TeleQnA benchmark.

Metrics (aligned with project KPIs):
    - Accuracy          : correct answer rate on MCQ questions
    - Top-k Accuracy    : correct answer in top-k retrieved docs
    - MRR               : Mean Reciprocal Rank of correct answer
    - Faithfulness      : via RAGAS (answer grounded in context)

Usage:
    python -m src.evaluator --n 100           # evaluate on 100 random questions
    python -m src.evaluator --category "Standards specifications"
    python -m src.evaluator --full            # full 10k evaluation (slow)
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
    reciprocal_rank: float   # 1/rank if found, 0 otherwise
    retrieved_sources: List[str] = field(default_factory=list)
    category: str = ""


# ── Core evaluation ───────────────────────────────────────────────────

class TelecomRAGEvaluator:
    def __init__(self, n_samples: int = 100, category: Optional[str] = None):
        self.rag      = TelecomRAG(use_mmr=True)
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
        Looks for 'option N' or the option text itself.
        """
        answer_lower = llm_answer.lower()
        # Try "option N" format
        match = re.search(r"\boption\s*([1-5])\b", answer_lower)
        if match:
            return f"option {match.group(1)}"

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

    def _compute_reciprocal_rank(self, sources: List, correct_answer: str) -> float:
        """Check if any retrieved source mentions the correct answer, return 1/rank."""
        for rank, doc in enumerate(sources, 1):
            if correct_answer.lower() in doc.page_content.lower():
                return 1.0 / rank
        return 0.0

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
            category       = entry.get("category", "")

            # Ask the RAG system
            try:
                result = self.rag.ask_mcq(question, options)
                predicted = self._parse_predicted_answer(result["answer"], options)
                is_correct = predicted.lower() == correct_answer.lower()
                rr = self._compute_reciprocal_rank(result["sources"], raw_answer)
                source_ids = [
                    d.metadata.get("question_id", d.metadata.get("source", ""))
                    for d in result["sources"]
                ]
            except Exception as e:
                provider_error = self._provider_error_message(e)
                if provider_error:
                    raise RuntimeError(provider_error) from e

                console.print(f"[red]Error on {qid}: {e}[/red]")
                predicted, is_correct, rr, source_ids = "error", False, 0.0, []

            results.append(EvalResult(
                question_id=qid,
                question=question,
                options=options,
                correct_answer=correct_answer,
                predicted_answer=predicted,
                is_correct=is_correct,
                reciprocal_rank=rr,
                retrieved_sources=source_ids,
                category=category,
            ))

        return results

    def report(self, results: List[EvalResult]):
        """Print a rich summary table of evaluation metrics."""
        n          = len(results)
        if n == 0:
            console.print("[yellow]No questions matched the evaluation filters.[/yellow]")
            return

        accuracy   = sum(r.is_correct for r in results) / n
        mrr        = sum(r.reciprocal_rank for r in results) / n
        top1_acc   = accuracy  # same as accuracy for MCQ
        # Top-5: answer is in sources
        top5_acc   = sum(r.reciprocal_rank > 0 for r in results) / n

        # Per-category breakdown
        categories: Dict[str, List] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r.is_correct)

        # ── Summary table ─────────────────────────────────────────────
        table = Table(title="TelecomRAG Evaluation Results", show_header=True)
        table.add_column("Metric",     style="cyan",  justify="left")
        table.add_column("Score",      style="green", justify="right")
        table.add_column("KPI Target", style="yellow",justify="right")
        table.add_column("Status",     justify="center")

        def status(val, target):
            return "✅" if val >= target else "❌"

        table.add_row("Accuracy",       f"{accuracy*100:.1f}%", "80%", status(accuracy, 0.80))
        table.add_row("Top-5 Accuracy", f"{top5_acc*100:.1f}%", "85%", status(top5_acc, 0.85))
        table.add_row("MRR",            f"{mrr*100:.1f}%",      "75%", status(mrr, 0.75))
        table.add_row("N questions",    str(n),                  "-",   "-")

        console.print(table)

        # Per-category
        cat_table = Table(title="Accuracy by Category")
        cat_table.add_column("Category",  style="cyan")
        cat_table.add_column("N",         style="dim",  justify="right")
        cat_table.add_column("Accuracy",  style="green",justify="right")
        for cat, correct_list in sorted(categories.items()):
            acc = sum(correct_list) / len(correct_list)
            cat_table.add_row(cat, str(len(correct_list)), f"{acc*100:.1f}%")
        console.print(cat_table)

        # Save results to JSON
        out_path = Path("./data/eval_results.json")
        with open(out_path, "w") as f:
            json.dump(
                [
                    {
                        "qid": r.question_id,
                        "question": r.question,
                        "correct": r.correct_answer,
                        "predicted": r.predicted_answer,
                        "is_correct": r.is_correct,
                        "mrr": r.reciprocal_rank,
                        "category": r.category,
                    }
                    for r in results
                ],
                f, indent=2
            )
        console.print(f"\n[dim]Full results saved → {out_path}[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TelecomRAG on TeleQnA")
    parser.add_argument("--n",        type=int, default=100, help="Number of questions")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--full",     action="store_true",   help="Run all 10k questions")
    args = parser.parse_args()

    if args.full:
        args.n = 10_000

    evaluator = TelecomRAGEvaluator(n_samples=args.n, category=args.category)
    results   = evaluator.run()
    evaluator.report(results)
