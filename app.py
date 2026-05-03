"""
app.py
Interactive CLI for the Telecom RAN Assistant.

Usage:
    python app.py
    python app.py --demo        # runs 5 pre-set questions automatically
"""
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

from rag_chain import TelecomRAG

console = Console()

DEMO_QUESTIONS = [
    "What is the functional split between CU and DU in O-RAN?",
    "Explain the concept of network slicing in 5G.",
    "What is the difference between eMBB, URLLC, and mMTC in 5G NR?",
    "How does beamforming work in massive MIMO systems?",
    "What is the role of the RIC (RAN Intelligent Controller) in O-RAN?",
]

BANNER = """
╔══════════════════════════════════════════════════════╗
║       📡  Telecom RAN Assistant  (RAG-powered)       ║
║  Powered by LangChain + ChromaDB + TeleQnA dataset   ║
╚══════════════════════════════════════════════════════╝
  Type your question and press Enter.
  Commands: 'quit' | 'sources' (toggle) | 'clear'
"""


def run_demo(rag: TelecomRAG):
    console.print(Rule("[bold yellow]Demo Mode — 5 Sample Questions[/bold yellow]"))
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"\n[bold cyan]Demo Q{i}:[/bold cyan] {q}")
        result = rag.ask(q)
        console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
        console.print(f"[dim]Retrieved {len(result['sources'])} source(s)[/dim]")
        for j, doc in enumerate(result["sources"][:3], 1):
            src = doc.metadata.get("source", "?")
            qid = doc.metadata.get("question_id", "")
            console.print(f"  [dim]{j}. {src}" + (f" | {qid}" if qid else "") + "[/dim]")


def run_interactive(rag: TelecomRAG):
    show_sources = True
    console.print(BANNER)

    while True:
        try:
            query = console.input("[bold magenta]You:[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if query.lower() == "clear":
            console.clear()
            console.print(BANNER)
            continue
        if query.lower() == "sources":
            show_sources = not show_sources
            console.print(f"[dim]Source display {'ON' if show_sources else 'OFF'}[/dim]")
            continue

        with console.status("[bold green]Thinking…[/bold green]"):
            result = rag.ask(query)

        console.print()
        console.print(Panel(Markdown(result["answer"]), title="[bold green]Assistant[/bold green]", border_style="green"))

        if show_sources and result["sources"]:
            console.print(f"[dim]📚 Sources ({len(result['sources'])} retrieved):[/dim]")
            for i, doc in enumerate(result["sources"], 1):
                src = doc.metadata.get("source", "?")
                qid = doc.metadata.get("question_id", "")
                cat = doc.metadata.get("category", "")
                label = f"  {i}. {src}"
                if qid: label += f" | QID: {qid}"
                if cat: label += f" | {cat}"
                console.print(f"[dim]{label}[/dim]")

        console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo questions")
    args = parser.parse_args()

    console.print("[dim]Loading RAG system…[/dim]")
    rag = TelecomRAG(use_mmr=True)

    if args.demo:
        run_demo(rag)
    else:
        run_interactive(rag)
