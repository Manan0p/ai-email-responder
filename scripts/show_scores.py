"""Show pre-computed sample evaluation results (no API calls needed)."""
import json
import sys
import os

# Fix Windows cp1252 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(highlight=False)

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_results.json"
)


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall = data["overall"]
    results = data["per_response"]

    console.print(Panel(
        "[bold]AI Email Suggested-Response System[/bold]\n"
        "Evaluation Results — Per-Response & Overall Scores",
        border_style="cyan", width=72,
    ))

    # ── Overall scores ──────────────────────────────────────────────
    console.print("\n[bold cyan]Overall System Scores[/bold cyan]\n")
    ov = Table(width=72, show_header=False)
    ov.add_column("Metric", style="white", width=36)
    ov.add_column("Score", style="bold green", justify="right")
    ov.add_row("Composite Score (weighted)",    f"{overall['composite_score']:.4f} / 1.00")
    ov.add_row("─" * 34, "─" * 10)
    ov.add_row("ROUGE-L (lexical overlap)",     f"{overall['rouge_l']:.4f}")
    ov.add_row("BERTScore (semantic, F1)",      f"{overall['bert_score']:.4f}")
    ov.add_row("Sentence Similarity (cosine)",  f"{overall['sentence_similarity']:.4f}")
    ov.add_row("Key Action Coverage",           f"{overall['key_action_coverage']:.4f}")
    ov.add_row("LLM Judge Average (1-5 scale)", f"{overall['llm_judge_avg']:.2f}")
    console.print(ov)

    # ── By category ─────────────────────────────────────────────────
    console.print("\n[bold]Composite Score by Category:[/bold]")
    cat = Table(width=72)
    cat.add_column("Category", style="cyan")
    cat.add_column("Composite", style="green", justify="right")
    cat.add_column("N", justify="right")
    for name, vals in overall["by_category"].items():
        cat.add_row(
            name.replace("_", " ").title(),
            f"{vals['composite']:.4f}",
            str(vals["count"]),
        )
    console.print(cat)

    # ── Per-response detail ─────────────────────────────────────────
    console.print("\n[bold cyan]Per-Response Scores[/bold cyan]\n")
    for r in results:
        scores = r["scores"]
        judge  = scores["llm_judge"]

        console.print(Panel(
            f"[bold]Email ID:[/bold] {r['email_id']}  |  "
            f"[bold]Category:[/bold] {r['category'].replace('_',' ').title()}  |  "
            f"[bold]Scenario:[/bold] {r['scenario'].replace('_',' ').title()}",
            border_style="yellow", width=72,
        ))

        # Subject / snippet
        console.print(f"  [dim]Subject:[/dim] {r['incoming_email']['subject']}")

        # Automated metrics
        auto = Table(width=72, show_header=True)
        auto.add_column("Metric", style="white")
        auto.add_column("Score", style="green", justify="right")
        auto.add_column("Weight", style="dim", justify="right")
        auto.add_row("ROUGE-L",            f"{scores['rouge_l']:.4f}",            "10%")
        auto.add_row("BERTScore",          f"{scores['bert_score']:.4f}",          "20%")
        auto.add_row("Sentence Similarity",f"{scores['sentence_similarity']:.4f}", "15%")
        auto.add_row("Key Action Coverage",f"{scores['key_action_coverage']:.4f}", "15%")
        console.print(auto)

        # LLM judge
        judge_t = Table(width=72, title="LLM-as-a-Judge (1–5)", title_style="bold")
        judge_t.add_column("Dimension", style="white")
        judge_t.add_column("Score", style="yellow", justify="right")
        judge_t.add_column("Reason", style="dim")
        dims = [
            ("semantic_accuracy", "Semantic Accuracy"),
            ("intent_alignment",  "Intent Alignment"),
            ("tone",              "Tone"),
            ("completeness",      "Completeness"),
            ("coherence",         "Coherence"),
        ]
        for key, label in dims:
            d = judge[key]
            judge_t.add_row(label, str(d["score"]), d["reason"][:55] + "…")
        console.print(judge_t)

        console.print(
            f"  [bold green]Composite Score:[/bold green] "
            f"[bold]{scores['composite_score']:.4f}[/bold] / 1.00"
            f"   |   [dim]Retrieved: {', '.join(r['retrieved_examples'])}[/dim]\n"
        )

    # ── Summary ─────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold green]Evaluation Complete[/bold green]\n\n"
        f"Emails evaluated: [bold]{len(results)}[/bold]\n"
        f"Overall Composite Score: [bold]{overall['composite_score']:.4f}[/bold] / 1.00\n\n"
        f"[dim]Full live evaluation: python scripts/run_evaluation.py[/dim]",
        border_style="green", width=72,
    ))


if __name__ == "__main__":
    main()
