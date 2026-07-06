import json
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from evaluation.evaluator import EvaluationReport


console = Console()


def print_report(report: EvaluationReport):
    auto_avgs = report.avg_automated_scores()
    judge_avgs = report.avg_judge_scores()
    by_cat = report.by_category()

    table = Table(title="AI Email Responder — Evaluation Report", show_lines=True, width=60)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", justify="right", width=25)

    table.add_row("Test Set Size", f"{len(report.results)} emails")
    table.add_row("Overall Composite", f"{report.overall_composite:.2f} / 1.00")
    table.add_row("", "")

    table.add_row("[bold]── Automated Metrics (Avg) ──[/bold]", "")
    for key, val in auto_avgs.items():
        label = key.replace("_", " ").title()
        table.add_row(f"  {label}", f"{val:.4f}")

    if judge_avgs:
        table.add_row("", "")
        table.add_row("[bold]── LLM Judge (Avg, 1-5) ──[/bold]", "")
        for key, val in judge_avgs.items():
            label = key.replace("_", " ").title()
            table.add_row(f"  {label}", f"{val:.1f}")

    table.add_row("", "")
    table.add_row("[bold]── By Category ──[/bold]", "")
    for cat, val in sorted(by_cat.items()):
        label = cat.replace("_", " ").title()
        table.add_row(f"  {label}", f"{val:.4f}")

    console.print()
    console.print(table)
    console.print()


def print_single_result(result_dict: dict):
    scores = result_dict.get("scores", {})

    console.print(Panel(
        f"[bold]Email ID:[/bold] {result_dict['email_id']}\n"
        f"[bold]Category:[/bold] {result_dict['category']}\n"
        f"[bold]Composite Score:[/bold] {scores.get('composite_score', 0):.4f}",
        title="Evaluation Result",
        width=60,
    ))

    auto_table = Table(title="Automated Metrics", width=50)
    auto_table.add_column("Metric", style="cyan")
    auto_table.add_column("Score", style="green", justify="right")

    for key in ["rouge_l", "bert_score", "sentence_similarity", "key_action_coverage"]:
        if key in scores:
            auto_table.add_row(key.replace("_", " ").title(), f"{scores[key]:.4f}")

    console.print(auto_table)

    if "llm_judge" in scores:
        judge = scores["llm_judge"]
        judge_table = Table(title="LLM Judge Scores", width=60)
        judge_table.add_column("Dimension", style="cyan")
        judge_table.add_column("Score", style="green", justify="right")
        judge_table.add_column("Reason", style="white", max_width=30)

        for dim in ["semantic_accuracy", "intent_alignment", "tone", "completeness", "coherence"]:
            if dim in judge:
                judge_table.add_row(
                    dim.replace("_", " ").title(),
                    str(judge[dim]["score"]),
                    judge[dim].get("reason", "")[:50],
                )
        console.print(judge_table)


def save_json_report(report: EvaluationReport, path: str):
    data = report.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]JSON report saved to: {path}[/green]")


def save_markdown_report(report: EvaluationReport, path: str):
    auto_avgs = report.avg_automated_scores()
    judge_avgs = report.avg_judge_scores()
    by_cat = report.by_category()

    lines = [
        "# AI Email Responder — Evaluation Report\n",
        f"**Test Set Size:** {len(report.results)} emails  ",
        f"**Overall Composite Score:** {report.overall_composite:.4f} / 1.00\n",
        "## Automated Metrics (Averages)\n",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for key, val in auto_avgs.items():
        lines.append(f"| {key.replace('_', ' ').title()} | {val:.4f} |")

    if judge_avgs:
        lines.extend([
            "\n## LLM Judge Scores (Averages, 1-5 scale)\n",
            "| Dimension | Score |",
            "|-----------|-------|",
        ])
        for key, val in judge_avgs.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {val:.1f} |")

    lines.extend([
        "\n## Scores by Category\n",
        "| Category | Composite Score |",
        "|----------|----------------|",
    ])
    for cat, val in sorted(by_cat.items()):
        lines.append(f"| {cat.replace('_', ' ').title()} | {val:.4f} |")

    lines.extend([
        "\n## Per-Response Details\n",
        "| Email ID | Category | Composite |",
        "|----------|----------|-----------|",
    ])
    for r in report.results:
        lines.append(f"| {r.email_id} | {r.category} | {r.composite_score:.4f} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    console.print(f"[green]Markdown report saved to: {path}[/green]")
