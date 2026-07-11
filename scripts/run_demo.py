"""CLI: Full end-to-end demo of the AI Email Responder system."""
import json
import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.pipeline import Pipeline
from evaluation.evaluator import Evaluator
from evaluation.report import print_report, save_json_report, save_markdown_report

console = Console()

DEMO_EMAILS = [
    {
        "from": "alex.chen@techstartup.io",
        "subject": "Urgent: Need to Reschedule Tomorrow's Product Review",
        "body": (
            "Hi,\n\n"
            "I'm really sorry for the last-minute notice, but I need to reschedule "
            "our product review meeting that's set for tomorrow at 2 PM. I have an "
            "unexpected conflict with a client demo that just came up.\n\n"
            "Would Thursday or Friday afternoon work instead? I'm flexible on the time.\n\n"
            "Thanks for understanding,\nAlex"
        ),
    },
    {
        "from": "maria.santos@globalretail.com",
        "subject": "Defective Product Received - Order #98712",
        "body": (
            "Dear Support Team,\n\n"
            "I received my order yesterday (Order #98712) and unfortunately the "
            "main item — a wireless keyboard — arrived with several keys not working. "
            "The 'A', 'S', and 'D' keys are completely unresponsive.\n\n"
            "I need this for work and would like either a replacement shipped "
            "urgently or a full refund. Please advise on the next steps.\n\n"
            "Thank you,\nMaria Santos"
        ),
    },
    {
        "from": "james.wilson@engineeringcorp.com",
        "subject": "Q3 Project Status Update Request",
        "body": (
            "Hi Team,\n\n"
            "As we approach the end of Q3, I'd like to get a comprehensive status "
            "update on the backend migration project. Specifically:\n\n"
            "1. What percentage of the migration is complete?\n"
            "2. Are there any blockers or risks we should be aware of?\n"
            "3. Is the timeline still on track for the October deadline?\n\n"
            "Please send your update by EOD Wednesday.\n\n"
            "Best,\nJames Wilson\nVP of Engineering"
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="Full end-to-end demo")
    parser.add_argument("--dataset", default=None, help="Path to dataset JSON")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-Judge")
    parser.add_argument("--eval-only", action="store_true", help="Skip demo emails, run evaluation only")
    parser.add_argument("--output-json", default=None, help="Save evaluation report as JSON")
    parser.add_argument("--output-md", default=None, help="Save evaluation report as Markdown")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]AI Email Suggested-Response System[/bold]\n"
        "End-to-end demonstration: RAG pipeline + Multi-dimensional evaluation",
        border_style="cyan",
        width=70,
    ))

    # Step 1: Load and index
    console.print("\n[bold cyan]Step 1: Loading Dataset & Building Index[/bold cyan]\n")
    pipeline = Pipeline()
    pipeline.load_dataset(args.dataset)
    total = len(pipeline.dataset)
    train_count = pipeline.index_dataset(split="train")
    test_count = len(pipeline.get_test_set())
    console.print(f"  Dataset: {total} email pairs ({train_count} train / {test_count} test)")
    console.print(f"  Vector store: {pipeline.vector_store.count()} emails indexed\n")

    # Step 2: Demo emails
    if not args.eval_only:
        console.print("[bold cyan]Step 2: Generating Replies for Demo Emails[/bold cyan]\n")
        for i, email in enumerate(DEMO_EMAILS, 1):
            console.print(f"[bold]Demo Email {i}:[/bold]")
            console.print(Panel(
                f"[bold]From:[/bold] {email['from']}\n"
                f"[bold]Subject:[/bold] {email['subject']}\n\n"
                f"{email['body']}",
                border_style="yellow",
                width=70,
            ))

            reply = pipeline.respond(
                from_addr=email["from"],
                subject=email["subject"],
                body=email["body"],
            )

            console.print(Panel(
                f"[bold]Subject:[/bold] Re: {email['subject']}\n\n"
                f"{reply.reply_text}",
                title=f"Generated Reply (latency: {reply.latency_seconds}s)",
                border_style="green",
                width=70,
            ))
            console.print(f"  [dim]Retrieved examples: {', '.join(reply.retrieved_ids)}[/dim]\n")

    # Step 3: Evaluate on test set
    console.print("[bold cyan]Step 3: Evaluating on Test Set[/bold cyan]\n")
    console.print("  Generating replies for all test emails (this will take ~1 minute to respect API rate limits)...")
    pipeline_results = pipeline.evaluate_test_set()
    console.print(f"  Generated {len(pipeline_results)} replies\n")

    console.print("  Running evaluation metrics...")
    evaluator = Evaluator(use_llm_judge=not args.no_judge)

    eval_items = []
    for pr in pipeline_results:
        item_data = next(d for d in pipeline.dataset if d["id"] == pr.email_id)
        eval_items.append({
            "email_id": pr.email_id,
            "category": item_data["category"],
            "scenario": item_data.get("scenario", ""),
            "incoming_email": pr.incoming_email,
            "generated_reply": pr.generated_reply.reply_text,
            "reference_reply": pr.expected_reply["body"],
            "key_actions": item_data.get("metadata", {}).get("key_actions", []),
            "retrieved_ids": pr.generated_reply.retrieved_ids,
        })

    report = evaluator.evaluate_batch(eval_items)

    # Step 4: Display report
    console.print("\n[bold cyan]Step 4: Evaluation Report[/bold cyan]")
    print_report(report)

    # Show a few sample results
    console.print("[bold]Sample Per-Response Scores:[/bold]\n")
    sample_table = Table(width=70)
    sample_table.add_column("Email ID", style="cyan")
    sample_table.add_column("Category", style="white")
    sample_table.add_column("Composite", style="green", justify="right")
    sample_table.add_column("BERTScore", style="yellow", justify="right")

    for r in report.results[:5]:
        sample_table.add_row(
            r.email_id,
            r.category.replace("_", " ").title(),
            f"{r.composite_score:.4f}",
            f"{r.automated_scores.bert_score:.4f}",
        )
    console.print(sample_table)

    if args.output_json:
        save_json_report(report, args.output_json)
    if args.output_md:
        save_markdown_report(report, args.output_md)

    console.print(Panel(
        f"[bold green]Demo Complete![/bold green]\n\n"
        f"Overall Composite Score: [bold]{report.overall_composite:.4f}[/bold] / 1.00\n"
        f"Emails Evaluated: {len(report.results)}",
        border_style="green",
        width=70,
    ))


if __name__ == "__main__":
    main()
