"""CLI: Evaluate the system on the test set."""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from src.pipeline import Pipeline
from evaluation.evaluator import Evaluator
from evaluation.report import print_report, print_single_result, save_json_report, save_markdown_report

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AI email responder on the test set")
    parser.add_argument("--dataset", default=None, help="Path to dataset JSON")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-Judge evaluation")
    parser.add_argument("--output-json", default=None, help="Save JSON report to path")
    parser.add_argument("--output-md", default=None, help="Save Markdown report to path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-response details")
    args = parser.parse_args()

    console.print("[bold cyan]AI Email Responder — Evaluation[/bold cyan]\n")

    console.print("Loading pipeline...")
    pipeline = Pipeline()
    pipeline.load_dataset(args.dataset)
    console.print(f"Loaded {len(pipeline.dataset)} email pairs")

    train_count = pipeline.index_dataset(split="train")
    console.print(f"Indexed {train_count} training emails\n")

    test_set = pipeline.get_test_set()
    console.print(f"[bold]Evaluating {len(test_set)} test emails...[/bold]\n")

    console.print("Generating replies for test set...")
    pipeline_results = pipeline.evaluate_test_set()

    console.print("Running evaluation metrics...\n")
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

    print_report(report)

    if args.verbose:
        console.print("\n[bold]Per-Response Details:[/bold]\n")
        for result in report.results:
            print_single_result(result.to_dict())
            console.print()

    if args.output_json:
        save_json_report(report, args.output_json)

    if args.output_md:
        save_markdown_report(report, args.output_md)


if __name__ == "__main__":
    main()
