"""CLI: Generate a reply for a single email."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from src.pipeline import Pipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Generate a reply for a single email")
    parser.add_argument("--from", dest="from_addr", default="user@example.com", help="Sender email")
    parser.add_argument("--subject", required=True, help="Email subject")
    parser.add_argument("--body", required=True, help="Email body")
    parser.add_argument("--dataset", default=None, help="Path to dataset JSON")
    parser.add_argument("--show-prompt", action="store_true", help="Show the full prompt sent to LLM")
    args = parser.parse_args()

    console.print("[bold]AI Email Responder[/bold]", style="cyan")
    console.print("Loading pipeline...\n")

    pipeline = Pipeline()
    pipeline.load_dataset(args.dataset)

    console.print(f"Loaded {len(pipeline.dataset)} email pairs")
    train_count = pipeline.index_dataset(split="train")
    console.print(f"Indexed {train_count} training emails in vector store\n")

    console.print("[bold]Generating reply...[/bold]\n")
    reply = pipeline.respond(
        from_addr=args.from_addr,
        subject=args.subject,
        body=args.body,
    )

    console.print(Panel(
        f"[bold]Subject:[/bold] Re: {args.subject}\n\n"
        f"{reply.reply_text}",
        title="Generated Reply",
        border_style="green",
    ))

    console.print(f"\n[dim]Model: {reply.model} | Latency: {reply.latency_seconds}s | "
                  f"Retrieved: {', '.join(reply.retrieved_ids)}[/dim]")

    if args.show_prompt:
        console.print(Panel(reply.prompt, title="Full Prompt", border_style="dim"))


if __name__ == "__main__":
    main()
