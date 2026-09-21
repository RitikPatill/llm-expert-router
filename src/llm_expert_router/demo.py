from __future__ import annotations

import argparse
import sys

import httpx
import rich.box
import rich.table
from rich.console import Console
from rich.progress import track
from rich.rule import Rule

SAMPLE_PROMPTS: list[dict[str, str]] = [
    {"prompt": "Write a Python function to check if a number is prime", "hint": "code"},
    {"prompt": "Debug this JS: `const x=[1,2,3]; console.log(x.lenght)`", "hint": "code"},
    {"prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?", "hint": "reasoning"},
    {"prompt": "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?", "hint": "reasoning"},
    {"prompt": "Summarise the CAP theorem in 3 bullet points", "hint": "summarisation"},
    {"prompt": "Translate 'Good morning, how are you?' into French", "hint": "translation"},
    {"prompt": "Write a short poem about a robot learning to feel emotions", "hint": "creative"},
    {"prompt": "What is the capital of Australia?", "hint": "general"},
    {"prompt": "Explain what a Python decorator does", "hint": "code"},
    {"prompt": "Give me a TL;DR of the French Revolution", "hint": "summarisation"},
]

_CATEGORY_STYLE: dict[str, str] = {
    "code": "cyan",
    "reasoning": "magenta",
    "creative": "yellow",
    "translation": "green",
    "summarisation": "blue",
    "general": "white",
}


def _truncate(text: str, width: int = 48) -> str:
    if len(text) > width:
        return text[: width - 1] + "\u2026"
    return text


def call_chat(prompt: str, base_url: str = "http://localhost:8000") -> dict:
    url = base_url.rstrip("/") + "/chat"
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json={"message": prompt})
        resp.raise_for_status()
        return resp.json()


def build_table(results: list[dict]) -> rich.table.Table:
    table = rich.table.Table(
        box=rich.box.SIMPLE_HEAVY,
        show_footer=False,
        title="LLM Expert Router — Demo Results",
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Prompt", min_width=20, max_width=50)
    table.add_column("Category", min_width=12)
    table.add_column("Expert", min_width=12)
    table.add_column("Model", min_width=14)
    table.add_column("Method", min_width=10)
    table.add_column("Latency(ms)", justify="right", min_width=11)
    table.add_column("Cost(USD)", justify="right", min_width=10)

    for i, r in enumerate(results, start=1):
        meta = r.get("routing_metadata", {})
        category = meta.get("task_category", "?")
        style = _CATEGORY_STYLE.get(category, "white")
        table.add_row(
            str(i),
            _truncate(r.get("_prompt", ""), 48),
            f"[{style}]{category}[/{style}]",
            meta.get("expert_name", "?"),
            meta.get("model", "?"),
            meta.get("classification_method", "?"),
            f"{meta.get('latency_ms', 0):.0f}",
            f"${meta.get('estimated_cost_usd', 0):.6f}",
        )

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Expert Router demo script")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the router server")
    args = parser.parse_args()

    console = Console()
    results: list[dict] = []

    console.print(f"\n[bold]Firing {len(SAMPLE_PROMPTS)} prompts at {args.url}[/bold]\n")

    for item in track(SAMPLE_PROMPTS, description="Calling /chat..."):
        try:
            data = call_chat(item["prompt"], base_url=args.url)
            data["_prompt"] = item["prompt"]
            results.append(data)
        except Exception:
            console.print_exception()
            sys.exit(1)

    console.print()
    console.print(build_table(results))

    total_cost = sum(r.get("routing_metadata", {}).get("estimated_cost_usd", 0) for r in results)
    console.print(Rule())
    console.print(f"[bold]{len(results)} calls[/bold] | total cost [bold green]${total_cost:.6f}[/bold green]")
    console.print()
