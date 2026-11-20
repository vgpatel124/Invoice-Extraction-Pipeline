"""Run the extractor on every document in the sample pack and write outputs/.

uv run python run_all.py            # all labelled + unlabelled documents
uv run python run_all.py --only U01 U02
uv run python run_all.py --model gemini-3.5-flash-lite

Without --model, each document is re-run with the model recorded in its existing
outputs/_meta/<ID>.json, so a re-run reuses the cache. New documents use GEMINI_MODEL.

outputs/<ID>.json holds only the 8 schema fields (written only for a completed extraction).
outputs/_meta/<ID>.json holds tokens, cost, latency, retries, flags and errors (always written).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src import budget
from src.config import META_DIR, OUTPUTS_DIR, data_dir, primary_model
from src.pipeline import Outcome, extract_document, to_json
from src.schema import schema_errors

DOC_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png")
SUBFOLDERS = ("labelled", "unlabelled")

console = Console()


def pack_documents(only: list[str] | None) -> list[Path]:
    docs = []
    for name in SUBFOLDERS:
        folder = data_dir() / name
        if not folder.is_dir():
            raise SystemExit(f"Missing {folder}. Put the sample pack in data/ (see README).")
        docs += [p for p in folder.iterdir() if p.suffix.lower() in DOC_SUFFIXES]
    return sorted(p for p in docs if not only or p.stem in only)


def write_outputs(doc_id: str, outcome: Outcome) -> None:
    out_path = OUTPUTS_DIR / f"{doc_id}.json"
    if outcome.result is not None:
        errors = schema_errors(outcome.result)
        if errors:
            raise RuntimeError(f"{doc_id}: refusing to write schema-invalid output: {errors}")
        out_path.write_text(to_json(outcome.result) + "\n")
    else:
        out_path.unlink(missing_ok=True)
    (META_DIR / f"{doc_id}.json").write_text(to_json(outcome.meta) + "\n")


def model_for(doc_id: str, override: str | None) -> str:
    if override:
        return override
    meta_path = META_DIR / f"{doc_id}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())["model"]
    return primary_model()


def status_label(outcome: Outcome) -> str:
    if outcome.meta["processing_error"]:
        return f"[red]error: {outcome.meta['processing_error']['type']}[/]"
    if outcome.needs_review:
        return "[yellow]needs review[/]"
    return "[green]ok[/]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract every sample-pack document to outputs/.")
    parser.add_argument("--only", nargs="*", help="subset of ids, e.g. U01 U02")
    parser.add_argument("--no-cache", action="store_true", help="force fresh model calls")
    parser.add_argument("--model", help="use this model for every document")
    args = parser.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    table = Table(title="run_all")
    for col in ("Doc", "Model", "Status", "is_invoice", "Rows", "Retries", "Cost USD", "Source"):
        table.add_column(col)

    failures = 0
    for doc in pack_documents(args.only):
        console.print(f"{doc.name} ...", end=" ", style="dim")
        model = model_for(doc.stem, args.model)
        outcome = extract_document(doc, model=model, use_cache=not args.no_cache)
        write_outputs(doc.stem, outcome)
        meta, result = outcome.meta, outcome.result or {}
        failures += result == {}
        console.print(status_label(outcome))
        table.add_row(
            doc.stem,
            model.removeprefix("gemini-"),
            status_label(outcome),
            str(result.get("is_invoice", "")),
            str(len(result.get("line_items") or [])) if result else "",
            str(meta["retry_count"]),
            f"{meta['estimated_cost_usd']:.5f}",
            "cache" if meta["cache_hit"] else "api",
        )

    console.print(table)
    console.print(f"Budget ledger: ${budget.spent_usd():.4f} of ${budget.MAX_BUDGET_USD:.2f}")
    if failures:
        console.print(f"[red]{failures} document(s) failed; see outputs/_meta/[/]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
