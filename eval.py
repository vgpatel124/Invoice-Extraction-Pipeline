"""One-command eval: uv run python eval.py

Runs the extractor on the labelled documents (cache-backed), compares each field with the
ground truth under Strict and Normalized rules (see src/matching.py) and prints the results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src import budget
from src.config import PROJECT_ROOT, data_dir, price_for, primary_model
from src.matching import EVAL_FIELDS, LineItemsResult, compare_line_items, field_match
from src.pipeline import INPUT_MODES, extract_document
from src.prompt import PROMPT_VERSION

DOC_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png")
RESULTS_DIR = PROJECT_ROOT / "eval_results"
LATER_PRICE_DATE = date(2027, 1, 1)
OK, BAD, NA = "[green]✓[/]", "[red]✗[/]", "[dim]·[/]"

console = Console()
err = Console(stderr=True)


@dataclass
class DocEval:
    doc_id: str
    expected: dict[str, Any]
    got: dict[str, Any] | None
    meta: dict[str, Any]
    strict: dict[str, bool]
    normalized: dict[str, bool]
    line_items: LineItemsResult | None

    @property
    def is_invoice(self) -> bool:
        return bool(self.expected["is_invoice"])

    @property
    def all_correct(self) -> bool:
        return all(self.normalized.values())

    @property
    def has_error(self) -> bool:
        return self.got is None or not self.all_correct


def labelled_documents(only: list[str] | None) -> list[tuple[Path, Path]]:
    folder = data_dir() / "labelled"
    if not folder.is_dir():
        raise SystemExit(f"No labelled data at {folder}. Put the sample pack in data/ (README).")
    pairs = []
    for doc in sorted(p for p in folder.iterdir() if p.suffix.lower() in DOC_SUFFIXES):
        truth = doc.with_suffix(".json")
        if truth.is_file() and (not only or doc.stem in only):
            pairs.append((doc, truth))
    return pairs


def score_document(doc_id: str, expected: dict, got: dict | None, meta: dict) -> DocEval:
    """Invoices: all 8 fields. Non-invoices: is_invoice plus 'all other fields null'."""
    candidate = got or {}
    names = EVAL_FIELDS if expected["is_invoice"] else ("is_invoice",)
    strict = {
        n: got is not None and field_match(n, expected[n], candidate.get(n), strict=True)
        for n in names
    }
    normalized = {
        n: got is not None and field_match(n, expected[n], candidate.get(n), strict=False)
        for n in names
    }
    if not expected["is_invoice"]:
        nulls_ok = got is not None and all(candidate.get(n) is None for n in EVAL_FIELDS[1:])
        strict["other_fields_null"] = normalized["other_fields_null"] = nulls_ok
    line_items = (
        compare_line_items(expected["line_items"], candidate.get("line_items"))
        if expected["is_invoice"]
        else None
    )
    return DocEval(doc_id, expected, got, meta, strict, normalized, line_items)


def extract_all(
    pairs: list[tuple[Path, Path]], model: str, use_cache: bool, input_mode: str
) -> list[DocEval]:
    results = []
    for index, (doc, truth) in enumerate(pairs, 1):
        err.print(f"[{index}/{len(pairs)}] {doc.name} ...", end=" ")
        outcome = extract_document(doc, model=model, use_cache=use_cache, input_mode=input_mode)
        state = (
            "cache" if outcome.meta["cache_hit"] else f"{outcome.meta['latency_ms'] / 1000:.0f}s"
        )
        if outcome.meta["processing_error"]:
            state += f" ERROR {outcome.meta['processing_error']['type']}"
        err.print(state)
        expected = json.loads(truth.read_text())
        results.append(score_document(doc.stem, expected, outcome.result, outcome.meta))
    return results


def percent(numerator: int, denominator: int) -> str:
    return (
        f"{numerator}/{denominator} ({100 * numerator / denominator:.0f}%)"
        if denominator
        else "n/a"
    )


def field_table(results: list[DocEval]) -> tuple[Table, dict[str, Any]]:
    invoices = [r for r in results if r.is_invoice]
    table = Table(title="Per-field accuracy (invoice fields scored on invoice documents only)")
    for col in ("Field", "Strict", "Normalized"):
        table.add_column(col, justify="left" if col == "Field" else "right")
    summary: dict[str, Any] = {}
    for name in EVAL_FIELDS:
        pool = results if name == "is_invoice" else invoices
        s = sum(r.strict[name] for r in pool)
        n = sum(r.normalized[name] for r in pool)
        table.add_row(name, percent(s, len(pool)), percent(n, len(pool)))
        summary[name] = {"strict": s, "normalized": n, "n": len(pool)}
    s_all = sum(all(r.strict.values()) for r in invoices)
    n_all = sum(r.all_correct for r in invoices)
    table.add_section()
    table.add_row(
        "all fields correct", percent(s_all, len(invoices)), percent(n_all, len(invoices))
    )
    summary["all_fields_correct"] = {"strict": s_all, "normalized": n_all, "n": len(invoices)}
    return table, summary


def document_table(results: list[DocEval]) -> Table:
    table = Table(title="Per-document results (Normalized)")
    table.add_column("Doc")
    for name in EVAL_FIELDS:
        table.add_column(name.replace("invoice_", "inv_"), justify="center")
    for col in ("rows ok", "retry", "review", "error"):
        table.add_column(col, justify="center")
    for r in results:
        if r.is_invoice:
            cells = [OK if r.normalized[n] else BAD for n in EVAL_FIELDS]
            li = r.line_items
            rows_ok = f"{li.correct_rows}/{li.expected_count}" if li else NA
        else:
            nulls = OK if r.normalized["other_fields_null"] else BAD
            cells = [OK if r.normalized["is_invoice"] else BAD] + [nulls] * (len(EVAL_FIELDS) - 1)
            rows_ok = NA
        error = r.meta["processing_error"]["type"] if r.meta["processing_error"] else ""
        table.add_row(
            f"{r.doc_id}{'' if r.is_invoice else ' (non-inv)'}",
            *cells,
            rows_ok,
            str(r.meta["retry_count"]),
            "yes" if r.meta["needs_review"] else "",
            f"[red]{error}[/]" if error else "",
        )
    return table


def line_item_breakdown(results: list[DocEval]) -> tuple[Table, dict[str, Any]]:
    items = [r.line_items for r in results if r.line_items is not None]
    expected = sum(li.expected_count for li in items)
    predicted = sum(li.got_count for li in items)
    correct = sum(li.correct_rows for li in items)
    precision = correct / predicted if predicted else 0.0
    recall = correct / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    stats = {
        "expected_rows": expected,
        "predicted_rows": predicted,
        "correct_rows": correct,
        "missing_rows": sum(li.count("missing") for li in items),
        "extra_rows": sum(li.count("extra") for li in items),
        "wrong_quantity": sum(li.count_wrong("quantity_ok") for li in items),
        "wrong_amount": sum(li.count_wrong("amount_ok") for li in items),
        "wrong_description": sum(li.count_wrong("description_ok") for li in items),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    table = Table(title="Line-item breakdown (all labelled invoices, rows aligned in order)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in stats.items():
        table.add_row(
            key.replace("_", " "), f"{value:.3f}" if isinstance(value, float) else str(value)
        )
    return table, stats


def review_table(results: list[DocEval]) -> tuple[Table, dict[str, int]]:
    completed = [r for r in results if r.got is not None]
    cells = {
        "flagged_with_error": sum(r.meta["needs_review"] and r.has_error for r in completed),
        "flagged_no_error": sum(r.meta["needs_review"] and not r.has_error for r in completed),
        "not_flagged_with_error": sum(
            not r.meta["needs_review"] and r.has_error for r in completed
        ),
        "not_flagged_no_error": sum(
            not r.meta["needs_review"] and not r.has_error for r in completed
        ),
        "processing_errors": len(results) - len(completed),
    }
    table = Table(title="Review flag vs actual errors (completed extractions)")
    table.add_column("")
    table.add_column("Has error", justify="right")
    table.add_column("No error", justify="right")
    table.add_row("needs_review", str(cells["flagged_with_error"]), str(cells["flagged_no_error"]))
    table.add_row(
        "not flagged", str(cells["not_flagged_with_error"]), str(cells["not_flagged_no_error"])
    )
    return table, cells


def cost_summary(results: list[DocEval], model: str) -> dict[str, Any]:
    attempts = [a for r in results for a in r.meta["attempts"]]
    later = price_for(model, LATER_PRICE_DATE)
    tokens_in = sum(a["input_tokens"] for a in attempts)
    tokens_out = sum(a["output_tokens"] for a in attempts)
    thinking = sum(a["thinking_tokens"] for a in attempts)
    cost_now = sum(r.meta["estimated_cost_usd"] for r in results)
    cost_later = (
        tokens_in * later.input_per_m + (tokens_out + thinking) * later.output_per_m
    ) / 1e6
    n = len(results)
    return {
        "documents": n,
        "api_calls_incl_retries": len(attempts),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "thinking_tokens": thinking,
        "avg_input_tokens": round(tokens_in / n, 1),
        "avg_output_tokens_incl_thinking": round((tokens_out + thinking) / n, 1),
        "total_cost_usd": round(cost_now, 6),
        "avg_cost_per_doc_usd": round(cost_now / n, 6),
        "avg_cost_per_doc_usd_2027_prices": round(cost_later / n, 6),
        "spent_this_run_usd": round(sum(r.meta["spent_this_run_usd"] for r in results), 6),
        "ledger_total_usd": round(budget.spent_usd(), 6),
        "budget_usd": budget.MAX_BUDGET_USD,
    }


def find_mismatches(results: list[DocEval]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        if r.got is None:
            out.append({"doc": r.doc_id, "field": "*", "error": r.meta["processing_error"]})
            continue
        for name, strict_ok in r.strict.items():
            if strict_ok:
                continue
            key = name if name in r.expected else None
            out.append(
                {
                    "doc": r.doc_id,
                    "field": name,
                    "rule": "strict_only" if r.normalized[name] else "normalized",
                    "expected": r.expected.get(key) if key else None,
                    "got": r.got.get(key) if key else {k: r.got[k] for k in EVAL_FIELDS[1:]},
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the extractor on the labelled set.")
    parser.add_argument("--no-cache", action="store_true", help="force fresh model calls")
    parser.add_argument("--model", default=primary_model())
    parser.add_argument("--docs", nargs="*", help="subset of ids, e.g. L01 L02")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="native",
        help="native = send the file (default); text = PyMuPDF text only (experiment)",
    )
    parser.add_argument("--tag", help="results file name (default: <model>_<prompt version>)")
    args = parser.parse_args()

    results = extract_all(
        labelled_documents(args.docs), args.model, not args.no_cache, args.input_mode
    )
    if not results:
        raise SystemExit("No labelled documents matched.")

    fields, field_summary = field_table(results)
    items, item_stats = line_item_breakdown(results)
    review, review_cells = review_table(results)
    costs = cost_summary(results, args.model)

    console.print(fields, document_table(results), items, review)
    for r in (r for r in results if not r.is_invoice):
        detected = "correct" if r.normalized["is_invoice"] else "WRONG"
        nulls = "yes" if r.normalized["other_fields_null"] else "NO"
        console.print(f"Non-invoice {r.doc_id}: is_invoice {detected}, other fields null: {nulls}")
    console.print("\n[bold]Cost[/] (paid-tier prices; thinking billed as output; retries included)")
    for key, value in costs.items():
        console.print(f"  {key}: {value}")

    tag = args.tag or f"{args.model}_{PROMPT_VERSION}_{args.input_mode}"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "input_mode": args.input_mode,
                "fields": field_summary,
                "line_items": item_stats,
                "review_flag": review_cells,
                "cost": costs,
                "per_document": {
                    r.doc_id: {
                        "normalized": r.normalized,
                        "strict": r.strict,
                        "retry_count": r.meta["retry_count"],
                        "needs_review": r.meta["needs_review"],
                        "review_reasons": r.meta["review_reasons"],
                        "validation_errors": r.meta["validation_errors"],
                        "processing_error": r.meta["processing_error"],
                    }
                    for r in results
                },
                "mismatches": find_mismatches(results),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    console.print(f"\nSaved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
