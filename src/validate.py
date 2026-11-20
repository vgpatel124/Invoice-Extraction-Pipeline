"""Schema, totals and sanity checks, and the feedback text sent on retry."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from src.schema import schema_errors

IssueKind = Literal["parse", "schema", "totals", "sanity"]


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    message: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


@dataclass(frozen=True)
class CheckResult:
    data: dict[str, Any] | None
    issues: list[Issue]

    @property
    def schema_valid(self) -> bool:
        return self.data is not None and not any(i.kind in ("parse", "schema") for i in self.issues)

    @property
    def clean(self) -> bool:
        return self.data is not None and not self.issues


def totals_tolerance(total: float) -> float:
    return max(0.02, 0.001 * abs(total))


def _money(x: float) -> str:
    return f"{x:,.2f}"


def check_totals(data: dict[str, Any]) -> list[Issue]:
    items, tax, total = data.get("line_items"), data.get("tax_total"), data.get("total")
    if not items or tax is None or total is None:
        return []
    items_sum = sum(item["amount"] for item in items)
    diff = items_sum + tax - total
    if abs(diff) <= totals_tolerance(total):
        return []
    return [
        Issue(
            "totals",
            f"line items sum to {_money(items_sum)}, tax_total is {_money(tax)}, total is "
            f"{_money(total)}: sum + tax = {_money(items_sum + tax)}, which differs from total "
            f"by {_money(diff)}. Re-read the item table (rows on every page, discount and charge "
            "rows, pre-tax amounts) and the tax lines, and copy numbers exactly as printed.",
        )
    ]


def _all_numbers(data: dict[str, Any]) -> list[float]:
    values = [data.get("tax_total"), data.get("total")]
    for item in data.get("line_items") or []:
        values += [item.get("amount"), item.get("quantity")]
    return [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]


def check_sanity(data: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    if any(not math.isfinite(v) for v in _all_numbers(data)):
        issues.append(Issue("sanity", "an amount is NaN or infinite"))
    if not data.get("is_invoice"):
        return issues
    raw_date = data.get("invoice_date")
    if raw_date is not None:
        try:
            date.fromisoformat(raw_date)
        except ValueError:
            issues.append(Issue("sanity", f"invoice_date {raw_date!r} is not a real calendar date"))
    total = data.get("total")
    if total is None or total <= 0:
        issues.append(Issue("sanity", f"total is {total!r}; an invoice total must be > 0"))
    if not data.get("line_items"):
        issues.append(Issue("sanity", "an invoice must have at least one line item"))
    return issues


def run_checks(text: str) -> CheckResult:
    """Run all checks in order: parse, schema, totals, sanity."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return CheckResult(None, [Issue("parse", f"response is not valid JSON: {exc}")])
    if not isinstance(data, dict):
        return CheckResult(None, [Issue("parse", "response is not a JSON object")])
    errors = schema_errors(data)
    if errors:
        return CheckResult(data, [Issue("schema", e) for e in errors])
    issues = check_sanity(data)
    if not any("NaN" in i.message for i in issues):
        issues = check_totals(data) + issues
    return CheckResult(data, issues)
