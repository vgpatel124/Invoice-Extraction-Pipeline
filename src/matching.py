"""Eval matching rules. Each field has a Strict and a Normalized comparison.

Strict: exact equality after trimming whitespace (numbers: exact float equality).
Normalized:
- Numbers: parse "1,200.00" / "₹1,200" style strings to floats; match within ±0.01.
- vendor: lowercase, strip punctuation and legal suffixes, collapse whitespace,
  then rapidfuzz ratio >= 90.
- invoice_number: case-insensitive exact match after trimming. Internal "/" and "-" are kept.
- invoice_date, currency: exact.
- line_items: rows aligned in printed order (order-preserving best-match alignment when the
  counts differ). A row is correct if description ratio >= 85 (whitespace and case normalised),
  quantity is equal (null must match null) and amount is within ±0.01. The field is correct
  only if the row count matches and every row is correct.
- null vs a value is always a mismatch.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

NUMBER_TOLERANCE = 0.01
VENDOR_THRESHOLD = 90
DESCRIPTION_THRESHOLD = 85
LEGAL_SUFFIXES = {"pvt", "private", "ltd", "limited", "llp", "inc", "llc", "co"}
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_NON_NUMERIC = re.compile(r"[^0-9.\-]")


def parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    cleaned = _NON_NUMERIC.sub("", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return None


def numbers_match(expected: Any, got: Any) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    a, b = parse_number(expected), parse_number(got)
    return a is not None and b is not None and abs(a - b) <= NUMBER_TOLERANCE + 1e-9


def strict_equal(expected: Any, got: Any) -> bool:
    if isinstance(expected, str) and isinstance(got, str):
        return expected.strip() == got.strip()
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected is got
    if isinstance(expected, int | float) and isinstance(got, int | float):
        return float(expected) == float(got)
    return expected == got


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_vendor(name: str) -> str:
    words = name.lower().translate(_PUNCT_TABLE).split()
    return " ".join(w for w in words if w not in LEGAL_SUFFIXES)


def vendors_match(expected: str | None, got: str | None) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    return fuzz.ratio(normalize_vendor(expected), normalize_vendor(got)) >= VENDOR_THRESHOLD


def invoice_numbers_match(expected: str | None, got: str | None) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    return expected.strip().casefold() == got.strip().casefold()


def descriptions_match(expected: str, got: str) -> bool:
    return fuzz.ratio(normalize_text(expected), normalize_text(got)) >= DESCRIPTION_THRESHOLD


@dataclass
class RowResult:
    expected: dict[str, Any] | None
    got: dict[str, Any] | None
    description_ok: bool = False
    quantity_ok: bool = False
    amount_ok: bool = False

    @property
    def status(self) -> str:
        if self.got is None:
            return "missing"
        if self.expected is None:
            return "extra"
        return "correct" if self.correct else "wrong"

    @property
    def correct(self) -> bool:
        return self.description_ok and self.quantity_ok and self.amount_ok


@dataclass
class LineItemsResult:
    rows: list[RowResult] = field(default_factory=list)
    expected_count: int = 0
    got_count: int = 0

    @property
    def correct_rows(self) -> int:
        return sum(r.status == "correct" for r in self.rows)

    @property
    def field_correct(self) -> bool:
        return self.expected_count == self.got_count and self.correct_rows == self.expected_count

    def count(self, status: str) -> int:
        return sum(r.status == status for r in self.rows)

    def count_wrong(self, attr: str) -> int:
        return sum(r.status == "wrong" and not getattr(r, attr) for r in self.rows)


def compare_row(expected: dict[str, Any], got: dict[str, Any]) -> RowResult:
    return RowResult(
        expected=expected,
        got=got,
        description_ok=descriptions_match(expected["description"], got["description"]),
        quantity_ok=numbers_match(expected.get("quantity"), got.get("quantity")),
        amount_ok=numbers_match(expected["amount"], got["amount"]),
    )


def _row_similarity(expected: dict[str, Any], got: dict[str, Any]) -> float:
    """How plausible it is that two rows are the same printed row (0 means never pair them)."""
    desc = fuzz.ratio(normalize_text(expected["description"]), normalize_text(got["description"]))
    amount_bonus = 50 if numbers_match(expected["amount"], got["amount"]) else 0
    score = desc + amount_bonus
    return score if score >= 60 else 0.0


def align_rows(
    expected: list[dict[str, Any]], got: list[dict[str, Any]]
) -> list[tuple[int | None, int | None]]:
    """Order-preserving alignment maximising total pair score (Needleman-Wunsch style)."""
    if len(expected) == len(got):
        return [(i, i) for i in range(len(expected))]
    n, m = len(expected), len(got)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            pair = _row_similarity(expected[i], got[j])
            take = best[i + 1][j + 1] + pair if pair else float("-inf")
            best[i][j] = max(take, best[i + 1][j], best[i][j + 1])
    pairs: list[tuple[int | None, int | None]] = []
    i = j = 0
    while i < n and j < m:
        pair = _row_similarity(expected[i], got[j])
        if pair and best[i][j] == best[i + 1][j + 1] + pair:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            pairs.append((i, None))
            i += 1
        else:
            pairs.append((None, j))
            j += 1
    pairs += [(k, None) for k in range(i, n)] + [(None, k) for k in range(j, m)]
    return pairs


def compare_line_items(
    expected: list[dict[str, Any]] | None, got: list[dict[str, Any]] | None
) -> LineItemsResult:
    expected, got = expected or [], got or []
    result = LineItemsResult(expected_count=len(expected), got_count=len(got))
    for i, j in align_rows(expected, got):
        if i is not None and j is not None:
            result.rows.append(compare_row(expected[i], got[j]))
        else:
            result.rows.append(
                RowResult(
                    expected=expected[i] if i is not None else None,
                    got=got[j] if j is not None else None,
                )
            )
    return result


def line_items_equal_strict(expected: list | None, got: list | None) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    if len(expected) != len(got):
        return False
    return all(
        all(strict_equal(e[k], g.get(k)) for k in ("description", "quantity", "amount"))
        for e, g in zip(expected, got, strict=True)
    )


SCALAR_FIELDS = ("is_invoice", "vendor", "invoice_number", "invoice_date", "currency")
NUMBER_FIELDS = ("tax_total", "total")
EVAL_FIELDS = (*SCALAR_FIELDS, "line_items", *NUMBER_FIELDS)


def field_match(name: str, expected: Any, got: Any, *, strict: bool) -> bool:
    if name == "line_items":
        if strict:
            return line_items_equal_strict(expected, got)
        if expected is None or got is None:
            return expected is None and got is None
        return compare_line_items(expected, got).field_correct
    if strict:
        return strict_equal(expected, got)
    if name == "vendor":
        return vendors_match(expected, got)
    if name == "invoice_number":
        return invoice_numbers_match(expected, got)
    if name in NUMBER_FIELDS:
        return numbers_match(expected, got)
    return strict_equal(expected, got)
