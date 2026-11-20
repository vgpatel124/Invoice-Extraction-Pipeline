"""needs_review heuristics for completed extractions (failures are processing_error instead).

Each rule returns a short reason code. A document is flagged if any rule fires.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any

from src.inputs import LoadedDocument, extract_text

MANY_LINE_ITEMS = 15
INVOICE_FIELDS = ("vendor", "invoice_number", "invoice_date", "currency", "tax_total", "total")
MONTHS = [name.lower() for name in calendar.month_name[1:]]
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})\b")


def _has_month_name_date(text: str, day: date) -> bool:
    """True if the date appears with its month spelled out (e.g. '5 Mar 2026', 'March 5, 2026')."""
    month = MONTHS[day.month - 1]
    name = rf"(?:{month}|{month[:3]})\.?"
    d = rf"0?{day.day}(?:st|nd|rd|th)?"
    patterns = (rf"\b{d}[\s\-]+{name}\b", rf"\b{name}[\s\-]+{d}\b")
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _date_order_on_page(text: str) -> str | None:
    """'day_first' / 'month_first' if some numeric date on the page can only be read one way."""
    orders = set()
    for first, second, _ in _NUMERIC_DATE.findall(text):
        a, b = int(first), int(second)
        if a > 12 >= b:
            orders.add("day_first")
        elif b > 12 >= a:
            orders.add("month_first")
    return orders.pop() if len(orders) == 1 else None


def _printed_in_order(text: str, day: date, order: str) -> bool:
    for first, second, _ in _NUMERIC_DATE.findall(text):
        a, b = int(first), int(second)
        dm = (a, b) if order == "day_first" else (b, a)
        if dm == (day.day, day.month):
            return True
    return False


def date_is_ambiguous(raw_date: str | None, text: str) -> bool:
    """Both day and month <= 12 (and differ) and the page text does not settle the order."""
    if raw_date is None:
        return False
    try:
        day = date.fromisoformat(raw_date)
    except ValueError:
        return False
    if day.day > 12 or day.day == day.month:
        return False
    if not text:
        return True
    if raw_date in text or _has_month_name_date(text, day):
        return False
    order = _date_order_on_page(text)
    return not (order and _printed_in_order(text, day, order))


def review_reasons(doc: LoadedDocument, data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    no_text_layer = doc.is_image or not doc.has_text_layer
    if not data.get("is_invoice"):
        if no_text_layer:
            reasons.append("non_invoice_without_text_layer")
        return reasons
    if no_text_layer:
        reasons.append("scan_or_photo")
    if date_is_ambiguous(data.get("invoice_date"), extract_text(doc)):
        reasons.append("ambiguous_date")
    if len(data.get("line_items") or []) > MANY_LINE_ITEMS:
        reasons.append("many_line_items")
    missing = [name for name in INVOICE_FIELDS if data.get(name) is None]
    if missing:
        reasons.append("missing_fields:" + ",".join(missing))
    return reasons
