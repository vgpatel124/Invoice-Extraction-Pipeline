from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

VALID_INVOICE: dict[str, Any] = {
    "is_invoice": True,
    "vendor": "Synthetic Test Traders",
    "invoice_number": "ST/2026/007",
    "invoice_date": "2026-03-05",
    "currency": "INR",
    "line_items": [
        {"description": "Test widget A", "quantity": 2, "amount": 1000.0},
        {"description": "Test service B", "quantity": None, "amount": 500.0},
        {"description": "Loyalty discount", "quantity": None, "amount": -100.0},
    ],
    "tax_total": 252.0,
    "total": 1652.0,
}

NON_INVOICE: dict[str, Any] = {
    "is_invoice": False,
    "vendor": None,
    "invoice_number": None,
    "invoice_date": None,
    "currency": None,
    "line_items": None,
    "tax_total": None,
    "total": None,
}


@pytest.fixture
def invoice() -> dict[str, Any]:
    return copy.deepcopy(VALID_INVOICE)


@pytest.fixture
def non_invoice() -> dict[str, Any]:
    return copy.deepcopy(NON_INVOICE)
