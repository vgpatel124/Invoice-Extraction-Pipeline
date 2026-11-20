from __future__ import annotations

import pytest

from src.matching import (
    align_rows,
    compare_line_items,
    field_match,
    invoice_numbers_match,
    numbers_match,
    parse_number,
    strict_equal,
    vendors_match,
)


def row(description: str, quantity: float | None, amount: float) -> dict:
    return {"description": description, "quantity": quantity, "amount": amount}


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1,200.00", 1200.0), ("₹1,200", 1200.0), ("$ 99.5", 99.5), ("-50", -50.0), (7, 7.0)],
)
def test_parse_number(raw, value):
    assert parse_number(raw) == value


def test_parse_number_rejects_garbage():
    assert parse_number("n/a") is None
    assert parse_number(None) is None


def test_numbers_match_within_tolerance():
    assert numbers_match("1,200.00", 1200)
    assert numbers_match(100.0, 100.01)
    assert not numbers_match(100.0, 100.02)


def test_null_vs_value_is_always_a_mismatch():
    assert not numbers_match(None, 0)
    assert not numbers_match(0, None)
    assert numbers_match(None, None)
    assert not vendors_match("Acme", None)
    assert not field_match("invoice_date", None, "2026-01-01", strict=False)


@pytest.mark.parametrize(
    ("expected", "got"),
    [
        ("Acme Supplies Pvt Ltd", "Acme Supplies"),
        ("Acme Supplies Pvt. Ltd.", "ACME SUPPLIES PRIVATE LIMITED"),
        ("Northwind Logistics Ltd", "Northwind  Logistics"),
        ("Smith & Co", "Smith"),
    ],
)
def test_vendor_variants_match(expected, got):
    assert vendors_match(expected, got)


def test_different_vendors_do_not_match():
    assert not vendors_match("Acme Supplies Pvt Ltd", "Apex Traders Pvt Ltd")


def test_invoice_number_keeps_internal_characters():
    assert invoice_numbers_match("INV/2026/07", " inv/2026/07 ")
    assert not invoice_numbers_match("INV/2026/07", "INV-2026-07")
    assert not invoice_numbers_match("INV/2026/07", "INV202607")


def test_strict_is_exact_after_trim():
    assert strict_equal(" Acme ", "Acme")
    assert not strict_equal("Acme Ltd", "Acme")
    assert strict_equal(3, 3.0)
    assert not strict_equal(True, 1)


def test_strict_vs_normalized_gap():
    assert not field_match("total", 1200.0, 1200.004, strict=True)
    assert field_match("total", 1200.0, 1200.004, strict=False)


def test_identical_line_items_are_correct():
    items = [row("Widget", 2, 100.0), row("Discount", None, -10.0)]
    result = compare_line_items(items, [dict(r) for r in items])
    assert result.field_correct
    assert result.correct_rows == 2


def test_description_case_and_spacing_are_ignored():
    result = compare_line_items(
        [row("A4 Paper  500 sheets", 1, 10)], [row("a4 paper 500 sheets", 1, 10)]
    )
    assert result.field_correct


def test_quantity_null_must_match_null():
    result = compare_line_items([row("Freight", None, 50)], [row("Freight", 1, 50)])
    assert not result.field_correct
    assert result.count_wrong("quantity_ok") == 1


def test_wrong_discount_sign_is_wrong_amount():
    result = compare_line_items([row("Discount", None, -10)], [row("Discount", None, 10)])
    assert result.count_wrong("amount_ok") == 1


def test_missing_row_is_detected_and_others_still_align():
    expected = [row("Alpha item", 1, 10), row("Bravo item", 1, 20), row("Charlie item", 1, 30)]
    got = [row("Alpha item", 1, 10), row("Charlie item", 1, 30)]
    result = compare_line_items(expected, got)
    assert not result.field_correct
    assert result.count("missing") == 1
    assert result.correct_rows == 2
    assert align_rows(expected, got) == [(0, 0), (1, None), (2, 1)]


def test_extra_row_is_detected():
    expected = [row("Alpha item", 1, 10)]
    got = [row("Alpha item", 1, 10), row("Subtotal", None, 10)]
    result = compare_line_items(expected, got)
    assert result.count("extra") == 1
    assert result.correct_rows == 1
    assert not result.field_correct


def test_non_invoice_null_line_items():
    assert field_match("line_items", None, None, strict=False)
    assert not field_match("line_items", None, [], strict=False)
