from __future__ import annotations

import json

import pytest

from src.schema import schema_errors
from src.validate import check_sanity, check_totals, run_checks, totals_tolerance


def test_valid_invoice_is_clean(invoice):
    assert run_checks(json.dumps(invoice)).clean


def test_non_invoice_with_all_nulls_is_clean(non_invoice):
    assert run_checks(json.dumps(non_invoice)).clean


def test_non_invoice_with_a_value_fails_schema(non_invoice):
    non_invoice["vendor"] = "Someone"
    result = run_checks(json.dumps(non_invoice))
    assert not result.schema_valid
    assert any("vendor" in str(i) for i in result.issues)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invoice_date", "05/03/2026"),
        ("currency", "inr"),
        ("currency", "RUPEES"),
        ("total", "1,652.00"),
        ("is_invoice", "yes"),
    ],
)
def test_schema_rejects_bad_formats(invoice, field, value):
    invoice[field] = value
    assert schema_errors(invoice)


def test_schema_rejects_extra_keys(invoice):
    invoice["confidence"] = 0.9
    assert schema_errors(invoice)


def test_schema_rejects_missing_line_item_amount(invoice):
    del invoice["line_items"][0]["amount"]
    assert schema_errors(invoice)


def test_unparseable_json_is_a_parse_issue():
    result = run_checks('{"is_invoice": true,')
    assert result.data is None
    assert result.issues[0].kind == "parse"


def test_json_array_is_rejected():
    assert run_checks("[]").issues[0].kind == "parse"


def test_totals_pass_within_rounding_tolerance(invoice):
    invoice["total"] = 1652.01
    assert check_totals(invoice) == []


def test_totals_mismatch_reports_the_numbers(invoice):
    invoice["tax_total"] = 0
    issues = check_totals(invoice)
    assert len(issues) == 1
    message = issues[0].message
    assert "1,400.00" in message and "1,652.00" in message and "-252.00" in message


def test_discount_with_wrong_sign_breaks_totals(invoice):
    invoice["line_items"][2]["amount"] = 100.0
    assert check_totals(invoice)


def test_tolerance_scales_with_large_totals():
    assert totals_tolerance(10) == 0.02
    assert totals_tolerance(1_000_000) == pytest.approx(1000)


def test_impossible_calendar_date_is_flagged(invoice):
    invoice["invoice_date"] = "2026-02-30"
    assert any("calendar" in i.message for i in check_sanity(invoice))


def test_zero_total_is_flagged(invoice):
    invoice["total"] = 0
    assert any("total" in i.message for i in check_sanity(invoice))


def test_invoice_without_line_items_is_flagged(invoice):
    invoice["line_items"] = []
    assert any("line item" in i.message for i in check_sanity(invoice))


def test_nan_amount_is_flagged(invoice):
    text = json.dumps(invoice).replace('"tax_total": 252.0', '"tax_total": NaN')
    result = run_checks(text)
    assert not result.clean


def test_gemini_response_schema_has_no_additional_properties():
    """Gemini's response_schema rejects additionalProperties (it made every call fail once)."""
    from src.schema import InvoiceExtraction

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    assert "additionalProperties" not in set(keys(InvoiceExtraction.model_json_schema()))
