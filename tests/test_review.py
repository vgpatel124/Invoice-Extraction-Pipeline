from __future__ import annotations

import copy

import pymupdf
from PIL import Image

from src.inputs import LoadedDocument, load_document
from src.review import date_is_ambiguous, review_reasons
from tests.conftest import FIXTURES, NON_INVOICE, VALID_INVOICE


def _pdf_with_text(tmp_path, lines: list[str]) -> LoadedDocument:
    pdf = pymupdf.open()
    page = pdf.new_page()
    for i, line in enumerate(lines):
        page.insert_text((60, 80 + 18 * i), line)
    path = tmp_path / "doc.pdf"
    pdf.save(path)
    return load_document(path)


def _photo_document(tmp_path) -> LoadedDocument:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (60, 60), "white").save(path)
    return load_document(path)


def test_day_above_twelve_is_never_ambiguous():
    assert not date_is_ambiguous("2026-03-25", "")


def test_same_day_and_month_is_not_ambiguous():
    assert not date_is_ambiguous("2026-04-04", "")


def test_low_day_without_text_is_ambiguous():
    assert date_is_ambiguous("2026-03-05", "")


def test_month_name_settles_the_date():
    assert not date_is_ambiguous("2026-03-05", "Invoice date: 5th Mar 2026")
    assert not date_is_ambiguous("2026-03-05", "Date: March 05, 2026")


def test_other_date_proves_day_first():
    text = "Invoice date 05/03/2026  Due date 25/03/2026"
    assert not date_is_ambiguous("2026-03-05", text)


def test_proven_order_contradicting_extraction_stays_ambiguous():
    text = "Invoice date 05/03/2026  Due date 25/03/2026"
    assert date_is_ambiguous("2026-05-03", text)


def test_numeric_date_alone_is_ambiguous():
    assert date_is_ambiguous("2026-03-05", "Invoice date 05/03/2026")


def test_clean_digital_invoice_is_not_flagged(tmp_path):
    doc = _pdf_with_text(tmp_path, ["Invoice ST/2026/007", "Date: 05 Mar 2026", "Total 1652.00"])
    assert review_reasons(doc, copy.deepcopy(VALID_INVOICE)) == []


def test_photo_is_flagged_as_scan(tmp_path):
    reasons = review_reasons(_photo_document(tmp_path), copy.deepcopy(VALID_INVOICE))
    assert "scan_or_photo" in reasons
    assert "ambiguous_date" in reasons


def test_non_invoice_from_photo_is_flagged(tmp_path):
    assert review_reasons(_photo_document(tmp_path), copy.deepcopy(NON_INVOICE)) == [
        "non_invoice_without_text_layer"
    ]


def test_non_invoice_with_text_layer_is_trusted():
    doc = load_document(FIXTURES / "synthetic_invoice.pdf")
    assert review_reasons(doc, copy.deepcopy(NON_INVOICE)) == []


def test_many_rows_and_missing_fields(tmp_path):
    doc = _pdf_with_text(tmp_path, ["Date: 05 Mar 2026"])
    data = copy.deepcopy(VALID_INVOICE)
    data["line_items"] = [{"description": f"r{i}", "quantity": 1, "amount": 1.0} for i in range(16)]
    data["invoice_number"] = None
    reasons = review_reasons(doc, data)
    assert "many_line_items" in reasons
    assert "missing_fields:invoice_number" in reasons
