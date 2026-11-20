"""Regenerate the bad-input and synthetic test fixtures (none come from the sample pack).

uv run python tests/make_fixtures.py [output_dir]   # default: tests/fixtures
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

SYNTHETIC_LINES = [
    "Synthetic Test Traders",
    "TAX INVOICE",
    "Invoice No: ST/2026/007    Date: 05/03/2026",
    "Bill To: Example Buyer Co",
    "",
    "Description            Qty    Amount (Rs)",
    "Test widget A           2      1,000.00",
    "Test service B          -        500.00",
    "Loyalty discount        -       -100.00",
    "",
    "Subtotal                        1,400.00",
    "CGST 9%                           126.00",
    "SGST 9%                           126.00",
    "Grand Total                     1,652.00",
]


def synthetic_invoice() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(SYNTHETIC_LINES):
        page.insert_text((60, 70 + 18 * i), line, fontname="cour", fontsize=10)
    return doc.tobytes()


def text_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "INVOICE INV-1 Widget 100.00 Total 100.00" * 2)
    return doc.tobytes()


def blank_pdf(pages: int) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    return doc.tobytes()


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    full = text_pdf()
    files = {
        "empty.pdf": b"",
        "truncated.pdf": full[: len(full) // 3],
        "not_an_image.jpg": b"This is a plain text file renamed to .jpg\n",
        "eleven_pages.pdf": blank_pdf(11),
        "synthetic_invoice.pdf": synthetic_invoice(),
    }
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        print(f"wrote {out_dir / name} ({len(data)} bytes)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fixtures")
