from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from src.inputs import InputError, load_document
from tests.conftest import FIXTURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("name", "error_type"),
    [
        ("empty.pdf", "empty_file"),
        ("truncated.pdf", "corrupt_file"),
        ("not_an_image.jpg", "unsupported_file"),
        ("eleven_pages.pdf", "too_many_pages"),
    ],
)
def test_bad_fixture_is_rejected(name, error_type):
    with pytest.raises(InputError) as exc_info:
        load_document(FIXTURES / name)
    assert exc_info.value.type == error_type


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(InputError) as exc_info:
        load_document(tmp_path / "nope.pdf")
    assert exc_info.value.type == "file_not_found"


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "invoice.docx"
    path.write_bytes(b"PK\x03\x04")
    with pytest.raises(InputError) as exc_info:
        load_document(path)
    assert exc_info.value.type == "unsupported_file"


def test_pdf_renamed_to_jpg_is_rejected(tmp_path):
    path = tmp_path / "invoice.jpg"
    path.write_bytes((FIXTURES / "synthetic_invoice.pdf").read_bytes())
    with pytest.raises(InputError, match="does not match"):
        load_document(path)


def test_truncated_jpeg_is_rejected(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (400, 400), "white").save(buffer, format="JPEG")
    path = tmp_path / "photo.jpg"
    path.write_bytes(buffer.getvalue()[:200])
    with pytest.raises(InputError) as exc_info:
        load_document(path)
    assert exc_info.value.type == "corrupt_file"


def test_digital_pdf_has_text_layer():
    doc = load_document(FIXTURES / "synthetic_invoice.pdf")
    assert doc.mime_type == "application/pdf"
    assert doc.page_count == 1
    assert doc.has_text_layer


def test_image_only_pdf_has_no_text_layer(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buffer, format="PNG")
    pdf = pymupdf.open()
    pdf.new_page().insert_image(pymupdf.Rect(0, 0, 200, 200), stream=buffer.getvalue())
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    assert not load_document(path).has_text_layer


def test_valid_png_loads(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (50, 50), "white").save(path)
    doc = load_document(path)
    assert doc.is_image and doc.mime_type == "image/png" and not doc.has_text_layer


def test_cli_bad_input_contract():
    """Exit 1, nothing on stdout, error JSON on stderr, no API call needed."""
    proc = subprocess.run(
        [sys.executable, "extract.py", str(FIXTURES / "empty.pdf")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert json.loads(proc.stderr)["error"]["type"] == "empty_file"
