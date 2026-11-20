"""File loading and bad-input detection, all before any API call."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, UnidentifiedImageError

from src.config import MAX_FILE_BYTES, MAX_PAGES

pymupdf.TOOLS.mupdf_display_errors(False)

PDF_MAGIC = b"%PDF-"
IMAGE_MAGIC = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG\r\n\x1a\n": "image/png"}
EXTENSION_MIME = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}
MIN_TEXT_LAYER_CHARS = 20


class InputError(Exception):
    """A document that cannot be processed. `type` is a stable machine-readable code."""

    def __init__(self, type_: str, message: str) -> None:
        super().__init__(message)
        self.type = type_
        self.message = message


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    data: bytes
    mime_type: str
    file_hash: str
    page_count: int
    has_text_layer: bool

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")


def detect_file_type(data: bytes) -> str | None:
    if data.startswith(PDF_MAGIC):
        return "application/pdf"
    for magic, mime in IMAGE_MAGIC.items():
        if data.startswith(magic):
            return mime
    return None


def _check_pdf(data: bytes) -> tuple[int, bool]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InputError("corrupt_file", f"PDF could not be opened: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise InputError("unsupported_file", "PDF is password protected")
        if doc.page_count == 0:
            raise InputError("corrupt_file", "PDF has no readable pages")
        if doc.page_count > MAX_PAGES:
            raise InputError(
                "too_many_pages", f"PDF has {doc.page_count} pages; the limit is {MAX_PAGES}"
            )
        text_chars = 0
        try:
            for page in doc:
                text_chars += len(page.get_text().strip())
        except Exception as exc:
            raise InputError("corrupt_file", f"PDF page could not be read: {exc}") from exc
        if doc.is_repaired and text_chars == 0:
            _check_pages_render(doc)
        return doc.page_count, text_chars >= MIN_TEXT_LAYER_CHARS


def _check_pages_render(doc: pymupdf.Document) -> None:
    try:
        for page in doc:
            page.get_pixmap(matrix=pymupdf.Matrix(0.2, 0.2))
    except Exception as exc:
        raise InputError("corrupt_file", f"PDF is damaged and cannot be rendered: {exc}") from exc


def _check_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InputError("corrupt_file", f"Image could not be decoded: {exc}") from exc


def extract_text(doc: LoadedDocument) -> str:
    """Text layer of a PDF via PyMuPDF (empty for images and scans). Experiment only."""
    if doc.is_image:
        return ""
    with pymupdf.open(stream=doc.data, filetype="pdf") as pdf:
        pages = [f"--- page {i} ---\n{page.get_text()}" for i, page in enumerate(pdf, 1)]
    return "\n".join(pages) if doc.has_text_layer else ""


def load_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    if not path.is_file():
        raise InputError("file_not_found", f"No such file: {path}")
    expected_mime = EXTENSION_MIME.get(path.suffix.lower())
    if expected_mime is None:
        raise InputError("unsupported_file", f"Unsupported extension {path.suffix!r}")
    data = path.read_bytes()
    if not data:
        raise InputError("empty_file", "File is empty (0 bytes)")
    if len(data) > MAX_FILE_BYTES:
        raise InputError("file_too_large", f"File is {len(data)} bytes; limit is {MAX_FILE_BYTES}")
    actual_mime = detect_file_type(data)
    if actual_mime is None:
        raise InputError("unsupported_file", "File content is not a PDF, JPEG or PNG")
    if actual_mime != expected_mime:
        raise InputError(
            "unsupported_file",
            f"Extension {path.suffix!r} does not match file content ({actual_mime})",
        )

    if actual_mime == "application/pdf":
        page_count, has_text_layer = _check_pdf(data)
    else:
        _check_image(data)
        page_count, has_text_layer = 1, False

    return LoadedDocument(
        path=path,
        data=data,
        mime_type=actual_mime,
        file_hash=hashlib.sha256(data).hexdigest(),
        page_count=page_count,
        has_text_layer=has_text_layer,
    )
