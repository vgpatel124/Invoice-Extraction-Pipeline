"""Output models mirroring schema.json, plus a check against the official JSON Schema."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

SCHEMA_PATH = Path(__file__).with_name("schema.json")
SCHEMA_VERSION = "pack-1"
FIELDS = (
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
)


class LineItem(BaseModel):
    description: str
    quantity: float | None
    amount: float


class InvoiceExtraction(BaseModel):
    """Kept Gemini-friendly: plain fields, Optional for nullable, no custom validators.

    No extra="forbid" here: it emits additionalProperties, which Gemini's response_schema
    rejects. Extra keys are still caught by the jsonschema check against schema.json.
    """

    is_invoice: bool
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None
    currency: str | None
    line_items: list[LineItem] | None
    tax_total: float | None
    total: float | None


@cache
def _schema_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def schema_errors(data: Any) -> list[str]:
    """Return every Pydantic and JSON Schema error for `data` (empty list means valid)."""
    errors: list[str] = []
    try:
        InvoiceExtraction.model_validate(data, strict=False)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            errors.append(f"pydantic: {loc}: {err['msg']}")
    for err in sorted(_schema_validator().iter_errors(data), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        errors.append(f"jsonschema: {loc}: {err.message}")
    return errors
