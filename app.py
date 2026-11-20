"""Bonus frontend: uv run --group ui streamlit run app.py

Upload a document, see it on the left and the extracted fields, checks and cost on the right.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf
import streamlit as st

from src import budget
from src.config import CACHE_DIR, baseline_model, primary_model
from src.pipeline import Outcome, extract_document, to_json
from src.schema import FIELDS

UPLOAD_DIR = CACHE_DIR / "uploads"
PREVIEW_DPI = 110
REASON_TEXT = {
    "validation_failed_after_retry": "Totals or sanity checks still fail after one retry",
    "scan_or_photo": "Scan or photo (no text layer): higher misread risk",
    "ambiguous_date": "Day and month could be swapped and the page does not settle it",
    "many_line_items": "More than 15 line items: rows are easy to drop or merge",
    "non_invoice_without_text_layer": "Marked 'not an invoice' but the page has no text layer",
}


def save_upload(name: str, data: bytes) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    path = UPLOAD_DIR / f"{hashlib.sha256(data).hexdigest()[:12]}_{safe_name}"
    path.write_bytes(data)
    return path


def show_preview(name: str, data: bytes) -> None:
    """Best effort: a broken file must not stop the pipeline from reporting why it is broken."""
    try:
        if not name.lower().endswith(".pdf"):
            st.image(data, use_container_width=True)
            return
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            for number, page in enumerate(pdf, 1):
                st.image(page.get_pixmap(dpi=PREVIEW_DPI).tobytes("png"), caption=f"Page {number}")
    except Exception:
        st.warning("Preview unavailable: the file could not be rendered.")


def describe_reason(reason: str) -> str:
    if reason.startswith("missing_fields:"):
        return "Fields left empty by the model: " + reason.split(":", 1)[1].replace(",", ", ")
    return REASON_TEXT.get(reason, reason)


def show_status(outcome: Outcome) -> None:
    meta = outcome.meta
    if meta["processing_error"]:
        error = meta["processing_error"]
        st.error(f"No extraction ({error['type']}): {error['message']}")
    elif outcome.needs_review:
        st.warning("Extracted, but needs human review")
    else:
        st.success("Extracted and passed every check")
    for reason in meta["review_reasons"]:
        st.markdown(f"- {describe_reason(reason)}")
    for issue in meta["validation_errors"]:
        st.caption(issue)


def show_fields(result: dict) -> None:
    if not result["is_invoice"]:
        st.info("Not an invoice. All other fields are null.")
        return
    header = {name: result[name] for name in FIELDS if name != "line_items"}
    st.table({"Field": list(header), "Value": [str(v) for v in header.values()]})
    st.dataframe(result["line_items"], use_container_width=True, hide_index=True)
    items = sum(item["amount"] for item in result["line_items"])
    st.caption(
        f"Line items {items:,.2f} + tax {result['tax_total']:,.2f} = "
        f"{items + result['tax_total']:,.2f} (printed total {result['total']:,.2f})"
    )


def show_usage(meta: dict) -> None:
    source = "from cache" if meta["cache_hit"] else f"{meta['latency_ms'] / 1000:.1f} s"
    st.markdown(
        f"**Cost** ${meta['estimated_cost_usd']:.5f} · **tokens** {meta['input_tokens']:,} in, "
        f"{meta['output_tokens']:,} out, {meta['thinking_tokens']:,} thinking · "
        f"**retries** {meta['retry_count']} · {source}"
    )


def main() -> None:
    st.set_page_config(page_title="Invoice Extractor", layout="wide")
    st.title("Invoice Extractor")
    with st.sidebar:
        model = st.selectbox("Model", [primary_model(), baseline_model()])
        use_cache = st.checkbox("Reuse cached responses", value=True)
        st.caption(f"Spend so far: ${budget.spent_usd():.4f} of ${budget.MAX_BUDGET_USD:.2f}")

    upload = st.file_uploader("Invoice (PDF, JPG or PNG)", type=["pdf", "jpg", "jpeg", "png"])
    if upload is None:
        return
    data = upload.getvalue()
    left, right = st.columns([1, 1], gap="large")
    with left:
        show_preview(upload.name, data)
    with right:
        key = (hashlib.sha256(data).hexdigest(), model, use_cache)
        if st.session_state.get("key") != key:
            with st.spinner("Extracting..."):
                path = save_upload(upload.name, data)
                st.session_state.update(
                    key=key, outcome=extract_document(path, model=model, use_cache=use_cache)
                )
        outcome: Outcome = st.session_state["outcome"]
        show_status(outcome)
        if outcome.result is not None:
            show_fields(outcome.result)
            st.download_button(
                "Download JSON", to_json(outcome.result), file_name=f"{Path(upload.name).stem}.json"
            )
        show_usage(outcome.meta)
        with st.expander("Raw metadata"):
            st.json(outcome.meta)


main()
