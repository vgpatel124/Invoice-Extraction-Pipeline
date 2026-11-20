"""Retry and flagging logic, with the model call replaced by scripted responses."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from src import budget, pipeline
from src.llm import CallResult, LLMError
from tests.conftest import FIXTURES

DOC = FIXTURES / "synthetic_invoice.pdf"


def fake_call_result(text: str, level: str) -> CallResult:
    return CallResult(
        text=text,
        model="gemini-3.8-flash",
        thinking_level=level,
        input_tokens=1000,
        output_tokens=100,
        thinking_tokens=50,
        latency_ms=10,
        cost_usd=0.001,
        cache_hit=False,
        finish_reason="STOP",
    )


@pytest.fixture(autouse=True)
def no_heuristics(monkeypatch):
    """Retry tests check validation logic only; heuristics are tested in test_review.py."""
    monkeypatch.setattr(pipeline, "review_reasons", lambda doc, data: [])


@pytest.fixture
def scripted(monkeypatch) -> Callable[[list], list[dict]]:
    """Replace call_model with a list of responses (str) or exceptions; return the call log."""

    def install(responses: list) -> list[dict]:
        calls: list[dict] = []

        def fake_call_model(doc, *, model, system, user_messages, thinking_level, **_):
            calls.append({"messages": user_messages, "thinking_level": thinking_level})
            item = responses[len(calls) - 1]
            if isinstance(item, Exception):
                raise item
            return fake_call_result(item, thinking_level)

        monkeypatch.setattr(pipeline, "call_model", fake_call_model)
        return calls

    return install


def test_clean_first_attempt_does_not_retry(scripted, invoice):
    calls = scripted([json.dumps(invoice)])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result == invoice
    assert len(calls) == 1
    assert outcome.meta["retry_count"] == 0
    assert not outcome.needs_review
    assert outcome.meta["estimated_cost_usd"] == pytest.approx(0.001)


def test_totals_failure_retries_with_numbers_and_higher_thinking(scripted, invoice):
    bad = {**invoice, "tax_total": 0}
    calls = scripted([json.dumps(bad), json.dumps(invoice)])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result == invoice
    assert not outcome.needs_review
    assert outcome.meta["retry_count"] == 1
    assert [c["thinking_level"] for c in calls] == ["low", "medium"]
    feedback = calls[1]["messages"][-1]
    assert "1,400.00" in feedback and "1,652.00" in feedback
    assert outcome.meta["input_tokens"] == 2000


def test_totals_still_failing_is_kept_but_flagged(scripted, invoice):
    bad = {**invoice, "tax_total": 0}
    scripted([json.dumps(bad), json.dumps(bad)])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result == bad
    assert outcome.needs_review
    assert outcome.meta["processing_error"] is None
    assert "validation_failed_after_retry" in outcome.meta["review_reasons"]
    assert any("totals" in e for e in outcome.meta["validation_errors"])


def test_schema_invalid_twice_is_a_processing_error(scripted):
    scripted(['{"is_invoice": "maybe"}', "not json"])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result is None
    assert outcome.meta["processing_error"]["type"] == "schema_invalid"
    assert not outcome.needs_review


def test_schema_invalid_then_flagged_keeps_valid_attempt(scripted, invoice):
    bad_totals = {**invoice, "total": 9999.0}
    scripted(["not json", json.dumps(bad_totals)])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result == bad_totals
    assert outcome.needs_review


def test_retry_that_gets_worse_keeps_better_attempt(scripted, invoice):
    one_issue = {**invoice, "tax_total": 0}
    two_issues = {**invoice, "tax_total": 0, "invoice_date": "2026-02-30"}
    scripted([json.dumps(one_issue), json.dumps(two_issues)])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result == one_issue


def test_non_invoice_passes_through(scripted, non_invoice):
    scripted([json.dumps(non_invoice)])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result == non_invoice
    assert not outcome.needs_review


def test_output_keys_follow_schema_order(scripted, invoice):
    shuffled = dict(reversed(list(invoice.items())))
    scripted([json.dumps(shuffled)])
    assert list(pipeline.extract_document(DOC).result) == list(invoice)


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (LLMError("api_error", "503 after backoff"), "api_error"),
        (budget.BudgetExceeded("over"), "budget_exceeded"),
    ],
)
def test_call_failures_become_processing_errors(scripted, error, error_type):
    scripted([error])
    outcome = pipeline.extract_document(DOC)
    assert outcome.result is None
    assert outcome.meta["processing_error"]["type"] == error_type


def test_bad_input_never_calls_the_model(scripted):
    calls = scripted([])
    outcome = pipeline.extract_document(FIXTURES / "empty.pdf")
    assert calls == []
    assert outcome.meta["processing_error"]["type"] == "empty_file"


def test_text_mode_sends_extracted_text_without_the_file(monkeypatch, invoice):
    seen: dict = {}

    def fake_call_model(doc, *, user_messages, attach_document, thinking_level, **_):
        seen.update(messages=user_messages, attach=attach_document)
        return fake_call_result(json.dumps(invoice), thinking_level)

    monkeypatch.setattr(pipeline, "call_model", fake_call_model)
    pipeline.extract_document(DOC, input_mode="text")
    assert seen["attach"] is False
    assert "Synthetic Test Traders" in seen["messages"][0]


def test_flash_lite_keeps_minimal_thinking(scripted, invoice):
    calls = scripted([json.dumps({**invoice, "tax_total": 0}), json.dumps(invoice)])
    pipeline.extract_document(DOC, model="gemini-3.5-flash-lite")
    assert [c["thinking_level"] for c in calls] == ["minimal", "low"]


def test_heuristic_reasons_set_needs_review(scripted, invoice, monkeypatch):
    monkeypatch.setattr(pipeline, "review_reasons", lambda doc, data: ["scan_or_photo"])
    scripted([json.dumps(invoice)])
    outcome = pipeline.extract_document(DOC)
    assert outcome.needs_review
    assert outcome.meta["review_reasons"] == ["scan_or_photo"]
    assert outcome.meta["validation_errors"] == []


def test_heuristics_never_apply_to_processing_errors(scripted, monkeypatch):
    monkeypatch.setattr(pipeline, "review_reasons", lambda doc, data: ["scan_or_photo"])
    scripted(["not json", "still not json"])
    outcome = pipeline.extract_document(DOC)
    assert outcome.meta["processing_error"] is not None
    assert not outcome.needs_review
