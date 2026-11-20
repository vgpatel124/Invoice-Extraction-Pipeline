from __future__ import annotations

from datetime import date

import pytest

from src import budget
from src import cache as response_cache
from src.config import price_for


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "LEDGER_PATH", tmp_path / "spend.json")
    monkeypatch.setattr(response_cache, "RESPONSES_DIR", tmp_path / "responses")


def test_thinking_tokens_are_billed_as_output():
    price = price_for("gemini-3.8-flash")
    expected = (1000 * price.input_per_m + (200 + 300) * price.output_per_m) / 1e6
    assert budget.cost_usd("gemini-3.8-flash", 1000, 200, 300) == pytest.approx(expected)


def test_flash_price_rises_in_2027():
    assert price_for("gemini-3.8-flash", date(2026, 12, 31)).input_per_m == 0.75
    assert price_for("gemini-3.8-flash", date(2027, 1, 1)).input_per_m == 1.50
    assert price_for("gemini-3.8-flash", date(2027, 1, 1)).output_per_m == 7.50


def test_unknown_model_has_no_price():
    with pytest.raises(KeyError):
        price_for("some-other-model")


def test_ledger_accumulates_and_persists():
    budget.record_spend(0.25, {"document": "a"})
    budget.record_spend(0.50, {"document": "b"})
    assert budget.spent_usd() == pytest.approx(0.75)
    assert budget.remaining_usd() == pytest.approx(budget.MAX_BUDGET_USD - 0.75)


def test_call_over_remaining_budget_is_refused():
    budget.record_spend(budget.MAX_BUDGET_USD - 0.01, {"document": "a"})
    budget.check_budget(0.005)
    with pytest.raises(budget.BudgetExceeded):
        budget.check_budget(0.02)


def make_cache_key(**overrides) -> str:
    params = {
        "file_hash": "abc",
        "model": "gemini-3.8-flash",
        "prompt_version": "v1",
        "prompt_hash": "p1",
        "thinking_level": "low",
        "temperature": None,
        "max_output_tokens": 8192,
        "schema_version": "pack-1",
    }
    return response_cache.cache_key(**{**params, **overrides})


def test_cache_key_is_stable():
    assert make_cache_key() == make_cache_key()


@pytest.mark.parametrize(
    "override",
    [
        {"file_hash": "def"},
        {"model": "gemini-3.5-flash-lite"},
        {"prompt_version": "v2"},
        {"prompt_hash": "p1-with-retry-feedback"},
        {"thinking_level": "medium"},
        {"temperature": 0.5},
        {"max_output_tokens": 4096},
        {"schema_version": "pack-2"},
    ],
)
def test_every_input_changes_the_cache_key(override):
    assert make_cache_key(**override) != make_cache_key()


def test_cache_round_trip():
    assert response_cache.load("k") is None
    response_cache.save("k", {"text": "{}", "cost_usd": 0.001})
    assert response_cache.load("k") == {"text": "{}", "cost_usd": 0.001}
