"""Hard spend cap with a persistent ledger. Checked before every API call, retries included."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src.config import CACHE_DIR, price_for

MAX_BUDGET_USD = 2.00
LEDGER_PATH = CACHE_DIR / "spend.json"


class BudgetExceeded(Exception):
    pass


def cost_usd(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    """Thinking tokens are billed at the output rate."""
    price = price_for(model)
    billed_output = output_tokens + thinking_tokens
    return (input_tokens * price.input_per_m + billed_output * price.output_per_m) / 1_000_000


def worst_case_cost(model: str, input_tokens: int, max_output_tokens: int) -> float:
    return cost_usd(model, input_tokens, max_output_tokens, 0)


def _load_ledger() -> dict[str, Any]:
    if LEDGER_PATH.is_file():
        return json.loads(LEDGER_PATH.read_text())
    return {"total_usd": 0.0, "calls": []}


def spent_usd() -> float:
    return float(_load_ledger()["total_usd"])


def remaining_usd() -> float:
    return MAX_BUDGET_USD - spent_usd()


def check_budget(estimated_usd: float) -> None:
    remaining = remaining_usd()
    if estimated_usd > remaining:
        raise BudgetExceeded(
            f"Call could cost up to ${estimated_usd:.4f} but only ${remaining:.4f} of the "
            f"${MAX_BUDGET_USD:.2f} budget remains"
        )


def record_spend(cost: float, details: dict[str, Any]) -> None:
    ledger = _load_ledger()
    ledger["total_usd"] = round(ledger["total_usd"] + cost, 8)
    ledger["calls"].append(
        {"at": datetime.now(UTC).isoformat(timespec="seconds"), "cost_usd": cost, **details}
    )
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    tmp.replace(LEDGER_PATH)
