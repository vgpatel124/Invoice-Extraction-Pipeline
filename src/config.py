"""Environment loading, model ids and token prices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CACHE_DIR = PROJECT_ROOT / ".cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
META_DIR = OUTPUTS_DIR / "_meta"

MAX_PAGES = 10
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_TOKENS = 8192
REQUEST_TIMEOUT_MS = 300_000
THINKING_LEVELS: dict[str, tuple[str, str]] = {"gemini-3.5-flash-lite": ("minimal", "low")}
DEFAULT_THINKING_LEVELS = ("low", "medium")

PRICES_SOURCE_URL = "https://aistudio.google.com/docs/pricing"
PRICES_CHECKED_ON = "2026-11-20"


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


@dataclass(frozen=True)
class ModelPricing:
    current: Price
    later: Price | None = None
    later_from: date | None = None

    def price_on(self, day: date) -> Price:
        if self.later and self.later_from and day >= self.later_from:
            return self.later
        return self.current


MODEL_PRICES: dict[str, ModelPricing] = {
    "gemini-3.8-flash": ModelPricing(
        current=Price(0.75, 3.75),
        later=Price(1.50, 7.50),
        later_from=date(2027, 1, 1),
    ),
    "gemini-3.5-flash-lite": ModelPricing(current=Price(0.30, 2.50)),
}


def price_for(model: str, day: date | None = None) -> Price:
    if model not in MODEL_PRICES:
        raise KeyError(f"No price configured for model {model!r}; add it to MODEL_PRICES")
    return MODEL_PRICES[model].price_on(day or date.today())


def thinking_levels(model: str) -> tuple[str, str]:
    return THINKING_LEVELS.get(model, DEFAULT_THINKING_LEVELS)


def gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or None


def primary_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def baseline_model() -> str:
    return os.getenv("GEMINI_BASELINE_MODEL", "gemini-3.5-flash-lite")


def data_dir() -> Path:
    raw = Path(os.getenv("DATA_DIR", "./data"))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw
