"""Disk cache of raw model responses, so re-running eval never re-bills."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR

RESPONSES_DIR = CACHE_DIR / "responses"


def cache_key(
    *,
    file_hash: str,
    model: str,
    prompt_version: str,
    prompt_hash: str,
    thinking_level: str,
    temperature: float | None,
    max_output_tokens: int,
    schema_version: str,
) -> str:
    payload = {
        "file_hash": file_hash,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "thinking_level": thinking_level,
        "temperature": "default" if temperature is None else temperature,
        "max_output_tokens": max_output_tokens,
        "schema_version": schema_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_file(key: str) -> Path:
    return RESPONSES_DIR / f"{key}.json"


def load(key: str) -> dict[str, Any] | None:
    path = _cache_file(key)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def save(key: str, entry: dict[str, Any]) -> None:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_file(key).with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, indent=2))
    tmp.replace(_cache_file(key))
