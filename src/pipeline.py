"""End-to-end extraction for one document: load, call, validate, retry once, decide."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import budget
from src.config import primary_model, thinking_levels
from src.inputs import InputError, LoadedDocument, extract_text, load_document
from src.llm import CallResult, LLMError, call_model
from src.prompt import PROMPT_VERSION, USER_INSTRUCTION, retry_message, system_prompt
from src.review import review_reasons
from src.schema import FIELDS
from src.validate import CheckResult, run_checks

INPUT_MODES = ("native", "text")
TEXT_HEADER = "DOCUMENT TEXT (extracted from the PDF text layer):\n"
NO_TEXT = "(No text could be extracted from this document.)"


@dataclass
class Outcome:
    """`result` is schema-valid invoice JSON, or None when processing_error is set."""

    result: dict[str, Any] | None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return bool(self.meta.get("needs_review"))


def _attempt_summary(call: CallResult, checked: CheckResult) -> dict[str, Any]:
    return {
        "model": call.model,
        "thinking_level": call.thinking_level,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "thinking_tokens": call.thinking_tokens,
        "cost_usd": round(call.cost_usd, 6),
        "latency_ms": call.latency_ms,
        "cache_hit": call.cache_hit,
        "finish_reason": call.finish_reason,
        "errors": [str(i) for i in checked.issues],
    }


def _empty_meta(
    path: Path, model: str, doc: LoadedDocument | None, input_mode: str
) -> dict[str, Any]:
    return {
        "document": path.name,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "input_mode": input_mode,
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "latency_ms": 0,
        "retry_count": 0,
        "cache_hit": False,
        "estimated_cost_usd": 0.0,
        "spent_this_run_usd": 0.0,
        "has_text_layer": doc.has_text_layer if doc else None,
        "page_count": doc.page_count if doc else None,
        "mime_type": doc.mime_type if doc else None,
        "needs_review": False,
        "review_reasons": [],
        "validation_errors": [],
        "processing_error": None,
        "attempts": [],
    }


def _total_usage(meta: dict[str, Any], calls: list[CallResult]) -> None:
    meta["input_tokens"] = sum(c.input_tokens for c in calls)
    meta["output_tokens"] = sum(c.output_tokens for c in calls)
    meta["thinking_tokens"] = sum(c.thinking_tokens for c in calls)
    meta["latency_ms"] = sum(c.latency_ms for c in calls)
    meta["estimated_cost_usd"] = round(sum(c.cost_usd for c in calls), 6)
    meta["spent_this_run_usd"] = round(sum(c.cost_usd for c in calls if not c.cache_hit), 6)
    meta["cache_hit"] = bool(calls) and all(c.cache_hit for c in calls)
    meta["retry_count"] = max(len(calls) - 1, 0)


def _best_attempt(attempts: list[CheckResult]) -> CheckResult | None:
    """Schema-valid attempt with the fewest issues; the later attempt wins ties."""
    valid = [a for a in attempts if a.schema_valid]
    if not valid:
        return None
    return min(reversed(valid), key=lambda a: len(a.issues))


def _in_schema_order(data: dict[str, Any]) -> dict[str, Any]:
    return {name: data[name] for name in FIELDS}


def _text_input(doc: LoadedDocument, input_mode: str) -> list[str]:
    """Native mode attaches the file itself; text mode sends PyMuPDF text instead (experiment)."""
    if input_mode == "native":
        return []
    text = extract_text(doc)
    return [TEXT_HEADER + text if text.strip() else NO_TEXT]


def extract_document(
    path: str | Path,
    *,
    model: str | None = None,
    use_cache: bool = True,
    input_mode: str = "native",
) -> Outcome:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    path = Path(path)
    model = model or primary_model()
    try:
        doc = load_document(path)
    except InputError as exc:
        meta = _empty_meta(path, model, None, input_mode)
        meta["processing_error"] = {"type": exc.type, "message": exc.message}
        return Outcome(None, meta)

    meta = _empty_meta(path, model, doc, input_mode)
    system = system_prompt()
    document = _text_input(doc, input_mode)
    messages = [*document, USER_INSTRUCTION]
    calls: list[CallResult] = []
    checks: list[CheckResult] = []

    for thinking_level in thinking_levels(model):
        try:
            call = call_model(
                doc,
                model=model,
                system=system,
                user_messages=messages,
                thinking_level=thinking_level,
                use_cache=use_cache,
                attach_document=input_mode == "native",
            )
        except (LLMError, budget.BudgetExceeded) as exc:
            error_type = exc.type if isinstance(exc, LLMError) else "budget_exceeded"
            _total_usage(meta, calls)
            meta["processing_error"] = {"type": error_type, "message": str(exc)}
            return Outcome(None, meta)
        checked = run_checks(call.text)
        calls.append(call)
        checks.append(checked)
        meta["attempts"].append(_attempt_summary(call, checked))
        if checked.clean:
            break
        previous = call.text if len(call.text) < 20_000 else call.text[:20_000] + "..."
        feedback = retry_message([str(i) for i in checked.issues], previous)
        messages = [*document, USER_INSTRUCTION, feedback]

    _total_usage(meta, calls)
    best = _best_attempt(checks)
    if best is None:
        meta["validation_errors"] = [str(i) for i in checks[-1].issues]
        meta["processing_error"] = {
            "type": "schema_invalid",
            "message": "model output failed schema validation after retry",
        }
        return Outcome(None, meta)

    if best.issues:
        meta["validation_errors"] = [str(i) for i in best.issues]
        meta["review_reasons"].append("validation_failed_after_retry")
    meta["review_reasons"] += review_reasons(doc, best.data)
    meta["needs_review"] = bool(meta["review_reasons"])
    return Outcome(_in_schema_order(best.data), meta)


def to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
