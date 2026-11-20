"""The one place that talks to the model provider (Gemini via google-genai)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cache

from google import genai
from google.genai import errors, types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src import budget
from src import cache as response_cache
from src.config import MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_MS, gemini_api_key
from src.inputs import LoadedDocument
from src.prompt import PROMPT_VERSION, prompt_hash
from src.schema import SCHEMA_VERSION, InvoiceExtraction

API_ATTEMPTS = 4
TEMPERATURE: float | None = None


class LLMError(Exception):
    """API failure after backoff, or missing configuration."""

    def __init__(self, type_: str, message: str) -> None:
        super().__init__(message)
        self.type = type_
        self.message = message


@dataclass(frozen=True)
class CallResult:
    text: str
    model: str
    thinking_level: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_ms: int
    cost_usd: float
    cache_hit: bool
    finish_reason: str | None


@cache
def _get_client() -> genai.Client:
    key = gemini_api_key()
    if not key:
        raise LLMError("config_error", "GEMINI_API_KEY is not set; copy .env.example to .env")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def _is_network_error(exc: BaseException) -> bool:
    """Timeouts and connection failures from the SDK's HTTP layer (httpx/httpx2) or the OS."""
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    return any(cls.__name__ == "TransportError" for cls in type(exc).__mro__)


def _is_daily_quota(exc: BaseException) -> bool:
    """Free-tier per-day quota: retrying within minutes cannot succeed."""
    return isinstance(exc, errors.ClientError) and exc.code == 429 and "PerDay" in str(exc)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, errors.ServerError) or _is_network_error(exc):
        return True
    is_rate_limit = isinstance(exc, errors.ClientError) and exc.code == 429
    return is_rate_limit and not _is_daily_quota(exc)


def _estimate_input_tokens(model: str, contents: list, system: str, doc: LoadedDocument) -> int:
    try:
        counted = _get_client().models.count_tokens(model=model, contents=contents).total_tokens
        return int(counted or 0) + len(system) // 3
    except errors.APIError:
        return doc.page_count * 2000 + (len(system) + 2000) // 3


def call_model(
    doc: LoadedDocument,
    *,
    model: str,
    system: str,
    user_messages: list[str],
    thinking_level: str,
    use_cache: bool = True,
    attach_document: bool = True,
) -> CallResult:
    """Send the document and instructions; return the raw JSON text plus usage."""
    key = response_cache.cache_key(
        file_hash=doc.file_hash,
        model=model,
        prompt_version=PROMPT_VERSION,
        prompt_hash=prompt_hash(
            system, *user_messages, *([] if attach_document else ["text-only"])
        ),
        thinking_level=thinking_level,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        schema_version=SCHEMA_VERSION,
    )
    if use_cache and (hit := response_cache.load(key)):
        return CallResult(**{**hit, "cache_hit": True})

    contents: list = []
    if attach_document:
        contents.append(types.Part.from_bytes(data=doc.data, mime_type=doc.mime_type))
    contents.extend(user_messages)
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=InvoiceExtraction,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(API_ATTEMPTS),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        reraise=True,
    )
    def _send_request() -> tuple[types.GenerateContentResponse, int]:
        estimate = budget.worst_case_cost(
            model, _estimate_input_tokens(model, contents, system, doc), MAX_OUTPUT_TOKENS
        )
        budget.check_budget(estimate)
        start = time.perf_counter()
        response = _get_client().models.generate_content(
            model=model, contents=contents, config=config
        )
        return response, int((time.perf_counter() - start) * 1000)

    try:
        response, latency_ms = _send_request()
    except budget.BudgetExceeded:
        raise
    except errors.APIError as exc:
        error_type = "quota_exhausted" if _is_daily_quota(exc) else "api_error"
        raise LLMError(error_type, f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        if not _is_network_error(exc):
            raise
        raise LLMError("api_timeout", f"{type(exc).__name__}: {exc}") from exc

    usage = response.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = (usage.candidates_token_count or 0) if usage else 0
    thinking_tokens = (usage.thoughts_token_count or 0) if usage else 0
    cost = budget.cost_usd(model, input_tokens, output_tokens, thinking_tokens)
    budget.record_spend(
        cost,
        {
            "document": doc.path.name,
            "model": model,
            "thinking_level": thinking_level,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
        },
    )

    candidate = response.candidates[0] if response.candidates else None
    finish = candidate.finish_reason.name if candidate and candidate.finish_reason else None
    entry = {
        "text": response.text or "",
        "model": model,
        "thinking_level": thinking_level,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "finish_reason": finish,
    }
    response_cache.save(key, entry)
    return CallResult(**entry, cache_hit=False)
