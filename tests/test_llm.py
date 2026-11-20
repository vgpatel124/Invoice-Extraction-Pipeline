from __future__ import annotations

from google.genai import errors

from src.llm import _is_daily_quota, _is_retryable


def _client_error(code: int, message: str) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"code": code, "message": message}})


def test_per_minute_rate_limit_is_retried():
    assert _is_retryable(_client_error(429, "GenerateRequestsPerMinutePerProjectPerModel"))


def test_daily_quota_is_not_retried():
    exc = _client_error(429, "quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert _is_daily_quota(exc)
    assert not _is_retryable(exc)


def test_server_errors_and_timeouts_are_retried():
    assert _is_retryable(errors.ServerError(503, {"error": {"message": "high demand"}}))
    assert _is_retryable(TimeoutError())


def test_bad_request_is_not_retried():
    assert not _is_retryable(_client_error(400, "INVALID_ARGUMENT"))
