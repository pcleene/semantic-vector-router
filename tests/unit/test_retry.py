"""Unit tests for the retry decorator and helpers.

Tests with_retry (async + sync), async_retry, exponential backoff, jitter,
Retry-After header handling, retryable exception filtering, and HTTP status checks.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx

from semantic_vector_router.utils.retry import (
    with_retry,
    async_retry,
    _is_retryable_http_status,
    _get_retry_after,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_status_error(status_code: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError with the given status code and optional headers."""
    response = httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("GET", "https://example.com"),
    )
    return httpx.HTTPStatusError(
        message=f"HTTP {status_code}",
        request=response.request,
        response=response,
    )


# ===========================================================================
# 1. test_async_retry_succeeds_on_first_try
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_async_retry_succeeds_on_first_try(mock_sleep):
    """When the function succeeds on the first call, no retry or sleep occurs."""

    @with_retry(max_attempts=3, base_delay=1.0)
    async def always_ok():
        return "ok"

    result = await always_ok()
    assert result == "ok"
    mock_sleep.assert_not_called()


# ===========================================================================
# 2. test_async_retry_succeeds_on_second_attempt
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_async_retry_succeeds_on_second_attempt(mock_sleep):
    """Function fails once then succeeds; asyncio.sleep is called exactly once."""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=1.0)
    async def fail_then_ok():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")
        return "recovered"

    result = await fail_then_ok()
    assert result == "recovered"
    assert call_count == 2
    assert mock_sleep.call_count == 1


# ===========================================================================
# 3. test_async_retry_exhausts_max_attempts_raises_last_exception
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_async_retry_exhausts_max_attempts_raises_last_exception(mock_sleep):
    """After all attempts fail the last exception is re-raised."""
    attempt = 0

    @with_retry(max_attempts=3, base_delay=0.1)
    async def always_fail():
        nonlocal attempt
        attempt += 1
        raise ValueError(f"fail-{attempt}")

    with pytest.raises(ValueError, match="fail-3"):
        await always_fail()

    # 3 attempts, 2 sleeps (no sleep after the final failure)
    assert attempt == 3
    assert mock_sleep.call_count == 2


# ===========================================================================
# 4. test_sync_retry_succeeds_on_second_attempt
# ===========================================================================

@patch("semantic_vector_router.utils.retry.time.sleep")
def test_sync_retry_succeeds_on_second_attempt(mock_sleep):
    """Sync wrapper: fails once, succeeds second time, time.sleep called once."""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=1.0)
    def fail_then_ok():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")
        return "recovered"

    result = fail_then_ok()
    assert result == "recovered"
    assert call_count == 2
    assert mock_sleep.call_count == 1


# ===========================================================================
# 5. test_exponential_backoff_delays
# ===========================================================================

@patch("semantic_vector_router.utils.retry.random.random", return_value=0.5)
@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_exponential_backoff_delays(mock_sleep, mock_random):
    """Verify delay pattern: base_delay * 2^attempt with jitter=0 when random()=0.5."""
    # When random() returns 0.5: jitter = delay * 0.25 * (2*0.5 - 1) = 0
    # So delays are exactly: base_delay*1, base_delay*2, base_delay*4

    attempt = 0

    @with_retry(max_attempts=4, base_delay=1.0, max_delay=100.0)
    async def always_fail():
        nonlocal attempt
        attempt += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        await always_fail()

    assert mock_sleep.call_count == 3
    delays = [c.args[0] for c in mock_sleep.call_args_list]
    assert delays == pytest.approx([1.0, 2.0, 4.0])


# ===========================================================================
# 6. test_jitter_within_bounds
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_jitter_within_bounds(mock_sleep):
    """Run multiple retries and verify jitter stays within +/-25% of the base delay."""
    # We'll use max_attempts=2 so there is exactly 1 sleep per call.
    # The nominal delay at attempt 0 is base_delay * 2^0 = 2.0
    # Jitter range: 2.0 +/- 0.5 => [1.5, 2.5]

    delays_seen = []

    async def _capture_delay(d):
        delays_seen.append(d)

    mock_sleep.side_effect = _capture_delay

    for _ in range(50):
        attempt = 0

        @with_retry(max_attempts=2, base_delay=2.0, max_delay=100.0)
        async def fail_once():
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise RuntimeError("transient")
            return "ok"

        await fail_once()

    assert len(delays_seen) == 50
    for d in delays_seen:
        assert 1.5 <= d <= 2.5, f"delay {d} outside jitter bounds [1.5, 2.5]"


# ===========================================================================
# 7. test_max_delay_cap
# ===========================================================================

@patch("semantic_vector_router.utils.retry.random.random", return_value=0.5)
@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_max_delay_cap(mock_sleep, mock_random):
    """Delay should never exceed max_delay (even before jitter, it is capped by min())."""
    # base_delay=1.0, max_delay=3.0
    # attempt 0: min(1*1, 3)=1.0, attempt 1: min(1*2, 3)=2.0, attempt 2: min(1*4, 3)=3.0
    # With random()=0.5, jitter=0

    attempt = 0

    @with_retry(max_attempts=4, base_delay=1.0, max_delay=3.0)
    async def always_fail():
        nonlocal attempt
        attempt += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        await always_fail()

    delays = [c.args[0] for c in mock_sleep.call_args_list]
    assert all(d <= 3.0 for d in delays), f"Delays {delays} exceed max_delay=3.0"
    # The third delay should be exactly 3.0 (capped)
    assert delays[2] == pytest.approx(3.0)


# ===========================================================================
# 8. test_retryable_exceptions_filter
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_retryable_exceptions_filter(mock_sleep):
    """Non-retryable exception type is raised immediately without retry."""
    call_count = 0

    @with_retry(max_attempts=3, retryable_exceptions=(ValueError,))
    async def raises_type_error():
        nonlocal call_count
        call_count += 1
        raise TypeError("not retryable")

    with pytest.raises(TypeError, match="not retryable"):
        await raises_type_error()

    assert call_count == 1
    mock_sleep.assert_not_called()


# ===========================================================================
# 9. test_http_status_429_retry_after_header_respected
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_http_status_429_retry_after_header_respected(mock_sleep):
    """HTTP 429 with Retry-After header uses that value as the sleep delay."""
    call_count = 0
    exc_429 = _make_http_status_error(429, headers={"retry-after": "7.5"})

    @with_retry(max_attempts=2, base_delay=1.0, retryable_exceptions=(httpx.HTTPStatusError,))
    async def rate_limited():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise exc_429
        return "ok"

    result = await rate_limited()
    assert result == "ok"
    assert call_count == 2
    # Sleep delay should be exactly 7.5 from Retry-After header, not exponential backoff
    mock_sleep.assert_called_once_with(7.5)


# ===========================================================================
# 10. test_http_status_400_not_retried
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_http_status_400_not_retried(mock_sleep):
    """HTTP 400 (Bad Request) is not a retryable status, should raise immediately."""
    call_count = 0
    exc_400 = _make_http_status_error(400)

    @with_retry(max_attempts=3, retryable_exceptions=(httpx.HTTPStatusError,))
    async def bad_request():
        nonlocal call_count
        call_count += 1
        raise exc_400

    with pytest.raises(httpx.HTTPStatusError):
        await bad_request()

    assert call_count == 1
    mock_sleep.assert_not_called()


# ===========================================================================
# 11. test_http_status_503_retried
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_http_status_503_retried(mock_sleep):
    """HTTP 503 (Service Unavailable) is retryable and should be retried."""
    call_count = 0
    exc_503 = _make_http_status_error(503)

    @with_retry(max_attempts=3, base_delay=0.5, retryable_exceptions=(httpx.HTTPStatusError,))
    async def service_unavailable():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise exc_503
        return "back_up"

    result = await service_unavailable()
    assert result == "back_up"
    assert call_count == 3
    assert mock_sleep.call_count == 2


# ===========================================================================
# 12. test_programmatic_async_retry
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_programmatic_async_retry(mock_sleep):
    """async_retry() programmatic helper works with args/kwargs."""
    call_count = 0

    async def flaky_add(a, b, offset=0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")
        return a + b + offset

    result = await async_retry(
        flaky_add,
        args=(10, 20),
        kwargs={"offset": 5},
        max_attempts=3,
        base_delay=0.1,
    )
    assert result == 35
    assert call_count == 2
    assert mock_sleep.call_count == 1


# ===========================================================================
# 13. test_with_retry_zero_max_attempts
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_with_retry_zero_max_attempts(mock_sleep):
    """max_attempts=0 should still run the function once (effective_max=1)."""
    call_count = 0

    @with_retry(max_attempts=0, base_delay=1.0)
    async def run_once():
        nonlocal call_count
        call_count += 1
        return "ran"

    result = await run_once()
    assert result == "ran"
    assert call_count == 1
    mock_sleep.assert_not_called()


@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_with_retry_zero_max_attempts_failure(mock_sleep):
    """max_attempts=0, function fails: should raise immediately with no retry."""
    call_count = 0

    @with_retry(max_attempts=0, base_delay=1.0)
    async def fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("only once")

    with pytest.raises(RuntimeError, match="only once"):
        await fail()

    assert call_count == 1
    mock_sleep.assert_not_called()


# ===========================================================================
# 14. test_decorator_preserves_function_name
# ===========================================================================

async def test_decorator_preserves_function_name():
    """@with_retry should preserve __name__ and __qualname__ via functools.wraps."""

    @with_retry(max_attempts=2)
    async def my_async_function():
        pass

    @with_retry(max_attempts=2)
    def my_sync_function():
        pass

    assert my_async_function.__name__ == "my_async_function"
    assert my_sync_function.__name__ == "my_sync_function"
    assert "my_async_function" in my_async_function.__qualname__
    assert "my_sync_function" in my_sync_function.__qualname__


# ===========================================================================
# Additional: _is_retryable_http_status unit tests
# ===========================================================================

class TestIsRetryableHttpStatus:
    """Direct unit tests for the _is_retryable_http_status helper."""

    def test_retryable_status_codes(self):
        for code in (429, 500, 502, 503, 504):
            exc = _make_http_status_error(code)
            assert _is_retryable_http_status(exc) is True, f"Expected {code} to be retryable"

    def test_non_retryable_status_codes(self):
        for code in (400, 401, 403, 404, 405, 409, 422):
            exc = _make_http_status_error(code)
            assert _is_retryable_http_status(exc) is False, f"Expected {code} to NOT be retryable"

    def test_non_http_exception_is_retryable(self):
        """Non-HTTPStatusError exceptions return True (always retryable)."""
        assert _is_retryable_http_status(RuntimeError("something")) is True


# ===========================================================================
# Additional: _get_retry_after unit tests
# ===========================================================================

class TestGetRetryAfter:
    """Direct unit tests for the _get_retry_after helper."""

    def test_returns_float_from_retry_after_header(self):
        exc = _make_http_status_error(429, headers={"retry-after": "3.5"})
        assert _get_retry_after(exc) == pytest.approx(3.5)

    def test_returns_int_as_float_from_retry_after_header(self):
        exc = _make_http_status_error(429, headers={"retry-after": "10"})
        assert _get_retry_after(exc) == pytest.approx(10.0)

    def test_returns_none_for_non_429(self):
        exc = _make_http_status_error(503, headers={"retry-after": "5"})
        assert _get_retry_after(exc) is None

    def test_returns_none_for_missing_header(self):
        exc = _make_http_status_error(429)
        assert _get_retry_after(exc) is None

    def test_returns_none_for_invalid_header_value(self):
        exc = _make_http_status_error(429, headers={"retry-after": "not-a-number"})
        assert _get_retry_after(exc) is None

    def test_returns_none_for_non_http_exception(self):
        assert _get_retry_after(RuntimeError("something")) is None


# ===========================================================================
# Additional: httpx.TimeoutException is retried
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_httpx_timeout_exception_retried(mock_sleep):
    """httpx.TimeoutException should be retried when listed in retryable_exceptions."""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.1, retryable_exceptions=(httpx.TimeoutException,))
    async def timeout_then_ok():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("timed out")
        return "ok"

    result = await timeout_then_ok()
    assert result == "ok"
    assert call_count == 2
    assert mock_sleep.call_count == 1


# ===========================================================================
# Additional: retryable_exceptions=None retries all
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_retryable_exceptions_none_retries_all(mock_sleep):
    """When retryable_exceptions is None (default), all exception types are retried."""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.1, retryable_exceptions=None)
    async def different_errors():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TypeError("type error")
        if call_count == 2:
            raise ValueError("value error")
        return "ok"

    result = await different_errors()
    assert result == "ok"
    assert call_count == 3
    assert mock_sleep.call_count == 2


# ===========================================================================
# Additional: sync retry also filters non-retryable exceptions
# ===========================================================================

@patch("semantic_vector_router.utils.retry.time.sleep")
def test_sync_retryable_exceptions_filter(mock_sleep):
    """Sync wrapper: non-retryable exception is raised immediately."""
    call_count = 0

    @with_retry(max_attempts=3, retryable_exceptions=(ValueError,))
    def raises_key_error():
        nonlocal call_count
        call_count += 1
        raise KeyError("not retryable")

    with pytest.raises(KeyError):
        raises_key_error()

    assert call_count == 1
    mock_sleep.assert_not_called()


# ===========================================================================
# Additional: sync HTTP 400 not retried
# ===========================================================================

@patch("semantic_vector_router.utils.retry.time.sleep")
def test_sync_http_status_400_not_retried(mock_sleep):
    """Sync wrapper: HTTP 400 is not retried even when HTTPStatusError is retryable."""
    call_count = 0
    exc_400 = _make_http_status_error(400)

    @with_retry(max_attempts=3, retryable_exceptions=(httpx.HTTPStatusError,))
    def bad_request():
        nonlocal call_count
        call_count += 1
        raise exc_400

    with pytest.raises(httpx.HTTPStatusError):
        bad_request()

    assert call_count == 1
    mock_sleep.assert_not_called()


# ===========================================================================
# Additional: warning logging on retry
# ===========================================================================

@patch("semantic_vector_router.utils.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_retry_logs_warning(mock_sleep):
    """Each retry attempt should log a WARNING message."""
    import logging

    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.1)
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient")
        return "ok"

    # Capture logs directly from the retry logger
    retry_logger = logging.getLogger("semantic_vector_router.utils.retry")
    records: list[logging.LogRecord] = []

    class RecordHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = RecordHandler()
    handler.setLevel(logging.WARNING)
    retry_logger.addHandler(handler)
    try:
        result = await flaky()
    finally:
        retry_logger.removeHandler(handler)

    assert result == "ok"
    # Two retries -> two warning log messages
    retry_messages = [r for r in records if "Retry" in r.getMessage()]
    assert len(retry_messages) == 2
    assert "RuntimeError" in retry_messages[0].getMessage()
    assert "transient" in retry_messages[0].getMessage()
