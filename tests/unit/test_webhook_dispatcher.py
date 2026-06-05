"""Comprehensive unit tests for WebhookDispatcher."""

import hashlib
import hmac
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.events.models import SVREvent, SVREventType
from semantic_vector_router.events.webhook import (
    WebhookConfig,
    WebhookDispatcher,
    WebhookTestResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: SVREventType = SVREventType.PARTITION_CREATED,
    partition: str | None = "part-1",
    severity: str = "info",
    details: dict | None = None,
) -> SVREvent:
    return SVREvent(
        event_type=event_type,
        partition=partition,
        severity=severity,
        details=details or {},
    )


def _make_webhook(**overrides) -> WebhookConfig:
    defaults = {
        "url": "https://example.com/webhook",
        "events": [],
        "secret": None,
        "timeout_seconds": 5,
        "retry_count": 3,
        "retry_delay_seconds": 0.0,  # no delay in tests
        "headers": {},
        "enabled": True,
    }
    defaults.update(overrides)
    return WebhookConfig(**defaults)


def _mock_response(status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _dispatcher_with_mock_client(
    webhooks: list[WebhookConfig],
) -> tuple["WebhookDispatcher", AsyncMock]:
    """Create a dispatcher and inject a mock httpx client."""
    dispatcher = WebhookDispatcher(webhooks)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response(200))
    mock_client.aclose = AsyncMock()
    dispatcher._client = mock_client
    return dispatcher, mock_client


# ---------------------------------------------------------------------------
# 1. Successful delivery (200)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_delivery_200():
    """POST is called with the correct URL, JSON payload, and standard headers."""
    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])
    event = _make_event()

    await dispatcher.handle_event(event)

    client.post.assert_awaited_once()
    call_kwargs = client.post.call_args
    assert call_kwargs.args[0] == "https://example.com/webhook"
    # Payload is bytes
    sent_payload = json.loads(call_kwargs.kwargs["content"].decode("utf-8"))
    assert sent_payload["event_type"] == SVREventType.PARTITION_CREATED.value
    # Standard headers present
    headers = call_kwargs.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert headers["User-Agent"] == "SVR-Webhook/1.0"


# ---------------------------------------------------------------------------
# 2. Event filtering with specific events list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_filtering_specific_events():
    """Webhook with specific events list only receives matching events."""
    wh = _make_webhook(events=[SVREventType.HEALTH_ALERT])
    dispatcher, client = _dispatcher_with_mock_client([wh])

    # Non-matching event
    await dispatcher.handle_event(_make_event(SVREventType.PARTITION_CREATED))
    client.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Empty events list = all events pass through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_events_list_passes_all():
    """Webhook with empty events list receives all event types."""
    wh = _make_webhook(events=[])
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event(SVREventType.PARTITION_CREATED))
    assert client.post.await_count == 1

    await dispatcher.handle_event(_make_event(SVREventType.HEALTH_ALERT))
    assert client.post.await_count == 2

    await dispatcher.handle_event(_make_event(SVREventType.JOB_COMPLETED))
    assert client.post.await_count == 3


# ---------------------------------------------------------------------------
# 4. Disabled webhook is skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_webhook_skipped():
    """Disabled webhook never triggers HTTP POST."""
    wh = _make_webhook(enabled=False)
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event())
    client.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. HMAC signature headers present when secret configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hmac_signature_present_when_secret_set():
    """X-SVR-Signature and X-SVR-Timestamp headers present when secret is set."""
    wh = _make_webhook(secret="<redacted>")
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event())

    headers = client.post.call_args.kwargs["headers"]
    assert "X-SVR-Signature" in headers
    assert "X-SVR-Timestamp" in headers


# ---------------------------------------------------------------------------
# 6. HMAC signature format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hmac_signature_format():
    """Signature follows sha256=<hex_digest> format and is verifiable."""
    secret="<redacted>"
    wh = _make_webhook(secret=secret)
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event())

    headers = client.post.call_args.kwargs["headers"]
    sig = headers["X-SVR-Signature"]
    assert sig.startswith("sha256=")
    hex_part = sig[len("sha256="):]
    # Verify it's valid hex
    int(hex_part, 16)

    # Verify HMAC is correct for the sent payload
    payload_bytes = client.post.call_args.kwargs["content"]
    expected = hmac.new(
        secret.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()
    assert hex_part == expected


# ---------------------------------------------------------------------------
# 7. No HMAC headers when no secret
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_hmac_headers_without_secret():
    """When no secret is configured, no signature headers are sent."""
    wh = _make_webhook(secret=None)
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event())

    headers = client.post.call_args.kwargs["headers"]
    assert "X-SVR-Signature" not in headers
    assert "X-SVR-Timestamp" not in headers


# ---------------------------------------------------------------------------
# 8. Retry on 500 response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_500():
    """5xx responses trigger retries up to retry_count."""
    wh = _make_webhook(retry_count=2, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(500))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await dispatcher.handle_event(_make_event())

    # 1 initial + 2 retries = 3 total attempts
    assert client.post.await_count == 3


# ---------------------------------------------------------------------------
# 9. Retry on 503 response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_503():
    """503 (server error) also triggers retries."""
    wh = _make_webhook(retry_count=1, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(503))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 2


# ---------------------------------------------------------------------------
# 10. Retry on network error (httpx.HTTPError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_network_error():
    """Network errors (httpx.HTTPError) trigger retries."""
    import httpx

    wh = _make_webhook(retry_count=2, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(side_effect=httpx.HTTPError("connection refused"))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 3


# ---------------------------------------------------------------------------
# 11. No retry on 400 response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retry_on_400():
    """400 (client error) is not retried."""
    wh = _make_webhook(retry_count=3, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(400))

    await dispatcher.handle_event(_make_event())

    # Only 1 attempt, no retries
    assert client.post.await_count == 1


# ---------------------------------------------------------------------------
# 12. No retry on 404 response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retry_on_404():
    """404 (client error) is not retried."""
    wh = _make_webhook(retry_count=3, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(404))

    await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 1


# ---------------------------------------------------------------------------
# 13. No retry on 422 response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retry_on_422():
    """422 (client error) is not retried."""
    wh = _make_webhook(retry_count=3, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(422))

    await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 1


# ---------------------------------------------------------------------------
# 14. Timeout handling (httpx.TimeoutException)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_retries():
    """Timeout exceptions trigger retries."""
    import httpx

    wh = _make_webhook(retry_count=2, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 3


# ---------------------------------------------------------------------------
# 15. Maximum retries exhausted (no crash)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_retries_exhausted_does_not_raise():
    """After all retries exhausted, handle_event doesn't raise."""
    wh = _make_webhook(retry_count=1, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(500))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Should not raise
        await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 2


# ---------------------------------------------------------------------------
# 16. test_webhook success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_webhook_success():
    """test_webhook returns successful result with status code and response time."""
    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(200))

    result = await dispatcher.test_webhook(wh)

    assert isinstance(result, WebhookTestResult)
    assert result.success is True
    assert result.status_code == 200
    assert result.response_time_ms is not None
    assert result.response_time_ms >= 0
    assert result.url == wh.url
    assert result.error is None


# ---------------------------------------------------------------------------
# 17. test_webhook failure (non-2xx)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_webhook_failure_status():
    """test_webhook returns failure for non-2xx status."""
    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(500))

    result = await dispatcher.test_webhook(wh)

    assert result.success is False
    assert result.status_code == 500


# ---------------------------------------------------------------------------
# 18. test_webhook failure (exception)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_webhook_network_error():
    """test_webhook returns failure result on exception without raising."""
    import httpx

    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    result = await dispatcher.test_webhook(wh)

    assert result.success is False
    assert result.status_code is None
    assert result.error is not None
    assert "refused" in result.error


# ---------------------------------------------------------------------------
# 19. Payload is valid JSON with required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payload_contains_required_fields():
    """Dispatched payload has event_type, timestamp, severity."""
    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])

    event = _make_event(
        event_type=SVREventType.HEALTH_ALERT,
        severity="warning",
        partition="alerts",
    )
    await dispatcher.handle_event(event)

    payload_bytes = client.post.call_args.kwargs["content"]
    payload = json.loads(payload_bytes.decode("utf-8"))
    assert payload["event_type"] == "health.alert"
    assert "timestamp" in payload
    assert payload["severity"] == "warning"
    assert payload["partition"] == "alerts"


# ---------------------------------------------------------------------------
# 20. close() closes the httpx client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_closes_client():
    """close() calls aclose() on the httpx client and sets it to None."""
    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.close()

    client.aclose.assert_awaited_once()
    assert dispatcher._client is None


# ---------------------------------------------------------------------------
# 21. close() when no client is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_without_client_is_noop():
    """close() when no client has been created is safe."""
    dispatcher = WebhookDispatcher([])
    assert dispatcher._client is None
    await dispatcher.close()  # should not raise
    assert dispatcher._client is None


# ---------------------------------------------------------------------------
# 22. Multiple webhooks dispatched to all matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_webhooks_all_receive():
    """Event dispatched to all matching webhooks."""
    wh1 = _make_webhook(url="https://a.com/hook")
    wh2 = _make_webhook(url="https://b.com/hook")
    dispatcher, client = _dispatcher_with_mock_client([wh1, wh2])

    await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 2
    urls = [call.args[0] for call in client.post.call_args_list]
    assert "https://a.com/hook" in urls
    assert "https://b.com/hook" in urls


# ---------------------------------------------------------------------------
# 23. Multiple webhooks, only matching receive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_webhooks_only_matching():
    """Only webhooks whose filter matches receive the event."""
    wh_health = _make_webhook(
        url="https://health.com/hook",
        events=[SVREventType.HEALTH_ALERT],
    )
    wh_partition = _make_webhook(
        url="https://partition.com/hook",
        events=[SVREventType.PARTITION_CREATED],
    )
    dispatcher, client = _dispatcher_with_mock_client([wh_health, wh_partition])

    await dispatcher.handle_event(_make_event(SVREventType.PARTITION_CREATED))

    assert client.post.await_count == 1
    assert client.post.call_args.args[0] == "https://partition.com/hook"


# ---------------------------------------------------------------------------
# 24. Custom headers merged into request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_headers_merged():
    """Extra headers from WebhookConfig are included in the request."""
    wh = _make_webhook(headers={"X-Custom": "value123", "Authorization": "Bearer tok"})
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event())

    headers = client.post.call_args.kwargs["headers"]
    assert headers["X-Custom"] == "value123"
    assert headers["Authorization"] == "Bearer tok"
    # Standard headers still present
    assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# 25. Exponential backoff delay calculation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exponential_backoff_delays():
    """Retry delays follow exponential backoff: delay * 2^attempt."""
    wh = _make_webhook(retry_count=3, retry_delay_seconds=1.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(500))

    sleep_calls: list[float] = []

    async def mock_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await dispatcher.handle_event(_make_event())

    # 4 attempts => 3 sleeps: 1*2^0=1, 1*2^1=2, 1*2^2=4
    assert sleep_calls == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# 26. Successful delivery after transient failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_after_transient_failure():
    """Succeeds after initial 5xx and does not retry further."""
    wh = _make_webhook(retry_count=3, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(
        side_effect=[_mock_response(502), _mock_response(200)]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await dispatcher.handle_event(_make_event())

    # 1 failure + 1 success = 2 total
    assert client.post.await_count == 2


# ---------------------------------------------------------------------------
# 27. Timeout passed to httpx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_value_passed():
    """Configured timeout_seconds is forwarded to httpx.post."""
    wh = _make_webhook(timeout_seconds=42)
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event())

    assert client.post.call_args.kwargs["timeout"] == 42


# ---------------------------------------------------------------------------
# 28. _ensure_client lazy init
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_client_lazy_init():
    """_ensure_client creates httpx.AsyncClient only once."""
    dispatcher = WebhookDispatcher([])
    assert dispatcher._client is None

    mock_async_client = MagicMock()
    with patch("httpx.AsyncClient", return_value=mock_async_client) as mock_cls:
        c1 = await dispatcher._ensure_client()
        c2 = await dispatcher._ensure_client()

    assert c1 is c2
    assert c1 is mock_async_client
    mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# 29. Event with details serialized in payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_details_in_payload():
    """Event details dict is serialized into the webhook payload."""
    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])

    event = _make_event(
        details={"doc_count": 100, "threshold": 0.85},
    )
    await dispatcher.handle_event(event)

    payload = json.loads(client.post.call_args.kwargs["content"].decode("utf-8"))
    assert payload["details"]["doc_count"] == 100
    assert payload["details"]["threshold"] == 0.85


# ---------------------------------------------------------------------------
# 30. Delivery failure logged but does not propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delivery_exception_caught():
    """Unexpected exception in _deliver is caught; handle_event doesn't raise."""
    wh = _make_webhook(retry_count=0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(side_effect=RuntimeError("unexpected"))

    # Should not raise
    await dispatcher.handle_event(_make_event())


# ---------------------------------------------------------------------------
# 31. WebhookConfig defaults
# ---------------------------------------------------------------------------


def test_webhook_config_defaults():
    """WebhookConfig has sensible defaults."""
    cfg = WebhookConfig(url="https://x.com/hook")
    assert cfg.events == []
    assert cfg.secret is None
    assert cfg.timeout_seconds == 10
    assert cfg.retry_count == 3
    assert cfg.retry_delay_seconds == 5.0
    assert cfg.headers == {}
    assert cfg.enabled is True


# ---------------------------------------------------------------------------
# 32. WebhookTestResult model
# ---------------------------------------------------------------------------


def test_webhook_test_result_model():
    """WebhookTestResult correctly stores all fields."""
    result = WebhookTestResult(
        url="https://x.com",
        success=True,
        status_code=200,
        response_time_ms=45.3,
        error=None,
    )
    assert result.url == "https://x.com"
    assert result.success is True
    assert result.status_code == 200
    assert result.response_time_ms == 45.3
    assert result.error is None


# ---------------------------------------------------------------------------
# 33. test_webhook sends JOB_COMPLETED event type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_webhook_sends_job_completed_event():
    """test_webhook sends a JOB_COMPLETED test event."""
    wh = _make_webhook()
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.test_webhook(wh)

    payload = json.loads(client.post.call_args.kwargs["content"].decode("utf-8"))
    assert payload["event_type"] == SVREventType.JOB_COMPLETED.value
    assert payload["details"]["test"] is True


# ---------------------------------------------------------------------------
# 34. Matching event filter accepts the event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_matching_filter_accepts():
    """Webhook with a matching filter list dispatches the event."""
    wh = _make_webhook(events=[SVREventType.PARTITION_CREATED])
    dispatcher, client = _dispatcher_with_mock_client([wh])

    await dispatcher.handle_event(_make_event(SVREventType.PARTITION_CREATED))

    assert client.post.await_count == 1


# ---------------------------------------------------------------------------
# 35. Disabled webhook among enabled ones
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_among_enabled():
    """Only enabled webhooks receive the event."""
    wh_on = _make_webhook(url="https://on.com/hook", enabled=True)
    wh_off = _make_webhook(url="https://off.com/hook", enabled=False)
    dispatcher, client = _dispatcher_with_mock_client([wh_on, wh_off])

    await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 1
    assert client.post.call_args.args[0] == "https://on.com/hook"


# ---------------------------------------------------------------------------
# 36. 2xx status codes are treated as success (e.g. 201, 204)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_2xx_status_success():
    """201 and 204 are treated as successful delivery (no retry)."""
    for status in (200, 201, 204, 301, 302):
        wh = _make_webhook(retry_count=2, retry_delay_seconds=0.0)
        dispatcher, client = _dispatcher_with_mock_client([wh])
        client.post = AsyncMock(return_value=_mock_response(status))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await dispatcher.handle_event(_make_event())

        assert client.post.await_count == 1, f"Status {status} should not retry"


# ---------------------------------------------------------------------------
# 37. Zero retry_count means exactly 1 attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_retry_count_single_attempt():
    """With retry_count=0, only 1 attempt is mParts Distributor even on 5xx."""
    wh = _make_webhook(retry_count=0, retry_delay_seconds=0.0)
    dispatcher, client = _dispatcher_with_mock_client([wh])
    client.post = AsyncMock(return_value=_mock_response(500))

    await dispatcher.handle_event(_make_event())

    assert client.post.await_count == 1
