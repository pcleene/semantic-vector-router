"""Webhook event dispatcher for SVR."""

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Optional

from pydantic import BaseModel

from semantic_vector_router.events.models import SVREvent, SVREventType

logger = logging.getLogger("semantic_vector_router.events.webhook")


class WebhookConfig(BaseModel):
    """Webhook endpoint configuration."""

    url: str
    events: list[SVREventType] = []  # Empty = all events
    secret: Optional[str] = None  # HMAC-SHA256 signing key
    timeout_seconds: int = 10
    retry_count: int = 3
    retry_delay_seconds: float = 5.0
    headers: dict[str, str] = {}
    enabled: bool = True


class WebhookTestResult(BaseModel):
    """Result of a webhook connectivity test."""

    url: str
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class WebhookDispatcher:
    """Delivers SVR events to configured HTTP endpoints.

    - Async HTTP via httpx
    - Payload signing with HMAC-SHA256 (X-SVR-Signature header)
    - Retry with exponential backoff on 5xx / network errors
    - Fire-and-forget: webhook failures don't affect SVR operations
    """

    def __init__(self, webhooks: list[WebhookConfig]) -> None:
        self._webhooks = webhooks
        self._client: Any = None  # httpx.AsyncClient, lazy init

    async def _ensure_client(self) -> Any:
        """Lazy-initialize httpx client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def handle_event(self, event: SVREvent) -> None:
        """Dispatch event to all matching webhooks."""
        for webhook in self._webhooks:
            if not webhook.enabled:
                continue

            # Check event filter
            if webhook.events and event.event_type not in webhook.events:
                continue

            try:
                await self._deliver(webhook, event)
            except Exception as e:
                logger.warning(
                    f"Webhook delivery failed for {webhook.url}: {e}"
                )

    async def _deliver(self, webhook: WebhookConfig, event: SVREvent) -> None:
        """Deliver event to a single webhook with retry."""
        import httpx

        client = await self._ensure_client()
        payload = json.dumps(event.to_dict(), default=str)
        payload_bytes = payload.encode("utf-8")

        # Build headers
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "SVR-Webhook/1.0",
        }
        headers.update(webhook.headers)

        # HMAC signature
        if webhook.secret:
            timestamp = str(int(time.time()))
            signature = hmac.new(
                webhook.secret.encode("utf-8"),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()
            headers["X-SVR-Signature"] = f"sha256={signature}"
            headers["X-SVR-Timestamp"] = timestamp

        # Retry loop
        last_error: Optional[Exception] = None
        for attempt in range(webhook.retry_count + 1):
            try:
                response = await client.post(
                    webhook.url,
                    content=payload_bytes,
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )

                if response.status_code < 400:
                    logger.debug(
                        f"Webhook delivered to {webhook.url}: "
                        f"{response.status_code}"
                    )
                    return

                # 4xx = client error, don't retry
                if 400 <= response.status_code < 500:
                    logger.warning(
                        f"Webhook {webhook.url} returned {response.status_code} "
                        f"(client error, not retrying)"
                    )
                    return

                # 5xx = server error, retry
                last_error = Exception(
                    f"HTTP {response.status_code}"
                )
                logger.warning(
                    f"Webhook {webhook.url} returned {response.status_code}, "
                    f"attempt {attempt + 1}/{webhook.retry_count + 1}"
                )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Webhook {webhook.url} timed out, "
                    f"attempt {attempt + 1}/{webhook.retry_count + 1}"
                )
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"Webhook {webhook.url} network error: {e}, "
                    f"attempt {attempt + 1}/{webhook.retry_count + 1}"
                )

            # Exponential backoff before retry
            if attempt < webhook.retry_count:
                import asyncio
                delay = webhook.retry_delay_seconds * (2 ** attempt)
                await asyncio.sleep(delay)

        logger.error(
            f"Webhook delivery to {webhook.url} failed after "
            f"{webhook.retry_count + 1} attempts: {last_error}"
        )

    async def test_webhook(self, webhook: WebhookConfig) -> WebhookTestResult:
        """Send a test event to verify endpoint connectivity."""

        test_event = SVREvent(
            event_type=SVREventType.JOB_COMPLETED,
            details={"test": True, "message": "SVR webhook connectivity test"},
        )

        client = await self._ensure_client()
        payload = json.dumps(test_event.to_dict(), default=str)

        start = time.perf_counter()
        try:
            response = await client.post(
                webhook.url,
                content=payload.encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SVR-Webhook/1.0",
                },
                timeout=webhook.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            return WebhookTestResult(
                url=webhook.url,
                success=response.status_code < 400,
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WebhookTestResult(
                url=webhook.url,
                success=False,
                response_time_ms=elapsed_ms,
                error=str(e),
            )
