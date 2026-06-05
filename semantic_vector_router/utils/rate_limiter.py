"""Token-bucket rate limiter for external API calls.

Protects embedding and reranking API calls from exceeding provider rate limits.
Async-compatible, per-provider, with configurable rates and burst capacity.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from semantic_vector_router.utils.logging import get_logger

if TYPE_CHECKING:
    from semantic_vector_router.models import RateLimitConfig

logger = get_logger(__name__)

# Default rate limits per provider (requests per second)
# Conservative defaults — users should tune based on their plan tier
DEFAULT_RATE_LIMITS: dict[str, float] = {
    "openai": 50.0,
    "voyage": 30.0,
    "cohere": 40.0,
    "huggingface": 100.0,
}

# Default burst multiplier (how many tokens can accumulate beyond base rate)
DEFAULT_BURST_MULTIPLIER: float = 2.0

# Effectively unlimited rate for disabled limiters
UNLIMITED_RATE: float = 10_000.0
UNLIMITED_BURST: int = 10_000


@dataclass
class RateLimiterStats:
    """Statistics for a rate limiter instance.

    Attributes:
        total_requests: Total number of acquire() calls.
        total_waited: Total number of times acquire() had to wait.
        total_wait_time_ms: Cumulative wait time in milliseconds.
        current_tokens: Current number of available tokens.
        tokens_per_second: Configured rate.
        burst: Configured burst capacity.
    """

    total_requests: int = 0
    total_waited: int = 0
    total_wait_time_ms: float = 0.0
    current_tokens: float = 0.0
    tokens_per_second: float = 0.0
    burst: int = 0


class TokenBucketRateLimiter:
    """Async-compatible token bucket rate limiter.

    Implements the token bucket algorithm:
    - Tokens are added at a fixed rate (tokens_per_second)
    - Tokens accumulate up to a burst capacity
    - Each request consumes one or more tokens
    - If no tokens are available, the request waits until tokens refill

    This is designed to be shared across concurrent async tasks.
    Uses asyncio.Lock for thread safety in async contexts.

    Args:
        tokens_per_second: Rate at which tokens are added to the bucket.
        burst: Maximum number of tokens the bucket can hold.
            Allows short bursts above the sustained rate.
    """

    def __init__(
        self,
        tokens_per_second: float,
        burst: int,
    ) -> None:
        self._tokens_per_second = tokens_per_second
        self._burst = burst
        self._tokens = float(burst)  # Start full
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

        # Stats tracking
        self._total_requests = 0
        self._total_waited = 0
        self._total_wait_time_ms = 0.0

    @property
    def tokens_per_second(self) -> float:
        """Configured token refill rate."""
        return self._tokens_per_second

    @property
    def burst(self) -> int:
        """Maximum token capacity."""
        return self._burst

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        If insufficient tokens are available, this coroutine sleeps until
        enough tokens have been refilled. Multiple concurrent callers are
        serialized by the internal lock.

        Args:
            tokens: Number of tokens to acquire (default 1).

        Returns:
            Wait time in seconds (0.0 if tokens were immediately available).

        Raises:
            ValueError: If tokens exceeds burst capacity.
        """
        if tokens > self._burst:
            raise ValueError(
                f"Requested {tokens} tokens exceeds burst capacity {self._burst}"
            )

        wait_time = 0.0
        async with self._lock:
            self._total_requests += 1
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait needed for enough tokens to refill
            deficit = tokens - self._tokens
            wait_time = deficit / self._tokens_per_second

        # Wait outside the lock so other tasks can proceed
        if wait_time > 0:
            self._total_waited += 1
            self._total_wait_time_ms += wait_time * 1000
            await asyncio.sleep(wait_time)

            # Re-acquire lock and consume tokens after waiting
            async with self._lock:
                self._refill()
                self._tokens -= tokens

        return wait_time

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self._burst),
            self._tokens + elapsed * self._tokens_per_second,
        )
        self._last_refill = now

    def stats(self) -> RateLimiterStats:
        """Return rate limiter statistics.

        Returns:
            RateLimiterStats with request counts and wait times.
        """
        return RateLimiterStats(
            total_requests=self._total_requests,
            total_waited=self._total_waited,
            total_wait_time_ms=self._total_wait_time_ms,
            current_tokens=self._tokens,
            tokens_per_second=self._tokens_per_second,
            burst=self._burst,
        )

    def reset(self) -> None:
        """Reset the bucket to full and clear statistics."""
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._total_requests = 0
        self._total_waited = 0
        self._total_wait_time_ms = 0.0


class RateLimiterRegistry:
    """Registry of per-provider rate limiters.

    Manages a collection of TokenBucketRateLimiter instances, one per
    provider name. Lazily creates limiters on first access using
    configured or default rate limits.

    Example:
        >>> registry = RateLimiterRegistry(config)
        >>> limiter = registry.get("voyage")
        >>> await limiter.acquire()
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        """Initialize the registry.

        Args:
            config: Rate limit configuration. If None, creates effectively
                unlimited limiters (for when rate limiting is disabled).
        """
        self._config = config
        self._limiters: dict[str, TokenBucketRateLimiter] = {}

    def get(self, provider: str) -> TokenBucketRateLimiter:
        """Get or create a rate limiter for the given provider.

        Args:
            provider: Provider name (e.g., "openai", "voyage", "cohere").

        Returns:
            TokenBucketRateLimiter for the provider.
        """
        if provider not in self._limiters:
            self._limiters[provider] = self._create_limiter(provider)
        return self._limiters[provider]

    def _create_limiter(self, provider: str) -> TokenBucketRateLimiter:
        """Create a rate limiter with provider-specific or default config."""
        if self._config is not None:
            # Check provider-specific config first
            if provider in self._config.providers:
                prov_config = self._config.providers[provider]
                return TokenBucketRateLimiter(
                    tokens_per_second=prov_config.tokens_per_second,
                    burst=prov_config.burst,
                )
            # Use default config values
            return TokenBucketRateLimiter(
                tokens_per_second=self._config.default_tokens_per_second,
                burst=self._config.default_burst,
            )
        else:
            # No config (disabled) — use high defaults (effectively unlimited)
            return TokenBucketRateLimiter(
                tokens_per_second=UNLIMITED_RATE,
                burst=UNLIMITED_BURST,
            )

    def stats(self) -> dict[str, RateLimiterStats]:
        """Return statistics for all active rate limiters.

        Returns:
            Dict mapping provider name to RateLimiterStats.
        """
        return {name: limiter.stats() for name, limiter in self._limiters.items()}
