"""Unit tests for token-bucket rate limiter (utils/rate_limiter.py)."""

import asyncio
import time

import pytest

from semantic_vector_router.models import ProviderRateLimit, RateLimitConfig
from semantic_vector_router.utils.rate_limiter import (
    DEFAULT_RATE_LIMITS,
    RateLimiterRegistry,
    RateLimiterStats,
    TokenBucketRateLimiter,
    UNLIMITED_BURST,
    UNLIMITED_RATE,
)


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_immediate_when_tokens_available(self):
        """Acquiring 1 token from a full bucket should return immediately."""
        limiter = TokenBucketRateLimiter(tokens_per_second=100.0, burst=10)
        wait = await limiter.acquire(1)
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Acquiring multiple tokens from a full bucket should work."""
        limiter = TokenBucketRateLimiter(tokens_per_second=100.0, burst=10)
        wait = await limiter.acquire(5)
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_drains_bucket(self):
        """After draining the bucket, acquiring more tokens should wait."""
        limiter = TokenBucketRateLimiter(tokens_per_second=100.0, burst=5)
        # Drain the bucket
        await limiter.acquire(5)
        # Next acquire should need to wait
        start = time.monotonic()
        wait = await limiter.acquire(1)
        elapsed = time.monotonic() - start
        assert wait > 0
        assert elapsed >= 0.005  # At least some wait happened

    @pytest.mark.asyncio
    async def test_acquire_exceeds_burst_raises(self):
        """Requesting more tokens than burst capacity should raise ValueError."""
        limiter = TokenBucketRateLimiter(tokens_per_second=100.0, burst=5)
        with pytest.raises(ValueError, match="exceeds burst capacity"):
            await limiter.acquire(6)

    @pytest.mark.asyncio
    async def test_stats_tracks_requests(self):
        """Stats should track total requests."""
        limiter = TokenBucketRateLimiter(tokens_per_second=100.0, burst=10)
        await limiter.acquire(1)
        await limiter.acquire(1)
        await limiter.acquire(1)

        stats = limiter.stats()
        assert stats.total_requests == 3
        assert stats.tokens_per_second == 100.0
        assert stats.burst == 10

    @pytest.mark.asyncio
    async def test_stats_tracks_waits(self):
        """Stats should track wait events."""
        limiter = TokenBucketRateLimiter(tokens_per_second=1000.0, burst=2)
        # Drain bucket
        await limiter.acquire(2)
        # This should wait
        await limiter.acquire(1)

        stats = limiter.stats()
        assert stats.total_requests == 2
        assert stats.total_waited >= 1
        assert stats.total_wait_time_ms > 0

    @pytest.mark.asyncio
    async def test_reset_clears_stats_and_refills(self):
        """Reset should clear stats and refill the bucket."""
        limiter = TokenBucketRateLimiter(tokens_per_second=100.0, burst=10)
        await limiter.acquire(5)
        limiter.reset()

        stats = limiter.stats()
        assert stats.total_requests == 0
        assert stats.total_waited == 0
        assert stats.total_wait_time_ms == 0.0
        assert stats.current_tokens == 10.0

    def test_properties(self):
        """Properties should return configured values."""
        limiter = TokenBucketRateLimiter(tokens_per_second=42.0, burst=7)
        assert limiter.tokens_per_second == 42.0
        assert limiter.burst == 7

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Tokens should refill over time up to burst capacity."""
        limiter = TokenBucketRateLimiter(tokens_per_second=10000.0, burst=10)
        await limiter.acquire(10)  # Drain fully

        # Wait a bit for refill
        await asyncio.sleep(0.01)

        stats = limiter.stats()
        # After waiting, tokens should have refilled (but we check via acquire)
        wait = await limiter.acquire(1)
        # Should be immediate or very fast because of high rate
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        """Multiple concurrent acquires should be serialized by lock."""
        limiter = TokenBucketRateLimiter(tokens_per_second=10000.0, burst=100)

        results = await asyncio.gather(
            *[limiter.acquire(1) for _ in range(50)]
        )

        # All should succeed
        assert len(results) == 50
        stats = limiter.stats()
        assert stats.total_requests == 50


# ---------------------------------------------------------------------------
# RateLimiterStats
# ---------------------------------------------------------------------------


class TestRateLimiterStats:
    def test_defaults(self):
        """Stats should have zero defaults."""
        stats = RateLimiterStats()
        assert stats.total_requests == 0
        assert stats.total_waited == 0
        assert stats.total_wait_time_ms == 0.0
        assert stats.current_tokens == 0.0
        assert stats.tokens_per_second == 0.0
        assert stats.burst == 0


# ---------------------------------------------------------------------------
# RateLimiterRegistry
# ---------------------------------------------------------------------------


class TestRateLimiterRegistry:
    """Tests for RateLimiterRegistry."""

    def test_get_creates_limiter_lazily(self):
        """get() should create a limiter on first access."""
        config = RateLimitConfig(
            enabled=True,
            default_tokens_per_second=50.0,
            default_burst=100,
        )
        registry = RateLimiterRegistry(config)

        limiter = registry.get("openai")
        assert isinstance(limiter, TokenBucketRateLimiter)
        assert limiter.tokens_per_second == 50.0
        assert limiter.burst == 100

    def test_get_returns_same_limiter(self):
        """get() should return the same limiter for the same provider."""
        config = RateLimitConfig(enabled=True)
        registry = RateLimiterRegistry(config)

        l1 = registry.get("openai")
        l2 = registry.get("openai")
        assert l1 is l2

    def test_get_different_providers(self):
        """get() should create different limiters for different providers."""
        config = RateLimitConfig(enabled=True)
        registry = RateLimiterRegistry(config)

        l1 = registry.get("openai")
        l2 = registry.get("voyage")
        assert l1 is not l2

    def test_provider_specific_config(self):
        """Provider-specific config should override defaults."""
        config = RateLimitConfig(
            enabled=True,
            default_tokens_per_second=50.0,
            default_burst=100,
            providers={
                "voyage": ProviderRateLimit(
                    tokens_per_second=30.0,
                    burst=60,
                ),
            },
        )
        registry = RateLimiterRegistry(config)

        voyage = registry.get("voyage")
        assert voyage.tokens_per_second == 30.0
        assert voyage.burst == 60

        # Unknown provider uses defaults
        openai = registry.get("openai")
        assert openai.tokens_per_second == 50.0
        assert openai.burst == 100

    def test_disabled_uses_unlimited(self):
        """When config is None (disabled), limiters should be effectively unlimited."""
        registry = RateLimiterRegistry(None)

        limiter = registry.get("openai")
        assert limiter.tokens_per_second == UNLIMITED_RATE
        assert limiter.burst == UNLIMITED_BURST

    def test_stats_returns_all_active(self):
        """stats() should return stats for all accessed providers."""
        config = RateLimitConfig(enabled=True)
        registry = RateLimiterRegistry(config)

        registry.get("openai")
        registry.get("voyage")

        stats = registry.stats()
        assert "openai" in stats
        assert "voyage" in stats
        assert isinstance(stats["openai"], RateLimiterStats)

    def test_stats_empty_when_no_limiters(self):
        """stats() should be empty when no limiters have been created."""
        registry = RateLimiterRegistry(None)
        assert registry.stats() == {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_rate_limits_keys(self):
        """Default rate limits should have common providers."""
        assert "openai" in DEFAULT_RATE_LIMITS
        assert "voyage" in DEFAULT_RATE_LIMITS
        assert "cohere" in DEFAULT_RATE_LIMITS
        assert "huggingface" in DEFAULT_RATE_LIMITS

    def test_unlimited_values(self):
        """Unlimited values should be very high."""
        assert UNLIMITED_RATE >= 10_000.0
        assert UNLIMITED_BURST >= 10_000
