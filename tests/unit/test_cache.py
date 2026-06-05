"""Unit tests for embedding cache (utils/cache.py)."""

import threading
import time
from unittest.mock import patch

import pytest

from semantic_vector_router.utils.cache import (
    CacheEntry,
    CacheKey,
    EmbeddingCache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key(text: str = "hello", model: str = "voyage-4-lite", dims: int = 1024) -> CacheKey:
    return CacheKey(text=text, model=model, dimensions=dims, input_type="query")


def _vector(dim: int = 3) -> list[float]:
    return [float(i) for i in range(dim)]


# ---------------------------------------------------------------------------
# CacheKey
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_frozen(self):
        key = _key()
        with pytest.raises(AttributeError):
            key.text = "new"

    def test_hash_deterministic(self):
        k1 = _key("test", "model-a", 512)
        k2 = _key("test", "model-a", 512)
        assert hash(k1) == hash(k2)

    def test_different_keys_different_hash(self):
        k1 = _key("hello")
        k2 = _key("world")
        assert hash(k1) != hash(k2)

    def test_equality(self):
        k1 = _key("test")
        k2 = _key("test")
        assert k1 == k2

    def test_input_type_in_key(self):
        k_query = CacheKey(text="hello", model="m", dimensions=1024, input_type="query")
        k_doc = CacheKey(text="hello", model="m", dimensions=1024, input_type="document")
        assert k_query != k_doc


# ---------------------------------------------------------------------------
# EmbeddingCache — basic operations
# ---------------------------------------------------------------------------


class TestEmbeddingCacheBasic:
    def test_put_and_get(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = _key()
        vec = _vector()
        cache.put(key, vec)
        result = cache.get(key)
        assert result == vec

    def test_get_miss(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        result = cache.get(_key("nonexistent"))
        assert result is None

    def test_size(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        assert cache.size == 0
        cache.put(_key("a"), _vector())
        assert cache.size == 1
        cache.put(_key("b"), _vector())
        assert cache.size == 2

    def test_clear(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        cache.put(_key("a"), _vector())
        cache.put(_key("b"), _vector())
        cache.clear()
        assert cache.size == 0


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_evicts_oldest_when_full(self):
        cache = EmbeddingCache(max_size=3, ttl_seconds=0)
        cache.put(_key("a"), [1.0])
        cache.put(_key("b"), [2.0])
        cache.put(_key("c"), [3.0])
        # Cache is full. Inserting a 4th evicts "a" (oldest).
        cache.put(_key("d"), [4.0])

        assert cache.get(_key("a")) is None
        assert cache.get(_key("b")) == [2.0]
        assert cache.get(_key("d")) == [4.0]
        assert cache.size == 3

    def test_get_refreshes_lru_order(self):
        cache = EmbeddingCache(max_size=3, ttl_seconds=0)
        cache.put(_key("a"), [1.0])
        cache.put(_key("b"), [2.0])
        cache.put(_key("c"), [3.0])

        # Access "a" to move it to end (most recently used)
        cache.get(_key("a"))

        # Insert "d" — should evict "b" (now the oldest)
        cache.put(_key("d"), [4.0])

        assert cache.get(_key("a")) == [1.0]
        assert cache.get(_key("b")) is None

    def test_evictions_counter(self):
        cache = EmbeddingCache(max_size=2, ttl_seconds=0)
        cache.put(_key("a"), [1.0])
        cache.put(_key("b"), [2.0])
        cache.put(_key("c"), [3.0])  # evicts "a"
        cache.put(_key("d"), [4.0])  # evicts "b"

        stats = cache.stats()
        assert stats["evictions"] == 2

    def test_update_existing_does_not_evict(self):
        cache = EmbeddingCache(max_size=2, ttl_seconds=0)
        cache.put(_key("a"), [1.0])
        cache.put(_key("b"), [2.0])
        # Update "a" with new vector — should not evict
        cache.put(_key("a"), [10.0])

        assert cache.size == 2
        assert cache.get(_key("a")) == [10.0]
        assert cache.get(_key("b")) == [2.0]


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


class TestTTLExpiration:
    def test_expired_entry_returns_none(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=1)
        key = _key()
        cache.put(key, _vector())

        # Patch time.time to simulate TTL expiration
        original_time = time.time
        with patch("semantic_vector_router.utils.cache.time.time", return_value=original_time() + 2):
            result = cache.get(key)

        assert result is None

    def test_non_expired_entry_returns_vector(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = _key()
        vec = _vector()
        cache.put(key, vec)
        result = cache.get(key)
        assert result == vec

    def test_ttl_zero_means_no_expiration(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=0)
        key = _key()
        vec = _vector()
        cache.put(key, vec)

        # Even far in the future, TTL=0 means no expiry
        original_time = time.time
        with patch("semantic_vector_router.utils.cache.time.time", return_value=original_time() + 999999):
            result = cache.get(key)

        assert result == vec


# ---------------------------------------------------------------------------
# Cache disabled
# ---------------------------------------------------------------------------


class TestCacheDisabled:
    def test_disabled_cache_get_returns_none(self):
        cache = EmbeddingCache(max_size=0, ttl_seconds=3600)
        assert cache.enabled is False
        cache.put(_key(), _vector())
        result = cache.get(_key())
        assert result is None

    def test_disabled_cache_put_is_noop(self):
        cache = EmbeddingCache(max_size=0, ttl_seconds=3600)
        cache.put(_key(), _vector())
        assert cache.size == 0


# ---------------------------------------------------------------------------
# Invalidate
# ---------------------------------------------------------------------------


class TestInvalidate:
    def test_invalidate_existing(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = _key()
        cache.put(key, _vector())
        assert cache.invalidate(key) is True
        assert cache.get(key) is None

    def test_invalidate_nonexistent(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        assert cache.invalidate(_key("missing")) is False


# ---------------------------------------------------------------------------
# Stats and hit rate
# ---------------------------------------------------------------------------


class TestStats:
    def test_hit_rate_no_lookups(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        assert cache.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = _key()
        cache.put(key, _vector())
        cache.get(key)
        cache.get(key)
        assert cache.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = _key()
        cache.put(key, _vector())
        cache.get(key)           # hit
        cache.get(_key("miss"))  # miss
        assert cache.hit_rate == 0.5

    def test_stats_dict(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = _key()
        cache.put(key, _vector())
        cache.get(key)  # hit
        cache.get(_key("miss"))  # miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["evictions"] == 0

    def test_entry_hits_counter(self):
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = _key()
        cache.put(key, _vector())
        cache.get(key)
        cache.get(key)
        cache.get(key)
        # Internal entry should have hits=3
        entry = cache._cache[key]
        assert entry.hits == 3


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_put_get(self):
        cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)
        errors = []

        def writer(start: int):
            try:
                for i in range(100):
                    key = _key(f"thread_{start}_{i}")
                    cache.put(key, [float(i)])
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(_key(f"thread_0_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size <= 1000
