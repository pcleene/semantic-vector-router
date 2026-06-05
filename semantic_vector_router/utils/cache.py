"""Embedding cache for Semantic Vector Router.

In-memory LRU cache with TTL for embedding vectors. Repeated identical
queries skip the embedding API call entirely.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key for embedding lookups."""

    text: str
    model: str
    dimensions: int
    input_type: str  # "query" or "document"

    def __hash__(self) -> int:
        return hash((self.text, self.model, self.dimensions, self.input_type))


@dataclass
class CacheEntry:
    """Cache entry with expiration tracking.

    Attributes:
        vector: The cached embedding vector.
        created_at: Epoch timestamp when the entry was created.
        hits: Number of times this entry has been retrieved.
    """

    vector: list[float]
    created_at: float
    hits: int = 0


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors.

    Args:
        max_size: Maximum number of entries. 0 = disabled.
        ttl_seconds: Time-to-live for entries. 0 = no expiration.
    """

    def __init__(self, max_size: int = 10_000, ttl_seconds: int = 3600) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        """Cache is enabled if max_size > 0."""
        return self._max_size > 0

    def get(self, key: CacheKey) -> Optional[list[float]]:
        """Get cached embedding vector, or None if not found/expired.

        Moves accessed entry to end of LRU order on a hit.

        Args:
            key: The cache key to look up.

        Returns:
            The cached vector, or None on miss or expiration.
        """
        if not self.enabled:
            self._misses += 1
            return None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            # Check TTL
            if self._ttl_seconds > 0 and (time.time() - entry.created_at) > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            return entry.vector

    def put(self, key: CacheKey, vector: list[float]) -> None:
        """Store embedding vector. Evicts LRU entry if at capacity.

        Args:
            key: The cache key.
            vector: The embedding vector to cache.
        """
        if not self.enabled:
            return
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache[key] = CacheEntry(vector=vector, created_at=time.time())
                self._cache.move_to_end(key)
            else:
                # Evict if at capacity
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)  # Remove oldest
                    self._evictions += 1
                self._cache[key] = CacheEntry(vector=vector, created_at=time.time())

    def invalidate(self, key: CacheKey) -> bool:
        """Remove a specific entry.

        Args:
            key: The cache key to remove.

        Returns:
            True if the entry was found and removed, False otherwise.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0). Returns 0.0 if no lookups."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict[str, int | float]:
        """Return cache statistics.

        Returns:
            Dict with keys: size, max_size, hits, misses, hit_rate, evictions.
        """
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "evictions": self._evictions,
        }
