"""Performance benchmarks for search operations.

These tests measure and assert performance characteristics.
They require MONGODB_URI and VOYAGE_API_KEY in .env.

Run with: .venv/bin/pytest tests/performance/ -v -s --timeout=600 -m performance
"""

import asyncio
import time

import pytest


@pytest.mark.performance
class TestSearchPerformance:
    """Benchmarks for search latency, cache speedup, and concurrency."""

    @pytest.mark.asyncio
    async def test_single_partition_search_latency(self, perf_client):
        """Single partition search should complete in <2s (excluding cold start)."""
        client = perf_client

        # Warm up — first call may be slower due to connection pool, cache, etc.
        await client.search("warmup query", partitions=["electronics"], limit=5)

        start = time.perf_counter()
        for _ in range(5):
            result = await client.search(
                "wireless headphones", partitions=["electronics"], limit=10
            )
        elapsed = (time.perf_counter() - start) / 5

        assert elapsed < 2.0, f"Average search latency {elapsed:.2f}s exceeds 2s threshold"
        assert len(result.hits) > 0

    @pytest.mark.asyncio
    async def test_multi_partition_fanout_latency(self, perf_client):
        """3-partition fan-out search should complete in <3s."""
        client = perf_client

        # Warm up
        await client.search(
            "warmup",
            partitions=["electronics", "furniture", "clothing"],
            limit=5,
        )

        start = time.perf_counter()
        result = await client.search(
            "comfortable products for home office",
            partitions=["electronics", "furniture", "clothing"],
            limit=10,
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"Fan-out search latency {elapsed:.2f}s exceeds 3s threshold"

    @pytest.mark.asyncio
    async def test_embedding_cache_speedup(self, perf_client):
        """Cache hit should be >3x faster than cache miss."""
        client = perf_client
        query = "unique cache test query for speedup measurement xyz123"

        # Clear any prior cached entry for this query
        client._embedding_cache.clear()

        # Cold (cache miss) — embedding API call required
        start = time.perf_counter()
        await client.search(query, partitions=["electronics"], limit=5)
        cold = time.perf_counter() - start

        # Warm (cache hit) — embedding served from cache
        start = time.perf_counter()
        await client.search(query, partitions=["electronics"], limit=5)
        warm = time.perf_counter() - start

        speedup = cold / warm if warm > 0 else float("inf")
        print(f"Cache speedup: {speedup:.1f}x (cold={cold:.3f}s, warm={warm:.3f}s)")
        assert speedup > 3.0, f"Cache speedup only {speedup:.1f}x (expected >3x)"

    @pytest.mark.asyncio
    async def test_concurrent_search_throughput(self, perf_client):
        """20 concurrent searches should complete without errors."""
        client = perf_client
        queries = [f"concurrent test query number {i}" for i in range(20)]

        async def search_one(q):
            return await client.search(q, partitions=["electronics"], limit=5)

        start = time.perf_counter()
        results = await asyncio.gather(*[search_one(q) for q in queries])
        elapsed = time.perf_counter() - start

        assert all(r.hits is not None for r in results)
        assert elapsed < 60.0, f"20 concurrent searches took {elapsed:.1f}s"
        print(f"Concurrent throughput: 20 searches in {elapsed:.1f}s")

    @pytest.mark.asyncio
    async def test_repeated_search_consistency(self, perf_client):
        """Repeated identical searches should return consistent results."""
        client = perf_client
        query = "bluetooth speaker portable"

        results = []
        for _ in range(3):
            result = await client.search(
                query, partitions=["electronics"], limit=5
            )
            results.append(result)

        # All runs should return the same hit IDs in the same order
        first_ids = [h.id for h in results[0].hits]
        for i, r in enumerate(results[1:], 2):
            ids = [h.id for h in r.hits]
            assert ids == first_ids, (
                f"Run {i} returned different results: {ids} vs {first_ids}"
            )

    @pytest.mark.asyncio
    async def test_search_with_filters_latency(self, perf_client):
        """Filtered search should complete in <2s."""
        client = perf_client

        # Warm up
        await client.search(
            "product", partitions=["electronics"], limit=5, filters={"in_stock": True}
        )

        start = time.perf_counter()
        result = await client.search(
            "wireless headphones",
            partitions=["electronics"],
            limit=5,
            filters={"in_stock": True},
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Filtered search latency {elapsed:.2f}s exceeds 2s threshold"
        print(f"Filtered search: {len(result.hits)} hits in {elapsed:.3f}s")
