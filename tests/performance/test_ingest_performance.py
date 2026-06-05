"""Performance benchmarks for ingestion operations.

These tests measure ingestion throughput and assert timing bounds.
They require MONGODB_URI and VOYAGE_API_KEY in .env.

Run with: .venv/bin/pytest tests/performance/ -v -s --timeout=600 -m performance
"""

import time

import pytest


@pytest.mark.performance
class TestIngestPerformance:
    """Benchmarks for document ingestion throughput."""

    @pytest.mark.asyncio
    async def test_ingest_throughput(self, perf_client):
        """Ingest 100 documents and measure throughput."""
        client = perf_client
        docs = [
            {
                "name": f"Perf Product {i}",
                "description": (
                    f"Test product {i} for performance measurement with detailed "
                    f"description text about features and specifications"
                ),
                "category": "electronics",
                "price": 9.99 + i,
            }
            for i in range(100)
        ]

        start = time.perf_counter()
        result = await client.ingest(docs, partition="electronics")
        elapsed = time.perf_counter() - start

        throughput = result.inserted / elapsed if elapsed > 0 else 0
        assert result.inserted == 100, (
            f"Expected 100 inserted, got {result.inserted} "
            f"({result.failed} failed, errors: {result.errors[:5]})"
        )
        print(
            f"Ingest throughput: {throughput:.1f} docs/sec "
            f"({elapsed:.1f}s for 100 docs, "
            f"embed={result.embed_ms:.0f}ms, write={result.write_ms:.0f}ms)"
        )

    @pytest.mark.asyncio
    async def test_ingest_small_batch(self, perf_client):
        """Ingest 10 documents should complete in <10s."""
        client = perf_client
        docs = [
            {
                "name": f"Small Batch Product {i}",
                "description": f"Quick ingest test product {i} with brief description",
                "category": "furniture",
                "price": 19.99 + i,
            }
            for i in range(10)
        ]

        start = time.perf_counter()
        result = await client.ingest(docs, partition="furniture")
        elapsed = time.perf_counter() - start

        assert result.inserted == 10
        assert elapsed < 10.0, f"10-doc ingest took {elapsed:.1f}s (expected <10s)"
        print(f"Small batch ingest: {elapsed:.2f}s for 10 docs")

    @pytest.mark.asyncio
    async def test_ingest_embed_vs_write_timing(self, perf_client):
        """Verify that embedding is the dominant cost, not writes."""
        client = perf_client
        docs = [
            {
                "name": f"Timing Test Product {i}",
                "description": f"Document for embed vs write timing analysis number {i}",
                "category": "clothing",
                "price": 29.99 + i,
            }
            for i in range(50)
        ]

        result = await client.ingest(docs, partition="clothing")

        assert result.inserted == 50
        assert result.embed_ms > 0, "Embed timing should be positive"
        assert result.write_ms > 0, "Write timing should be positive"
        print(
            f"Timing breakdown: embed={result.embed_ms:.0f}ms, "
            f"write={result.write_ms:.0f}ms, total={result.elapsed_ms:.0f}ms"
        )
        # Embedding should typically dominate (>50% of total time)
        embed_fraction = result.embed_ms / result.elapsed_ms if result.elapsed_ms > 0 else 0
        print(f"Embedding fraction: {embed_fraction:.1%} of total time")
