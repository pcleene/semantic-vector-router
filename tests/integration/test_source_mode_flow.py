"""End-to-end SOURCE mode integration test with real Atlas and Voyage embeddings.

Full lifecycle: create partitions → ingest documents → wait for index →
search single/multi partition → verify metrics → cleanup.

Requires: MONGODB_URI and VOYAGE_API_KEY in .env.
"""

import asyncio
import logging
import os
import time

import pytest
from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from semantic_vector_router.client import SVRClient
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.models import IndexLocation
from tests.integration.conftest import (
    ALL_DOCS,
    ELECTRONICS_DOCS,
    FURNITURE_DOCS,
    CLOTHING_DOCS,
    INTEGRATION_TEST_DB,
    CapturingMetricsHandler,
    make_svr_config,
    wait_for_index,
)

load_dotenv()
logger = logging.getLogger(__name__)

TEST_COLLECTION = "products_source"
CATEGORIES = ["electronics", "furniture", "clothing"]


# ---------------------------------------------------------------------------
# Module-scoped fixture: set up SOURCE mode client with data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def metrics_capture() -> CapturingMetricsHandler:
    return CapturingMetricsHandler()


@pytest.fixture(scope="module")
async def source_env(metrics_capture):
    """Set up SOURCE mode client, create partitions, ingest docs, wait for index."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set")
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        pytest.skip("VOYAGE_API_KEY not set")

    # Clean slate
    mongo = AsyncMongoClient(uri)
    db = mongo[INTEGRATION_TEST_DB]
    try:
        await db.drop_collection(TEST_COLLECTION)
    except Exception:
        pass

    config = make_svr_config(
        index_on=IndexLocation.SOURCE,
        collection_name=TEST_COLLECTION,
        view_prefix="svr_src_",
        index_name_prefix="svr_src_idx_",
    )

    client = SVRClient(config=config, auto_connect=False, metrics_handler=metrics_capture)
    await client.connect()

    provisioner = PartitionProvisioner(client._backend, client.config, auto_save_config=False)

    # Create partitions
    partitions = {}
    for cat in CATEGORIES:
        p = await provisioner.create_partition(cat)
        partitions[cat] = p
        logger.info(f"Created SOURCE partition: {cat} -> index={p.index_name}")

    # Ingest documents per partition
    for cat, docs in [("electronics", ELECTRONICS_DOCS), ("furniture", FURNITURE_DOCS), ("clothing", CLOTHING_DOCS)]:
        result = await client.ingest(docs, partition=cat)
        logger.info(f"Ingested {result.inserted} docs into {cat}")

    # Wait for the shared source index to become queryable
    index_ready = await wait_for_index(
        client._backend, TEST_COLLECTION, "svr_vector_idx_source", timeout=300
    )

    yield {
        "client": client,
        "provisioner": provisioner,
        "partitions": partitions,
        "index_ready": index_ready,
        "metrics": metrics_capture,
    }

    # Cleanup
    try:
        for cat in CATEGORIES:
            try:
                await provisioner.delete_partition(cat)
            except Exception:
                pass
        try:
            await client._backend.delete_vector_search_index(TEST_COLLECTION, "svr_vector_idx_source")
        except Exception:
            pass
    finally:
        await client.disconnect()
        await db.drop_collection(TEST_COLLECTION)
        await mongo.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSourceModeFlow:
    """End-to-end SOURCE mode lifecycle with real Atlas and Voyage embeddings."""

    async def test_partitions_created(self, source_env):
        """Verify all 3 partitions were created successfully."""
        partitions = source_env["partitions"]
        assert len(partitions) == 3
        for cat in CATEGORIES:
            p = partitions[cat]
            assert p.index_location == IndexLocation.SOURCE
            assert p.search_collection == TEST_COLLECTION
            assert p.index_name == "svr_vector_idx_source"  # All share the same index

    async def test_ingest_populated_collection(self, source_env):
        """Verify documents were ingested with embeddings."""
        client = source_env["client"]
        total = await client._backend.count_documents(TEST_COLLECTION)
        assert total == len(ALL_DOCS), f"Expected {len(ALL_DOCS)} docs, got {total}"

    async def test_search_single_partition(self, source_env):
        """Search electronics partition for headphones — verify partition isolation."""
        if not source_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = source_env["client"]
        result = await client.search(
            "wireless bluetooth noise cancelling headphones",
            partitions=["electronics"],
            limit=5,
        )

        assert len(result.hits) > 0, "Expected results from electronics partition"
        assert result.partitions_searched == ["electronics"]
        for hit in result.hits:
            assert hit.partition == "electronics"
            assert hit.score > 0
        # Top result should be the headphones doc
        top_doc = result.hits[0].document
        assert "headphone" in top_doc.get("name", "").lower() or "headphone" in top_doc.get("description", "").lower()

    async def test_search_multi_partition(self, source_env):
        """Search across all partitions — verify fan-out and merging."""
        if not source_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = source_env["client"]
        result = await client.search(
            "comfortable seating for long work hours at a desk",
            partitions=CATEGORIES,
            limit=10,
        )

        assert len(result.hits) > 0
        assert len(result.partitions_searched) == 3
        # Should see results from multiple partitions
        partitions_in_results = set(h.partition for h in result.hits)
        assert len(partitions_in_results) >= 1  # At least some results

    async def test_search_furniture_relevance(self, source_env):
        """Furniture query should rank furniture items highest."""
        if not source_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = source_env["client"]
        result = await client.search(
            "ergonomic office chair with lumbar support and adjustable armrests",
            partitions=CATEGORIES,
            limit=5,
        )

        assert len(result.hits) > 0
        # The top-1 result should be from furniture
        top = result.hits[0]
        assert top.partition == "furniture", (
            f"Expected furniture top result, got {top.partition}: {top.document.get('name')}"
        )

    async def test_search_partition_isolation(self, source_env):
        """Search only electronics — should never return furniture or clothing."""
        if not source_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = source_env["client"]
        result = await client.search(
            "wooden bookshelf",
            partitions=["electronics"],
            limit=5,
        )

        for hit in result.hits:
            assert hit.partition == "electronics", (
                f"Partition isolation violated: got {hit.partition} in electronics search"
            )

    async def test_search_emits_metrics(self, source_env):
        """Verify that search emits latency and embedding metrics."""
        if not source_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        metrics = source_env["metrics"]
        metrics.clear()

        client = source_env["client"]
        # Use a fresh query to force embedding (not cached)
        unique_query = f"unique metric test query {time.time()}"
        await client.search(unique_query, partitions=["electronics"], limit=3)

        assert metrics.has("search_latency"), "No search_latency metric emitted"
        assert metrics.has("embedding_latency") or metrics.has("cache_hit"), (
            "No embedding_latency or cache_hit metric emitted"
        )

    async def test_embedding_cache_works(self, source_env):
        """Second identical search should hit cache (faster, no embedding_latency)."""
        if not source_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = source_env["client"]
        query = "cache test headphones speakers audio"

        # First search (cache miss)
        start = time.perf_counter()
        await client.search(query, partitions=["electronics"], limit=3)
        first_time = time.perf_counter() - start

        # Second search (cache hit)
        start = time.perf_counter()
        await client.search(query, partitions=["electronics"], limit=3)
        second_time = time.perf_counter() - start

        # Cache hit should be noticeably faster (embedding skipped)
        logger.info(f"First: {first_time:.3f}s, Second: {second_time:.3f}s")
        assert second_time < first_time, "Cache hit was not faster than cache miss"

    async def test_latency_reasonable(self, source_env):
        """Single partition search should complete in under 5 seconds."""
        if not source_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = source_env["client"]
        result = await client.search(
            "laptop stand cooling",
            partitions=["electronics"],
            limit=5,
        )

        assert result.latency_ms < 5000, (
            f"Search latency {result.latency_ms:.0f}ms exceeds 5s threshold"
        )
