"""End-to-end centroid routing integration test with real Atlas and Voyage embeddings.

Full lifecycle: create partitions -> ingest documents -> wait for index ->
compute centroids -> search WITHOUT explicit partition hints -> verify correct
partition was selected via centroid similarity.

Validates that the centroid routing cascParts Distributor (filter-map -> centroid -> fallback)
routes queries to the semantically correct partition using precomputed centroid
embeddings, reducing fan-out from O(N) to O(log N).

Requires: MONGODB_URI and VOYAGE_API_KEY in .env.
Run with: .venv/bin/pytest tests/integration/test_centroid_routing_flow.py -v -s --timeout=600 -m integration
"""

import asyncio
import logging
import os

import pytest
from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from semantic_vector_router.client import SVRClient
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.models import (
    CentroidRoutingConfig,
    IndexLocation,
    RoutingConfig,
    RoutingMode,
)
from semantic_vector_router.routing.centroid import compute_partition_centroid
from tests.integration.conftest import (
    ALL_DOCS,
    CLOTHING_DOCS,
    ELECTRONICS_DOCS,
    FURNITURE_DOCS,
    INTEGRATION_TEST_DB,
    CapturingMetricsHandler,
    make_svr_config,
    wait_for_index,
)

load_dotenv()
logger = logging.getLogger(__name__)

TEST_COLLECTION = "centroid_routing_test"
CATEGORIES = ["electronics", "furniture", "clothing"]

# Use module-scoped event loop for all async fixtures and tests in this module
pytestmark = pytest.mark.asyncio(loop_scope="module")


# ---------------------------------------------------------------------------
# Module-scoped fixture: set up SOURCE mode client with centroid routing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metrics_capture() -> CapturingMetricsHandler:
    return CapturingMetricsHandler()


@pytest.fixture(scope="module")
async def centroid_env(metrics_capture):
    """Set up SOURCE mode client with centroid routing enabled.

    Creates partitions, ingests docs with real Voyage embeddings, waits for
    the index to become queryable, computes centroids for all partitions,
    and stores them in metadata.
    """
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
        await db.drop_collection("svr_metadata")
    except Exception:
        pass

    # Ensure the collection exists (Atlas requires it before index creation)
    await db.create_collection(TEST_COLLECTION)

    # Build config with centroid routing enabled and AUTO mode
    config = make_svr_config(
        index_on=IndexLocation.SOURCE,
        collection_name=TEST_COLLECTION,
        view_prefix="svr_crt_",
        index_name_prefix="svr_crt_idx_",
    )
    config.routing = RoutingConfig(
        mode=RoutingMode.AUTO,
        default_partitions="all",
        max_partitions_per_query=5,
        centroid_routing=CentroidRoutingConfig(
            enabled=True,
            relative_threshold=0.5,
            min_score=0.10,
            max_probe_partitions=3,
            sample_size=50,
        ),
    )

    client = SVRClient(config=config, auto_connect=False, metrics_handler=metrics_capture)
    await client.connect()

    provisioner = PartitionProvisioner(
        client._backend, client.config, auto_save_config=False
    )

    # Create partitions and save to metadata store (resolver reads from metadata)
    partitions = {}
    for cat in CATEGORIES:
        p = await provisioner.create_partition(cat)
        partitions[cat] = p
        if client._metadata is not None:
            await client._metadata.save_partition(p)
        logger.info(f"Created centroid routing partition: {cat} -> index={p.index_name}")

    # Ingest documents per partition (7 docs each, 21 total)
    for cat, docs in [
        ("electronics", ELECTRONICS_DOCS),
        ("furniture", FURNITURE_DOCS),
        ("clothing", CLOTHING_DOCS),
    ]:
        result = await client.ingest(docs, partition=cat)
        logger.info(f"Ingested {result.inserted} docs into {cat}")

    # Wait for the shared source index to become queryable
    index_ready = await wait_for_index(
        client._backend, TEST_COLLECTION, "svr_vector_idx_source", timeout=300
    )

    # Compute and store centroids for all 3 partitions
    centroids_computed = {}
    collection = client._backend.db[TEST_COLLECTION]
    embedding_field = config.vector_search.embedding_field

    for cat in CATEGORIES:
        partition_filter = {config.partitioning.field: cat}
        centroid = await compute_partition_centroid(
            collection=collection,
            embedding_field=embedding_field,
            partition_filter=partition_filter,
            sample_size=50,
        )
        if centroid and client._metadata is not None:
            await client._metadata.update_centroid(cat, centroid)
            centroids_computed[cat] = len(centroid)
            logger.info(
                f"Computed centroid for {cat}: {len(centroid)} dimensions"
            )
        else:
            logger.warning(f"Failed to compute centroid for {cat}")

    yield {
        "client": client,
        "provisioner": provisioner,
        "partitions": partitions,
        "index_ready": index_ready,
        "metrics": metrics_capture,
        "centroids_computed": centroids_computed,
    }

    # Cleanup
    try:
        for cat in CATEGORIES:
            try:
                await provisioner.delete_partition(cat)
            except Exception:
                pass
        try:
            await client._backend.delete_vector_search_index(
                TEST_COLLECTION, "svr_vector_idx_source"
            )
        except Exception:
            pass
    finally:
        await client.disconnect()
        await db.drop_collection(TEST_COLLECTION)
        await db.drop_collection("svr_metadata")
        await mongo.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCentroidRoutingFlow:
    """End-to-end centroid routing with real Atlas and Voyage embeddings."""

    async def test_centroids_computed(self, centroid_env):
        """Verify centroids were computed for all 3 partitions."""
        centroids = centroid_env["centroids_computed"]
        assert len(centroids) == 3, f"Expected 3 centroids, got {len(centroids)}"
        for cat in CATEGORIES:
            assert cat in centroids, f"Missing centroid for {cat}"
            assert centroids[cat] > 0, f"Centroid for {cat} has 0 dimensions"

    async def test_search_routes_to_electronics(self, centroid_env):
        """Search for headphones without partition hint routes to electronics."""
        if not centroid_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = centroid_env["client"]
        metrics = centroid_env["metrics"]
        metrics.clear()

        result = await client.search(
            "wireless bluetooth headphones noise cancelling audio",
            partitions="all",
            limit=5,
        )

        assert len(result.hits) > 0, "Expected results from centroid routing"

        # The majority of top results should be from electronics
        electronics_hits = [h for h in result.hits if h.partition == "electronics"]
        assert len(electronics_hits) >= 1, (
            f"Expected electronics results for headphones query, got partitions: "
            f"{[h.partition for h in result.hits]}"
        )

        # Top result should be electronics
        top_hit = result.hits[0]
        assert top_hit.partition == "electronics", (
            f"Expected top result from electronics, got {top_hit.partition}: "
            f"{top_hit.document.get('name')}"
        )

        # Verify centroid routing metrics were emitted
        assert metrics.has("centroid_route_latency"), (
            "No centroid_route_latency metric emitted"
        )
        assert metrics.has("centroid_route_partitions"), (
            "No centroid_route_partitions metric emitted"
        )

    async def test_search_routes_to_furniture(self, centroid_env):
        """Search for desk/chair without partition hint routes to furniture."""
        if not centroid_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = centroid_env["client"]

        result = await client.search(
            "ergonomic office desk chair with lumbar support",
            partitions="all",
            limit=5,
        )

        assert len(result.hits) > 0, "Expected results for furniture query"

        # The majority of top results should be from furniture
        furniture_hits = [h for h in result.hits if h.partition == "furniture"]
        assert len(furniture_hits) >= 1, (
            f"Expected furniture results for desk/chair query, got partitions: "
            f"{[h.partition for h in result.hits]}"
        )

        # Top result should be furniture
        top_hit = result.hits[0]
        assert top_hit.partition == "furniture", (
            f"Expected top result from furniture, got {top_hit.partition}: "
            f"{top_hit.document.get('name')}"
        )

    async def test_explicit_partition_bypasses_centroid(self, centroid_env):
        """Explicit partition list bypasses centroid routing (backward compat)."""
        if not centroid_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = centroid_env["client"]
        metrics = centroid_env["metrics"]
        metrics.clear()

        # Search for headphones but explicitly target furniture only
        result = await client.search(
            "wireless bluetooth headphones",
            partitions=["furniture"],
            limit=5,
        )

        assert len(result.hits) > 0, "Expected results from furniture partition"
        assert result.partitions_searched == ["furniture"]

        # All results must be from furniture (centroid routing bypassed)
        for hit in result.hits:
            assert hit.partition == "furniture", (
                f"Partition isolation violated: got {hit.partition} in furniture search"
            )

        # Centroid routing metrics should NOT be emitted for explicit partition
        assert not metrics.has("centroid_route_latency"), (
            "centroid_route_latency should not be emitted for explicit partition"
        )

    async def test_filter_map_routing_takes_priority(self, centroid_env):
        """Filter-map routing (category filter) takes priority over centroid."""
        if not centroid_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = centroid_env["client"]
        metrics = centroid_env["metrics"]
        metrics.clear()

        # Search with an explicit category filter - should use filter-map routing,
        # not centroid routing, even though centroid routing is enabled
        result = await client.search(
            "comfortable premium product",
            partitions="all",
            filters={"category": "clothing"},
            limit=5,
        )

        assert len(result.hits) > 0, "Expected results from clothing partition"

        # All results should be from clothing (filter-map routing)
        for hit in result.hits:
            assert hit.partition == "clothing", (
                f"Filter routing violated: got {hit.partition} when filtering for clothing"
            )

        # Centroid routing metrics should NOT be emitted when filter-map resolves
        assert not metrics.has("centroid_route_latency"), (
            "centroid_route_latency should not be emitted when filter-map resolves"
        )

    async def test_latency_reasonable(self, centroid_env):
        """Centroid-routed search should complete in under 5 seconds."""
        if not centroid_env["index_ready"]:
            pytest.skip("Index not queryable in time")

        client = centroid_env["client"]

        result = await client.search(
            "hiking jacket waterproof outdoor gear",
            partitions="all",
            limit=5,
        )

        assert result.latency_ms < 5000, (
            f"Search latency {result.latency_ms:.0f}ms exceeds 5s threshold"
        )
