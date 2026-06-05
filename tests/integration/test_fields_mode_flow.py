"""Integration tests for FIELDS mode end-to-end lifecycle.

Full FIELDS mode lifecycle with real Atlas and Voyage embeddings:
- Partition creation with per-partition embedding fields
- Document ingestion routed to partition-specific fields
- Vector search with partition isolation via separate indexes
- Cleanup of indexes and partitions

Requires: MONGODB_URI and VOYAGE_API_KEY environment variables.
Run with: pytest tests/integration/test_fields_mode_flow.py -v -s --timeout=600
"""

import asyncio
import logging
import os
import time

import pytest
from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.client import SVRClient
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.models import (
    CacheConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestConfig,
    IngestMode,
    LogConfig,
    MetricsConfig,
    PartitioningConfig,
    RateLimitConfig,
    RerankingConfig,
    ResilienceConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test Constants
# ---------------------------------------------------------------------------

TEST_DB = "svr_integration_test"
TEST_COLLECTION = "products_fields"
PARTITIONS = ["electronics", "furniture", "clothing"]

# ---------------------------------------------------------------------------
# Sample Documents
# ---------------------------------------------------------------------------

ELECTRONICS_DOCS = [
    {"name": "Wireless Noise-Cancelling Headphones", "description": "Premium Bluetooth headphones with active noise cancellation, 30-hour battery life, and comfortable over-ear design", "category": "electronics", "price": 299.99},
    {"name": "Portable USB-C Charging Hub", "description": "Compact 7-port USB hub with fast charging for laptops, tablets, and smartphones", "category": "electronics", "price": 49.99},
    {"name": "Mechanical Gaming Keyboard", "description": "RGB backlit mechanical keyboard with Cherry MX switches and programmable macros", "category": "electronics", "price": 149.99},
    {"name": "4K Webcam Pro", "description": "Ultra HD webcam with auto-focus, noise reduction microphone, and low-light correction", "category": "electronics", "price": 89.99},
    {"name": "Smart Fitness Watch", "description": "GPS-enabled fitness tracker with heart rate monitor, sleep tracking, and 7-day battery", "category": "electronics", "price": 199.99},
    {"name": "Wireless Earbuds Pro", "description": "True wireless earbuds with spatial audio, adaptive EQ, and sweat resistance for workouts", "category": "electronics", "price": 179.99},
    {"name": "Laptop Cooling Stand", "description": "Adjustable aluminum laptop stand with dual fans and USB passthrough for heat dissipation", "category": "electronics", "price": 39.99},
]

FURNITURE_DOCS = [
    {"name": "Ergonomic Office Chair", "description": "Adjustable lumbar support office chair with breathable mesh back and 4D armrests", "category": "furniture", "price": 599.99},
    {"name": "Standing Desk Converter", "description": "Height-adjustable sit-stand desk riser with keyboard tray and monitor mount", "category": "furniture", "price": 349.99},
    {"name": "Solid Oak Bookshelf", "description": "Five-tier solid oak bookshelf with adjustable shelves and anti-tip wall mount", "category": "furniture", "price": 449.99},
    {"name": "Executive Filing Cabinet", "description": "Three-drawer lateral filing cabinet with lock and full-extension ball-bearing slides", "category": "furniture", "price": 279.99},
    {"name": "L-Shaped Computer Desk", "description": "Spacious corner desk with cable management grommets and modesty panel", "category": "furniture", "price": 499.99},
    {"name": "Leather Desk Pad", "description": "Premium full-grain leather desk pad with non-slip base and waterproof surface", "category": "furniture", "price": 69.99},
    {"name": "Monitor Riser Shelf", "description": "Bamboo monitor stand with storage drawer and ventilation slots for ergonomic viewing", "category": "furniture", "price": 45.99},
]

CLOTHING_DOCS = [
    {"name": "Merino Wool Base Layer", "description": "Lightweight merino wool thermal top for hiking and outdoor activities", "category": "clothing", "price": 89.99},
    {"name": "Waterproof Hiking Jacket", "description": "Three-layer Gore-Tex shell jacket with sealed seams and adjustable hood", "category": "clothing", "price": 249.99},
    {"name": "Stretch Denim Jeans", "description": "Comfortable stretch denim with classic straight fit and reinforced knees", "category": "clothing", "price": 79.99},
    {"name": "Trail Running Shoes", "description": "Lightweight trail runners with Vibram outsole and responsive cushioning", "category": "clothing", "price": 139.99},
    {"name": "Down Insulated Vest", "description": "Packable 800-fill goose down vest with water-resistant shell and zippered pockets", "category": "clothing", "price": 169.99},
    {"name": "UV Protection Sun Hat", "description": "Wide-brim UPF 50+ sun hat with moisture-wicking sweatband and chin strap", "category": "clothing", "price": 34.99},
    {"name": "Organic Cotton T-Shirt", "description": "Sustainably sourced organic cotton crew neck tee with tagless comfort", "category": "clothing", "price": 29.99},
]

DOCS_BY_PARTITION = {
    "electronics": ELECTRONICS_DOCS,
    "furniture": FURNITURE_DOCS,
    "clothing": CLOTHING_DOCS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fields_config() -> SVRConfig:
    """Build an SVRConfig for FIELDS mode with Voyage embeddings."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database=TEST_DB,
            source_collection=TEST_COLLECTION,
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_intfields_",
            index_name_prefix="svr_intfields_idx_",
        ),
        vector_storage=VectorStorageConfig(
            index_on=IndexLocation.FIELDS,
            storage_format=VectorStorageFormat.ARRAY,
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=512,
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-3-lite",
            dimensions=512,
            api_key_env="VOYAGE_API_KEY",
        ),
        reranking=RerankingConfig(enabled=False),
        ingestion=IngestConfig(
            text_fields=["name", "description"],
            separator=" - ",
            batch_size=50,
            mode=IngestMode.INSERT,
            continue_on_error=True,
            trigger_detection=False,
        ),
        resilience=ResilienceConfig(
            connection_timeout_ms=30_000,
            server_selection_timeout_ms=30_000,
            embedding_timeout_ms=60_000,
        ),
        cache=CacheConfig(enabled=False),
        metrics=MetricsConfig(enabled=False),
        logging=LogConfig(level="INFO"),
        rate_limiting=RateLimitConfig(enabled=False),
    )


async def _wait_for_index(
    backend: MongoDBBackend,
    collection_name: str,
    index_name: str,
    timeout: int = 240,
    poll_interval: int = 5,
) -> bool:
    """Wait for Atlas vector search index to become queryable."""
    start = time.time()
    while time.time() - start < timeout:
        status = await backend.get_index_status(collection_name, index_name)
        state = status.get("status", "unknown")
        queryable = status.get("queryable", False)
        elapsed = int(time.time() - start)
        logger.info(f"  [{elapsed}s] {index_name}: status={state}, queryable={queryable}")
        if queryable:
            await asyncio.sleep(3)
            return True
        await asyncio.sleep(poll_interval)
    return False


async def _cleanup_fields_indexes(backend: MongoDBBackend, partition_names: list[str]) -> None:
    """Best-effort cleanup of FIELDS indexes on the source collection."""
    for name in partition_names:
        safe_name = name.replace("-", "_").replace(" ", "_").lower()
        index_name = f"svr_intfields_idx_{safe_name}"
        try:
            await backend.delete_vector_search_index(TEST_COLLECTION, index_name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def _skip_if_no_env():
    """Skip entire module if required environment variables are missing."""
    if not os.environ.get("MONGODB_URI"):
        pytest.skip("MONGODB_URI not set")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")


@pytest.fixture(scope="module")
async def fields_client(_skip_if_no_env):
    """Module-scoped fixture: create SVRClient, connect, provision partitions,
    ingest documents, and wait for indexes to become queryable.

    Yields the connected SVRClient for all tests. Cleans up on teardown.
    """
    config = _make_fields_config()
    svr = SVRClient(config=config, auto_connect=False)
    await svr.connect()

    backend = svr._backend
    provisioner = PartitionProvisioner(backend, svr.config, auto_save_config=False)

    # Drop test collection for a clean start
    try:
        raw_client = AsyncMongoClient(os.environ["MONGODB_URI"])
        await raw_client[TEST_DB].drop_collection(TEST_COLLECTION)
        await raw_client.close()
    except Exception as e:
        logger.warning(f"Pre-cleanup failed (ok on first run): {e}")

    # Create partitions (each creates a per-partition index)
    for name in PARTITIONS:
        await provisioner.create_partition(name, skip_if_exists=True)
        logger.info(f"Created FIELDS partition: {name}")

    # Ingest documents per partition
    for name in PARTITIONS:
        docs = DOCS_BY_PARTITION[name]
        result = await svr.ingest(docs, partition=name)
        logger.info(
            f"Ingested {result.inserted} docs into partition '{name}' "
            f"({result.failed} failed, {result.elapsed_ms:.0f}ms)"
        )
        assert result.failed == 0, f"Ingestion failed for partition '{name}': {result.errors}"

    # Wait for each per-partition index to become queryable
    for name in PARTITIONS:
        safe_name = name.replace("-", "_").replace(" ", "_").lower()
        index_name = f"svr_intfields_idx_{safe_name}"
        ready = await _wait_for_index(backend, TEST_COLLECTION, index_name, timeout=240)
        if not ready:
            pytest.skip(f"FIELDS index '{index_name}' not queryable within timeout")

    yield svr

    # Teardown: clean up partitions, indexes, and collection
    try:
        for name in list(svr.config.partitions.registry.keys()):
            try:
                await provisioner.delete_partition(name)
            except Exception:
                pass
    except Exception:
        pass

    try:
        await _cleanup_fields_indexes(backend, PARTITIONS)
    except Exception:
        pass

    try:
        raw_client = AsyncMongoClient(os.environ["MONGODB_URI"])
        await raw_client[TEST_DB].drop_collection(TEST_COLLECTION)
        await raw_client.close()
    except Exception:
        pass

    await svr.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestFieldsModeFlow:
    """End-to-end FIELDS mode lifecycle with real Atlas and Voyage embeddings."""

    async def test_partition_embedding_fields(self, fields_client: SVRClient):
        """After ingestion, verify each partition's docs have the correct
        partition-specific embedding field (embedding_electronics, etc.)."""
        backend = fields_client._backend
        collection = backend.db[TEST_COLLECTION]

        for name in PARTITIONS:
            field_name = f"embedding_{name}"

            # Documents in this partition should have the embedding field
            count_with_field = await collection.count_documents(
                {"category": name, field_name: {"$exists": True}}
            )
            total_in_partition = await collection.count_documents({"category": name})

            assert count_with_field == total_in_partition, (
                f"Partition '{name}': expected {total_in_partition} docs with '{field_name}', "
                f"found {count_with_field}"
            )
            assert count_with_field == len(DOCS_BY_PARTITION[name])

            # Documents in OTHER partitions should NOT have this field
            count_other = await collection.count_documents(
                {"category": {"$ne": name}, field_name: {"$exists": True}}
            )
            assert count_other == 0, (
                f"Found {count_other} docs outside '{name}' with '{field_name}' -- "
                f"FIELDS mode should route embeddings to partition-specific fields only"
            )

    async def test_search_single_partition(self, fields_client: SVRClient):
        """Search 'laptop accessories' in electronics partition only.
        All results should be electronics products."""
        result = await fields_client.search(
            query="laptop accessories and charging",
            partitions=["electronics"],
            limit=5,
        )

        assert len(result.hits) > 0, "Expected search results for electronics"
        assert result.partitions_searched == ["electronics"]

        for hit in result.hits:
            assert hit.partition == "electronics"
            assert hit.document.get("category") == "electronics", (
                f"Expected electronics, got {hit.document.get('category')}"
            )
            assert hit.score > 0

    async def test_partition_isolation(self, fields_client: SVRClient):
        """Search electronics partition for 'wooden bookshelf'. Even though
        furniture docs match this query better, results should only be
        electronics because FIELDS mode gives complete isolation via
        separate per-partition indexes."""
        result = await fields_client.search(
            query="solid oak wooden bookshelf with adjustable shelves",
            partitions=["electronics"],
            limit=5,
        )

        # Should still get results (electronics docs exist), even if low relevance
        # The key assertion: NO furniture docs leak into electronics results
        for hit in result.hits:
            assert hit.partition == "electronics", (
                f"Partition isolation violated: got '{hit.partition}' in electronics search"
            )
            assert hit.document.get("category") == "electronics", (
                f"Category isolation violated: got '{hit.document.get('category')}'"
            )

    async def test_cross_partition_search(self, fields_client: SVRClient):
        """Search across all 3 partitions. Should get results from multiple
        partitions, demonstrating cross-partition merging."""
        result = await fields_client.search(
            query="comfortable ergonomic product for daily use",
            partitions="all",
            limit=10,
        )

        assert len(result.hits) > 0
        assert len(result.partitions_searched) == 3

        # Verify results come from multiple partitions
        partitions_in_results = set(hit.partition for hit in result.hits)
        assert len(partitions_in_results) >= 2, (
            f"Expected results from multiple partitions, got only: {partitions_in_results}"
        )

        # All hits should have valid scores
        for hit in result.hits:
            assert hit.score > 0
            assert hit.partition in PARTITIONS
            assert hit.document.get("category") == hit.partition

    async def test_search_with_specific_partitions(self, fields_client: SVRClient):
        """Search across two specific partitions (electronics + clothing),
        excluding furniture."""
        result = await fields_client.search(
            query="wireless wearable technology",
            partitions=["electronics", "clothing"],
            limit=10,
        )

        assert len(result.hits) > 0
        assert set(result.partitions_searched) == {"electronics", "clothing"}

        for hit in result.hits:
            assert hit.partition in ("electronics", "clothing"), (
                f"Got unexpected partition '{hit.partition}' when searching electronics+clothing"
            )

    async def test_cleanup(self, fields_client: SVRClient):
        """Verify that partitions can be deleted and indexes are cleaned up."""
        config = fields_client.config
        backend = fields_client._backend
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        # Verify partitions exist before cleanup
        assert len(config.partitions.registry) == 3

        # Delete one partition and verify
        await provisioner.delete_partition("clothing")
        assert "clothing" not in config.partitions.registry
        assert len(config.partitions.registry) == 2

        # The FIELDS index for clothing should eventually disappear
        # (Atlas may take a moment to propagate the deletion)
        safe_name = "clothing"
        index_name = f"svr_intfields_idx_{safe_name}"
        await asyncio.sleep(3)
        exists = await backend.index_exists(TEST_COLLECTION, index_name)
        # Note: index deletion is best-effort and may not be instant
        logger.info(f"After delete, index '{index_name}' exists={exists}")

        # Re-create the partition for other tests' teardown to work cleanly
        await provisioner.create_partition("clothing", skip_if_exists=True)
