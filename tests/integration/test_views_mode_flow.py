"""Integration tests for VIEWS mode end-to-end lifecycle.

Full VIEWS mode lifecycle with real Atlas and Voyage embeddings:
- Partition creation with views for counting/browsing
- Shared source index for search (Atlas doesn't support search indexes on views)
- Document ingestion to the standard embedding field
- Vector search with partition pre-filtering
- Cleanup of views, indexes, and partitions

Requires: MONGODB_URI and VOYAGE_API_KEY environment variables.
Run with: pytest tests/integration/test_views_mode_flow.py -v -s --timeout=600
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
TEST_COLLECTION = "products_views"
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


def _make_views_config() -> SVRConfig:
    """Build an SVRConfig for VIEWS mode with Voyage embeddings."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database=TEST_DB,
            source_collection=TEST_COLLECTION,
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_intviews_",
            index_name_prefix="svr_intviews_idx_",
        ),
        vector_storage=VectorStorageConfig(
            index_on=IndexLocation.VIEWS,
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


async def _cleanup_views_and_indexes(
    backend: MongoDBBackend,
    partition_names: list[str],
) -> None:
    """Best-effort cleanup of views and the shared source index."""
    for name in partition_names:
        view_name = f"svr_intviews_{name}"
        try:
            await backend.delete_partition_view(view_name)
        except Exception:
            pass
    # VIEWS mode uses a shared source index
    try:
        await backend.delete_vector_search_index(TEST_COLLECTION, "svr_vector_idx_source")
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
async def views_client(_skip_if_no_env):
    """Module-scoped fixture: create SVRClient with VIEWS config, connect,
    provision partitions, ingest documents, and wait for the shared source
    index to become queryable.

    Yields the connected SVRClient for all tests. Cleans up on teardown.
    """
    config = _make_views_config()
    svr = SVRClient(config=config, auto_connect=False)
    await svr.connect()

    backend = svr._backend
    provisioner = PartitionProvisioner(backend, svr.config, auto_save_config=False)

    # Drop test collection for a clean start
    try:
        raw_client = AsyncMongoClient(os.environ["MONGODB_URI"])
        db = raw_client[TEST_DB]
        await db.drop_collection(TEST_COLLECTION)
        # Also drop any leftover views from previous runs
        for name in PARTITIONS:
            view_name = f"svr_intviews_{name}"
            try:
                await db.drop_collection(view_name)
            except Exception:
                pass
        await raw_client.close()
    except Exception as e:
        logger.warning(f"Pre-cleanup failed (ok on first run): {e}")

    # Ingest all documents first (VIEWS mode uses standard embedding field)
    all_docs = []
    for name in PARTITIONS:
        all_docs.extend(DOCS_BY_PARTITION[name])

    result = await svr.ingest(all_docs)
    logger.info(
        f"Ingested {result.inserted} docs total "
        f"({result.failed} failed, {result.elapsed_ms:.0f}ms)"
    )
    assert result.failed == 0, f"Ingestion failed: {result.errors}"

    # Create partitions (each creates a view + shared source index)
    for name in PARTITIONS:
        await provisioner.create_partition(name, skip_if_exists=True)
        logger.info(f"Created VIEWS partition: {name}")

    # Wait for shared source index to become queryable
    ready = await _wait_for_index(
        backend, TEST_COLLECTION, "svr_vector_idx_source", timeout=240
    )
    if not ready:
        pytest.skip("Shared source index not queryable within timeout")

    yield svr

    # Teardown: clean up partitions, views, indexes, and collection
    try:
        for name in list(svr.config.partitions.registry.keys()):
            try:
                await provisioner.delete_partition(name)
            except Exception:
                pass
    except Exception:
        pass

    try:
        await _cleanup_views_and_indexes(backend, PARTITIONS)
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
class TestViewsModeFlow:
    """End-to-end VIEWS mode lifecycle with real Atlas and Voyage embeddings."""

    async def test_views_created(self, views_client: SVRClient):
        """Verify MongoDB views exist for each partition."""
        backend = views_client._backend

        for name in PARTITIONS:
            view_name = f"svr_intviews_{name}"
            exists = await backend.view_exists(view_name)
            assert exists, f"View '{view_name}' should exist for partition '{name}'"

    async def test_view_counting(self, views_client: SVRClient):
        """Verify view-based counting returns correct per-partition counts.
        Each view should filter to only its partition's documents."""
        backend = views_client._backend

        for name in PARTITIONS:
            view_name = f"svr_intviews_{name}"
            expected_count = len(DOCS_BY_PARTITION[name])

            view_count = await backend.count_documents(view_name)
            assert view_count == expected_count, (
                f"View '{view_name}' count mismatch: expected {expected_count}, got {view_count}"
            )

        # Total across all views should equal total documents
        total_via_views = 0
        for name in PARTITIONS:
            view_name = f"svr_intviews_{name}"
            total_via_views += await backend.count_documents(view_name)

        total_source = await backend.count_documents(TEST_COLLECTION)
        assert total_via_views == total_source, (
            f"Sum of view counts ({total_via_views}) should equal source count ({total_source})"
        )

    async def test_partition_info_correct(self, views_client: SVRClient):
        """Verify PartitionInfo fields are correct for VIEWS mode."""
        config = views_client.config

        for name in PARTITIONS:
            partition = config.partitions.registry[name]
            assert partition.index_location == IndexLocation.VIEWS
            assert partition.view_name == f"svr_intviews_{name}"
            # VIEWS mode on <8.1: uses shared source index, searches on source
            assert partition.index_name == "svr_vector_idx_source"
            assert partition.search_collection == TEST_COLLECTION
            assert partition.embedding_field is None  # No per-partition fields in VIEWS mode
            assert partition.filter_value == name

    async def test_search_single_partition(self, views_client: SVRClient):
        """Search in electronics partition only. All results should be electronics."""
        result = await views_client.search(
            query="wireless bluetooth headphones and audio accessories",
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

    async def test_search_furniture_isolation(self, views_client: SVRClient):
        """Search furniture partition for a furniture query. Results should
        only come from furniture, demonstrating partition pre-filtering."""
        result = await views_client.search(
            query="adjustable standing desk and office furniture",
            partitions=["furniture"],
            limit=5,
        )

        assert len(result.hits) > 0, "Expected search results for furniture"
        assert result.partitions_searched == ["furniture"]

        for hit in result.hits:
            assert hit.partition == "furniture"
            assert hit.document.get("category") == "furniture", (
                f"Expected furniture, got {hit.document.get('category')}"
            )

    async def test_search_multi_partition(self, views_client: SVRClient):
        """Search across all partitions. Should get results from multiple
        partitions, demonstrating cross-partition result merging."""
        result = await views_client.search(
            query="comfortable durable premium product for everyday use",
            partitions="all",
            limit=15,
        )

        assert len(result.hits) > 0
        assert len(result.partitions_searched) == 3

        # Results should span multiple partitions
        partitions_in_results = set(hit.partition for hit in result.hits)
        assert len(partitions_in_results) >= 2, (
            f"Expected results from multiple partitions, got only: {partitions_in_results}"
        )

        # Verify data integrity
        for hit in result.hits:
            assert hit.score > 0
            assert hit.partition in PARTITIONS
            assert hit.document.get("category") == hit.partition

    async def test_search_two_partitions(self, views_client: SVRClient):
        """Search electronics + clothing, excluding furniture."""
        result = await views_client.search(
            query="waterproof wearable gear for outdoor activities",
            partitions=["electronics", "clothing"],
            limit=10,
        )

        assert len(result.hits) > 0
        assert set(result.partitions_searched) == {"electronics", "clothing"}

        for hit in result.hits:
            assert hit.partition in ("electronics", "clothing"), (
                f"Got unexpected partition '{hit.partition}'"
            )

    async def test_search_relevance_ranking(self, views_client: SVRClient):
        """Search with a query closely matching one partition and verify
        that top results are from the expected category."""
        result = await views_client.search(
            query="hiking boots trail running shoes outdoor footwear",
            partitions="all",
            limit=10,
        )

        assert len(result.hits) > 0

        # The top result should ideally be clothing (trail running shoes, hiking jacket, etc.)
        # At minimum, clothing items should appear in results
        clothing_hits = [h for h in result.hits if h.partition == "clothing"]
        assert len(clothing_hits) > 0, (
            "Expected at least one clothing result for a footwear/hiking query"
        )

    async def test_cleanup(self, views_client: SVRClient):
        """Verify that partitions can be deleted and views are removed."""
        config = views_client.config
        backend = views_client._backend
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        # Verify partitions and views exist before cleanup
        assert len(config.partitions.registry) == 3

        for name in PARTITIONS:
            view_name = f"svr_intviews_{name}"
            assert await backend.view_exists(view_name), (
                f"View '{view_name}' should exist before cleanup"
            )

        # Delete one partition and verify
        await provisioner.delete_partition("clothing")
        assert "clothing" not in config.partitions.registry
        assert len(config.partitions.registry) == 2

        # The view for clothing should be gone
        view_gone = not await backend.view_exists("svr_intviews_clothing")
        assert view_gone, "View 'svr_intviews_clothing' should be deleted"

        # Other views should still exist
        assert await backend.view_exists("svr_intviews_electronics")
        assert await backend.view_exists("svr_intviews_furniture")

        # Re-create the partition for other tests' teardown to work cleanly
        await provisioner.create_partition("clothing", skip_if_exists=True)
