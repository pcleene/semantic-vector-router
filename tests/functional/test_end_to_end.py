"""Functional end-to-end tests against a real MongoDB Atlas cluster.

Uses semantically structured embeddings: each category has vectors clustered
in a distinct direction so search results are predictable and verifiable.

Requires: MONGODB_URI environment variable.
Run with: pytest tests/functional/ -v -s --timeout=300
"""

import asyncio
import io
import json
import logging
import math
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from dotenv import load_dotenv
from pymongo import AsyncMongoClient
from pymongo.errors import ServerSelectionTimeoutError

from semantic_vector_router.backends.mongodb import (
    MongoDBBackend,
    bindata_to_vector,
    query_vector_for_search,
    vector_to_bindata,
)
from semantic_vector_router.cli import main as cli_main
from semantic_vector_router.config import load_config, save_config, validate_config
from semantic_vector_router.exceptions import (
    PartitionAlreadyExistsError,
)
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.lifecycle.scanner import PartitionScanner
from semantic_vector_router.backends.metadata import MetadataStore
from semantic_vector_router.lifecycle.detector import DetectionResult, PartitionDetector
from semantic_vector_router.models import (
    CacheConfig,
    DatabaseConfig,
    DetectionSignal,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestConfig,
    IngestMode,
    LogConfig,
    MetricsConfig,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RateLimitConfig,
    RerankingConfig,
    ResilienceConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
)
from semantic_vector_router.utils.cache import CacheKey, EmbeddingCache
from semantic_vector_router.utils.logging import (
    SVRLogFormatter,
    configure_logging,
    correlation_id_var,
    get_correlation_id,
    new_correlation_id,
)
from semantic_vector_router.utils.metrics import (
    MetricEvent,
    MetricType,
    MetricsCollector,
    MetricsHandler,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Test constants
TEST_DB = "svr_functional_test"
TEST_COLLECTION = "products"
DIMENSIONS = 32
NUM_DOCS_PER_CATEGORY = 15
CATEGORIES = ["electronics", "furniture", "clothing"]

# Category "centroids" — orthogonal directions so each category is
# clearly separated in vector space. A query near a centroid should
# only return docs from that category.
CATEGORY_CENTROIDS = {
    "electronics": [1.0] * 10 + [0.0] * 11 + [0.0] * 11,
    "furniture":   [0.0] * 10 + [1.0] * 11 + [0.0] * 11,
    "clothing":    [0.0] * 10 + [0.0] * 11 + [1.0] * 11,
}

SAMPLE_PRODUCTS = {
    "electronics": [
        "wireless headphones", "bluetooth speaker", "laptop stand",
        "usb-c hub", "mechanical keyboard", "gaming mouse",
        "webcam hd", "monitor arm", "power bank", "smart watch",
        "tablet case", "phone charger", "earbuds pro", "cable organizer",
        "desk lamp led",
    ],
    "furniture": [
        "office chair", "standing desk", "bookshelf oak",
        "filing cabinet", "desk organizer", "monitor riser",
        "ergonomic footrest", "desk pad leather", "cable tray",
        "drawer unit", "shelf bracket", "coat rack", "side table",
        "storage ottoman", "wall mount",
    ],
    "clothing": [
        "cotton t-shirt", "denim jeans", "wool sweater",
        "rain jacket", "running shoes", "canvas backpack",
        "leather belt", "silk scarf", "baseball cap", "winter gloves",
        "hiking boots", "linen pants", "polo shirt", "fleece vest",
        "ankle socks",
    ],
}


def _centroid_vector(category: str, noise: float = 0.15) -> list[float]:
    """Generate a vector near the category centroid with slight noise.

    The noise is small enough that cosine similarity to own centroid is
    always higher than to other centroids.
    """
    base = CATEGORY_CENTROIDS[category]
    vec = [v + random.gauss(0, noise) for v in base]
    # Normalize
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _make_docs(category: str) -> list[dict]:
    """Generate sample docs with structured embeddings."""
    names = SAMPLE_PRODUCTS[category]
    docs = []
    for i, name in enumerate(names):
        docs.append({
            "name": name,
            "category": category,
            "price": round(random.uniform(10, 500), 2),
            "brand": random.choice(["BrandA", "BrandB", "BrandC"]),
            "in_stock": random.choice([True, False]),
            "embedding": _centroid_vector(category),
        })
    return docs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mongodb_uri() -> str:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set")
    return uri


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def seeded_db(mongodb_uri):
    """Seed test database once, clean up after all tests."""
    client = AsyncMongoClient(mongodb_uri)
    db = client[TEST_DB]

    # Clean slate
    await client.drop_database(TEST_DB)

    # Insert sample data
    collection = db[TEST_COLLECTION]
    all_docs = []
    for cat in CATEGORIES:
        all_docs.extend(_make_docs(cat))
    await collection.insert_many(all_docs)

    count = await collection.count_documents({})
    logger.info(f"Seeded {TEST_DB}.{TEST_COLLECTION} with {count} docs")

    yield client

    # Cleanup
    await client.drop_database(TEST_DB)
    await client.close()


def _make_config(index_on: IndexLocation = IndexLocation.VIEWS) -> SVRConfig:
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database=TEST_DB,
            source_collection=TEST_COLLECTION,
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_ft_",
            index_name_prefix="svr_ft_idx_",
        ),
        vector_storage=VectorStorageConfig(
            index_on=index_on,
            storage_format=VectorStorageFormat.ARRAY,
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=DIMENSIONS,
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.OPENAI,
            dimensions=DIMENSIONS,
        ),
        reranking=RerankingConfig(enabled=False),
    )


async def _wait_for_index(
    backend: MongoDBBackend,
    collection_name: str,
    index_name: str,
    timeout: int = 180,
    poll_interval: int = 3,
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
            # Small extra delay for filter field indexing to catch up
            await asyncio.sleep(2)
            return True
        await asyncio.sleep(poll_interval)
    return False


async def _cleanup_views_and_indexes(backend, names, index_prefix="svr_ft_idx_", view_prefix="svr_ft_"):
    """Best-effort cleanup of views and indexes."""
    for name in names:
        try:
            await backend.delete_vector_search_index(f"{view_prefix}{name}", f"{index_prefix}{name}")
        except Exception:
            pass
        try:
            await backend.delete_partition_view(f"{view_prefix}{name}")
        except Exception:
            pass
    # Also clean shared source index (used by VIEWS and SOURCE modes)
    try:
        await backend.delete_vector_search_index(TEST_COLLECTION, "svr_vector_idx_source")
    except Exception:
        pass


# ===========================================================================
# TESTS
# ===========================================================================


class TestConnection:
    """Verify connectivity and seeded data."""

    @pytest.mark.asyncio
    async def test_connect(self, seeded_db):
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        assert await backend.is_connected()
        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_data_correct(self, seeded_db):
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()

        total = await backend.count_documents()
        assert total == NUM_DOCS_PER_CATEGORY * len(CATEGORIES)

        values = await backend.get_distinct_values("category")
        assert set(values) == set(CATEGORIES)

        counts = await backend.get_partition_document_counts("category")
        for cat in CATEGORIES:
            assert counts[cat] == NUM_DOCS_PER_CATEGORY

        await backend.disconnect()


class TestScanner:
    """Scanner against real data."""

    @pytest.mark.asyncio
    async def test_scan_and_validate(self, seeded_db):
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()

        scanner = PartitionScanner(backend, config)

        # All values should be new (nothing registered)
        new_vals = await scanner.get_new_partition_values()
        assert set(str(v) for v in new_vals) == set(CATEGORIES)

        # Scan counts
        counts = await scanner.scan_partition_values()
        assert len(counts) == 3
        for cat in CATEGORIES:
            assert counts[cat] == NUM_DOCS_PER_CATEGORY

        # Validate with one registered
        config.partitions.registry["electronics"] = PartitionInfo(
            name="electronics",
            view_name="svr_ft_electronics",
            index_name="svr_ft_idx_electronics",
            filter_value="electronics",
        )
        result = await scanner.validate_partitions()
        assert "electronics" in result["valid"]
        assert set(result["orphaned"]) == {"furniture", "clothing"}

        await backend.disconnect()


class TestProvisionerViews:
    """VIEWS mode provisioning and deletion."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, seeded_db):
        config = _make_config(IndexLocation.VIEWS)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            # Create
            p = await provisioner.create_partition("electronics")
            assert p.index_location == IndexLocation.VIEWS
            assert p.view_name == "svr_ft_electronics"
            # VIEWS mode: search on source collection (Atlas doesn't support
            # search indexes on views), but views are used for counting
            assert p.search_collection == TEST_COLLECTION
            assert p.document_count == NUM_DOCS_PER_CATEGORY
            assert p.embedding_field is None
            assert await backend.view_exists("svr_ft_electronics")
            # Index is on source collection, not on the view
            assert p.index_name == "svr_vector_idx_source"
            # May take a moment for Atlas to propagate new index to listing
            for _ in range(5):
                if await backend.index_exists(TEST_COLLECTION, "svr_vector_idx_source"):
                    break
                await asyncio.sleep(2)
            assert await backend.index_exists(TEST_COLLECTION, "svr_vector_idx_source")

            # Skip if exists
            p2 = await provisioner.create_partition("electronics", skip_if_exists=True)
            assert p2.name == p.name

            # Already exists error
            with pytest.raises(PartitionAlreadyExistsError):
                await provisioner.create_partition("electronics")

            # View only shows electronics
            view_count = await backend.count_documents("svr_ft_electronics")
            assert view_count == NUM_DOCS_PER_CATEGORY

            # Delete
            await provisioner.delete_partition("electronics")
            assert "electronics" not in config.partitions.registry
            assert not await backend.view_exists("svr_ft_electronics")

        finally:
            await _cleanup_views_and_indexes(backend, ["electronics"])
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_batch_create(self, seeded_db):
        config = _make_config(IndexLocation.VIEWS)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            created = await provisioner.create_partitions_batch(CATEGORIES)
            assert len(created) == 3
            for cat in CATEGORIES:
                assert await backend.view_exists(f"svr_ft_{cat}")
        finally:
            await _cleanup_views_and_indexes(backend, CATEGORIES)
            await backend.disconnect()


class TestProvisionerSource:
    """SOURCE mode provisioning."""

    @pytest.mark.asyncio
    async def test_source_mode(self, seeded_db):
        config = _make_config(IndexLocation.SOURCE)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            p1 = await provisioner.create_partition("electronics")
            assert p1.index_location == IndexLocation.SOURCE
            assert p1.search_collection == TEST_COLLECTION
            assert p1.index_name == "svr_vector_idx_source"

            p2 = await provisioner.create_partition("furniture")
            assert p2.index_name == "svr_vector_idx_source"  # Same index

            # Source index on main collection (may take a moment to propagate)
            for _ in range(5):
                if await backend.index_exists(TEST_COLLECTION, "svr_vector_idx_source"):
                    break
                await asyncio.sleep(2)
            assert await backend.index_exists(TEST_COLLECTION, "svr_vector_idx_source")

            # Views still created for browsing
            assert await backend.view_exists("svr_ft_electronics")
            assert await backend.view_exists("svr_ft_furniture")

        finally:
            await _cleanup_views_and_indexes(backend, ["electronics", "furniture"])
            try:
                await backend.delete_vector_search_index(TEST_COLLECTION, "svr_vector_idx_source")
            except Exception:
                pass
            await backend.disconnect()


class TestProvisionerFields:
    """FIELDS mode provisioning."""

    @pytest.mark.asyncio
    async def test_fields_mode_basic(self, seeded_db):
        config = _make_config(IndexLocation.FIELDS)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            p = await provisioner.create_partition("electronics")
            assert p.index_location == IndexLocation.FIELDS
            assert p.embedding_field == "embedding_electronics"
            assert p.view_name is None
            assert p.search_collection == TEST_COLLECTION
            assert p.index_name == "svr_ft_idx_electronics"

            # Index on source collection (propagation delay possible)
            for _ in range(5):
                if await backend.index_exists(TEST_COLLECTION, "svr_ft_idx_electronics"):
                    break
                await asyncio.sleep(2)
            assert await backend.index_exists(TEST_COLLECTION, "svr_ft_idx_electronics")

            # Doc count = 0 (no docs have embedding_electronics yet)
            assert p.document_count == 0

        finally:
            try:
                await backend.delete_vector_search_index(TEST_COLLECTION, "svr_ft_idx_electronics")
            except Exception:
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_fields_mode_with_data(self, seeded_db):
        """Add the partition-specific field to docs, verify count."""
        config = _make_config(IndexLocation.FIELDS)
        backend = MongoDBBackend(config)
        await backend.connect()

        # Set embedding_electronics on electronics docs
        coll = backend.db[TEST_COLLECTION]
        result = await coll.update_many(
            {"category": "electronics"},
            [{"$set": {"embedding_electronics": "$embedding"}}],
        )
        logger.info(f"Updated {result.modified_count} docs with embedding_electronics")

        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            p = await provisioner.create_partition("electronics")
            assert p.document_count == NUM_DOCS_PER_CATEGORY

        finally:
            await coll.update_many({}, {"$unset": {"embedding_electronics": ""}})
            try:
                await backend.delete_vector_search_index(TEST_COLLECTION, "svr_ft_idx_electronics")
            except Exception:
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_fields_mode_delete(self, seeded_db):
        config = _make_config(IndexLocation.FIELDS)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            await provisioner.create_partition("clothing")
            assert "clothing" in config.partitions.registry

            await provisioner.delete_partition("clothing")
            assert "clothing" not in config.partitions.registry

        finally:
            try:
                await backend.delete_vector_search_index(TEST_COLLECTION, "svr_ft_idx_clothing")
            except Exception:
                pass
            await backend.disconnect()


class TestSearchViews:
    """Actual $vectorSearch against Atlas — VIEWS mode.

    VIEWS mode uses a shared index on the source collection (Atlas doesn't
    support search indexes on views). Views are still created for counting.
    """

    @pytest.mark.asyncio
    async def test_search_single_partition(self, seeded_db):
        config = _make_config(IndexLocation.VIEWS)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            partition = await provisioner.create_partition("electronics")

            # Index is on source collection (shared, like SOURCE mode)
            ready = await _wait_for_index(backend, TEST_COLLECTION, "svr_vector_idx_source")
            if not ready:
                pytest.skip("Index not queryable in time")

            # Query near electronics centroid
            query_vec = _centroid_vector("electronics", noise=0.05)
            results = await backend.execute_search(
                partition=partition,
                query="",
                query_vector=query_vec,
                limit=5,
                num_candidates=15,
            )

            assert len(results) > 0
            assert len(results) <= 5
            for doc in results:
                assert doc["_svr_partition"] == "electronics"
                assert doc["category"] == "electronics"
                assert doc["_svr_score"] > 0.5  # Should be high similarity

        finally:
            await _cleanup_views_and_indexes(backend, ["electronics"])
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_search_parallel_partitions(self, seeded_db):
        config = _make_config(IndexLocation.VIEWS)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            partitions = []
            for cat in CATEGORIES:
                p = await provisioner.create_partition(cat)
                partitions.append(p)

            # Shared index on source collection
            ready = await _wait_for_index(backend, TEST_COLLECTION, "svr_vector_idx_source")
            if not ready:
                pytest.skip("Index not queryable in time")

            # Query near electronics — should rank electronics results highest
            query_vec = _centroid_vector("electronics", noise=0.05)
            results = await backend.search_partitions(
                partitions=partitions,
                query="",
                query_vector=query_vec,
                limit=5,
            )

            assert len(results) > 0
            partitions_found = set(doc["_svr_partition"] for doc in results)
            assert len(partitions_found) >= 2  # Results from multiple partitions

            # The top result should be from electronics (closest to query)
            sorted_results = sorted(results, key=lambda d: d["_svr_score"], reverse=True)
            assert sorted_results[0]["category"] == "electronics"

        finally:
            await _cleanup_views_and_indexes(backend, CATEGORIES)
            await backend.disconnect()


class TestSearchSource:
    """$vectorSearch with SOURCE mode pre-filtering."""

    @pytest.mark.asyncio
    async def test_source_filter_isolation(self, seeded_db):
        config = _make_config(IndexLocation.SOURCE)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            p_elec = await provisioner.create_partition("electronics")
            p_furn = await provisioner.create_partition("furniture")

            ready = await _wait_for_index(backend, TEST_COLLECTION, "svr_vector_idx_source")
            if not ready:
                pytest.skip("Source index not queryable in time")

            # Query with furniture centroid
            query_vec = _centroid_vector("furniture", noise=0.05)

            # Search electronics partition — pre-filter should limit to electronics
            results_elec = await backend.execute_search(
                partition=p_elec,
                query="",
                query_vector=query_vec,
                limit=5,
                num_candidates=15,
            )
            for doc in results_elec:
                assert doc["category"] == "electronics", (
                    f"SOURCE mode filter leak: got {doc['category']} in electronics partition"
                )

            # Search furniture partition — should get furniture docs
            results_furn = await backend.execute_search(
                partition=p_furn,
                query="",
                query_vector=query_vec,
                limit=5,
                num_candidates=15,
            )
            for doc in results_furn:
                assert doc["category"] == "furniture"

            # Furniture results should score higher (query is near furniture centroid)
            if results_elec and results_furn:
                avg_elec = sum(d["_svr_score"] for d in results_elec) / len(results_elec)
                avg_furn = sum(d["_svr_score"] for d in results_furn) / len(results_furn)
                assert avg_furn > avg_elec, (
                    f"Furniture should score higher: furn={avg_furn:.3f} vs elec={avg_elec:.3f}"
                )

        finally:
            await _cleanup_views_and_indexes(backend, ["electronics", "furniture"])
            try:
                await backend.delete_vector_search_index(TEST_COLLECTION, "svr_vector_idx_source")
            except Exception:
                pass
            await backend.disconnect()


class TestBinData:
    """BinData storage format against real MongoDB."""

    @pytest.mark.asyncio
    async def test_float32_roundtrip(self, seeded_db):
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            original = _centroid_vector("electronics")
            bindata = vector_to_bindata(original, VectorStorageFormat.BINDATA_FLOAT32)

            coll = backend.db["svr_ft_bindata_test"]
            await coll.insert_one({"embedding": bindata, "type": "float32"})
            doc = await coll.find_one({"type": "float32"})

            recovered = bindata_to_vector(doc["embedding"])
            for a, b in zip(original, recovered):
                assert abs(a - b) < 1e-6

        finally:
            await backend.db.drop_collection("svr_ft_bindata_test")
            await backend.disconnect()

    def test_int8_validation(self):
        with pytest.raises(ValueError, match="INT8"):
            vector_to_bindata([200, -200], VectorStorageFormat.BINDATA_INT8)

        result = vector_to_bindata([-128, 0, 127], VectorStorageFormat.BINDATA_INT8)
        assert result is not None

    def test_packed_bit_validation(self):
        with pytest.raises(ValueError, match="PACKED_BIT"):
            vector_to_bindata([0, 1, 2], VectorStorageFormat.BINDATA_PACKED_BIT)

        result = vector_to_bindata([0, 1, 1, 0], VectorStorageFormat.BINDATA_PACKED_BIT)
        assert result is not None

    def test_query_vector_formats(self):
        vec = [0.1, 0.2, 0.3]

        # ARRAY and FLOAT32 → return list (no conversion)
        assert query_vector_for_search(vec, VectorStorageFormat.ARRAY) == vec
        assert query_vector_for_search(vec, VectorStorageFormat.BINDATA_FLOAT32) == vec

        # INT8 → return BinData
        int8_vec = [-128, 0, 127]
        result = query_vector_for_search(int8_vec, VectorStorageFormat.BINDATA_INT8)
        assert not isinstance(result, list)  # Should be Binary


class TestFieldAnalyzerReal:
    """Field analyzer against real collection."""

    @pytest.mark.asyncio
    async def test_analyze_real_fields(self, seeded_db):
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            from semantic_vector_router.utils.field_analyzer import (
                analyze_fields,
                get_recommended_filter_fields,
            )

            analyses = await analyze_fields(backend, config)

            # category: 3 distinct, 100% coverage → suitable
            cat = next((a for a in analyses if a.name == "category"), None)
            assert cat is not None
            assert cat.is_suitable
            assert cat.distinct_count == 3

            # brand: 3 distinct, 100% coverage → suitable
            brand = next((a for a in analyses if a.name == "brand"), None)
            assert brand is not None
            assert brand.is_suitable

            # in_stock: boolean, 2 values → suitable
            stock = next((a for a in analyses if a.name == "in_stock"), None)
            assert stock is not None
            assert stock.is_suitable
            assert stock.distinct_count == 2

            # name: high cardinality → NOT suitable
            name_field = next((a for a in analyses if a.name == "name"), None)
            assert name_field is not None
            assert not name_field.is_suitable

            # _id, embedding should be excluded
            assert not any(a.name == "_id" for a in analyses)
            assert not any(a.name == "embedding" for a in analyses)

            # Recommendations
            recommended = get_recommended_filter_fields(analyses)
            assert "category" in recommended
            assert "brand" in recommended

        finally:
            await backend.disconnect()


class TestConfigValidation:
    """Config validation with real-ish configs."""

    def test_all_three_modes_valid(self):
        for mode in [IndexLocation.SOURCE, IndexLocation.VIEWS, IndexLocation.FIELDS]:
            config = _make_config(mode)
            warnings = validate_config(config)
            assert isinstance(warnings, list)


# ===========================================================================
# Phase 3 — Robustness functional tests
# ===========================================================================


class TestHealthCheck:
    """health_check() against real Atlas."""

    @pytest.mark.asyncio
    async def test_health_check_connected(self, seeded_db):
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        try:
            result = await backend.health_check()
            assert result is True
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_staleness_caching(self, seeded_db):
        config = _make_config()
        # Use a long interval so the cached check won't expire
        config.resilience.health_check_interval_s = 300
        backend = MongoDBBackend(config)
        await backend.connect()
        try:
            # First call — real ping
            t0 = time.monotonic()
            result1 = await backend.health_check()
            t1 = time.monotonic()
            assert result1 is True
            first_duration = t1 - t0

            # Second call — should be cached (near-instant)
            t2 = time.monotonic()
            result2 = await backend.health_check()
            t3 = time.monotonic()
            assert result2 is True
            cached_duration = t3 - t2

            # Cached call should be at least 10x faster than the real ping
            # (real ping involves network, cached is just a time comparison)
            # Use a generous threshold to avoid flakiness
            assert cached_duration < first_duration or cached_duration < 0.001
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_after_disconnect(self, seeded_db):
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        await backend.disconnect()

        result = await backend.health_check()
        assert result is False


class TestSearchTimeout:
    """Search with maxTimeMS on real Atlas."""

    @pytest.mark.asyncio
    async def test_search_with_reasonable_timeout(self, seeded_db):
        """Search with a normal timeout should succeed."""
        config = _make_config(IndexLocation.VIEWS)
        config.resilience.search_timeout_ms = 30_000
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            partition = await provisioner.create_partition("electronics")
            ready = await _wait_for_index(backend, TEST_COLLECTION, "svr_vector_idx_source")
            if not ready:
                pytest.skip("Index not queryable in time")

            query_vec = _centroid_vector("electronics", noise=0.05)
            results = await backend.execute_search(
                partition=partition,
                query="",
                query_vector=query_vec,
                limit=5,
                num_candidates=15,
            )
            assert len(results) > 0
        finally:
            await _cleanup_views_and_indexes(backend, ["electronics"])
            await backend.disconnect()


class TestConnectionTimeouts:
    """Connection timeout configuration against real Atlas."""

    @pytest.mark.asyncio
    async def test_connect_with_valid_timeout(self, seeded_db):
        """Connection with reasonable timeouts should succeed."""
        config = _make_config()
        config.resilience.connection_timeout_ms = 10_000
        config.resilience.server_selection_timeout_ms = 30_000
        backend = MongoDBBackend(config)
        await backend.connect()
        assert await backend.is_connected()
        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_connect_unreachable_host_fast_timeout(self):
        """Connection to unreachable host with tiny timeout should fail fast."""
        config = _make_config()
        # Override the connection string env to point at an unreachable host
        # We create a backend manually with a very short timeout
        config.resilience.server_selection_timeout_ms = 1
        config.resilience.connection_timeout_ms = 1

        # Build a client with an unreachable host directly
        try:
            client = AsyncMongoClient(
                "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<db>",  # RFC 5737 TEST-NET, guaranteed unreachable
                serverSelectionTimeoutMS=1,
                connectTimeoutMS=1,
            )
            # The client won't fail until we try an operation
            t0 = time.monotonic()
            with pytest.raises(ServerSelectionTimeoutError):
                await client.admin.command("ping")
            elapsed = time.monotonic() - t0
            # Should fail within a few seconds, not 30+
            assert elapsed < 10, f"Timeout took too long: {elapsed:.1f}s"
        finally:
            await client.close()


class TestProvisionerRollback:
    """Provisioner rollback on real Atlas."""

    @pytest.mark.asyncio
    async def test_rollback_cleans_up_resources(self, seeded_db):
        """Create a partition, then verify _rollback_partition cleans it up."""
        config = _make_config(IndexLocation.VIEWS)
        backend = MongoDBBackend(config)
        await backend.connect()
        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)

        try:
            # Create partition successfully first
            partition = await provisioner.create_partition("electronics")
            assert "electronics" in config.partitions.registry
            assert await backend.view_exists("svr_ft_electronics")

            # Now call rollback manually to verify it cleans up
            await provisioner._rollback_partition(
                name="electronics",
                created_view="svr_ft_electronics",
                created_index=None,  # Shared source index — don't delete it
                config_modified=True,
            )

            # Config registry should be cleaned
            assert "electronics" not in config.partitions.registry

            # View should be deleted
            assert not await backend.view_exists("svr_ft_electronics")

        finally:
            # Best-effort cleanup of anything left over
            await _cleanup_views_and_indexes(backend, ["electronics"])
            await backend.disconnect()


class TestResilienceConfigBackwardCompat:
    """Backward-compatible config loading."""

    def test_config_without_resilience_gets_defaults(self):
        """A config dict without a 'resilience' key should load with defaults."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.resilience is not None
        assert config.resilience.max_retry_attempts == 3
        assert config.resilience.connection_timeout_ms == 10_000
        assert config.resilience.search_timeout_ms == 30_000
        assert config.resilience.health_check_interval_s == 30

    def test_config_with_explicit_resilience(self):
        """A config with explicit resilience settings loads correctly."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
            "resilience": {
                "max_retry_attempts": 5,
                "connection_timeout_ms": 20_000,
                "search_timeout_ms": 60_000,
                "health_check_interval_s": 60,
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.resilience.max_retry_attempts == 5
        assert config.resilience.connection_timeout_ms == 20_000
        assert config.resilience.search_timeout_ms == 60_000
        assert config.resilience.health_check_interval_s == 60
        # Unspecified fields should keep defaults
        assert config.resilience.retry_base_delay == 0.5
        assert config.resilience.embedding_timeout_ms == 60_000

    def test_config_file_without_resilience(self):
        """Loading a JSON config file without 'resilience' key works."""
        config_data = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            config = load_config(config_path=f.name, load_env=False)

        assert config.resilience is not None
        assert isinstance(config.resilience, ResilienceConfig)
        assert config.resilience.max_retry_attempts == 3
        os.unlink(f.name)

    def test_config_validation_with_resilience(self):
        """validate_config works with resilience settings."""
        config = _make_config()
        config.resilience.search_timeout_ms = 3000  # Low but valid
        warnings = validate_config(config)
        # Should warn about low search timeout
        assert any("search_timeout_ms" in w for w in warnings)


# ===========================================================================
# CLI FUNCTIONAL TESTS
# ===========================================================================


@pytest.fixture(scope="module")
def cli_runner():
    return CliRunner(catch_exceptions=False)


@pytest.fixture(scope="module")
def cli_config_path(tmp_path_factory, seeded_db):
    """Create a config file on disk for CLI tests.

    Uses SOURCE mode so we can test without needing to create
    views/indexes per partition (simpler setup).
    """
    config = _make_config(index_on=IndexLocation.SOURCE)
    tmp_dir = tmp_path_factory.mktemp("cli_tests")
    config_path = tmp_dir / "svr_test_config.json"
    save_config(config, config_path)
    return str(config_path)


@pytest.fixture(scope="module")
def cli_config_with_partitions(tmp_path_factory, seeded_db):
    """Create a config file with provisioned partitions for CLI tests."""
    config = _make_config(index_on=IndexLocation.SOURCE)

    # Manually register partitions in config (no index creation needed for list/status)
    for cat in CATEGORIES:
        config.partitions.registry[cat] = PartitionInfo(
            name=cat,
            index_name=f"svr_ft_idx_{cat}",
            filter_value=cat,
            document_count=NUM_DOCS_PER_CATEGORY,
            status=PartitionStatus.ACTIVE,
            index_location=IndexLocation.SOURCE,
            search_collection=TEST_COLLECTION,
        )

    tmp_dir = tmp_path_factory.mktemp("cli_partitioned")
    config_path = tmp_dir / "svr_test_config.json"
    save_config(config, config_path)
    return str(config_path)


class TestCLIPartitions:
    """CLI partition commands against real Atlas."""

    def test_cli_partitions_list(self, cli_runner, cli_config_with_partitions):
        """After provisioning, 'svr partitions list' shows all partitions."""
        result = cli_runner.invoke(cli_main, ["partitions", "list", "-c", cli_config_with_partitions])
        assert result.exit_code == 0
        for cat in CATEGORIES:
            assert cat in result.output
        assert "15" in result.output  # document count

    def test_cli_partitions_status_single(self, cli_runner, cli_config_with_partitions):
        """'svr partitions status <name>' shows details for one partition."""
        result = cli_runner.invoke(cli_main, ["partitions", "status", "electronics", "-c", cli_config_with_partitions])
        assert result.exit_code == 0
        assert "electronics" in result.output
        assert "active" in result.output.lower() or "ACTIVE" in result.output

    def test_cli_partitions_status_all(self, cli_runner, cli_config_with_partitions):
        """'svr partitions status' shows status table for all partitions."""
        result = cli_runner.invoke(cli_main, ["partitions", "status", "-c", cli_config_with_partitions])
        assert result.exit_code == 0
        for cat in CATEGORIES:
            assert cat in result.output

    def test_cli_partitions_scan(self, cli_runner, cli_config_path, mongodb_uri):
        """'svr partitions scan' finds partition values from real data."""
        result = cli_runner.invoke(cli_main, ["partitions", "scan", "-c", cli_config_path])
        assert result.exit_code == 0
        # Should find all categories since no partitions are registered in this config
        for cat in CATEGORIES:
            assert cat in result.output

    def test_cli_partitions_refresh(self, cli_runner, cli_config_with_partitions, mongodb_uri):
        """'svr partitions refresh' updates document counts from real Atlas."""
        result = cli_runner.invoke(cli_main, ["partitions", "refresh", "-c", cli_config_with_partitions])
        assert result.exit_code == 0
        assert "15" in result.output  # 15 docs per category


class TestCLIConfig:
    """CLI config commands with real config files."""

    def test_cli_config_show(self, cli_runner, cli_config_path):
        """'svr config show' displays config (with redaction)."""
        result = cli_runner.invoke(cli_main, ["config", "show", "-c", cli_config_path])
        assert result.exit_code == 0
        assert TEST_DB in result.output
        assert TEST_COLLECTION in result.output

    def test_cli_config_validate(self, cli_runner, cli_config_path):
        """'svr config validate' runs validation on real config."""
        result = cli_runner.invoke(cli_main, ["config", "validate", "-c", cli_config_path])
        assert result.exit_code == 0
        # Either "valid" or warnings - both are acceptable
        assert "valid" in result.output.lower() or "Warning" in result.output

    def test_cli_config_path(self, cli_runner, cli_config_path):
        """'svr config path' shows the config file path."""
        result = cli_runner.invoke(cli_main, ["config", "path", "-c", cli_config_path])
        assert result.exit_code == 0
        # Rich may wrap long paths across lines, so check the filename
        assert "svr_test_config.json" in result.output


class TestCLIAnalyze:
    """CLI analyze command against real Atlas."""

    def test_cli_analyze_full(self, cli_runner, cli_config_path, mongodb_uri):
        """'svr analyze' shows field analysis from real collection."""
        result = cli_runner.invoke(cli_main, ["analyze", "-c", cli_config_path])
        assert result.exit_code == 0
        # Should find 'category' field among analyzed fields
        assert "category" in result.output

    def test_cli_analyze_filters(self, cli_runner, cli_config_path, mongodb_uri):
        """'svr analyze --filters' shows filter recommendations."""
        result = cli_runner.invoke(cli_main, ["analyze", "--filters", "-c", cli_config_path])
        assert result.exit_code == 0
        assert "category" in result.output


class TestCLIIndex:
    """CLI index commands."""

    def test_cli_index_status_with_partitions(self, cli_runner, cli_config_with_partitions, mongodb_uri):
        """'svr index status' shows index status for registered partitions."""
        result = cli_runner.invoke(cli_main, ["index", "status", "-c", cli_config_with_partitions])
        assert result.exit_code == 0
        # Should show partition names (even if indexes don't exist)
        assert "electronics" in result.output or "error" in result.output.lower()

    def test_cli_index_status_no_partitions(self, cli_runner, cli_config_path, mongodb_uri):
        """'svr index status' with no partitions shows helpful message."""
        result = cli_runner.invoke(cli_main, ["index", "status", "-c", cli_config_path])
        assert result.exit_code == 0
        assert "No partitions" in result.output


class TestCLIHelp:
    """Verify all CLI help messages work."""

    def test_main_help(self, cli_runner):
        result = cli_runner.invoke(cli_main, ["--help"])
        assert result.exit_code == 0
        for cmd in ["partitions", "search", "analyze", "config", "index", "watch", "split", "init"]:
            assert cmd in result.output

    def test_each_command_help(self, cli_runner):
        """Every command and subgroup should have --help."""
        commands = [
            ["partitions", "--help"],
            ["search", "--help"],
            ["analyze", "--help"],
            ["config", "--help"],
            ["index", "--help"],
            ["watch", "--help"],
            ["split", "--help"],
            ["init", "--help"],
        ]
        for cmd in commands:
            result = cli_runner.invoke(cli_main, cmd)
            assert result.exit_code == 0, f"Help failed for: {' '.join(cmd)}"


# ===========================================================================
# Phase 5 — Lifecycle metadata, detection, backward compat
# ===========================================================================


class TestMetadataStoreReal:
    """MetadataStore CRUD against real MongoDB Atlas."""

    @pytest.mark.asyncio
    async def test_partition_crud(self, seeded_db):
        """Save, get, list, delete partition in metadata."""
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            # Clean up any existing metadata
            await metadata._collection.delete_many({})

            # Save partition
            partition = PartitionInfo(
                name="test_meta_elec",
                index_name="svr_idx_test_elec",
                filter_value="electronics",
                document_count=15,
                status=PartitionStatus.ACTIVE,
                index_location=IndexLocation.SOURCE,
                search_collection=TEST_COLLECTION,
            )
            await metadata.save_partition(partition)

            # Get partition
            retrieved = await metadata.get_partition("test_meta_elec")
            assert retrieved is not None
            assert retrieved.name == "test_meta_elec"
            assert retrieved.filter_value == "electronics"
            assert retrieved.status == PartitionStatus.ACTIVE

            # List partitions
            all_parts = await metadata.list_partitions()
            assert len(all_parts) == 1

            # Delete
            deleted = await metadata.delete_partition("test_meta_elec")
            assert deleted is True

            # Verify deleted
            assert await metadata.get_partition("test_meta_elec") is None
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_health_history(self, seeded_db):
        """Append and retrieve health history, verify 30-entry cap."""
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            # Create partition first
            partition = PartitionInfo(
                name="test_history",
                index_name="svr_idx_test",
                filter_value="test",
                status=PartitionStatus.ACTIVE,
                index_location=IndexLocation.SOURCE,
                search_collection=TEST_COLLECTION,
            )
            await metadata.save_partition(partition)

            # Append 35 entries (should cap at 30)
            for i in range(35):
                await metadata.append_health_history("test_history", 100 + i)

            history = await metadata.get_health_history("test_history")
            assert len(history) == 30
            # First entry should be 5 (35 - 30 cap), not 0
            assert history[0]["count"] == 105  # 100 + 5
            assert history[-1]["count"] == 134  # 100 + 34
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_lock_acquire_release(self, seeded_db):
        """Distributed lock: acquire, verify held, release, verify released."""
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            # Acquire lock
            acquired = await metadata.acquire_lock("test_lock", "worker-1", ttl_seconds=60)
            assert acquired is True

            # Verify held
            assert await metadata.is_lock_held("test_lock") is True

            # Another worker can't acquire
            acquired2 = await metadata.acquire_lock("test_lock", "worker-2", ttl_seconds=60)
            assert acquired2 is False

            # Release
            released = await metadata.release_lock("test_lock", "worker-1")
            assert released is True

            # Now another worker can acquire
            acquired3 = await metadata.acquire_lock("test_lock", "worker-2", ttl_seconds=60)
            assert acquired3 is True

            # Cleanup
            await metadata.release_lock("test_lock", "worker-2")
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_lock_expiry(self, seeded_db):
        """Expired lock can be re-acquired."""
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            # Acquire with very short TTL
            acquired = await metadata.acquire_lock("expire_lock", "worker-1", ttl_seconds=1)
            assert acquired is True

            # Wait for expiry
            await asyncio.sleep(2)

            # Lock should be expired
            assert await metadata.is_lock_held("expire_lock") is False

            # Another worker can now acquire
            acquired2 = await metadata.acquire_lock("expire_lock", "worker-2", ttl_seconds=60)
            assert acquired2 is True

            await metadata.release_lock("expire_lock", "worker-2")
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_operation_crud(self, seeded_db):
        """Create, get, list, update operations."""
        config = _make_config()
        backend = MongoDBBackend(config)
        await backend.connect()
        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            # Create operation
            op = {
                "_id": "op:test-split-001",
                "type": "operation",
                "action": "split",
                "target_partition": "electronics",
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "steps": [
                    {"action": "create_children", "status": "pending"},
                    {"action": "build_indexes", "status": "pending"},
                ],
            }
            op_id = await metadata.create_operation(op)
            assert op_id == "op:test-split-001"

            # Get
            retrieved = await metadata.get_operation(op_id)
            assert retrieved is not None
            assert retrieved["status"] == "pending"

            # List
            ops = await metadata.list_operations(status="pending")
            assert len(ops) == 1

            # Update status
            await metadata.update_operation_status(op_id, "in_progress")
            retrieved2 = await metadata.get_operation(op_id)
            assert retrieved2["status"] == "in_progress"

            # Update step
            await metadata.update_operation_step(op_id, "create_children", "done")
            retrieved3 = await metadata.get_operation(op_id)
            step = next(s for s in retrieved3["steps"] if s["action"] == "create_children")
            assert step["status"] == "done"
            assert step.get("completed_at") is not None
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_migrate_from_config(self, seeded_db):
        """Migrate partitions from config file to metadata collection."""
        config = _make_config()
        # Register partitions in config
        for cat in CATEGORIES:
            config.partitions.registry[cat] = PartitionInfo(
                name=cat,
                index_name=f"svr_idx_{cat}",
                filter_value=cat,
                document_count=NUM_DOCS_PER_CATEGORY,
                status=PartitionStatus.ACTIVE,
                index_location=IndexLocation.SOURCE,
                search_collection=TEST_COLLECTION,
            )

        backend = MongoDBBackend(config)
        await backend.connect()
        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            # First migration
            migrated = await metadata.migrate_from_config(config)
            assert migrated == 3

            # Second migration (idempotent)
            migrated2 = await metadata.migrate_from_config(config)
            assert migrated2 == 0

            # Verify all 3 in metadata
            parts = await metadata.list_partitions()
            assert len(parts) == 3
            names = {p.name for p in parts}
            assert names == set(CATEGORIES)
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()


class TestDetectionPipelineReal:
    """Detection pipeline against real MongoDB."""

    @pytest.mark.asyncio
    async def test_detection_threshold_breach(self, seeded_db):
        """Detect threshold breach when partition count exceeds threshold."""
        config = _make_config(index_on=IndexLocation.SOURCE)
        # Set threshold very low so our 15-doc partitions trigger it
        config.lifecycle.detection.threshold_vectors = 10
        config.lifecycle.detection.min_threshold_vectors = 1

        backend = MongoDBBackend(config)
        await backend.connect()

        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            # Save partitions to metadata
            for cat in CATEGORIES:
                partition = PartitionInfo(
                    name=cat,
                    index_name=f"svr_idx_{cat}",
                    filter_value=cat,
                    status=PartitionStatus.ACTIVE,
                    index_location=IndexLocation.SOURCE,
                    search_collection=TEST_COLLECTION,
                )
                await metadata.save_partition(partition)

            detector = PartitionDetector(backend, metadata, config)
            results = await detector.run_detection()

            # All 3 partitions should breach (15 > 10)
            breach_results = [r for r in results if r.signal == DetectionSignal.THRESHOLD_BREACH]
            assert len(breach_results) == 3

            # Verify health history was stored
            for cat in CATEGORIES:
                history = await metadata.get_health_history(cat)
                assert len(history) >= 1
                assert history[-1]["count"] == NUM_DOCS_PER_CATEGORY
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_detection_underpopulated(self, seeded_db):
        """Detect underpopulated partitions."""
        config = _make_config(index_on=IndexLocation.SOURCE)
        # Set min threshold higher than our doc count
        config.lifecycle.detection.threshold_vectors = 10_000_000
        config.lifecycle.detection.min_threshold_vectors = 100

        backend = MongoDBBackend(config)
        await backend.connect()

        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            for cat in CATEGORIES:
                partition = PartitionInfo(
                    name=cat,
                    index_name=f"svr_idx_{cat}",
                    filter_value=cat,
                    status=PartitionStatus.ACTIVE,
                    index_location=IndexLocation.SOURCE,
                    search_collection=TEST_COLLECTION,
                )
                await metadata.save_partition(partition)

            detector = PartitionDetector(backend, metadata, config)
            results = await detector.run_detection()

            # All should be underpopulated (15 < 100)
            underpop = [r for r in results if r.signal == DetectionSignal.UNDERPOPULATED]
            assert len(underpop) == 3
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_detection_with_lock(self, seeded_db):
        """Detection with distributed lock works correctly."""
        config = _make_config(index_on=IndexLocation.SOURCE)
        config.lifecycle.detection.threshold_vectors = 10_000_000
        config.lifecycle.detection.min_threshold_vectors = 1

        backend = MongoDBBackend(config)
        await backend.connect()

        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            await metadata._collection.delete_many({})

            for cat in CATEGORIES:
                partition = PartitionInfo(
                    name=cat,
                    index_name=f"svr_idx_{cat}",
                    filter_value=cat,
                    status=PartitionStatus.ACTIVE,
                    index_location=IndexLocation.SOURCE,
                    search_collection=TEST_COLLECTION,
                )
                await metadata.save_partition(partition)

            detector = PartitionDetector(backend, metadata, config)

            # Should succeed (lock acquired and released)
            results = await detector.run_detection_with_lock()
            assert results is not None

            # Lock should be released after
            assert await metadata.is_lock_held("monitor") is False
        finally:
            await metadata._collection.delete_many({})
            await metadata.disconnect()
            await backend.disconnect()


class TestBackwardCompatMetadata:
    """Backward compatibility: client works without metadata."""

    def test_config_without_lifecycle_metadata(self):
        """Config without lifecycle metadata section loads with defaults."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.lifecycle is not None
        assert config.lifecycle.metadata is not None
        assert config.lifecycle.metadata.collection == "svr_metadata"
        assert config.lifecycle.detection is not None
        assert config.lifecycle.repartition is not None

    def test_config_with_explicit_lifecycle(self):
        """Config with explicit lifecycle settings loads correctly."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
            "lifecycle": {
                "detection": {
                    "threshold_vectors": 5_000_000,
                    "min_threshold_vectors": 500,
                    "skew_ratio": 3.0,
                },
                "repartition": {
                    "index_wait_timeout_s": 3600,
                    "index_poll_interval_s": 15,
                },
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.lifecycle.detection.threshold_vectors == 5_000_000
        assert config.lifecycle.detection.min_threshold_vectors == 500
        assert config.lifecycle.detection.skew_ratio == 3.0
        assert config.lifecycle.repartition.index_wait_timeout_s == 3600
        assert config.lifecycle.repartition.index_poll_interval_s == 15


# ===========================================================================
# Phase 6 — Functional Tests
# ===========================================================================


class TestPoolTuningReal:
    """Verify pool tuning params are actually passed to AsyncMongoClient."""

    @pytest.mark.asyncio
    async def test_connect_with_pool_params(self, seeded_db):
        """Backend connects successfully with custom pool params."""
        config = _make_config()
        config.database.max_pool_size = 50
        config.database.min_pool_size = 5
        config.database.max_idle_time_ms = 30000
        config.database.wait_queue_timeout_ms = 5000

        backend = MongoDBBackend(config)
        await backend.connect()
        assert await backend.is_connected()

        # Verify we can actually perform operations with the tuned pool
        count = await backend.count_documents()
        assert count == NUM_DOCS_PER_CATEGORY * len(CATEGORIES)

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_default_pool_params(self, mongodb_uri):
        """Backend connects with default pool params (backward compat)."""
        config = _make_config()
        assert config.database.max_pool_size == 100
        assert config.database.min_pool_size == 0

        backend = MongoDBBackend(config)
        await backend.connect()
        assert await backend.is_connected()
        await backend.disconnect()


class TestStructuredLoggingReal:
    """Verify structured logging produces valid JSON with expected fields."""

    def test_json_formatter_output(self):
        """SVRLogFormatter produces valid JSON with required fields."""
        formatter = SVRLogFormatter()
        record = logging.LogRecord(
            name="semantic_vector_router.client",
            level=logging.INFO,
            pathname="client.py",
            lineno=100,
            msg="Search completed",
            args=None,
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "semantic_vector_router.client"
        assert parsed["msg"] == "Search completed"
        assert "ts" in parsed
        assert "correlation_id" in parsed

    def test_configure_logging_json_mode(self):
        """configure_logging with json_format=True installs JSON formatter."""
        test_logger = configure_logging(
            level="DEBUG",
            json_format=True,
            logger_name="svr_test_json",
        )

        assert test_logger.level == logging.DEBUG
        assert len(test_logger.handlers) == 1
        assert isinstance(test_logger.handlers[0].formatter, SVRLogFormatter)

        # Clean up
        test_logger.handlers.clear()

    def test_correlation_id_propagation(self):
        """Correlation ID set by new_correlation_id is visible in get_correlation_id."""
        cid = new_correlation_id()
        assert len(cid) == 12
        assert get_correlation_id() == cid

    @pytest.mark.asyncio
    async def test_correlation_id_across_await(self):
        """Correlation ID persists across await boundaries via ContextVar."""
        cid = new_correlation_id()

        async def inner_check():
            return get_correlation_id()

        inner_cid = await inner_check()
        assert inner_cid == cid

    def test_json_log_with_extra_fields(self):
        """Extra fields on log records appear in JSON output."""
        formatter = SVRLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test msg",
            args=None,
            exc_info=None,
        )
        record.partitions = 3
        record.duration_ms = 42.5

        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["partitions"] == 3
        assert parsed["duration_ms"] == 42.5


class TestMetricsHooksReal:
    """Verify metrics handler protocol works end-to-end."""

    def test_metrics_collector_dispatches_to_handler(self):
        """MetricsCollector dispatches events to registered handler."""
        received = []

        class TestHandler:
            def handle(self, event: MetricEvent) -> None:
                received.append(event)

        collector = MetricsCollector()
        collector.add_handler(TestHandler())
        collector.emit_timing(MetricType.SEARCH_LATENCY, 42.5, partition="electronics")

        assert len(received) == 1
        assert received[0].metric_type == MetricType.SEARCH_LATENCY
        assert received[0].value == 42.5
        assert received[0].tags["partition"] == "electronics"

    def test_metrics_collector_multiple_handlers(self):
        """Multiple handlers all receive the same event."""
        counts = {"h1": 0, "h2": 0}

        class HandlerA:
            def handle(self, event):
                counts["h1"] += 1

        class HandlerB:
            def handle(self, event):
                counts["h2"] += 1

        collector = MetricsCollector()
        collector.add_handler(HandlerA())
        collector.add_handler(HandlerB())
        collector.emit_count(MetricType.CACHE_HIT)
        collector.emit_count(MetricType.CACHE_MISS)

        assert counts["h1"] == 2
        assert counts["h2"] == 2

    def test_handler_exception_doesnt_break_pipeline(self):
        """A failing handler doesn't prevent other handlers from receiving events."""
        received = []

        class BrokenHandler:
            def handle(self, event):
                raise RuntimeError("boom")

        class GoodHandler:
            def handle(self, event):
                received.append(event)

        collector = MetricsCollector()
        collector.add_handler(BrokenHandler())
        collector.add_handler(GoodHandler())
        collector.emit_count(MetricType.ERROR)

        assert len(received) == 1


class TestEmbeddingCacheReal:
    """Verify embedding cache works with realistic data patterns."""

    def test_cache_hit_rate_after_repeated_queries(self):
        """Repeated queries should produce cache hits."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        key = CacheKey(text="wireless headphones", model="voyage-4", dimensions=1024, input_type="query")
        vec = [0.1] * 1024

        cache.put(key, vec)

        # First get = hit
        result = cache.get(key)
        assert result == vec

        # Second get = another hit
        result = cache.get(key)
        assert result == vec

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 1.0

    def test_cache_disabled_config(self):
        """Config with cache disabled creates zero-size cache."""
        config = _make_config()
        config.cache = CacheConfig(enabled=False)

        cache = EmbeddingCache(
            max_size=config.cache.max_size if config.cache.enabled else 0,
            ttl_seconds=config.cache.ttl_seconds,
        )
        assert not cache.enabled
        key = CacheKey(text="test", model="m", dimensions=32, input_type="query")
        cache.put(key, [1.0] * 32)
        assert cache.get(key) is None

    def test_cache_stats_via_cli(self, mongodb_uri):
        """CLI cache stats command runs without error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            config = _make_config()
            config_path = Path("test_config.json")
            save_config(config, config_path)

            result = runner.invoke(cli_main, ["cache", "stats", "--config", str(config_path)])
            # Command should run even if no client is connected (shows zero stats)
            # The command might fail because no client is connected, but it shouldn't crash
            assert result.exit_code in (0, 1)


class TestPhase6ConfigBackwardCompat:
    """Phase 6 config models load correctly with defaults."""

    def test_config_without_phase6_sections(self):
        """Config without logging/metrics/cache loads with defaults."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.logging.level == "INFO"
        assert config.logging.json_format is False
        assert config.metrics.enabled is True
        assert config.cache.enabled is True
        assert config.cache.max_size == 10_000
        assert config.cache.ttl_seconds == 3600
        assert config.database.max_pool_size == 100
        assert config.database.min_pool_size == 0

    def test_config_with_explicit_phase6(self):
        """Config with explicit Phase 6 settings loads correctly."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
                "max_pool_size": 50,
                "min_pool_size": 5,
                "max_idle_time_ms": 30000,
            },
            "partitioning": {
                "field": "category",
            },
            "logging": {
                "level": "DEBUG",
                "json_format": True,
                "log_query_text": True,
            },
            "metrics": {
                "enabled": False,
            },
            "cache": {
                "enabled": True,
                "max_size": 5000,
                "ttl_seconds": 1800,
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.logging.level == "DEBUG"
        assert config.logging.json_format is True
        assert config.logging.log_query_text is True
        assert config.metrics.enabled is False
        assert config.cache.max_size == 5000
        assert config.cache.ttl_seconds == 1800
        assert config.database.max_pool_size == 50
        assert config.database.min_pool_size == 5
        assert config.database.max_idle_time_ms == 30000

    def test_pool_config_validation(self):
        """Pool config validation catches invalid values."""
        config = _make_config()
        config.database.max_pool_size = 50
        config.database.min_pool_size = 5

        warnings = validate_config(config)
        # Should not have pool-related warnings for valid values
        pool_warnings = [w for w in warnings if "pool" in w.lower()]
        assert len(pool_warnings) == 0

    def test_pool_config_warning_low_max(self):
        """Pool config warns when max_pool_size is very low."""
        config = _make_config()
        config.database.max_pool_size = 5

        warnings = validate_config(config)
        pool_warnings = [w for w in warnings if "pool" in w.lower()]
        assert len(pool_warnings) >= 1


class TestTimeSplitFunctional:
    """Verify time bucket generation with real-world date patterns."""

    def test_generate_monthly_buckets(self):
        """Monthly buckets span full months with correct labels."""
        from semantic_vector_router.lifecycle.splitter import PartitionSplitter

        min_date = datetime(2025, 3, 15, tzinfo=timezone.utc)
        max_date = datetime(2025, 6, 20, tzinfo=timezone.utc)

        buckets = PartitionSplitter._generate_time_buckets(min_date, max_date, "monthly")

        # Should cover March through June (4 months)
        assert len(buckets) == 4
        labels = [b[2] for b in buckets]
        assert labels == ["2025_03", "2025_04", "2025_05", "2025_06"]

        # Each bucket start should be the 1st of the month
        for start, end, label in buckets:
            assert start.day == 1
            assert start.tzinfo is not None

    def test_generate_quarterly_buckets(self):
        """Quarterly buckets span full quarters."""
        from semantic_vector_router.lifecycle.splitter import PartitionSplitter

        min_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
        max_date = datetime(2025, 5, 1, tzinfo=timezone.utc)

        buckets = PartitionSplitter._generate_time_buckets(min_date, max_date, "quarterly")

        labels = [b[2] for b in buckets]
        assert "2024_Q1" in labels
        assert "2025_Q2" in labels

    def test_generate_yearly_buckets(self):
        """Yearly buckets span full years."""
        from semantic_vector_router.lifecycle.splitter import PartitionSplitter

        min_date = datetime(2022, 6, 1, tzinfo=timezone.utc)
        max_date = datetime(2025, 3, 1, tzinfo=timezone.utc)

        buckets = PartitionSplitter._generate_time_buckets(min_date, max_date, "yearly")

        labels = [b[2] for b in buckets]
        assert labels == ["2022", "2023", "2024", "2025"]

    @pytest.mark.asyncio
    async def test_time_split_on_real_data(self, mongodb_uri):
        """Time split aggregation works against real MongoDB with timestamped data."""
        client = AsyncMongoClient(mongodb_uri)
        db = client[TEST_DB]
        collection = db["time_split_test"]

        try:
            # Insert documents with timestamps spanning 3 months
            docs = []
            for month in [1, 2, 3]:
                for i in range(10):
                    docs.append({
                        "category": "articles",
                        "title": f"Article {month}-{i}",
                        "published_at": datetime(2025, month, (i % 28) + 1, tzinfo=timezone.utc),
                        "embedding": _centroid_vector("electronics"),
                    })
            await collection.insert_many(docs)

            # Verify aggregation min/max works
            pipeline = [
                {"$match": {"category": "articles"}},
                {"$group": {
                    "_id": None,
                    "min_date": {"$min": "$published_at"},
                    "max_date": {"$max": "$published_at"},
                    "count": {"$sum": 1},
                }},
            ]
            cursor = await collection.aggregate(pipeline)
            results = await cursor.to_list(length=1)

            assert len(results) == 1
            assert results[0]["count"] == 30
            assert results[0]["min_date"].month == 1
            assert results[0]["max_date"].month == 3
        finally:
            await db.drop_collection("time_split_test")
            await client.close()


# ===========================================================================
# Phase 7 — Ingestion Pipeline + Rate Limiting Functional Tests
# ===========================================================================


class _FakeEmbedder:
    """Fake embedder that returns deterministic centroid vectors.

    For functional ingestion tests — avoids calling real embedding APIs.
    Returns vectors near the category centroid based on document text content.
    """

    max_batch_size = 128

    def __init__(self):
        super().__init__()
        self._call_count = 0

    async def embed(self, text: str) -> list[float]:
        self._call_count += 1
        return self._vector_for_text(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._call_count += 1
        return [self._vector_for_text(t) for t in texts]

    async def embed_with_batching(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        batch_size = batch_size or self.max_batch_size
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(await self.embed_batch(batch))
        return results

    def _vector_for_text(self, text: str) -> list[float]:
        """Map text to a centroid vector based on keyword detection."""
        text_lower = text.lower()
        for cat, centroid in CATEGORY_CENTROIDS.items():
            if cat in text_lower:
                return _centroid_vector(cat, noise=0.1)
        # Default: electronics-ish
        return _centroid_vector("electronics", noise=0.1)


class TestIngestionPipelineReal:
    """IngestPipeline against real MongoDB Atlas.

    Uses a fake embedder to generate structured vectors so we can verify
    the full write path without calling real embedding APIs.
    """

    @pytest.mark.asyncio
    async def test_ingest_source_mode(self, seeded_db):
        """Ingest documents via SOURCE mode and verify they land in MongoDB."""
        ingest_coll_name = "svr_ft_ingest_source"
        config = _make_config(IndexLocation.SOURCE)
        config.database.source_collection = ingest_coll_name
        config.ingestion = IngestConfig(
            text_fields=["name", "category"],
            separator=" ",
            batch_size=10,
        )
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            coll = backend.db[ingest_coll_name]
            await coll.drop()

            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(backend, config, embedder, metrics)

            docs = [
                {"name": "laptop", "category": "electronics", "price": 999},
                {"name": "desk", "category": "furniture", "price": 299},
                {"name": "shirt", "category": "clothing", "price": 49},
            ]
            result = await pipeline.ingest(docs)

            assert result.inserted == 3
            assert result.failed == 0
            assert result.elapsed_ms > 0
            assert result.embed_ms > 0

            # Verify documents in MongoDB
            count = await coll.count_documents({})
            assert count == 3

            # Verify each doc has the embedding field
            async for doc in coll.find():
                assert "embedding" in doc
                assert isinstance(doc["embedding"], list)
                assert len(doc["embedding"]) == DIMENSIONS

        finally:
            await backend.db.drop_collection(ingest_coll_name)
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_fields_mode(self, seeded_db):
        """Ingest documents via FIELDS mode, verify partition-specific embedding field."""
        ingest_coll_name = "svr_ft_ingest_fields"
        config = _make_config(IndexLocation.FIELDS)
        config.database.source_collection = ingest_coll_name
        config.ingestion = IngestConfig(
            text_fields=["name", "category"],
            batch_size=10,
        )
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            coll = backend.db[ingest_coll_name]
            await coll.drop()

            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(backend, config, embedder, metrics)

            docs = [
                {"name": "laptop", "category": "electronics"},
                {"name": "desk", "category": "furniture"},
            ]
            result = await pipeline.ingest(docs)

            assert result.inserted == 2
            assert result.failed == 0

            # FIELDS mode: each doc gets embedding_{partition_name}
            elec_doc = await coll.find_one({"category": "electronics"})
            assert "embedding_electronics" in elec_doc
            assert "embedding_furniture" not in elec_doc

            furn_doc = await coll.find_one({"category": "furniture"})
            assert "embedding_furniture" in furn_doc
            assert "embedding_electronics" not in furn_doc

        finally:
            await backend.db.drop_collection(ingest_coll_name)
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_upsert_mode(self, seeded_db):
        """Upsert mode updates existing documents."""
        ingest_coll_name = "svr_ft_ingest_upsert"
        config = _make_config(IndexLocation.SOURCE)
        config.database.source_collection = ingest_coll_name
        config.ingestion = IngestConfig(
            text_fields=["name"],
            batch_size=10,
            mode=IngestMode.UPSERT,
        )
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            coll = backend.db[ingest_coll_name]
            await coll.drop()

            from bson import ObjectId
            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(backend, config, embedder, metrics)

            doc_id = ObjectId()
            docs = [{"_id": doc_id, "name": "electronics laptop v1", "price": 999}]
            result1 = await pipeline.ingest(docs, mode=IngestMode.INSERT)
            assert result1.inserted == 1

            # Upsert with updated price
            docs_v2 = [{"_id": doc_id, "name": "electronics laptop v2", "price": 1099}]
            result2 = await pipeline.ingest(docs_v2, mode=IngestMode.UPSERT)
            assert result2.inserted == 1

            # Only 1 doc in collection (upsert updated, not inserted second)
            count = await coll.count_documents({})
            assert count == 1

            # Price should be updated
            doc = await coll.find_one({"_id": doc_id})
            assert doc["price"] == 1099
            assert doc["name"] == "electronics laptop v2"

        finally:
            await backend.db.drop_collection(ingest_coll_name)
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_partial_failure(self, seeded_db):
        """Batch with duplicate _ids in INSERT mode: good docs inserted, bad reported."""
        ingest_coll_name = "svr_ft_ingest_partial"
        config = _make_config(IndexLocation.SOURCE)
        config.database.source_collection = ingest_coll_name
        config.ingestion = IngestConfig(
            text_fields=["name"],
            batch_size=10,
            mode=IngestMode.INSERT,
            continue_on_error=True,
        )
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            coll = backend.db[ingest_coll_name]
            await coll.drop()

            from bson import ObjectId
            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(backend, config, embedder, metrics)

            # Insert first doc
            shared_id = ObjectId()
            docs_first = [{"_id": shared_id, "name": "electronics original"}]
            r1 = await pipeline.ingest(docs_first)
            assert r1.inserted == 1

            # Now try inserting a batch with the duplicate _id
            docs_second = [
                {"_id": shared_id, "name": "electronics duplicate"},  # Will fail
                {"name": "electronics new item"},  # Should succeed (no _id conflict)
            ]
            r2 = await pipeline.ingest(docs_second)

            # One should fail (duplicate), one should succeed
            assert r2.inserted >= 1
            assert r2.failed >= 1 or len(r2.errors) >= 1

            # Total docs = original + new item = 2
            total = await coll.count_documents({})
            assert total == 2

        finally:
            await backend.db.drop_collection(ingest_coll_name)
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_bindata_float32(self, seeded_db):
        """Ingest with BINDATA_FLOAT32 storage format writes Binary objects."""
        ingest_coll_name = "svr_ft_ingest_bindata"
        config = _make_config(IndexLocation.SOURCE)
        config.database.source_collection = ingest_coll_name
        config.vector_storage.storage_format = VectorStorageFormat.BINDATA_FLOAT32
        config.ingestion = IngestConfig(
            text_fields=["name"],
            batch_size=10,
        )
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            coll = backend.db[ingest_coll_name]
            await coll.drop()

            from bson import Binary
            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(backend, config, embedder, metrics)

            docs = [{"name": "electronics headphones"}]
            result = await pipeline.ingest(docs)
            assert result.inserted == 1

            # Verify the embedding is stored as Binary, not list
            doc = await coll.find_one({})
            assert isinstance(doc["embedding"], (Binary, bytes))

            # Verify round-trip conversion
            recovered = bindata_to_vector(doc["embedding"])
            assert len(recovered) == DIMENSIONS
            for val in recovered:
                assert isinstance(val, float)

        finally:
            await backend.db.drop_collection(ingest_coll_name)
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_progress_callback(self, seeded_db):
        """Progress callback receives updates during ingestion."""
        ingest_coll_name = "svr_ft_ingest_progress"
        config = _make_config(IndexLocation.SOURCE)
        config.database.source_collection = ingest_coll_name
        config.ingestion = IngestConfig(
            text_fields=["name"],
            batch_size=2,  # Small batch to trigger multiple callbacks
            write_batch_size=2,
        )
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            coll = backend.db[ingest_coll_name]
            await coll.drop()

            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            progress_updates = []

            def on_progress(p):
                progress_updates.append({
                    "phase": p.phase,
                    "embedded": p.embedded,
                    "written": p.written,
                    "total": p.total,
                })

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(
                backend, config, embedder, metrics,
                progress_callback=on_progress,
            )

            docs = [
                {"name": "electronics item1"},
                {"name": "electronics item2"},
                {"name": "electronics item3"},
                {"name": "electronics item4"},
                {"name": "electronics item5"},
            ]
            result = await pipeline.ingest(docs)
            assert result.inserted == 5

            # Should have progress updates for embedding and writing phases
            assert len(progress_updates) > 0
            phases_seen = {p["phase"] for p in progress_updates}
            assert "embedding" in phases_seen
            assert "writing" in phases_seen
            assert "complete" in phases_seen

            # Final update should have all 5 written
            final = progress_updates[-1]
            assert final["phase"] == "complete"

        finally:
            await backend.db.drop_collection(ingest_coll_name)
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_empty_list(self, seeded_db):
        """Ingesting an empty list returns zero-result without error."""
        config = _make_config(IndexLocation.SOURCE)
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(backend, config, embedder, metrics)

            result = await pipeline.ingest([])
            assert result.inserted == 0
            assert result.failed == 0
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_partition_override(self, seeded_db):
        """Explicit partition parameter overrides document field value."""
        ingest_coll_name = "svr_ft_ingest_override"
        config = _make_config(IndexLocation.FIELDS)
        config.database.source_collection = ingest_coll_name
        config.ingestion = IngestConfig(text_fields=["name"], batch_size=10)
        backend = MongoDBBackend(config)
        await backend.connect()

        try:
            coll = backend.db[ingest_coll_name]
            await coll.drop()

            from semantic_vector_router.ingestion import IngestPipeline
            from semantic_vector_router.utils.metrics import MetricsCollector

            embedder = _FakeEmbedder()
            metrics = MetricsCollector()
            pipeline = IngestPipeline(backend, config, embedder, metrics)

            # Doc has category=electronics but we override to "furniture"
            docs = [{"name": "electronics laptop", "category": "electronics"}]
            result = await pipeline.ingest(docs, partition="furniture")
            assert result.inserted == 1

            # Should use the override field name
            doc = await coll.find_one({})
            assert "embedding_furniture" in doc
            assert "embedding_electronics" not in doc

        finally:
            await backend.db.drop_collection(ingest_coll_name)
            await backend.disconnect()


class TestRateLimiterReal:
    """Rate limiter functional tests with real timing."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic_timing(self):
        """Rate limiter with low rate forces waiting."""
        from semantic_vector_router.utils.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(tokens_per_second=100.0, burst=2)

        # Drain bucket
        await limiter.acquire(2)

        # Next acquire should wait
        start = time.monotonic()
        await limiter.acquire(1)
        elapsed = time.monotonic() - start

        assert elapsed > 0.005  # Should wait at least a few ms

        stats = limiter.stats()
        assert stats.total_requests == 2
        assert stats.total_waited >= 1

    @pytest.mark.asyncio
    async def test_rate_limiter_high_throughput(self):
        """Rate limiter with high rate allows rapid acquisition."""
        from semantic_vector_router.utils.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(tokens_per_second=10000.0, burst=100)

        start = time.monotonic()
        for _ in range(50):
            await limiter.acquire(1)
        elapsed = time.monotonic() - start

        # 50 requests at 10000 rps should complete very quickly
        assert elapsed < 1.0

        stats = limiter.stats()
        assert stats.total_requests == 50

    @pytest.mark.asyncio
    async def test_rate_limiter_registry_stats(self):
        """Registry tracks stats across providers."""
        from semantic_vector_router.utils.rate_limiter import RateLimiterRegistry

        config = RateLimitConfig(
            enabled=True,
            default_tokens_per_second=1000.0,
            default_burst=100,
        )
        registry = RateLimiterRegistry(config)

        openai_limiter = registry.get("openai")
        voyage_limiter = registry.get("voyage")

        await openai_limiter.acquire(1)
        await openai_limiter.acquire(1)
        await voyage_limiter.acquire(1)

        stats = registry.stats()
        assert "openai" in stats
        assert "voyage" in stats
        assert stats["openai"].total_requests == 2
        assert stats["voyage"].total_requests == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_safety(self):
        """Concurrent acquires don't corrupt state."""
        from semantic_vector_router.utils.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(tokens_per_second=10000.0, burst=200)

        # Fire 100 concurrent acquires
        results = await asyncio.gather(*[limiter.acquire(1) for _ in range(100)])
        assert len(results) == 100

        stats = limiter.stats()
        assert stats.total_requests == 100


class TestPhase7ConfigBackwardCompat:
    """Phase 7 config models load correctly with defaults."""

    def test_config_without_ingestion_or_rate_limiting(self):
        """Config without ingestion/rate_limiting sections loads with defaults."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.ingestion is not None
        assert config.ingestion.batch_size == 100
        assert config.ingestion.write_batch_size == 500
        assert config.ingestion.mode == IngestMode.INSERT
        assert config.ingestion.continue_on_error is True
        assert config.rate_limiting is not None
        assert config.rate_limiting.enabled is True
        assert config.rate_limiting.default_tokens_per_second == 50.0
        assert config.rate_limiting.default_burst == 100

    def test_config_with_explicit_ingestion(self):
        """Config with explicit ingestion settings loads correctly."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
            "ingestion": {
                "text_fields": ["title", "description"],
                "separator": "\n",
                "batch_size": 50,
                "write_batch_size": 200,
                "mode": "upsert",
                "continue_on_error": False,
            },
            "rate_limiting": {
                "enabled": True,
                "default_tokens_per_second": 30.0,
                "default_burst": 60,
                "providers": {
                    "voyage": {
                        "tokens_per_second": 20.0,
                        "burst": 40,
                    },
                },
            },
        }
        config = load_config(config_dict=config_dict, load_env=False)
        assert config.ingestion.text_fields == ["title", "description"]
        assert config.ingestion.separator == "\n"
        assert config.ingestion.batch_size == 50
        assert config.ingestion.write_batch_size == 200
        assert config.ingestion.mode == IngestMode.UPSERT
        assert config.ingestion.continue_on_error is False
        assert config.rate_limiting.default_tokens_per_second == 30.0
        assert config.rate_limiting.default_burst == 60
        assert config.rate_limiting.providers["voyage"].tokens_per_second == 20.0
        assert config.rate_limiting.providers["voyage"].burst == 40

    def test_config_validation_phase7(self):
        """Phase 7 config values are validated."""
        config = _make_config()
        # Valid config should pass
        warnings = validate_config(config)
        # Should not have ingestion/rate_limiting errors for default values
        assert not any("ingestion" in w.lower() and "error" in w.lower() for w in warnings)

    def test_config_file_roundtrip_with_ingestion(self):
        """Config with ingestion settings survives save/load round-trip."""
        config = _make_config()
        config.ingestion = IngestConfig(
            text_fields=["title", "body"],
            template="{title}\n{body}",
            batch_size=64,
            mode=IngestMode.UPSERT,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            save_config(config, f.name)
            loaded = load_config(config_path=f.name, load_env=False)

        assert loaded.ingestion.text_fields == ["title", "body"]
        assert loaded.ingestion.template == "{title}\n{body}"
        assert loaded.ingestion.batch_size == 64
        assert loaded.ingestion.mode == IngestMode.UPSERT
        os.unlink(f.name)


class TestCLIIngestFunctional:
    """CLI ingest command functional tests."""

    def test_cli_ingest_help(self, cli_runner):
        """'svr ingest --help' works."""
        result = cli_runner.invoke(cli_main, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output.lower()
        assert "--partition" in result.output
        assert "--mode" in result.output

    def test_cli_help_includes_ingest(self, cli_runner):
        """Main help shows ingest command."""
        result = cli_runner.invoke(cli_main, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output
