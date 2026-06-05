"""Integration tests: ingest documents via real Voyage embeddings, then search.

Validates the full ingest -> embed -> store -> search -> retrieve pipeline
against real MongoDB Atlas with real Voyage AI embeddings.

Requires: MONGODB_URI and VOYAGE_API_KEY environment variables.
Run with: .venv/bin/pytest tests/integration/test_ingest_search_roundtrip.py -v -s --timeout=300
"""

import asyncio
import logging
import os
import time

import pytest
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.client import SVRClient
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestConfig,
    IngestMode,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
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
# Constants
# ---------------------------------------------------------------------------

TEST_DB = "svr_integration_test"
ROUNDTRIP_COLLECTION = "products_roundtrip"
DIMENSIONS = 512
INDEX_WAIT_TIMEOUT = 180  # seconds
INDEX_POLL_INTERVAL = 3  # seconds

CATEGORIES = ["electronics", "food", "outdoor"]

SAMPLE_DOCS = {
    "electronics": [
        {
            "name": "Wireless Headphones",
            "description": "Premium Bluetooth noise-cancelling headphones with 30-hour battery life",
            "category": "electronics",
            "price": 299.99,
        },
        {
            "name": "USB-C Hub",
            "description": "Multi-port USB-C docking station with HDMI, ethernet and SD card slots",
            "category": "electronics",
            "price": 49.99,
        },
        {
            "name": "Mechanical Keyboard",
            "description": "Cherry MX Brown mechanical keyboard with RGB backlighting",
            "category": "electronics",
            "price": 159.99,
        },
    ],
    "food": [
        {
            "name": "Artisan Pasta",
            "description": "HandmParts Distributor Italian pappardelle pasta from durum wheat semolina",
            "category": "food",
            "price": 12.99,
        },
        {
            "name": "Olive Oil",
            "description": "Extra virgin cold-pressed olive oil from Tuscany Italy",
            "category": "food",
            "price": 24.99,
        },
        {
            "name": "Truffle Salt",
            "description": "Black truffle infused sea salt for gourmet cooking and seasoning",
            "category": "food",
            "price": 18.99,
        },
    ],
    "outdoor": [
        {
            "name": "Hiking Boots",
            "description": "Waterproof mountain hiking trail boots with vibram sole for rough terrain",
            "category": "outdoor",
            "price": 189.99,
        },
        {
            "name": "Camping Tent",
            "description": "Two person lightweight backpacking tent for mountain camping trips",
            "category": "outdoor",
            "price": 249.99,
        },
        {
            "name": "Trekking Poles",
            "description": "Adjustable carbon fiber trekking poles for mountain hiking trails",
            "category": "outdoor",
            "price": 79.99,
        },
    ],
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _make_integration_config(
    collection_name: str = ROUNDTRIP_COLLECTION,
    index_on: IndexLocation = IndexLocation.SOURCE,
) -> SVRConfig:
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database=TEST_DB,
            source_collection=collection_name,
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_int_rt_",
            index_name_prefix="svr_int_rt_idx_",
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
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-3-lite",
            dimensions=DIMENSIONS,
            api_key_env="VOYAGE_API_KEY",
        ),
        reranking=RerankingConfig(enabled=False),
        ingestion=IngestConfig(
            text_fields=["description"],
            batch_size=50,
        ),
        resilience=ResilienceConfig(embedding_timeout_ms=30000),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_index(
    backend: MongoDBBackend,
    collection_name: str,
    index_name: str,
    timeout: int = INDEX_WAIT_TIMEOUT,
    poll_interval: int = INDEX_POLL_INTERVAL,
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
            await asyncio.sleep(2)
            return True
        await asyncio.sleep(poll_interval)
    return False


async def _cleanup_indexes(backend: MongoDBBackend, collection_name: str) -> None:
    """Best-effort cleanup of indexes on a collection."""
    try:
        await backend.delete_vector_search_index(collection_name, "svr_vector_idx_source")
    except Exception:
        pass
    for cat in CATEGORIES:
        try:
            await backend.delete_vector_search_index(collection_name, f"svr_int_rt_idx_{cat}")
        except Exception:
            pass
        try:
            await backend.delete_partition_view(f"svr_int_rt_{cat}")
        except Exception:
            pass


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
def voyage_api_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        pytest.skip("VOYAGE_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def seeded_roundtrip_db(mongodb_uri, voyage_api_key):
    """Seed the roundtrip test database once for all tests in this module.

    Creates the collection, seeds placeholder docs, creates partitions with
    indexes, ingests all sample docs with real Voyage embeddings, and waits
    for the index to become queryable. Tests create their own fresh SVRClient
    per test to avoid event loop mismatch.

    Yields a dict with the config and info about what was seeded.
    """
    config = _make_integration_config()

    # Clean slate
    raw_client = AsyncMongoClient(mongodb_uri)
    db = raw_client[TEST_DB]
    await db.drop_collection(ROUNDTRIP_COLLECTION)
    await db.drop_collection("svr_metadata")

    # Seed placeholder documents so the collection exists before creating indexes.
    coll = db[ROUNDTRIP_COLLECTION]
    seed_docs = []
    for cat in CATEGORIES:
        for doc in SAMPLE_DOCS[cat]:
            seed_docs.append({
                "name": doc["name"],
                "description": doc["description"],
                "category": doc["category"],
                "price": doc["price"],
            })
    await coll.insert_many(seed_docs)
    await raw_client.close()

    # Create SVRClient, connect, create partitions
    client = SVRClient(config=config, auto_connect=False)
    await client.connect()

    # Create partitions for each category
    for cat in CATEGORIES:
        await client.create_partition(cat)

    # Wait for the source index to become queryable
    backend = client._backend
    ready = await _wait_for_index(backend, ROUNDTRIP_COLLECTION, "svr_vector_idx_source")
    if not ready:
        await client.disconnect()
        pytest.skip("Index not queryable in time")

    # Ingest all sample documents with real Voyage embeddings
    for cat in CATEGORIES:
        result = await client.ingest(SAMPLE_DOCS[cat], partition=cat)
        logger.info(
            f"Seeded {cat}: inserted={result.inserted}, failed={result.failed}"
        )

    # Wait for index to catch up with newly embedded docs
    await asyncio.sleep(5)

    # Disconnect the module-scoped client; tests will create their own
    await client.disconnect()

    yield {
        "config": config,
        "categories": CATEGORIES,
        "sample_docs": SAMPLE_DOCS,
    }

    # Teardown: clean up indexes and collections
    try:
        cleanup_backend = MongoDBBackend(config)
        await cleanup_backend.connect()
        await _cleanup_indexes(cleanup_backend, ROUNDTRIP_COLLECTION)
        await cleanup_backend.disconnect()
    except Exception:
        pass
    try:
        raw_client2 = AsyncMongoClient(mongodb_uri)
        await raw_client2[TEST_DB].drop_collection(ROUNDTRIP_COLLECTION)
        await raw_client2[TEST_DB].drop_collection("svr_metadata")
        await raw_client2.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIngestSearchRoundtrip:
    """Ingest documents, then search to verify correct retrieval."""

    @pytest.mark.asyncio
    async def test_ingest_and_search(self, seeded_roundtrip_db):
        """Search pre-ingested documents and verify ranking."""
        config = seeded_roundtrip_db["config"]

        # Create a fresh client for this test
        client = SVRClient(config=config, auto_connect=False)
        await client.connect()

        try:
            # Search for bluetooth headphones - the Wireless Headphones doc should rank high
            search_result = await client.search(
                query="bluetooth wireless headphones",
                partitions=["electronics"],
                limit=5,
            )

            assert len(search_result.hits) > 0
            assert search_result.latency_ms > 0
            assert "electronics" in search_result.partitions_searched

            # The headphones doc should be in top 3
            top_3_names = [hit.document.get("name", "") for hit in search_result.hits[:3]]
            assert "Wireless Headphones" in top_3_names, (
                f"Expected 'Wireless Headphones' in top 3, got: {top_3_names}"
            )
        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_insert_vs_upsert(self, seeded_roundtrip_db):
        """INSERT mode inserts, UPSERT mode with same _id does not duplicate."""
        config = seeded_roundtrip_db["config"]

        client = SVRClient(config=config, auto_connect=False)
        await client.connect()

        try:
            # Create docs with explicit _ids
            doc_id_1 = ObjectId()
            doc_id_2 = ObjectId()
            docs_insert = [
                {
                    "_id": doc_id_1,
                    "name": "Test Gadget A",
                    "description": "A small electronic testing gadget for unit tests",
                    "category": "electronics",
                    "price": 39.99,
                },
                {
                    "_id": doc_id_2,
                    "name": "Test Gadget B",
                    "description": "Another electronic device for integration testing",
                    "category": "electronics",
                    "price": 59.99,
                },
            ]

            # INSERT mode
            r1 = await client.ingest(
                docs_insert, partition="electronics", mode=IngestMode.INSERT
            )
            assert r1.inserted == 2
            assert r1.failed == 0

            # Count docs with these IDs
            coll = client._backend.db[ROUNDTRIP_COLLECTION]
            count_before = await coll.count_documents(
                {"_id": {"$in": [doc_id_1, doc_id_2]}}
            )
            assert count_before == 2

            # UPSERT with same IDs, updated descriptions
            docs_upsert = [
                {
                    "_id": doc_id_1,
                    "name": "Test Gadget A Updated",
                    "description": "An updated electronic gadget for upsert testing",
                    "category": "electronics",
                    "price": 44.99,
                },
                {
                    "_id": doc_id_2,
                    "name": "Test Gadget B Updated",
                    "description": "Another updated electronic device",
                    "category": "electronics",
                    "price": 64.99,
                },
            ]
            r2 = await client.ingest(
                docs_upsert, partition="electronics", mode=IngestMode.UPSERT
            )
            # Upsert should succeed (matched + modified or upserted)
            assert r2.failed == 0

            # Count should still be 2 (not 4)
            count_after = await coll.count_documents(
                {"_id": {"$in": [doc_id_1, doc_id_2]}}
            )
            assert count_after == 2

            # Verify updated content
            doc = await coll.find_one({"_id": doc_id_1})
            assert doc["name"] == "Test Gadget A Updated"
            assert doc["price"] == 44.99
        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_progress_callback(self, seeded_roundtrip_db):
        """Progress callback is called with embedding and writing phases."""
        config = seeded_roundtrip_db["config"]

        client = SVRClient(config=config, auto_connect=False)
        await client.connect()

        try:
            progress_updates = []

            def on_progress(p):
                progress_updates.append({
                    "phase": p.phase,
                    "embedded": p.embedded,
                    "written": p.written,
                    "total": p.total,
                    "failed": p.failed,
                })

            docs = [
                {
                    "name": "Progress Test Item 1",
                    "description": "First item for testing progress callbacks during ingestion",
                    "category": "electronics",
                    "price": 10.00,
                },
                {
                    "name": "Progress Test Item 2",
                    "description": "Second item for testing progress callbacks during ingestion",
                    "category": "electronics",
                    "price": 20.00,
                },
                {
                    "name": "Progress Test Item 3",
                    "description": "Third item for testing progress callbacks during ingestion",
                    "category": "electronics",
                    "price": 30.00,
                },
            ]

            result = await client.ingest(
                docs,
                partition="electronics",
                progress_callback=on_progress,
            )
            assert result.inserted == 3

            # Verify callback was called
            assert len(progress_updates) > 0

            # Should have seen embedding and writing phases
            phases_seen = {p["phase"] for p in progress_updates}
            assert "embedding" in phases_seen
            assert "writing" in phases_seen
            assert "complete" in phases_seen
        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_ingest_error_handling(self, seeded_roundtrip_db):
        """Documents missing text fields are reported as failures, others succeed."""
        config = seeded_roundtrip_db["config"]

        client = SVRClient(config=config, auto_connect=False)
        await client.connect()

        try:
            docs = [
                {
                    "name": "Good Document",
                    "description": "This document has a valid description field for embedding",
                    "category": "electronics",
                    "price": 99.99,
                },
                {
                    # Missing 'description' field entirely - should fail text extraction
                    "name": "Bad Document",
                    "category": "electronics",
                    "price": 49.99,
                },
            ]

            # The config uses text_fields=["description"], so the doc without
            # 'description' should fail text extraction.
            result = await client.ingest(docs, partition="electronics")

            # At least the good document should succeed
            assert result.inserted >= 1

            # The bad document should have failed
            assert result.failed >= 1
            assert len(result.errors) >= 1

            # Verify the error message references the document index
            error_indices = [idx for idx, msg in result.errors]
            assert 1 in error_indices  # Index 1 is the bad doc
        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_search_relevance(self, seeded_roundtrip_db):
        """Search within specific partitions, verify semantic ranking is correct."""
        config = seeded_roundtrip_db["config"]

        client = SVRClient(config=config, auto_connect=False)
        await client.connect()

        try:
            # Search for "Italian pasta" within food partition -- Artisan Pasta should rank top
            search_food = await client.search(
                query="Italian pasta recipe cooking",
                partitions=["food"],
                limit=3,
            )

            assert len(search_food.hits) > 0
            assert "food" in search_food.partitions_searched

            # Artisan Pasta should be #1 within the food partition
            top_food = search_food.hits[0]
            assert top_food.document.get("name") == "Artisan Pasta", (
                f"Expected 'Artisan Pasta' as top food hit, got: "
                f"{top_food.document.get('name', 'unknown')}"
            )

            # Search for "mountain hiking trails" within outdoor partition
            search_outdoor = await client.search(
                query="mountain hiking trails outdoor adventure",
                partitions=["outdoor"],
                limit=3,
            )

            assert len(search_outdoor.hits) > 0
            assert "outdoor" in search_outdoor.partitions_searched

            # Top hit should be hiking-related (Hiking Boots or Trekking Poles)
            top_outdoor = search_outdoor.hits[0]
            outdoor_name = top_outdoor.document.get("name", "")
            assert outdoor_name in ("Hiking Boots", "Trekking Poles"), (
                f"Expected hiking-related top outdoor hit, got: {outdoor_name}"
            )

            # Cross-partition search: verify results come from multiple partitions
            search_all = await client.search(
                query="products for outdoor activities",
                partitions="all",
                limit=9,
            )

            assert len(search_all.hits) > 0
            # Should have results from at least 2 partitions
            partitions_in_results = {hit.partition for hit in search_all.hits}
            assert len(partitions_in_results) >= 2, (
                f"Expected results from >= 2 partitions, got: {partitions_in_results}"
            )
        finally:
            await client.disconnect()
