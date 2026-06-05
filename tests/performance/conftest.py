"""Performance test fixtures.

Module-scoped fixtures that:
- Load .env for credentials
- Skip if MONGODB_URI or VOYAGE_API_KEY not set
- Create a shared SVRClient with SOURCE mode config
- Pre-seed with data (50+ products across categories)
- Wait for index to be queryable
- Cleanup after module completes
"""

import asyncio
import logging
import os
import random
import time

import pytest
from dotenv import load_dotenv
from pymongo import AsyncMongoClient

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
    PartitioningConfig,
    RateLimitConfig,
    RerankingConfig,
    ResilienceConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Test constants
PERF_DB = "svr_performance_test"
PERF_COLLECTION = "products_perf"
CATEGORIES = ["electronics", "furniture", "clothing"]

SAMPLE_PRODUCTS = {
    "electronics": [
        "wireless noise-canceling headphones with bluetooth 5.0",
        "portable bluetooth speaker with deep bass",
        "adjustable laptop stand for ergonomic setup",
        "usb-c multiport hub with hdmi and ethernet",
        "mechanical keyboard with cherry mx switches",
        "ergonomic gaming mouse with adjustable dpi",
        "hd webcam with built-in microphone",
        "dual monitor arm with cable management",
        "high capacity power bank 20000mah",
        "smart fitness watch with heart rate monitor",
        "protective tablet case with kickstand",
        "fast wireless phone charger pad",
        "premium earbuds with active noise cancellation",
        "desktop cable organizer and management kit",
        "led desk lamp with adjustable brightness",
        "usb microphone for podcasting and streaming",
        "wireless charging mouse pad",
    ],
    "furniture": [
        "ergonomic mesh office chair with lumbar support",
        "electric standing desk with memory settings",
        "solid oak bookshelf with five shelves",
        "metal filing cabinet with lock",
        "bamboo desk organizer with drawers",
        "wooden monitor riser with storage",
        "ergonomic footrest for under desk",
        "leather desk pad for writing",
        "under desk cable management tray",
        "rolling drawer unit on casters",
        "floating wall shelf bracket set",
        "modern coat rack with hooks",
        "mid-century side table",
        "storage ottoman with lid",
        "tv wall mount bracket adjustable",
        "standing desk converter for existing desk",
        "ergonomic kneeling chair",
    ],
    "clothing": [
        "organic cotton crew neck t-shirt",
        "slim fit stretch denim jeans",
        "merino wool pullover sweater",
        "waterproof rain jacket with hood",
        "lightweight running shoes with arch support",
        "canvas backpack with laptop compartment",
        "genuine leather reversible belt",
        "silk scarf with printed pattern",
        "adjustable baseball cap",
        "insulated winter gloves touchscreen compatible",
        "waterproof hiking boots with ankle support",
        "breathable linen pants for summer",
        "classic polo shirt in cotton pique",
        "fleece vest with zippered pockets",
        "cushioned ankle socks pack of six",
        "softshell jacket windproof",
        "moisture wicking performance t-shirt",
    ],
}


def _make_perf_config() -> SVRConfig:
    """Create performance test configuration."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database=PERF_DB,
            source_collection=PERF_COLLECTION,
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_perf_",
            index_name_prefix="svr_perf_idx_",
        ),
        vector_storage=VectorStorageConfig(
            index_on=IndexLocation.SOURCE,
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
        ingestion=IngestConfig(text_fields=["description"]),
        resilience=ResilienceConfig(embedding_timeout_ms=30000),
        cache=CacheConfig(enabled=True, max_size=200),
        rate_limiting=RateLimitConfig(enabled=True),
    )


async def _wait_for_index(
    backend,
    collection_name: str,
    index_name: str,
    timeout: int = 300,
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
            await asyncio.sleep(2)
            return True
        await asyncio.sleep(poll_interval)
    return False


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for module-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def perf_client():
    """Create a shared SVRClient for performance tests.

    - Skips if MONGODB_URI or VOYAGE_API_KEY not set
    - Seeds 51 products across 3 categories
    - Creates partitions and waits for index
    - Cleans up after all tests complete
    """
    mongodb_uri = os.environ.get("MONGODB_URI")
    voyage_key = os.environ.get("VOYAGE_API_KEY")

    if not mongodb_uri:
        pytest.skip("MONGODB_URI not set")
    if not voyage_key:
        pytest.skip("VOYAGE_API_KEY not set")

    # Clean slate
    raw_client = AsyncMongoClient(mongodb_uri)
    await raw_client.drop_database(PERF_DB)

    # Seed data
    db = raw_client[PERF_DB]
    collection = db[PERF_COLLECTION]

    all_docs = []
    for category, products in SAMPLE_PRODUCTS.items():
        for i, desc in enumerate(products):
            all_docs.append({
                "name": f"{category}_{i}",
                "description": desc,
                "category": category,
                "price": round(random.uniform(10.0, 500.0), 2),
                "brand": random.choice(["BrandA", "BrandB", "BrandC"]),
                "in_stock": random.choice([True, False]),
            })

    await collection.insert_many(all_docs)
    count = await collection.count_documents({})
    logger.info(f"Seeded {PERF_DB}.{PERF_COLLECTION} with {count} docs")
    await raw_client.close()

    # Create SVRClient and connect
    config = _make_perf_config()
    client = SVRClient(config=config, auto_connect=False)
    await client.connect()

    # Create partitions
    provisioner = PartitionProvisioner(client._backend, client.config, auto_save_config=False)
    await provisioner.ensure_source_index()

    for category in CATEGORIES:
        await provisioner.create_partition(
            name=category,
            filter_value=category,
            skip_if_exists=True,
            create_view=False,
        )

    # Wait for index to be queryable
    from semantic_vector_router.lifecycle.provisioner import SOURCE_INDEX_NAME

    queryable = await _wait_for_index(
        client._backend, PERF_COLLECTION, SOURCE_INDEX_NAME
    )
    if not queryable:
        await client.disconnect()
        pytest.skip("Index did not become queryable within timeout")

    # Ingest documents to add embeddings
    docs_for_ingest = []
    for category, products in SAMPLE_PRODUCTS.items():
        for i, desc in enumerate(products):
            docs_for_ingest.append({
                "name": f"{category}_{i}",
                "description": desc,
                "category": category,
                "price": round(random.uniform(10.0, 500.0), 2),
            })

    result = await client.ingest(docs_for_ingest)
    logger.info(
        f"Ingested {result.inserted} docs, {result.failed} failed "
        f"in {result.elapsed_ms:.0f}ms"
    )

    # Wait a bit for index to reflect new embeddings
    await asyncio.sleep(5)

    yield client

    # Cleanup
    await client.disconnect()
    cleanup_client = AsyncMongoClient(mongodb_uri)
    await cleanup_client.drop_database(PERF_DB)
    await cleanup_client.close()
