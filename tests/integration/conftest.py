"""Integration test fixtures and helpers for Semantic Vector Router.

Provides shared configuration, wait-for-index polling, metrics capture,
and database cleanup for integration tests that run against real MongoDB
Atlas and Voyage AI endpoints.

Requires: MONGODB_URI and VOYAGE_API_KEY environment variables.
Run with: pytest tests/integration/ -v -s --timeout=600 -m integration
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional

import pytest
from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.models import (
    CacheConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestConfig,
    LogConfig,
    MetricsConfig,
    PartitioningConfig,
    RerankingConfig,
    ResilienceConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
)
from semantic_vector_router.utils.metrics import MetricEvent, MetricsHandler

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTEGRATION_TEST_DB = "svr_integration_test"
INTEGRATION_TEST_COLLECTION = "products"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mongodb_uri() -> str:
    """Return MONGODB_URI from environment, skip if missing."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set")
    return uri


@pytest.fixture(scope="module")
def voyage_api_key() -> str:
    """Return VOYAGE_API_KEY from environment, skip if missing."""
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        pytest.skip("VOYAGE_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def cleanup_db(mongodb_uri):
    """Drop the integration test database after the module completes."""
    yield
    client = AsyncMongoClient(mongodb_uri)
    await client.drop_database(INTEGRATION_TEST_DB)
    await client.close()
    logger.info(f"Dropped integration test database: {INTEGRATION_TEST_DB}")


# ---------------------------------------------------------------------------
# Metrics capture handler
# ---------------------------------------------------------------------------


class CapturingMetricsHandler:
    """MetricsHandler that stores events for test assertions."""

    def __init__(self) -> None:
        self.events: list[MetricEvent] = []

    def handle(self, event: MetricEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()

    def find(self, metric_type: str) -> list[MetricEvent]:
        """Return all events matching the given metric type value."""
        return [e for e in self.events if e.metric_type.value == metric_type]

    def has(self, metric_type: str) -> bool:
        """Check if at least one event of the given type was captured."""
        return len(self.find(metric_type)) > 0


@pytest.fixture(scope="module")
def metrics_handler() -> CapturingMetricsHandler:
    """Module-scoped capturing metrics handler for assertions."""
    return CapturingMetricsHandler()


# ---------------------------------------------------------------------------
# Helper: wait for Atlas index to become queryable
# ---------------------------------------------------------------------------


async def wait_for_index(
    backend: MongoDBBackend,
    collection_name: str,
    index_name: str,
    timeout: int = 300,
    poll_interval: int = 5,
) -> bool:
    """Poll until an Atlas vector search index becomes queryable.

    Args:
        backend: Connected MongoDBBackend.
        collection_name: Collection the index lives on.
        index_name: Name of the vector search index.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between polls.

    Returns:
        True if the index became queryable within the timeout, False otherwise.
    """
    start = time.time()
    while time.time() - start < timeout:
        status = await backend.get_index_status(collection_name, index_name)
        state = status.get("status", "unknown")
        queryable = status.get("queryable", False)
        elapsed = int(time.time() - start)
        logger.info(
            f"  [{elapsed}s] {index_name}: status={state}, queryable={queryable}"
        )
        if queryable:
            # Extra delay for filter field indexing to catch up
            await asyncio.sleep(3)
            return True
        await asyncio.sleep(poll_interval)
    return False


# ---------------------------------------------------------------------------
# Helper: config factory
# ---------------------------------------------------------------------------


def make_svr_config(
    index_on: IndexLocation,
    db_name: str = INTEGRATION_TEST_DB,
    collection_name: str = INTEGRATION_TEST_COLLECTION,
    dimensions: int = 512,
    partition_field: str = "category",
    view_prefix: str = "svr_int_",
    index_name_prefix: str = "svr_int_idx_",
    reranking_enabled: bool = False,
    text_fields: Optional[list[str]] = None,
) -> SVRConfig:
    """Build an SVRConfig suitable for integration tests.

    Uses Voyage AI (voyage-3-lite) at 512 dimensions by default.
    """
    if text_fields is None:
        text_fields = ["name", "description"]

    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database=db_name,
            source_collection=collection_name,
        ),
        partitioning=PartitioningConfig(
            field=partition_field,
            view_prefix=view_prefix,
            index_name_prefix=index_name_prefix,
        ),
        vector_storage=VectorStorageConfig(
            index_on=index_on,
            storage_format=VectorStorageFormat.ARRAY,
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=dimensions,
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-3-lite",
            api_key_env="VOYAGE_API_KEY",
            dimensions=dimensions,
        ),
        reranking=RerankingConfig(enabled=reranking_enabled),
        resilience=ResilienceConfig(
            embedding_timeout_ms=30000,
        ),
        metrics=MetricsConfig(enabled=True),
        logging=LogConfig(level="INFO"),
        cache=CacheConfig(
            enabled=True,
            max_size=100,
        ),
        ingestion=IngestConfig(
            text_fields=text_fields,
            separator=" | ",
            batch_size=50,
        ),
    )


# ---------------------------------------------------------------------------
# Sample documents for integration tests
# ---------------------------------------------------------------------------

ELECTRONICS_DOCS = [
    {"name": "Wireless Noise-Cancelling Headphones", "description": "Premium Bluetooth headphones with active noise cancellation, 30-hour battery life, and comfortable over-ear design for immersive audio experience", "category": "electronics", "price": 299.99},
    {"name": "Portable USB-C Charging Hub", "description": "Compact 7-port USB hub with fast charging for laptops, tablets, and smartphones with power delivery support", "category": "electronics", "price": 49.99},
    {"name": "Mechanical Gaming Keyboard", "description": "RGB backlit mechanical keyboard with Cherry MX switches, programmable macros, and aluminum frame construction", "category": "electronics", "price": 149.99},
    {"name": "4K Ultra HD Webcam", "description": "Professional webcam with auto-focus, dual noise reduction microphones, low-light correction, and wide-angle lens", "category": "electronics", "price": 89.99},
    {"name": "Smart Fitness Watch", "description": "GPS-enabled fitness tracker with heart rate monitor, blood oxygen sensor, sleep tracking, and 7-day battery life", "category": "electronics", "price": 199.99},
    {"name": "Wireless Earbuds Pro", "description": "True wireless earbuds with spatial audio, adaptive EQ, active noise cancellation, and sweat resistance for workouts", "category": "electronics", "price": 179.99},
    {"name": "Laptop Cooling Stand", "description": "Adjustable aluminum laptop stand with dual cooling fans, USB passthrough, and ergonomic height adjustment for heat dissipation", "category": "electronics", "price": 39.99},
]

FURNITURE_DOCS = [
    {"name": "Ergonomic Office Chair", "description": "Adjustable lumbar support office chair with breathable mesh back, 4D armrests, and recline function for all-day comfort", "category": "furniture", "price": 599.99},
    {"name": "Standing Desk Converter", "description": "Height-adjustable sit-stand desk riser with keyboard tray, monitor mount, and smooth pneumatic lift mechanism", "category": "furniture", "price": 349.99},
    {"name": "Solid Oak Bookshelf", "description": "Five-tier solid oak bookshelf with adjustable shelves, anti-tip wall mount, and natural wood grain finish", "category": "furniture", "price": 449.99},
    {"name": "Executive Filing Cabinet", "description": "Three-drawer lateral filing cabinet with lock, full-extension ball-bearing slides, and commercial-grParts Distributor steel construction", "category": "furniture", "price": 279.99},
    {"name": "L-Shaped Computer Desk", "description": "Spacious corner desk with cable management grommets, modesty panel, and scratch-resistant laminate surface", "category": "furniture", "price": 499.99},
    {"name": "Leather Desk Pad", "description": "Premium full-grain leather desk pad with non-slip base, waterproof surface, and stitched edges for professional workspace", "category": "furniture", "price": 69.99},
    {"name": "Monitor Riser Shelf", "description": "Bamboo monitor stand with storage drawer, ventilation slots, and ergonomic height for comfortable viewing angle", "category": "furniture", "price": 45.99},
]

CLOTHING_DOCS = [
    {"name": "Merino Wool Base Layer", "description": "Lightweight merino wool thermal top for hiking, skiing, and outdoor winter activities with moisture wicking", "category": "clothing", "price": 89.99},
    {"name": "Waterproof Hiking Jacket", "description": "Three-layer Gore-Tex shell jacket with sealed seams, adjustable hood, and pit zips for breathable waterproof protection", "category": "clothing", "price": 249.99},
    {"name": "Stretch Denim Jeans", "description": "Comfortable stretch denim with classic straight fit, reinforced knees, and sustainable organic cotton blend", "category": "clothing", "price": 79.99},
    {"name": "Trail Running Shoes", "description": "Lightweight trail runners with Vibram outsole, responsive cushioning, and rock plate for technical terrain", "category": "clothing", "price": 139.99},
    {"name": "Down Insulated Vest", "description": "Packable 800-fill goose down vest with water-resistant shell, zippered pockets, and snap collar", "category": "clothing", "price": 169.99},
    {"name": "UV Protection Sun Hat", "description": "Wide-brim UPF 50+ sun hat with moisture-wicking sweatband, chin strap, and packable design for travel", "category": "clothing", "price": 34.99},
    {"name": "Organic Cotton T-Shirt", "description": "Sustainably sourced organic cotton crew neck tee with tagless comfort and pre-shrunk construction", "category": "clothing", "price": 29.99},
]

ALL_DOCS = ELECTRONICS_DOCS + FURNITURE_DOCS + CLOTHING_DOCS
