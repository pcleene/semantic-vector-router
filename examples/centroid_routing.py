"""Centroid routing example -- intelligent partition selection.

Demonstrates how SVR routes queries to the most relevant partitions
using precomputed centroid embeddings, reducing fan-out from O(N) to O(log N).

Without centroid routing, a search with partitions="all" fans out to every
partition. With centroid routing enabled, SVR computes cosine similarity
between the query embedding and each partition's centroid, then only searches
the partitions above a dynamic threshold.

Resolution cascParts Distributor (first match wins):
  1. Explicit partition names -> resolve directly (bypass routing)
  2. Filter-map routing -> match filters against partition field
  3. Centroid routing -> walk partition tree using embedding similarity
  4. Fallback -> fan-out to all partitions

Prerequisites:
  1. pip install semantic-vector-router[voyage]
  2. Set MONGODB_URI and VOYAGE_API_KEY in .env
  3. Run: python examples/centroid_routing.py
"""

import asyncio

from semantic_vector_router import SVRClient
from semantic_vector_router.models import (
    CentroidRoutingConfig,
    RoutingConfig,
    RoutingMode,
)


async def main() -> None:
    # Step 1: Configure with centroid routing enabled
    config = {
        "version": "1.0",
        "database": {
            "connection_string_env": "MONGODB_URI",
            "database": "product_catalog",
            "source_collection": "products",
        },
        "partitioning": {
            "field": "category",
            "strategy": "exact_match",
        },
        "vector_storage": {
            "index_on": "source",
            "embedding_field": "embedding",
            "dimensions": 512,
            "similarity": "cosine",
        },
        "embedding": {
            "mode": "byom",
            "provider": "voyage",
            "model": "voyage-3-lite",
            "api_key_env": "VOYAGE_API_KEY",
            "dimensions": 512,
        },
        "routing": {
            "mode": "auto",  # Enables the full routing cascade
            "default_partitions": "all",
            "centroid_routing": {
                "enabled": True,
                "relative_threshold": 0.5,   # Prune partitions below 50% of max score
                "min_score": 0.15,            # Absolute floor for relevance
                "max_probe_partitions": 3,    # Search at most 3 partitions
                "sample_size": 500,           # Docs to sample for centroid computation
            },
        },
        "ingestion": {
            "text_fields": ["name", "description"],
        },
    }

    svr = SVRClient(config=config, auto_connect=False)
    await svr.connect()

    # Step 2: Create partitions and ingest documents
    for category in ["electronics", "furniture", "clothing"]:
        await svr.create_partition(category)

    electronics_docs = [
        {"name": "Wireless Headphones", "description": "Bluetooth ANC headphones", "category": "electronics"},
        {"name": "Mechanical Keyboard", "description": "Cherry MX RGB keyboard", "category": "electronics"},
    ]
    furniture_docs = [
        {"name": "Ergonomic Chair", "description": "Lumbar support mesh chair", "category": "furniture"},
        {"name": "Standing Desk", "description": "Height-adjustable sit-stand desk", "category": "furniture"},
    ]
    clothing_docs = [
        {"name": "Hiking Jacket", "description": "Waterproof Gore-Tex shell", "category": "clothing"},
        {"name": "Trail Shoes", "description": "Vibram sole trail runners", "category": "clothing"},
    ]

    await svr.ingest(electronics_docs, partition="electronics")
    await svr.ingest(furniture_docs, partition="furniture")
    await svr.ingest(clothing_docs, partition="clothing")
    print("Ingested documents into 3 partitions")

    # Step 3: Compute centroids (happens automatically after ingest when
    # centroid_routing.enabled=True, or via CLI: svr partitions compute-centroids)
    print("Centroids computed automatically after ingest")

    # Step 4: Search WITHOUT specifying a partition
    # SVR uses the routing cascade: since partitions="all" and centroid routing
    # is enabled, it scores each partition's centroid against the query embedding
    # and only searches the most relevant partitions.
    print("\n--- Centroid-routed searches ---")

    result = await svr.search(
        query="noise cancelling bluetooth headphones",
        partitions="all",  # Let centroid routing decide
        limit=3,
    )
    print(f"\nQuery: 'noise cancelling bluetooth headphones'")
    print(f"  Partitions searched: {result.partitions_searched}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    for hit in result.hits:
        print(f"  {hit.score:.3f} [{hit.partition}] {hit.document.get('name')}")

    result = await svr.search(
        query="comfortable office desk for remote work",
        partitions="all",
        limit=3,
    )
    print(f"\nQuery: 'comfortable office desk for remote work'")
    print(f"  Partitions searched: {result.partitions_searched}")
    for hit in result.hits:
        print(f"  {hit.score:.3f} [{hit.partition}] {hit.document.get('name')}")

    # Step 5: Explicit partition still works (bypasses centroid routing)
    result = await svr.search(
        query="headphones",
        partitions=["furniture"],  # Explicit = no centroid routing
        limit=3,
    )
    print(f"\nQuery: 'headphones' (explicit furniture partition)")
    print(f"  Partitions searched: {result.partitions_searched}")
    for hit in result.hits:
        print(f"  {hit.score:.3f} [{hit.partition}] {hit.document.get('name')}")

    await svr.disconnect()
    print("\nDone.")


asyncio.run(main())
