"""FIELDS mode: per-partition embedding fields with dedicated HNSW indexes.

In FIELDS mode, each partition gets its own embedding field on the source
collection (e.g., embedding_electronics, embedding_furniture) with a
separate vector search index. This gives each partition its own HNSW graph
without needing MongoDB views.

Tradeoffs:
  - Maximum query performance (no view pipeline overhead)
  - Limited to 50 partitions (Atlas 64-index cap minus headroom)
  - Requires partition-specific embedding fields in your documents

Prerequisites:
  1. pip install semantic-vector-router[voyage]
  2. Configure with index_on: "fields" in .svr/config.json
  3. Set MONGODB_URI and VOYAGE_API_KEY in .env
"""
import asyncio

from semantic_vector_router import SVRClient


async def main():
    # Configure FIELDS mode via dict (or use .svr/config.json)
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
            "index_on": "fields",  # Each partition gets its own embedding field + index
            "embedding_field": "embedding",
            "dimensions": 1024,
            "similarity": "cosine",
        },
        "embedding": {
            "mode": "byom",
            "provider": "voyage",
            "model": "voyage-4-lite",
            "api_key_env": "VOYAGE_API_KEY",
            "voyage_output_dimension": 1024,
        },
    }

    svr = SVRClient(config=config, auto_connect=False)
    await svr.connect()

    # Search in the electronics partition
    # In FIELDS mode, this queries the embedding_electronics field's index
    results = await svr.search(
        query="lightweight laptop with long battery life",
        partitions=["electronics"],
        limit=5,
    )

    print("Electronics results:")
    for hit in results.hits:
        print(f"  {hit.score:.3f} — {hit.document.get('title', 'N/A')}")

    await svr.disconnect()


asyncio.run(main())
