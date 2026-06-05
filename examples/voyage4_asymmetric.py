"""Voyage 4 asymmetric embeddings: different models for docs vs queries.

Voyage 4's shared embedding space allows using different models for indexing
and querying without re-embedding your documents:

  - Documents: voyage-4-large (highest accuracy, MoE architecture, 40% lower cost)
  - Queries: voyage-4-lite (fastest, real-time latency)

Both models produce vectors in the same embedding space, so documents
embedded with voyage-4-large can be queried with voyage-4-lite -- no
re-indexing required.

Prerequisites:
  1. pip install semantic-vector-router[voyage]
  2. Set MONGODB_URI and VOYAGE_API_KEY in .env
"""
import asyncio

from semantic_vector_router import SVRClient


async def main():
    config = {
        "version": "1.0",
        "database": {
            "connection_string_env": "MONGODB_URI",
            "database": "knowledge_base",
            "source_collection": "articles",
        },
        "partitioning": {
            "field": "topic",
            "strategy": "exact_match",
        },
        "vector_storage": {
            "index_on": "views",
            "embedding_field": "embedding",
            "dimensions": 1024,
            "similarity": "cosine",
        },
        "embedding": {
            "mode": "byom",
            "provider": "voyage",
            "model": "voyage-4-lite",           # Fast model for queries
            "document_model": "voyage-4-large",  # Accurate model for indexing
            "api_key_env": "VOYAGE_API_KEY",
            "voyage_output_dimension": 1024,
        },
        "reranking": {
            "enabled": True,
            "provider": "voyage",
            "model": "rerank-2",
            "api_key_env": "VOYAGE_API_KEY",
        },
    }

    svr = SVRClient(config=config, auto_connect=False)
    await svr.connect()

    # Search uses voyage-4-lite (fast) for query embedding
    # Documents were indexed with voyage-4-large (accurate)
    # Both share the same embedding space -- no mismatch
    results = await svr.search(
        query="how to fine-tune large language models",
        partitions=["machine_learning"],
        limit=5,
    )

    print("Results (query: voyage-4-lite, docs: voyage-4-large):")
    for hit in results.hits:
        print(f"  {hit.score:.3f} — {hit.document.get('title', 'N/A')}")

    # Ingest new documents using voyage-4-large automatically
    new_docs = [
        {
            "title": "Transformer Architecture Explained",
            "topic": "machine_learning",
            "content": "The transformer architecture uses self-attention mechanisms...",
        },
    ]
    result = await svr.ingest(new_docs)
    print(f"\nIngested: {result.inserted} documents")

    await svr.disconnect()


asyncio.run(main())
