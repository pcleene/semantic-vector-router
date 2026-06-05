"""Cross-partition search with reranking.

When searching multiple partitions, SVR fans out queries in parallel,
collects candidates from each partition, merges them by normalized score,
and optionally applies a cross-encoder reranker for final relevance ordering.

Reranking is especially valuable for cross-partition queries because
score distributions differ between indexes -- a 0.92 in electronics
isn't directly comparable to a 0.88 in furniture. The reranker produces
a unified relevance ranking across all partitions.

Prerequisites:
  1. pip install semantic-vector-router[voyage]
  2. svr init (with reranking enabled)
  3. Set MONGODB_URI and VOYAGE_API_KEY in .env
"""
import asyncio

from semantic_vector_router import SVRClient


async def main():
    svr = SVRClient(auto_connect=False)
    await svr.connect()

    # Search across multiple partitions without reranking
    results_no_rerank = await svr.search(
        query="ergonomic setup for remote work",
        partitions=["electronics", "furniture", "accessories"],
        limit=10,
        rerank=False,
    )

    print("Without reranking (merged by normalized score):")
    for hit in results_no_rerank.hits:
        partition = hit.document.get("category", "?")
        title = hit.document.get("title", "N/A")
        print(f"  {hit.score:.3f} [{partition}] {title}")

    print(f"\n  Partitions searched: {results_no_rerank.partitions_searched}")
    print(f"  Total candidates: {results_no_rerank.total_candidates}")

    # Same query with reranking for unified relevance
    results_reranked = await svr.search(
        query="ergonomic setup for remote work",
        partitions=["electronics", "furniture", "accessories"],
        limit=10,
        rerank=True,
    )

    print("\nWith reranking (cross-encoder unified scoring):")
    for hit in results_reranked.hits:
        partition = hit.document.get("category", "?")
        title = hit.document.get("title", "N/A")
        print(f"  {hit.score:.3f} [{partition}] {title}")

    print(f"\n  Reranked: {results_reranked.reranked}")
    print(f"  Latency: {results_reranked.latency_ms:.0f}ms")

    await svr.disconnect()


asyncio.run(main())
