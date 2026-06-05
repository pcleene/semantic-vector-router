"""Ingest documents with automatic embedding and partition routing.

Shows the full write path: text extraction -> embedding -> BinData conversion ->
partition-aware field routing -> MongoDB bulk write.

The pipeline handles:
  - Text extraction from configurable fields (title + description by default)
  - Batch embedding via the configured provider
  - BinData conversion for efficient MongoDB storage
  - Field routing: FIELDS mode writes to embedding_{partition}, others to standard field
  - Bulk inserts with per-document error handling

Prerequisites:
  1. pip install semantic-vector-router[voyage]
  2. svr init
  3. Set MONGODB_URI and VOYAGE_API_KEY in .env
"""
import asyncio

from semantic_vector_router import SVRClient
from semantic_vector_router.models import IngestMode


async def main():
    svr = SVRClient(auto_connect=False)
    await svr.connect()

    documents = [
        {
            "title": "iPhone 15 Pro",
            "category": "electronics",
            "description": "Latest Apple smartphone with A17 Pro chip",
        },
        {
            "title": "Standing Desk",
            "category": "furniture",
            "description": "Adjustable height desk for home office",
        },
        {
            "title": "MacBook Air M3",
            "category": "electronics",
            "description": "Lightweight laptop with Apple silicon",
        },
        {
            "title": "Ergonomic Chair",
            "category": "furniture",
            "description": "Lumbar support mesh chair with adjustable armrests",
        },
    ]

    # Ingest with progress tracking
    def on_progress(p):
        print(
            f"  [{p.phase}] embedded={p.embedded}/{p.total} "
            f"written={p.written} failed={p.failed}"
        )

    result = await svr.ingest(
        documents=documents,
        mode=IngestMode.INSERT,
        progress_callback=on_progress,
    )

    print(f"\nInserted: {result.inserted}, Failed: {result.failed}")
    print(
        f"Timing: embed={result.embed_ms:.0f}ms, write={result.write_ms:.0f}ms, "
        f"total={result.elapsed_ms:.0f}ms"
    )

    # Verify via search
    search_results = await svr.search(
        query="Apple laptop",
        partitions=["electronics"],
        limit=3,
    )
    print("\nSearch results after ingestion:")
    for hit in search_results.hits:
        print(f"  {hit.score:.3f} — {hit.document.get('title')}")

    await svr.disconnect()


asyncio.run(main())
