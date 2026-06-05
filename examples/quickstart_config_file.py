"""Quickstart: search a partitioned collection in 10 lines.

Prerequisites:
  1. pip install semantic-vector-router[voyage]
  2. svr init   (or create .svr/config.json manually)
  3. Set MONGODB_URI and VOYAGE_API_KEY in .env
"""
import asyncio

from semantic_vector_router import SVRClient


async def main():
    svr = SVRClient(auto_connect=False)
    await svr.connect()

    results = await svr.search(
        query="wireless noise-cancelling headphones",
        partitions=["electronics"],
        limit=5,
    )

    for hit in results.hits:
        print(f"  {hit.score:.3f} — {hit.document.get('title', 'N/A')}")

    await svr.disconnect()


asyncio.run(main())
