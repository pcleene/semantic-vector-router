"""Quickstart: search a collection in 5 lines.

Prerequisites:
  1. pip install semantic-vector-router[openai]
  2. Set in .env or environment:
     MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-host>/<db>
     OPENAI_API_KEY=sk-...

That's it. No config file needed.
"""
import asyncio

from semantic_vector_router import SVRClient


async def main():
    svr = await SVRClient.quickstart(
        database="my_store",
        collection="products",
        partition_field="category",
    )

    results = await svr.search("wireless noise-cancelling headphones", limit=5)

    for hit in results.hits:
        print(f"  {hit.score:.3f} — {hit.document.get('title', 'N/A')}")

    await svr.disconnect()


asyncio.run(main())
