"""Quickstart (sync): zero async boilerplate.

Prerequisites:
  1. pip install semantic-vector-router[openai]
  2. Set MONGODB_URI and OPENAI_API_KEY in .env
"""
from semantic_vector_router import SVRClient

svr = SVRClient.quickstart_sync(
    database="my_store",
    collection="products",
    partition_field="category",
)

results = svr.search_sync("wireless noise-cancelling headphones", limit=5)

for hit in results.hits:
    print(f"  {hit.score:.3f} — {hit.document.get('title', 'N/A')}")
