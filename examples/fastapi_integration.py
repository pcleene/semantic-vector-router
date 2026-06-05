"""FastAPI integration: vector search as an API endpoint.

Demonstrates how to integrate SVR into a FastAPI application with:
  - Proper async lifecycle management (startup/shutdown)
  - Structured JSON logging for production
  - Search endpoint with query parameters

Run:
  uvicorn examples.fastapi_integration:app --reload

Test:
  curl "http://localhost:8000/search?q=wireless+headphones&partition=electronics&limit=5"
"""
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, Query

from semantic_vector_router import SVRClient
from semantic_vector_router.utils.logging import configure_logging

# Enable structured JSON logging for production
configure_logging(json_format=True, level="INFO")

svr: SVRClient = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage SVR client lifecycle with FastAPI."""
    global svr
    svr = SVRClient(auto_connect=False)
    await svr.connect()
    yield
    await svr.disconnect()


app = FastAPI(
    title="SVR Search API",
    description="Vector search powered by Semantic Vector Router",
    lifespan=lifespan,
)


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query text"),
    partition: str = Query("all", description="Partition name or 'all'"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    rerank: bool = Query(False, description="Apply cross-encoder reranking"),
):
    """Execute a vector search across partitions."""
    partitions: Union[list[str], str] = [partition] if partition != "all" else "all"
    results = await svr.search(
        query=q,
        partitions=partitions,
        limit=limit,
        rerank=rerank,
    )
    return {
        "query": q,
        "partition": partition,
        "total_candidates": results.total_candidates,
        "latency_ms": round(results.latency_ms, 1),
        "reranked": results.reranked,
        "hits": [
            {"score": round(h.score, 4), "document": h.document}
            for h in results.hits
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "connected": svr._connected if svr else False}
