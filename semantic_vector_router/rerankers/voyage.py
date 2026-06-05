"""Voyage AI reranker implementation."""

from typing import Optional

import httpx

from semantic_vector_router.exceptions import RerankingError
from semantic_vector_router.models import SearchHit
from semantic_vector_router.rerankers.base import BaseReranker
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.retry import with_retry

logger = get_logger(__name__)


class VoyageReranker(BaseReranker):
    """Voyage AI reranker (rerank-2, rerank-2-lite).

    Uses the Voyage AI reranking API to compute cross-encoder
    relevance scores for query-document pairs.
    """

    # Max documents per request
    MAX_DOCUMENTS = 1000

    def __init__(
        self,
        model: str = "rerank-2",
        api_key: Optional[str] = None,
        base_url: str = "https://api.voyageai.com/v1",
        timeout: float = 60.0,
        truncation: bool = True,
    ):
        """Initialize Voyage AI reranker.

        Args:
            model: Model name ("rerank-2", "rerank-2-lite").
            api_key: Voyage AI API key.
            base_url: Base URL for API calls.
            timeout: Request timeout in seconds.
            truncation: Whether to truncate long documents.
        """
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.truncation = truncation

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[float]:
        """Rerank documents by relevance to query. Retries on transient HTTP errors."""
        if not documents:
            return []

        if not self.api_key:
            raise RerankingError("Voyage AI API key not provided")

        @with_retry(
            max_attempts=3,
            base_delay=0.5,
            max_delay=30.0,
            retryable_exceptions=(httpx.TimeoutException, httpx.HTTPStatusError),
        )
        async def _do_rerank() -> list[float]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "truncation": self.truncation,
                }

                if top_k is not None:
                    payload["top_k"] = top_k

                response = await client.post(
                    f"{self.base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                results = data.get("data", [])

                scores = [0.0] * len(documents)
                for result in results:
                    idx = result.get("index", 0)
                    score = result.get("relevance_score", 0.0)
                    if idx < len(scores):
                        scores[idx] = score

                return scores

        try:
            return await _do_rerank()
        except httpx.HTTPStatusError as e:
            raise RerankingError(
                f"Voyage AI rerank error: {e.response.status_code}",
                details={"status": e.response.status_code, "error": e.response.text}
            )
        except httpx.HTTPError as e:
            raise RerankingError(f"HTTP error calling Voyage AI rerank: {e}")
        except KeyError as e:
            raise RerankingError(f"Unexpected response format from Voyage AI: {e}")

    async def rerank_hits(
        self,
        query: str,
        hits: list[SearchHit],
        text_field: str = "text",
        fallback_fields: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[SearchHit]:
        """Rerank SearchHit objects."""
        if not hits:
            return []

        # Set default fallback fields
        if fallback_fields is None:
            fallback_fields = ["description", "content", "body", "title"]

        # Extract text from each hit
        documents = [
            self._extract_text(hit.document, text_field, fallback_fields)
            for hit in hits
        ]

        # Get rerank scores
        scores = await self.rerank(query, documents, top_k=None)

        # Apply scores to hits
        for hit, score in zip(hits, scores):
            hit.rerank_score = score

        # Sort by rerank score
        hits.sort(key=lambda h: h.rerank_score or 0, reverse=True)

        # Limit if top_k specified
        if top_k is not None:
            hits = hits[:top_k]

        return hits

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def max_documents(self) -> int:
        """Return max documents per request."""
        return self.MAX_DOCUMENTS
