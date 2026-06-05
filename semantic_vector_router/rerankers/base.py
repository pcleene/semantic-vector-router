"""Abstract base class for reranking providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from semantic_vector_router.models import SearchHit

if TYPE_CHECKING:
    from semantic_vector_router.utils.rate_limiter import TokenBucketRateLimiter


class BaseReranker(ABC):
    """Abstract base class for reranking providers.

    Rerankers use cross-encoder models to re-score documents
    based on query-document relevance. This produces more accurate
    relevance scores than embedding similarity alone, especially
    when combining results from multiple partitions.
    """

    def __init__(self) -> None:
        self._rate_limiter: Optional[TokenBucketRateLimiter] = None

    def set_rate_limiter(self, limiter: TokenBucketRateLimiter) -> None:
        """Set the rate limiter for this reranker.

        Args:
            limiter: Token bucket rate limiter instance.
        """
        self._rate_limiter = limiter

    async def _acquire_rate_limit(self, tokens: int = 1) -> None:
        """Acquire rate limit tokens before making an API call."""
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(tokens)

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[float]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: If provided, only return scores for top_k documents.

        Returns:
            List of relevance scores (0-1, higher is more relevant).
            Scores are in the same order as input documents.

        Raises:
            RerankingError: If reranking fails.
        """
        pass

    @abstractmethod
    async def rerank_hits(
        self,
        query: str,
        hits: list[SearchHit],
        text_field: str = "text",
        fallback_fields: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[SearchHit]:
        """Rerank SearchHit objects.

        Args:
            query: The search query.
            hits: List of SearchHit objects to rerank.
            text_field: Primary field to extract text from document.
            fallback_fields: Fallback fields if text_field is missing.
            top_k: If provided, only return top_k hits.

        Returns:
            Reranked list of SearchHit objects with rerank_score set.

        Raises:
            RerankingError: If reranking fails.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the reranker model name."""
        pass

    @property
    def max_documents(self) -> int:
        """Return the maximum number of documents that can be reranked at once."""
        return 100

    def _extract_text(
        self,
        document: dict,
        text_field: str,
        fallback_fields: Optional[list[str]] = None,
    ) -> str:
        """Extract text from a document for reranking.

        Args:
            document: Document dictionary.
            text_field: Primary field to extract.
            fallback_fields: Fallback fields if primary is missing.

        Returns:
            Extracted text string.
        """
        # Try primary field
        text = document.get(text_field)
        if text:
            return str(text)

        # Try fallback fields
        if fallback_fields:
            for field in fallback_fields:
                text = document.get(field)
                if text:
                    return str(text)

        # Fall back to string representation
        return str(document)
