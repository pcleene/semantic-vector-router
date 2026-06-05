"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from semantic_vector_router.utils.rate_limiter import TokenBucketRateLimiter


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers.

    Embedders are responsible for converting text into vector embeddings.
    They support both single text embedding and batch embedding for efficiency.
    """

    def __init__(self) -> None:
        self._rate_limiter: Optional[TokenBucketRateLimiter] = None

    def set_rate_limiter(self, limiter: TokenBucketRateLimiter) -> None:
        """Set the rate limiter for this embedder.

        Args:
            limiter: Token bucket rate limiter instance.
        """
        self._rate_limiter = limiter

    async def _acquire_rate_limit(self, tokens: int = 1) -> None:
        """Acquire rate limit tokens before making an API call.

        No-op if no rate limiter is configured.
        """
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(tokens)

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding fails.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding fails.
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions for this model.

        Returns:
            Number of dimensions in the embedding vector.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            Name of the embedding model.
        """
        pass

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size for this provider.

        Returns:
            Maximum number of texts that can be embedded in a single batch.
        """
        return 100

    @property
    def max_tokens(self) -> Optional[int]:
        """Return the maximum tokens per text for this provider.

        Returns:
            Maximum tokens, or None if not applicable.
        """
        return None

    async def embed_with_batching(
        self, texts: list[str], batch_size: Optional[int] = None
    ) -> list[list[float]]:
        """Embed texts with automatic batching.

        Args:
            texts: List of texts to embed.
            batch_size: Optional batch size override.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        batch_size = batch_size or self.max_batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await self.embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings
