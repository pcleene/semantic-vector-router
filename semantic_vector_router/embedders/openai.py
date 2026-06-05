"""OpenAI embedding provider."""

from typing import Optional

import httpx

from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.exceptions import EmbeddingError
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.retry import with_retry

logger = get_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002).

    Supports the OpenAI embeddings API with configurable dimensions for
    text-embedding-3-* models.
    """

    # Default dimensions for each model
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Max batch sizes
    MAX_BATCH_SIZE = 2048

    # Max tokens per input
    MAX_TOKENS = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }

    def __init__(
        self,
        model: str,
        api_key: str,
        dimensions: Optional[int] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        """Initialize OpenAI embedder.

        Args:
            model: Model name (e.g., "text-embedding-3-small").
            api_key: OpenAI API key.
            dimensions: Override dimensions (only for text-embedding-3-* models).
            base_url: Base URL for API calls.
            timeout: Request timeout in seconds.
        """
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Set dimensions (custom or default)
        if dimensions is not None:
            self._dimensions = dimensions
        else:
            self._dimensions = self.DIMENSIONS.get(model, 1536)

        # Only text-embedding-3-* models support custom dimensions
        self._supports_custom_dimensions = model.startswith("text-embedding-3")

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a batch. Retries on transient HTTP errors."""
        if not texts:
            return []

        @with_retry(
            max_attempts=3,
            base_delay=0.5,
            max_delay=30.0,
            retryable_exceptions=(httpx.TimeoutException, httpx.HTTPStatusError),
        )
        async def _do_embed() -> list[list[float]]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload: dict = {
                    "model": self.model,
                    "input": texts,
                }

                if self._supports_custom_dimensions and self._dimensions:
                    payload["dimensions"] = self._dimensions

                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [e["embedding"] for e in embeddings]

        try:
            return await _do_embed()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"OpenAI API error: {e.response.status_code}",
                details={"status": e.response.status_code, "error": e.response.text}
            )
        except httpx.HTTPError as e:
            raise EmbeddingError(f"HTTP error calling OpenAI API: {e}")
        except KeyError as e:
            raise EmbeddingError(f"Unexpected response format from OpenAI: {e}")

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size."""
        return self.MAX_BATCH_SIZE

    @property
    def max_tokens(self) -> Optional[int]:
        """Return the maximum tokens per text."""
        return self.MAX_TOKENS.get(self.model)
