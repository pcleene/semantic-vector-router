"""Cohere embedding provider."""

from typing import Literal

import httpx

from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.exceptions import EmbeddingError
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.retry import with_retry

logger = get_logger(__name__)


class CohereEmbedder(BaseEmbedder):
    """Cohere embeddings (embed-english-v3.0, embed-multilingual-v3.0, etc.).

    Supports the Cohere embeddings API with input type specification
    for optimized query vs document embeddings.
    """

    # Default dimensions for each model
    DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v2.0": 768,
    }

    # Max batch sizes
    MAX_BATCH_SIZE = 96

    def __init__(
        self,
        model: str,
        api_key: str,
        input_type: Literal[
            "search_query", "search_document", "classification", "clustering"
        ] = "search_query",
        truncate: Literal["NONE", "START", "END"] = "END",
        base_url: str = "https://api.cohere.ai/v1",
        timeout: float = 60.0,
    ):
        """Initialize Cohere embedder.

        Args:
            model: Model name (e.g., "embed-english-v3.0").
            api_key: Cohere API key.
            input_type: Type of input text. Use "search_query" for queries
                        and "search_document" for documents being indexed.
            truncate: How to handle text that exceeds max tokens.
            base_url: Base URL for API calls.
            timeout: Request timeout in seconds.
        """
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.input_type = input_type
        self.truncate = truncate
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._dimensions = self.DIMENSIONS.get(model, 1024)

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
                response = await client.post(
                    f"{self.base_url}/embed",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "texts": texts,
                        "input_type": self.input_type,
                        "truncate": self.truncate,
                    },
                )
                response.raise_for_status()

                data = response.json()
                return data["embeddings"]

        try:
            return await _do_embed()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"Cohere API error: {e.response.status_code}",
                details={"status": e.response.status_code, "error": e.response.text}
            )
        except httpx.HTTPError as e:
            raise EmbeddingError(f"HTTP error calling Cohere API: {e}")
        except KeyError as e:
            raise EmbeddingError(f"Unexpected response format from Cohere: {e}")

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

    def for_documents(self) -> "CohereEmbedder":
        """Return a copy configured for document embedding.

        Returns:
            New CohereEmbedder with input_type="search_document".
        """
        return CohereEmbedder(
            model=self.model,
            api_key=self.api_key,
            input_type="search_document",
            truncate=self.truncate,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def for_queries(self) -> "CohereEmbedder":
        """Return a copy configured for query embedding.

        Returns:
            New CohereEmbedder with input_type="search_query".
        """
        return CohereEmbedder(
            model=self.model,
            api_key=self.api_key,
            input_type="search_query",
            truncate=self.truncate,
            base_url=self.base_url,
            timeout=self.timeout,
        )
