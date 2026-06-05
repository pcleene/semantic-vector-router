"""Voyage AI embedding provider.

Supports both Voyage 3.x and Voyage 4.x model families.

Voyage 4 introduces:
- Shared embedding space across all models (voyage-4-large, voyage-4, voyage-4-lite, voyage-4-nano)
- Asymmetric embeddings: use different models for queries vs documents
- Flexible dimensions via Matryoshka learning (256, 512, 1024, 2048)
- Quantization options (float, int8, uint8, binary, ubinary)
"""

from typing import Literal, Optional, Union

import httpx

from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.exceptions import EmbeddingError
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.retry import with_retry

logger = get_logger(__name__)

# Type aliases for Voyage 4 options
OutputDimension = Literal[256, 512, 1024, 2048]
OutputDtype = Literal["float", "int8", "uint8", "binary", "ubinary"]
InputType = Literal["query", "document", None]


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI embeddings supporting both Voyage 3.x and 4.x model families.

    Voyage 4 Features:
    - **Shared Embedding Space**: All Voyage 4 models produce compatible embeddings.
      You can embed documents with voyage-4-large and queries with voyage-4-lite,
      and they'll work together without re-indexing.
    - **Asymmetric Embeddings**: Optimize for accuracy (documents) vs latency (queries)
    - **Flexible Dimensions**: Matryoshka learning enables 256, 512, 1024, or 2048 dims
    - **Quantization**: int8, uint8, binary, ubinary for reduced storage

    Example - Asymmetric Embeddings:
        >>> # For documents (one-time indexing) - use the best model
        >>> doc_embedder = VoyageEmbedder(
        ...     model="voyage-4-large",
        ...     api_key=key,
        ...     input_type="document",
        ... )
        >>> doc_vector = await doc_embedder.embed("Product description...")
        >>>
        >>> # For queries (real-time) - use faster model, same embedding space!
        >>> query_embedder = VoyageEmbedder(
        ...     model="voyage-4-lite",
        ...     api_key=key,
        ...     input_type="query",
        ... )
        >>> query_vector = await query_embedder.embed("wireless headphones")
    """

    # Default dimensions for each model
    DIMENSIONS = {
        # Voyage 4.x - all support 256, 512, 1024 (default), 2048
        "voyage-4-large": 1024,
        "voyage-4": 1024,
        "voyage-4-lite": 1024,
        "voyage-4-nano": 1024,
        # Voyage 3.x (legacy)
        "voyage-3-large": 1024,
        "voyage-3-lite": 512,
        "voyage-3": 1024,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-multilingual-2": 1024,
    }

    # Token limits per request by model
    TOKEN_LIMITS = {
        "voyage-4-large": 120_000,
        "voyage-4": 320_000,
        "voyage-4-lite": 1_000_000,
        "voyage-4-nano": 320_000,
    }

    # Voyage 4 models that support shared embedding space
    VOYAGE_4_MODELS = {"voyage-4-large", "voyage-4", "voyage-4-lite", "voyage-4-nano"}

    # Valid dimension options for Voyage 4
    VALID_DIMENSIONS = {256, 512, 1024, 2048}

    # Max batch sizes (in texts)
    MAX_BATCH_SIZE = 128

    def __init__(
        self,
        model: str,
        api_key: str,
        input_type: InputType = "query",
        output_dimension: Optional[OutputDimension] = None,
        output_dtype: OutputDtype = "float",
        base_url: str = "https://api.voyageai.com/v1",
        timeout: float = 60.0,
    ):
        """Initialize Voyage AI embedder.

        Args:
            model: Model name. Voyage 4 models: "voyage-4-large", "voyage-4",
                   "voyage-4-lite", "voyage-4-nano". Legacy: "voyage-3-large", etc.
            api_key: Voyage AI API key.
            input_type: Type of input for retrieval optimization:
                - "query": For search queries (adds query-optimized prefix)
                - "document": For documents being indexed (adds doc prefix)
                - None: No prefix added
            output_dimension: Target embedding dimension (Voyage 4 only).
                Options: 256, 512, 1024 (default), 2048.
                Uses Matryoshka learning for dimension reduction.
            output_dtype: Output data type for quantization (Voyage 4 only):
                - "float": Full precision (default)
                - "int8"/"uint8": 8-bit integer quantization
                - "binary"/"ubinary": Binary quantization (1-bit)
            base_url: Base URL for API calls.
            timeout: Request timeout in seconds.
        """
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.input_type = input_type
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Validate Voyage 4-specific options
        if model not in self.VOYAGE_4_MODELS:
            if output_dimension is not None:
                logger.warning(
                    f"output_dimension is only supported for Voyage 4 models, "
                    f"ignoring for {model}"
                )
                self.output_dimension = None
            if output_dtype != "float":
                logger.warning(
                    f"output_dtype quantization is only supported for Voyage 4 models, "
                    f"using 'float' for {model}"
                )
                self.output_dtype = "float"

        # Validate dimension if specified
        if self.output_dimension is not None and self.output_dimension not in self.VALID_DIMENSIONS:
            raise EmbeddingError(
                f"Invalid output_dimension {self.output_dimension}. "
                f"Valid options: {sorted(self.VALID_DIMENSIONS)}"
            )

        # Set effective dimensions
        if self.output_dimension is not None:
            self._dimensions: int = self.output_dimension
        else:
            self._dimensions = self.DIMENSIONS.get(model, 1024)

    @property
    def is_voyage_4(self) -> bool:
        """Check if using a Voyage 4 model with shared embedding space."""
        return self.model in self.VOYAGE_4_MODELS

    async def embed(self, text: str) -> list[Union[float, int]]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector. Type depends on output_dtype:
            - float for "float"
            - int for "int8", "uint8"
            - int (0 or 1) for "binary", "ubinary"
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[Union[float, int]]]:
        """Embed multiple texts in a batch. Retries on transient HTTP errors."""
        if not texts:
            return []

        @with_retry(
            max_attempts=3,
            base_delay=0.5,
            max_delay=30.0,
            retryable_exceptions=(httpx.TimeoutException, httpx.HTTPStatusError),
        )
        async def _do_embed() -> list[list[Union[float, int]]]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "input": texts,
                }

                if self.input_type is not None:
                    payload["input_type"] = self.input_type

                if self.is_voyage_4:
                    if self.output_dimension is not None:
                        payload["output_dimension"] = self.output_dimension  # type: ignore[assignment]
                    if self.output_dtype != "float":
                        payload["output_dtype"] = self.output_dtype

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

                if "data" in data:
                    return [e["embedding"] for e in data["data"]]
                elif "embeddings" in data:
                    return data["embeddings"]
                else:
                    raise EmbeddingError(
                        "Unexpected response format from Voyage AI",
                        details={"response": data}
                    )

        try:
            return await _do_embed()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"Voyage AI API error: {e.response.status_code}",
                details={"status": e.response.status_code, "error": e.response.text}
            )
        except httpx.HTTPError as e:
            raise EmbeddingError(f"HTTP error calling Voyage AI API: {e}")
        except KeyError as e:
            raise EmbeddingError(f"Unexpected response format from Voyage AI: {e}")

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

    def for_documents(self, model: Optional[str] = None) -> "VoyageEmbedder":
        """Return a copy configured for document embedding.

        For Voyage 4 asymmetric embeddings, you can optionally specify a
        different model for documents (e.g., voyage-4-large for best accuracy).

        Args:
            model: Optional different model for documents. If None, uses same model.
                   For Voyage 4 shared embedding space, you can use voyage-4-large
                   for documents even if using voyage-4-lite for queries.

        Returns:
            New VoyageEmbedder configured for document embedding.

        Example:
            >>> query_embedder = VoyageEmbedder(model="voyage-4-lite", ...)
            >>> doc_embedder = query_embedder.for_documents(model="voyage-4-large")
        """
        target_model = model or self.model

        # Validate model compatibility for asymmetric embeddings
        if model is not None and self.is_voyage_4:
            if target_model not in self.VOYAGE_4_MODELS:
                raise EmbeddingError(
                    f"Cannot use {target_model} for documents when using Voyage 4 "
                    f"model {self.model} for queries. Use another Voyage 4 model."
                )

        return VoyageEmbedder(
            model=target_model,
            api_key=self.api_key,
            input_type="document",
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def for_queries(self, model: Optional[str] = None) -> "VoyageEmbedder":
        """Return a copy configured for query embedding.

        For Voyage 4 asymmetric embeddings, you can optionally specify a
        different (typically faster) model for queries.

        Args:
            model: Optional different model for queries. If None, uses same model.
                   For Voyage 4 shared embedding space, you can use voyage-4-lite
                   for queries even if documents were embedded with voyage-4-large.

        Returns:
            New VoyageEmbedder configured for query embedding.

        Example:
            >>> doc_embedder = VoyageEmbedder(model="voyage-4-large", ...)
            >>> query_embedder = doc_embedder.for_queries(model="voyage-4-lite")
        """
        target_model = model or self.model

        # Validate model compatibility for asymmetric embeddings
        if model is not None and self.is_voyage_4:
            if target_model not in self.VOYAGE_4_MODELS:
                raise EmbeddingError(
                    f"Cannot use {target_model} for queries when using Voyage 4 "
                    f"model {self.model} for documents. Use another Voyage 4 model."
                )

        return VoyageEmbedder(
            model=target_model,
            api_key=self.api_key,
            input_type="query",
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def with_dimensions(self, dimensions: OutputDimension) -> "VoyageEmbedder":
        """Return a copy with different output dimensions.

        Uses Matryoshka learning for dimension reduction (Voyage 4 only).

        Args:
            dimensions: Target dimensions (256, 512, 1024, or 2048).

        Returns:
            New VoyageEmbedder with specified dimensions.

        Raises:
            EmbeddingError: If model doesn't support dimension selection.
        """
        if not self.is_voyage_4:
            raise EmbeddingError(
                f"Dimension selection is only supported for Voyage 4 models, "
                f"not {self.model}"
            )

        return VoyageEmbedder(
            model=self.model,
            api_key=self.api_key,
            input_type=self.input_type,
            output_dimension=dimensions,
            output_dtype=self.output_dtype,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def with_quantization(self, dtype: OutputDtype) -> "VoyageEmbedder":
        """Return a copy with different quantization.

        Args:
            dtype: Output data type:
                - "float": Full precision (default)
                - "int8"/"uint8": 8-bit integer quantization
                - "binary"/"ubinary": Binary quantization

        Returns:
            New VoyageEmbedder with specified quantization.

        Raises:
            EmbeddingError: If model doesn't support quantization.
        """
        if not self.is_voyage_4:
            raise EmbeddingError(
                f"Quantization is only supported for Voyage 4 models, not {self.model}"
            )

        return VoyageEmbedder(
            model=self.model,
            api_key=self.api_key,
            input_type=self.input_type,
            output_dimension=self.output_dimension,
            output_dtype=dtype,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def create_asymmetric_pair(
        cls,
        api_key: str,
        query_model: str = "voyage-4-lite",
        document_model: str = "voyage-4-large",
        output_dimension: Optional[OutputDimension] = None,
        output_dtype: OutputDtype = "float",
        base_url: str = "https://api.voyageai.com/v1",
        timeout: float = 60.0,
    ) -> tuple["VoyageEmbedder", "VoyageEmbedder"]:
        """Create a pair of embedders for asymmetric retrieval.

        Creates optimized embedder configurations for Voyage 4's shared
        embedding space: a high-accuracy model for documents (indexed once)
        and a fast model for queries (real-time).

        Args:
            api_key: Voyage AI API key.
            query_model: Model for queries (default: voyage-4-lite for speed).
            document_model: Model for documents (default: voyage-4-large for accuracy).
            output_dimension: Shared dimension for both embedders.
            output_dtype: Shared quantization for both embedders.
            base_url: Base URL for API calls.
            timeout: Request timeout in seconds.

        Returns:
            Tuple of (query_embedder, document_embedder).

        Example:
            >>> query_embedder, doc_embedder = VoyageEmbedder.create_asymmetric_pair(
            ...     api_key=api_key,
            ...     query_model="voyage-4-lite",      # Fast for real-time queries
            ...     document_model="voyage-4-large",  # Accurate for indexing
            ... )
            >>> # Index documents once with high accuracy
            >>> doc_vectors = await doc_embedder.embed_batch(documents)
            >>> # Query with low latency
            >>> query_vector = await query_embedder.embed("search query")
        """
        # Validate both models are Voyage 4
        if query_model not in cls.VOYAGE_4_MODELS:
            raise EmbeddingError(
                f"Asymmetric embeddings require Voyage 4 models. "
                f"'{query_model}' is not a Voyage 4 model."
            )
        if document_model not in cls.VOYAGE_4_MODELS:
            raise EmbeddingError(
                f"Asymmetric embeddings require Voyage 4 models. "
                f"'{document_model}' is not a Voyage 4 model."
            )

        query_embedder = cls(
            model=query_model,
            api_key=api_key,
            input_type="query",
            output_dimension=output_dimension,
            output_dtype=output_dtype,
            base_url=base_url,
            timeout=timeout,
        )

        document_embedder = cls(
            model=document_model,
            api_key=api_key,
            input_type="document",
            output_dimension=output_dimension,
            output_dtype=output_dtype,
            base_url=base_url,
            timeout=timeout,
        )

        return query_embedder, document_embedder
