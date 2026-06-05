"""HuggingFace sentence-transformers embedding provider."""

import asyncio
from typing import Any, Optional

from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.exceptions import EmbeddingError
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace sentence-transformers embeddings (runs locally).

    Loads and runs sentence-transformer models locally using the
    sentence-transformers library. Supports CPU and GPU inference.
    """

    # Common models and their dimensions
    DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "intfloat/e5-small-v2": 384,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
    }

    # Default batch size for local inference
    MAX_BATCH_SIZE = 32

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        """Initialize HuggingFace embedder.

        Args:
            model: Model name or path (e.g., "sentence-transformers/all-MiniLM-L6-v2").
            device: Device to run inference on ("cpu", "cuda", "mps").
            normalize_embeddings: Whether to normalize embeddings to unit length.
        """
        super().__init__()
        self.model_id = model
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self._model: Any = None
        self._dimensions: Optional[int] = None
        self._executor = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Install with: pip install semantic-vector-router[huggingface]"
            )

        logger.info(f"Loading model {self.model_id} on {self.device}")
        self._model = SentenceTransformer(self.model_id, device=self.device)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimensions: {self._dimensions}")

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous encoding (for use in thread pool)."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.tolist()

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a batch."""
        if not texts:
            return []

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                self._encode_sync,
                texts,
            )
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"HuggingFace embedding error: {e}")

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        # Try to get from known models
        if self.model_id in self.DIMENSIONS:
            return self.DIMENSIONS[self.model_id]

        # Load model to get dimensions
        self._load_model()
        return self._dimensions  # type: ignore

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model_id

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size."""
        return self.MAX_BATCH_SIZE

    def set_device(self, device: str) -> None:
        """Change the device (requires model reload).

        Args:
            device: New device ("cpu", "cuda", "mps").
        """
        if self._model is not None and self.device != device:
            logger.info(f"Changing device from {self.device} to {device}")
            self._model = None
            self.device = device
            self._load_model()
        else:
            self.device = device
