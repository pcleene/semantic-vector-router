"""Semantic Vector Router - Automatic vector index partitioning and query routing."""

from semantic_vector_router.client import SVRClient
from semantic_vector_router.config import load_config, save_config
from semantic_vector_router.exceptions import (
    APIKeyError,
    ConfigurationError,
    ConnectionError,
    EmbeddingError,
    IndexCreationError,
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
    PartitionProvisioningError,
    RerankingError,
    SearchError,
    SplitError,
    SVRException,
    ValidationError,
    ViewCreationError,
)
from semantic_vector_router.models import (
    PartitionInfo,
    SearchHit,
    SearchResult,
    SVRConfig,
)
from semantic_vector_router.presets import get_preset

__version__ = "0.1.0"

__all__ = [
    # Main client
    "SVRClient",
    # Configuration
    "SVRConfig",
    "load_config",
    "save_config",
    "get_preset",
    # Models
    "SearchHit",
    "SearchResult",
    "PartitionInfo",
    # Exceptions
    "SVRException",
    "ConfigurationError",
    "PartitionNotFoundError",
    "PartitionProvisioningError",
    "PartitionAlreadyExistsError",
    "SearchError",
    "EmbeddingError",
    "RerankingError",
    "ConnectionError",
    "ValidationError",
    "APIKeyError",
    "IndexCreationError",
    "ViewCreationError",
    "SplitError",
]
