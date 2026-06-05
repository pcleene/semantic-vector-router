"""Custom exceptions for Semantic Vector Router."""

from typing import Any, Optional


class SVRException(Exception):
    """Base exception for Semantic Vector Router."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(SVRException):
    """Invalid or missing configuration."""

    pass


class PartitionNotFoundError(SVRException):
    """Requested partition does not exist."""

    pass


class PartitionProvisioningError(SVRException):
    """Failed to create partition view or index."""

    pass


class PartitionAlreadyExistsError(SVRException):
    """Partition already exists."""

    pass


class SearchError(SVRException):
    """Vector search query failed."""

    pass


class EmbeddingError(SVRException):
    """Embedding generation failed."""

    pass


class RerankingError(SVRException):
    """Reranking API call failed."""

    pass


class ConnectionError(SVRException):
    """Database connection failed."""

    pass


class ValidationError(SVRException):
    """Input validation failed."""

    pass


class APIKeyError(SVRException):
    """API key is missing or invalid."""

    pass


class IndexCreationError(SVRException):
    """Failed to create vector search index."""

    pass


class ViewCreationError(SVRException):
    """Failed to create MongoDB view."""

    pass


class ChangeStreamError(SVRException):
    """Change stream operation failed."""

    pass


class SplitError(SVRException):
    """Partition split operation failed."""

    pass


class MonitoringError(SVRException):
    """Monitoring operation failed."""

    pass


class ScanError(SVRException):
    """Partition scan operation failed."""

    pass


class MetadataError(SVRException):
    """Metadata store operation failed."""

    pass


class DetectionError(SVRException):
    """Detection pipeline operation failed."""

    pass


class RepartitionError(SVRException):
    """Repartition operation failed."""

    pass


class IngestionError(SVRException):
    """Document ingestion operation failed."""

    pass
