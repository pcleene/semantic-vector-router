"""Backward-compatible re-exports for all model classes."""

# Backend models (universal)
# Postgres config models
from semantic_vector_router.backends.postgres.config import (  # noqa: F401
    HnswConfig,
    IvfflatConfig,
    PgDistanceMetric,
    PgIndexType,
    PostgresBackendConfig,
)
from semantic_vector_router.models.backend import (  # noqa: F401
    IndexStatus,
    PartitionStorageResult,
)

# Enums
# Config models
from semantic_vector_router.models.config import (  # noqa: F401
    AutoSplitConfig,
    CacheConfig,
    CentroidRoutingConfig,
    DatabaseConfig,
    DetectionConfig,
    EmbeddingConfig,
    IngestConfig,
    LifecycleConfig,
    LogConfig,
    MaintenanceWindow,
    MetadataConfig,
    MetricsConfig,
    PartitioningConfig,
    ProviderRateLimit,
    RateLimitConfig,
    RepartitionConfig,
    RerankingConfig,
    ResilienceConfig,
    RoutingConfig,
    SplitScheduleConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)
from semantic_vector_router.models.enums import (  # noqa: F401
    BackendType,
    DetectionSignal,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestMode,
    MongoDBIndexQuantization,
    PartitionStatus,
    PartitionStrategy,
    RerankerProvider,
    RoutingMode,
    SimilarityMetric,
    SplitStrategy,
    VectorStorageFormat,
    VectorStorageMode,
    VoyageQuantization,
)

# Partition models
from semantic_vector_router.models.partition import (  # noqa: F401
    PartitionInfo,
    PartitionsRegistry,
)

# Result/status models
from semantic_vector_router.models.results import (  # noqa: F401
    IngestProgress,
    IngestResult,
    PartitionHealthStatus,
    WatcherStatus,
)

# Scheduler/events models
from semantic_vector_router.models.scheduler import (  # noqa: F401
    EventsConfig,
    SchedulerConfig,
    WebhookConfig,
)

# Search models
from semantic_vector_router.models.search import (  # noqa: F401
    SearchHit,
    SearchResult,
)

# SVRConfig (top-level)
from semantic_vector_router.models.svr_config import SVRConfig  # noqa: F401

__all__ = [
    # Backend models
    "IndexStatus",
    "PartitionStorageResult",
    # Postgres config
    "HnswConfig",
    "IvfflatConfig",
    "PgDistanceMetric",
    "PgIndexType",
    "PostgresBackendConfig",
    # Enums
    "BackendType",
    "DetectionSignal",
    "EmbeddingMode",
    "EmbeddingProvider",
    "IndexLocation",
    "IngestMode",
    "MongoDBIndexQuantization",
    "PartitionStatus",
    "PartitionStrategy",
    "RerankerProvider",
    "RoutingMode",
    "SimilarityMetric",
    "SplitStrategy",
    "VectorStorageFormat",
    "VectorStorageMode",
    "VoyageQuantization",
    # Config models
    "AutoSplitConfig",
    "CacheConfig",
    "CentroidRoutingConfig",
    "DatabaseConfig",
    "DetectionConfig",
    "EmbeddingConfig",
    "IngestConfig",
    "LifecycleConfig",
    "LogConfig",
    "MaintenanceWindow",
    "MetadataConfig",
    "MetricsConfig",
    "PartitioningConfig",
    "ProviderRateLimit",
    "RateLimitConfig",
    "RepartitionConfig",
    "RerankingConfig",
    "ResilienceConfig",
    "RoutingConfig",
    "SplitScheduleConfig",
    "SVRConfig",
    "VectorSearchConfig",
    "VectorStorageConfig",
    # Partition models
    "PartitionInfo",
    "PartitionsRegistry",
    # Search models
    "SearchHit",
    "SearchResult",
    # Result/status models
    "IngestProgress",
    "IngestResult",
    "PartitionHealthStatus",
    "WatcherStatus",
    # Scheduler/events models
    "EventsConfig",
    "SchedulerConfig",
    "WebhookConfig",
]
