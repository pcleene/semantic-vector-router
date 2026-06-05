"""SVRConfig — the top-level configuration model."""

from typing import Optional

from pydantic import BaseModel, Field

from semantic_vector_router.backends.postgres.config import PostgresBackendConfig
from semantic_vector_router.models.config import (
    CacheConfig,
    DatabaseConfig,
    EmbeddingConfig,
    IngestConfig,
    LifecycleConfig,
    LogConfig,
    MetricsConfig,
    PartitioningConfig,
    RateLimitConfig,
    RerankingConfig,
    ResilienceConfig,
    RoutingConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)
from semantic_vector_router.models.partition import PartitionsRegistry
from semantic_vector_router.models.scheduler import (
    EventsConfig,
    SchedulerConfig,
)


class SVRConfig(BaseModel):
    """Main configuration for Semantic Vector Router."""

    version: str = "1.0"
    database: DatabaseConfig
    partitioning: PartitioningConfig
    vector_storage: VectorStorageConfig = Field(default_factory=VectorStorageConfig)
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    partitions: PartitionsRegistry = Field(default_factory=PartitionsRegistry)
    logging: LogConfig = Field(default_factory=LogConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ingestion: IngestConfig = Field(default_factory=IngestConfig)
    rate_limiting: RateLimitConfig = Field(default_factory=RateLimitConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    postgres: Optional[PostgresBackendConfig] = None
