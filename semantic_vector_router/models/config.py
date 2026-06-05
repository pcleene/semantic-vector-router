"""Configuration Pydantic models."""

from typing import Optional

from pydantic import BaseModel, Field

from semantic_vector_router.models.enums import (
    BackendType,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestMode,
    MongoDBIndexQuantization,
    PartitionStrategy,
    RerankerProvider,
    RoutingMode,
    SimilarityMetric,
    SplitStrategy,
    VectorStorageFormat,
    VectorStorageMode,
    VoyageQuantization,
)


class DatabaseConfig(BaseModel):
    """Database configuration."""

    backend: BackendType = BackendType.MONGODB
    connection_string_env: str = "MONGODB_URI"
    database: str
    source_collection: str
    max_pool_size: int = 100
    min_pool_size: int = 0
    max_idle_time_ms: int = 0
    wait_queue_timeout_ms: int = 0


class PartitioningConfig(BaseModel):
    """Partitioning configuration."""

    field: str
    fields: Optional[list[str]] = None
    strategy: PartitionStrategy = PartitionStrategy.EXACT_MATCH
    separator: str = "_"
    view_prefix: str = "svr_partition_"
    index_name_prefix: str = "svr_vector_idx_"


class VectorStorageConfig(BaseModel):
    """Vector storage configuration."""

    mode: VectorStorageMode = VectorStorageMode.EMBEDDED
    embeddings_collection: Optional[str] = None
    reference_field: str = "source_id"
    index_on: IndexLocation = IndexLocation.VIEWS
    storage_format: VectorStorageFormat = VectorStorageFormat.BINDATA_FLOAT32
    index_quantization: MongoDBIndexQuantization = MongoDBIndexQuantization.NONE


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""

    embedding_field: str = "embedding"
    index_type: str = "vectorSearch"
    dimensions: int = 1536
    similarity: SimilarityMetric = SimilarityMetric.COSINE
    num_candidates_multiplier: int = 10


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    mode: EmbeddingMode = EmbeddingMode.BYOM
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model: str = "text-embedding-3-small"
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    dimensions: int = 1536
    batch_size: int = 100
    source_fields: Optional[list[str]] = None
    separator: str = " | "
    template: Optional[str] = None
    computed_field: str = "embedding_text"
    input_type: Optional[str] = None
    device: Optional[str] = "cpu"
    local: bool = False
    document_model: Optional[str] = None
    voyage_quantization: VoyageQuantization = VoyageQuantization.FLOAT
    voyage_output_dimension: Optional[int] = None

    @property
    def is_asymmetric(self) -> bool:
        """Check if using asymmetric embeddings."""
        return self.document_model is not None and self.document_model != self.model

    @property
    def query_model(self) -> str:
        """Get the model used for query embeddings."""
        return self.model

    @property
    def effective_document_model(self) -> str:
        """Get the model used for document embeddings."""
        return self.document_model or self.model


class CentroidRoutingConfig(BaseModel):
    """Configuration for centroid-based partition routing."""

    enabled: bool = False
    relative_threshold: float = 0.5
    min_score: float = 0.15
    max_probe_partitions: int = 5
    sample_size: int = 500
    centroid_ttl_seconds: int = 3600
    registry_ttl_seconds: int = 60


class RoutingConfig(BaseModel):
    """Routing configuration."""

    mode: RoutingMode = RoutingMode.EXPLICIT
    default_partitions: str = "all"
    max_partitions_per_query: int = 5
    centroid_routing: CentroidRoutingConfig = Field(
        default_factory=CentroidRoutingConfig
    )


class RerankingConfig(BaseModel):
    """Reranking configuration."""

    enabled: bool = True
    provider: RerankerProvider = RerankerProvider.VOYAGE
    model: str = "rerank-2"
    api_key_env: Optional[str] = "VOYAGE_API_KEY"
    top_k_per_partition: int = 20
    final_top_k: int = 10


class SplitScheduleConfig(BaseModel):
    """Auto-split schedule configuration.

    Also serves as the base for MaintenanceWindow in the scheduler.
    """

    allowed_days: list[str] = Field(default_factory=lambda: ["saturday", "sunday"])
    allowed_hours: dict[str, int] = Field(default_factory=lambda: {"start": 2, "end": 6})
    timezone: str = "UTC"


# Alias for use by the scheduler
MaintenanceWindow = SplitScheduleConfig


class AutoSplitConfig(BaseModel):
    """Auto-split configuration."""

    enabled: bool = False
    threshold_vectors: int = 10_000_000
    threshold_check_interval: str = "daily"
    split_strategy: SplitStrategy = SplitStrategy.SECONDARY_FIELD
    secondary_field: Optional[str] = None
    num_shards: int = 4
    time_field: Optional[str] = None
    bucket: str = "yearly"
    fallback_strategy: SplitStrategy = SplitStrategy.ALERT_ONLY
    schedule: Optional[SplitScheduleConfig] = None


class ResilienceConfig(BaseModel):
    """Resilience and timeout configuration."""

    max_retry_attempts: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 30.0
    connection_timeout_ms: int = 10_000
    server_selection_timeout_ms: int = 30_000
    search_timeout_ms: int = 30_000
    embedding_timeout_ms: int = 60_000
    reranking_timeout_ms: int = 60_000
    health_check_interval_s: int = 30
    watcher_max_retries: int = 10
    watcher_base_delay: float = 1.0
    watcher_max_delay: float = 60.0


class MetadataConfig(BaseModel):
    """Configuration for the metadata collection."""

    connection_string_env: Optional[str] = None
    database: Optional[str] = None
    collection: str = "svr_metadata"


class DetectionConfig(BaseModel):
    """Configuration for the detection pipeline."""

    enabled: bool = True
    interval: str = "1h"
    threshold_vectors: int = 10_000_000
    min_threshold_vectors: int = 1_000
    skew_ratio: float = 5.0
    trend_window_days: int = 30
    auto_split_on_breach: bool = False
    auto_schedule_approaching: bool = True


class RepartitionConfig(BaseModel):
    """Configuration for the repartition workflow engine."""

    schedule: Optional[SplitScheduleConfig] = None
    index_wait_timeout_s: int = 1800
    index_poll_interval_s: int = 10
    auto_cleanup_retired: bool = True


class LifecycleConfig(BaseModel):
    """Lifecycle management configuration."""

    auto_provision: bool = True
    confirmation_required: bool = False
    change_stream_enabled: bool = True
    catchall_partition: str = "_default"
    auto_split: Optional[AutoSplitConfig] = None
    pending_partitions: list[str] = Field(default_factory=list)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    repartition: RepartitionConfig = Field(default_factory=RepartitionConfig)


class LogConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    json_format: bool = False
    log_query_text: bool = False
    log_embeddings: bool = False


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    enabled: bool = True
    include_partition_tags: bool = True
    include_query_tags: bool = False


class CacheConfig(BaseModel):
    """Embedding cache configuration."""

    enabled: bool = True
    max_size: int = 10_000
    ttl_seconds: int = 3600


class IngestConfig(BaseModel):
    """Document ingestion configuration."""

    text_fields: list[str] = Field(default_factory=lambda: ["text"])
    separator: str = " "
    template: Optional[str] = None
    batch_size: int = 100
    write_batch_size: int = 500
    mode: IngestMode = IngestMode.INSERT
    continue_on_error: bool = True
    trigger_detection: bool = True


class ProviderRateLimit(BaseModel):
    """Rate limit configuration for a single provider."""

    tokens_per_second: float = 50.0
    burst: int = 100


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for external API calls."""

    enabled: bool = True
    providers: dict[str, ProviderRateLimit] = Field(default_factory=dict)
    default_tokens_per_second: float = 50.0
    default_burst: int = 100


