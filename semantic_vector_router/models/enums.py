"""Enum types for the Semantic Vector Router."""

from enum import Enum


class EmbeddingMode(str, Enum):
    """Embedding mode options."""

    AUTO = "auto"
    BYOM = "byom"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    VOYAGE = "voyage"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    ATLAS_VOYAGE = "atlas_voyage"


class RerankerProvider(str, Enum):
    """Supported reranker providers."""

    VOYAGE = "voyage"
    COHERE = "cohere"


class BackendType(str, Enum):
    """Supported database backends."""

    MONGODB = "mongodb"
    POSTGRES = "postgres"


class PartitionStrategy(str, Enum):
    """Partitioning strategy options."""

    EXACT_MATCH = "exact_match"
    COMPOSITE = "composite"


class VectorStorageMode(str, Enum):
    """Vector storage mode options."""

    EMBEDDED = "embedded"
    SEPARATE = "separate"
    VIEWS_ONLY = "views_only"


class IndexLocation(str, Enum):
    """Where to create vector search indexes.

    - SOURCE: Single index on source collection with partition field as filter.
    - VIEWS: Separate index per partition view.
    - FIELDS: Separate embedding field per partition on source collection.
    """

    SOURCE = "source"
    VIEWS = "views"
    FIELDS = "fields"


class SimilarityMetric(str, Enum):
    """Vector similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dotProduct"


class RoutingMode(str, Enum):
    """Query routing mode."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    AUTO = "auto"


class SplitStrategy(str, Enum):
    """Partition split strategies."""

    SECONDARY_FIELD = "secondary_field"
    HASH = "hash"
    TIME = "time"
    ALERT_ONLY = "alert_only"


class PartitionStatus(str, Enum):
    """Partition status."""

    ACTIVE = "active"
    PENDING_SPLIT = "pending_split"
    SPLIT = "split"
    SPLITTING = "splitting"
    MIGRATING = "migrating"
    RETIRED = "retired"
    DISABLED = "disabled"


class DetectionSignal(str, Enum):
    """Detection pipeline signal types."""

    THRESHOLD_BREACH = "threshold_breach"
    APPROACHING_THRESHOLD = "approaching_threshold"
    SEVERE_SKEW = "severe_skew"
    UNDERPOPULATED = "underpopulated"
    STALE = "stale"


class IngestMode(str, Enum):
    """Document ingestion mode."""

    INSERT = "insert"
    UPSERT = "upsert"


class VectorStorageFormat(str, Enum):
    """How vectors are stored in MongoDB."""

    ARRAY = "array"
    BINDATA_FLOAT32 = "bindata_float32"
    BINDATA_INT8 = "bindata_int8"
    BINDATA_PACKED_BIT = "bindata_packed_bit"


class MongoDBIndexQuantization(str, Enum):
    """MongoDB Atlas Vector Search index quantization mode."""

    NONE = "none"
    SCALAR = "scalar"
    BINARY = "binary"


class VoyageQuantization(str, Enum):
    """Voyage 4 quantization options."""

    FLOAT = "float"
    INT8 = "int8"
    UINT8 = "uint8"
    BINARY = "binary"
    UBINARY = "ubinary"
