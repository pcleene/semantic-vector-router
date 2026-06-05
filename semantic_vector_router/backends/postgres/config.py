"""PostgreSQL backend configuration models."""

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class PgIndexType(str, Enum):
    """PostgreSQL vector index algorithm."""

    HNSW = "hnsw"
    IVFFLAT = "ivfflat"


class PgDistanceMetric(str, Enum):
    """pgvector distance operator mapping."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "ip"


# Regex for valid SQL identifiers (schema, table prefix)
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class HnswConfig(BaseModel):
    """HNSW index tuning parameters."""

    m: int = Field(default=16, ge=2, le=100)
    ef_construction: int = Field(default=64, ge=10, le=2000)
    ef_search: int = Field(default=40, ge=10, le=1000)


class IvfflatConfig(BaseModel):
    """IVFFlat index tuning parameters."""

    lists: int = Field(default=100, ge=1, le=10000)
    probes: int = Field(default=10, ge=1, le=500)


class PostgresBackendConfig(BaseModel):
    """PostgreSQL-specific backend configuration.

    Used when ``database.backend == "postgres"``.
    """

    connection_string_env: str = "POSTGRES_URI"
    schema_name: str = Field(default="public", alias="schema")
    table_prefix: str = "svr_"
    index_type: PgIndexType = PgIndexType.HNSW
    distance_metric: PgDistanceMetric = PgDistanceMetric.COSINE
    pool_min_size: int = Field(default=5, ge=0, le=100)
    pool_max_size: int = Field(default=20, ge=1, le=500)
    hnsw: HnswConfig = Field(default_factory=HnswConfig)
    ivfflat: IvfflatConfig = Field(default_factory=IvfflatConfig)
    statement_timeout_ms: int = Field(default=30_000, ge=1000, le=600_000)
    vector_dimensions: Optional[int] = None

    model_config = {"populate_by_name": True}

    @field_validator("schema_name")
    @classmethod
    def validate_schema_name(cls, v: str) -> str:
        """Ensure schema name is a valid SQL identifier."""
        if not _IDENTIFIER_RE.match(v):
            raise ValueError(
                f"Invalid schema name '{v}': must match [a-zA-Z_][a-zA-Z0-9_]*"
            )
        return v

    @field_validator("table_prefix")
    @classmethod
    def validate_table_prefix(cls, v: str) -> str:
        """Ensure table prefix is a valid SQL identifier prefix."""
        if v and not _IDENTIFIER_RE.match(v.rstrip("_")):
            raise ValueError(
                f"Invalid table_prefix '{v}': must match [a-zA-Z_][a-zA-Z0-9_]*"
            )
        return v

    @field_validator("pool_max_size")
    @classmethod
    def validate_pool_sizes(cls, v: int, info: Any) -> int:
        """Ensure pool_max_size >= pool_min_size."""
        min_size = info.data.get("pool_min_size", 0)
        if v < min_size:
            raise ValueError(
                f"pool_max_size ({v}) must be >= pool_min_size ({min_size})"
            )
        return v
