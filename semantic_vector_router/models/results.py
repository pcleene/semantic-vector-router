"""Result and status models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PartitionHealthStatus(BaseModel):
    """Health status for a partition."""

    partition: str
    vector_count: int
    threshold: int
    utilization: float
    status: str


class WatcherStatus(BaseModel):
    """Status of the partition watcher."""

    running: bool
    last_event: Optional[datetime] = None
    partitions_created: int = 0
    errors: list[str] = Field(default_factory=list)


class IngestProgress(BaseModel):
    """Progress state during ingestion."""

    phase: str = "embedding"
    embedded: int = 0
    written: int = 0
    failed: int = 0
    total: int = 0


class IngestResult(BaseModel):
    """Result of a document ingestion operation."""

    inserted: int = 0
    failed: int = 0
    errors: list[tuple[int, str]] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    embed_ms: float = 0.0
    write_ms: float = 0.0
