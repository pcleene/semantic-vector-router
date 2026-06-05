"""Partition data models."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from semantic_vector_router.models.enums import IndexLocation, PartitionStatus


class PartitionInfo(BaseModel):
    """Information about a single partition."""

    name: str
    view_name: Optional[str] = None
    index_name: str = ""
    filter_value: Any = None
    filter_expression: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    document_count: int = 0
    status: PartitionStatus = PartitionStatus.ACTIVE
    parent_partition: Optional[str] = None
    child_partitions: list[str] = Field(default_factory=list)
    last_count_update: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    search_collection: Optional[str] = None
    index_location: IndexLocation = IndexLocation.VIEWS
    embedding_field: Optional[str] = None
    centroid: Optional[list[float]] = None
    centroid_updated_at: Optional[datetime] = None

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_datetime(cls, v: Any) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class PartitionsRegistry(BaseModel):
    """Registry of all partitions."""

    registry: dict[str, PartitionInfo] = Field(default_factory=dict)
