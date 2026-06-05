"""Universal backend models — shared across all backend implementations."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IndexStatus(str, Enum):
    """Universal index status across all backends."""

    READY = "ready"
    BUILDING = "building"
    ERROR = "error"
    NOT_FOUND = "not_found"


class PartitionStorageResult(BaseModel):
    """Result of creating partition storage.

    Each backend creates its own kind of partition-scoped storage:
    - MongoDB VIEWS mode: creates a MongoDB view
    - MongoDB FIELDS mode: registers a partition-specific embedding field
    - MongoDB SOURCE mode: uses source collection directly
    - Postgres (future): creates a filtered table or schema
    """

    storage_name: str
    storage_type: str  # "view", "table", "namespace", "field", "source"
    view_name: str | None = None
    search_collection: str | None = None
    embedding_field: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
