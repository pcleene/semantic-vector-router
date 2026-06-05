"""Backward-compatible re-export. Moved to backends/mongodb/index_manager.py."""

from semantic_vector_router.backends.mongodb.index_manager import (  # noqa: F401
    MAX_FIELDS_PARTITIONS,
    SOURCE_INDEX_NAME,
    IndexManager,
)
