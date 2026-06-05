"""Atlas Search index lifecycle management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from semantic_vector_router.models import SVRConfig

if TYPE_CHECKING:
    from semantic_vector_router.backends.mongodb.backend import MongoDBBackend
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)

# Name for the single source collection index (used in SOURCE mode)
SOURCE_INDEX_NAME = "svr_vector_idx_source"

# Maximum partitions in FIELDS mode (Atlas allows 64 indexes per collection)
MAX_FIELDS_PARTITIONS = 50


class IndexManager:
    """Manages Atlas Search index creation, deletion, and verification."""

    def __init__(self, backend: MongoDBBackend, config: SVRConfig):
        self.backend = backend
        self.config = config
        self._source_index_created = False

    async def ensure_source_index(
        self,
        auto_detect_filters: bool = False,
        extra_filter_fields: Optional[list[str]] = None,
    ) -> str:
        """Ensure the source collection has a vector search index with filter fields.

        Used when index_on=IndexLocation.SOURCE. Creates a single index
        on the source collection with the partition field as a filter field.

        Args:
            auto_detect_filters: If True, analyze collection fields and add
                suitable ones as filter fields alongside the partition field.
            extra_filter_fields: Additional field names to include as filter fields.

        Returns:
            The index name.
        """
        if self._source_index_created:
            return SOURCE_INDEX_NAME

        source_collection = self.config.database.source_collection
        partition_field = self.config.partitioning.field

        # Check if index already exists
        index_exists = await self.backend.index_exists(
            source_collection, SOURCE_INDEX_NAME
        )

        if not index_exists:
            # Build filter fields list
            filter_fields = [partition_field]

            if extra_filter_fields:
                for f in extra_filter_fields:
                    if f not in filter_fields:
                        filter_fields.append(f)

            if auto_detect_filters:
                try:
                    from semantic_vector_router.utils.field_analyzer import (
                        analyze_fields,
                        get_recommended_filter_fields,
                    )
                    analyses = await analyze_fields(self.backend, self.config)
                    recommended = get_recommended_filter_fields(analyses)
                    for f in recommended:
                        if f not in filter_fields:
                            filter_fields.append(f)
                    if recommended:
                        logger.info(
                            f"Auto-detected filter fields: {recommended}"
                        )
                except Exception as e:
                    logger.warning(f"Filter field auto-detection failed: {e}")

            # Create index with filter fields
            await self.backend.create_vector_search_index(
                collection_name=source_collection,
                index_name=SOURCE_INDEX_NAME,
                embedding_field=self.config.vector_search.embedding_field,
                dimensions=self.config.vector_search.dimensions,
                similarity=self.config.vector_search.similarity.value,
                filter_fields=filter_fields,
                quantization=self.config.vector_storage.index_quantization,
            )
            logger.info(
                f"Created source index '{SOURCE_INDEX_NAME}' on '{source_collection}' "
                f"with filter fields {filter_fields}"
            )
        else:
            logger.info(f"Source index '{SOURCE_INDEX_NAME}' already exists")

        self._source_index_created = True
        return SOURCE_INDEX_NAME

    async def create_views_index(
        self,
        view_name: str,
        index_name: str,
    ) -> None:
        """Create a vector search index on a partition view (8.1+)."""
        await self.backend.create_vector_search_index(
            collection_name=view_name,
            index_name=index_name,
            embedding_field=self.config.vector_search.embedding_field,
            dimensions=self.config.vector_search.dimensions,
            similarity=self.config.vector_search.similarity.value,
            quantization=self.config.vector_storage.index_quantization,
        )

    async def create_fields_index(
        self,
        embedding_field: str,
        index_name: str,
    ) -> None:
        """Create a vector search index for a FIELDS mode partition."""
        await self.backend.create_vector_search_index(
            collection_name=self.config.database.source_collection,
            index_name=index_name,
            embedding_field=embedding_field,
            dimensions=self.config.vector_search.dimensions,
            similarity=self.config.vector_search.similarity.value,
            quantization=self.config.vector_storage.index_quantization,
        )

    async def delete_index(self, collection_name: str, index_name: str) -> None:
        """Delete a vector search index."""
        await self.backend.delete_vector_search_index(collection_name, index_name)

    async def verify_index(
        self, collection_name: str, index_name: str
    ) -> dict[str, Any]:
        """Verify an index exists and return its status."""
        return await self.backend.get_index_status(collection_name, index_name)
