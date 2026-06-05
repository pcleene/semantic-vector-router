"""MongoDB view lifecycle management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from semantic_vector_router.models import SVRConfig
from semantic_vector_router.utils.logging import get_logger

if TYPE_CHECKING:
    from semantic_vector_router.backends.mongodb.backend import MongoDBBackend

logger = get_logger(__name__)


class ViewManager:
    """Manages MongoDB view creation and deletion for partition isolation."""

    def __init__(self, backend: MongoDBBackend, config: SVRConfig):
        self.backend = backend
        self.config = config

    async def create_partition_view(
        self,
        partition_name: str,
        filter_value: Any,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a MongoDB view for a partition.

        Args:
            partition_name: Name of the partition.
            filter_value: Value to filter on.
            filter_expression: Custom filter expression.

        Returns:
            The view name.
        """
        return await self.backend.create_partition_view(
            partition_name=partition_name,
            filter_value=filter_value,
            filter_expression=filter_expression,
        )

    async def delete_view(self, view_name: str) -> None:
        """Delete a partition view."""
        await self.backend.delete_partition_view(view_name)

    async def view_exists(self, view_name: str) -> bool:
        """Check if a view exists."""
        return await self.backend.view_exists(view_name)
