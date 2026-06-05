"""Abstract base class for database backends.

Defines a capability-based, backend-agnostic interface for vector search.
Backend-specific capabilities (change streams, auto-embedding) are expressed
through Protocol mixins rather than boolean flags.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from semantic_vector_router.models.backend import IndexStatus, PartitionStorageResult
from semantic_vector_router.models.partition import PartitionInfo
from semantic_vector_router.models.svr_config import SVRConfig


class BaseBackend(ABC):
    """Abstract backend for vector search operations.

    All backends must implement these core operations.
    Backend-specific capabilities (change streams, auto-embedding)
    are expressed through Protocol mixins, not boolean flags.
    """

    def __init__(self, config: SVRConfig):
        """Initialize the backend.

        Args:
            config: SVR configuration.
        """
        self.config = config

    # ── Connection lifecycle ──────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to the database."""
        ...

    async def health_check(self) -> bool:
        """Default: delegate to is_connected(). Override for richer checks."""
        return await self.is_connected()

    # ── Partition storage ─────────────────────────────────────────
    # Backend-agnostic partition storage lifecycle.
    # MongoDB creates views or adds fields. Postgres creates tables.

    @abstractmethod
    async def create_partition_storage(
        self,
        partition: PartitionInfo,
        config: SVRConfig,
    ) -> PartitionStorageResult:
        """Create storage for a partition.

        What gets created depends on the backend:
        - MongoDB VIEWS: a MongoDB view
        - MongoDB FIELDS: registers embedding field (no-op storage)
        - MongoDB SOURCE: no-op (uses source collection)
        - Postgres: a filtered table

        Args:
            partition: Partition to create storage for.
            config: SVR configuration.

        Returns:
            Result describing what was created.
        """
        ...

    @abstractmethod
    async def delete_partition_storage(
        self,
        partition: PartitionInfo,
    ) -> None:
        """Delete storage for a partition."""
        ...

    @abstractmethod
    async def partition_storage_exists(
        self,
        partition: PartitionInfo,
    ) -> bool:
        """Check if partition storage exists."""
        ...

    # ── Index lifecycle ───────────────────────────────────────────

    @abstractmethod
    async def create_partition_index(
        self,
        partition: PartitionInfo,
        config: SVRConfig,
    ) -> str:
        """Create a vector search index for a partition.

        Each backend knows how to build its own index type.

        Args:
            partition: Partition to index.
            config: SVR configuration.

        Returns:
            The index name/identifier.
        """
        ...

    @abstractmethod
    async def delete_partition_index(
        self,
        partition: PartitionInfo,
    ) -> None:
        """Delete a partition's vector search index."""
        ...

    @abstractmethod
    async def get_partition_index_status(
        self,
        partition: PartitionInfo,
    ) -> IndexStatus:
        """Get the status of a partition's index.

        Returns:
            Universal IndexStatus enum value.
        """
        ...

    async def wait_for_index_ready(
        self,
        partition: PartitionInfo,
        timeout_s: float = 300.0,
        poll_interval_s: float = 5.0,
    ) -> bool:
        """Wait for index to become ready.

        Default implementation polls get_partition_index_status().
        Override for backends with synchronous index builds (e.g. Postgres).

        Args:
            partition: Partition whose index to wait for.
            timeout_s: Maximum wait time in seconds.
            poll_interval_s: Time between polls.

        Returns:
            True if index is ready, False if timeout or error.
        """
        elapsed = 0.0
        while elapsed < timeout_s:
            status = await self.get_partition_index_status(partition)
            if status == IndexStatus.READY:
                return True
            if status == IndexStatus.ERROR:
                return False
            await asyncio.sleep(poll_interval_s)
            elapsed += poll_interval_s
        return False

    # ── Search ────────────────────────────────────────────────────

    @abstractmethod
    async def execute_search(
        self,
        partition: PartitionInfo,
        query_vector: list[float],
        limit: int,
        num_candidates: int,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
        pre_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Execute vector search on a single partition.

        For auto-embedding backends, use the AutoEmbeddingCapable protocol's
        execute_search_with_query() method instead.

        Args:
            partition: Partition to search.
            query_vector: Query embedding vector (required).
            limit: Maximum results to return.
            num_candidates: Number of candidates for ANN search.
            filters: Additional filters to apply.
            exact: If True, use brute-force exact search (no ANN).
            post_native: Backend-specific post-processing.
                MongoDB: ``list[dict]`` — aggregation stages appended after
                ``$vectorSearch`` + ``$addFields``.
                PostgreSQL: ``str`` — raw SQL appended as CTE consumer.
                **Runs per-partition before merge.** Trusted application code
                only — never pass user-generated input.
            pre_native: Backend-specific pre-filter conditions.
                MongoDB: Ignored (``$vectorSearch`` doesn't accept arbitrary
                expressions in its filter).
                PostgreSQL: ``str`` — additional WHERE conditions AND-joined
                with translated filters. Trusted application code only.

        Returns:
            List of search results with _svr_score and _svr_partition.
        """
        ...

    @abstractmethod
    async def search_partitions(
        self,
        partitions: list[PartitionInfo],
        limit: int,
        query_vector: Optional[list[float]] = None,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
        pre_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Fan-out search across multiple partitions.

        Each backend implements this to dispatch execute_search() (or
        execute_search_with_query() for auto-embedding backends) across
        the given partitions and combine results.

        ``post_native`` and ``pre_native`` run **per-partition** before
        merging. Aggregations in ``post_native`` operate on per-partition
        results, not the merged global result set.

        Args:
            partitions: Partitions to search.
            limit: Maximum results per partition.
            query_vector: Query embedding vector.
            query: Query string (for auto-embedding backends).
            filters: Additional filters to apply.
            exact: If True, use brute-force exact search (no ANN).
            post_native: Backend-specific post-processing (per-partition).
            pre_native: Backend-specific pre-filter conditions (per-partition).

        Returns:
            Combined list of search results from all partitions.
        """
        ...

    # ── Document writes ──────────────────────────────────────────

    @abstractmethod
    async def insert_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """Insert documents into the backend storage.

        Args:
            documents: List of document dicts to insert. Each should have
                an ``_id`` key. Backend-specific fields (embedding, partition_name)
                should already be set by the caller.
            collection_name: Target collection/table. If None, uses the
                backend's default storage location.

        Returns:
            Number of documents successfully inserted.
        """
        ...

    # ── Data operations ───────────────────────────────────────────

    @abstractmethod
    async def get_distinct_values(
        self,
        field: str,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """Get distinct values for a field."""
        ...

    @abstractmethod
    async def count_documents(
        self,
        collection_name: Optional[str] = None,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> int:
        """Count documents in a collection/table."""
        ...

    @abstractmethod
    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get collection/table statistics."""
        ...

    @abstractmethod
    async def get_partition_document_counts(
        self,
        field: str,
    ) -> dict[str, int]:
        """Get document counts grouped by partition field."""
        ...

    # ── Filter translation ────────────────────────────────────────

    def translate_filters(
        self, svr_filters: Optional[dict[str, Any]]
    ) -> Any:
        """Translate SVR universal filter format to backend-native format.

        Override per backend. Default returns filters unchanged
        (MongoDB pass-through since SVR syntax IS MongoDB syntax).
        """
        return svr_filters


# ── Capability Protocols ──────────────────────────────────────────
# Use isinstance() checks instead of boolean flags.
# MongoDB implements both. Other backends implement as needed.


@runtime_checkable
class ChangeStreamCapable(Protocol):
    """Backend that supports real-time change stream monitoring."""

    def watch_collection(
        self,
        pipeline: Optional[list[dict[str, Any]]] = None,
    ) -> Any:
        """Watch for changes. Returns async context manager / iterator."""
        ...


@runtime_checkable
class AutoEmbeddingCapable(Protocol):
    """Backend that supports server-side auto-embedding (e.g., Atlas + Voyage)."""

    async def execute_search_with_query(
        self,
        partition: PartitionInfo,
        query: str,
        limit: int,
        num_candidates: int,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Execute search using server-side embedding of query string."""
        ...
