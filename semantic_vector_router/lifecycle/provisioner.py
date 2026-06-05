"""Partition provisioner — orchestrates partition lifecycle.

Backend-agnostic: uses BaseBackend abstract operations for all storage
and index management. No direct dependency on IndexManager or ViewManager.
"""

from datetime import datetime
from typing import Any, Optional

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.backends.mongodb.index_manager import (
    MAX_FIELDS_PARTITIONS,
    SOURCE_INDEX_NAME,  # noqa: F401 — re-exported for backward compat
)
from semantic_vector_router.config import save_config
from semantic_vector_router.exceptions import (
    PartitionAlreadyExistsError,
    PartitionProvisioningError,
)
from semantic_vector_router.models import (
    IndexLocation,
    PartitionInfo,
    PartitionStatus,
    SVRConfig,
)
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class PartitionProvisioner:
    """Provisions partitions via abstract backend operations.

    Supports event emission when an EventBus is provided via set_event_bus().

    The provisioner is backend-agnostic: it calls
    ``backend.create_partition_storage()`` and
    ``backend.create_partition_index()`` — each backend decides *how* to
    implement those operations.

    Supports three index location modes (backend handles the details):

    1. **VIEWS mode** — per-partition view + index
    2. **SOURCE mode** — shared index on the source collection
    3. **FIELDS mode** — per-partition embedding field + index (max 50)
    """

    def __init__(
        self,
        backend: BaseBackend,
        config: SVRConfig,
        auto_save_config: bool = True,
    ):
        self.backend = backend
        self.config = config
        self.auto_save_config = auto_save_config
        self._event_bus: Any = None

    def set_event_bus(self, event_bus: Any) -> None:
        """Set the event bus for emitting partition lifecycle events."""
        self._event_bus = event_bus

    async def _emit_event(
        self, event_type_str: str, partition: str, **details: Any
    ) -> None:
        """Emit a partition lifecycle event."""
        if self._event_bus is None:
            return
        try:
            from semantic_vector_router.events.models import SVREvent, SVREventType
            event = SVREvent(
                event_type=SVREventType(event_type_str),
                partition=partition,
                details=details,
            )
            await self._event_bus.emit(event)
        except Exception as e:
            logger.warning(f"Failed to emit event: {e}")

    async def ensure_source_index(
        self,
        auto_detect_filters: bool = False,
        extra_filter_fields: Optional[list[str]] = None,
    ) -> str:
        """Ensure the source collection has a vector search index.

        Delegates to the backend's ensure_source_index() if available
        (e.g. MongoDBBackend). Non-MongoDB backends may not need this.

        Args:
            auto_detect_filters: If True, analyze collection fields for filters.
            extra_filter_fields: Additional field names to include as filter fields.

        Returns:
            The index name.
        """
        if hasattr(self.backend, "ensure_source_index"):
            result: str = await self.backend.ensure_source_index(
                auto_detect_filters=auto_detect_filters,
                extra_filter_fields=extra_filter_fields,
            )
            return result
        return ""

    async def create_partition(
        self,
        name: str,
        filter_value: Optional[Any] = None,
        filter_expression: Optional[dict[str, Any]] = None,
        skip_if_exists: bool = False,
        create_view: bool = True,
    ) -> PartitionInfo:
        """Create a new partition with storage and index.

        Delegates storage and index creation to the backend via abstract
        operations. The backend handles mode-specific details internally.

        Args:
            name: Partition name.
            filter_value: Value to filter on (defaults to name).
            filter_expression: Custom filter expression.
            skip_if_exists: If True, return existing partition instead of error.
            create_view: Whether to create view (legacy, ignored in abstract flow).

        Returns:
            Created or existing PartitionInfo.

        Raises:
            PartitionAlreadyExistsError: If partition exists and skip_if_exists is False.
            PartitionProvisioningError: If creation fails or FIELDS partition cap is reached.
        """
        # Check if partition already exists
        if name in self.config.partitions.registry:
            if skip_if_exists:
                logger.info(f"Partition '{name}' already exists, skipping")
                return self.config.partitions.registry[name]
            raise PartitionAlreadyExistsError(f"Partition '{name}' already exists")

        # Use name as filter value if not specified
        if filter_value is None and filter_expression is None:
            filter_value = name

        index_location = self.config.vector_storage.index_on
        source_collection = self.config.database.source_collection

        # FIELDS mode partition cap
        if index_location == IndexLocation.FIELDS:
            fields_count = sum(
                1 for p in self.config.partitions.registry.values()
                if p.index_location == IndexLocation.FIELDS
            )
            if fields_count >= MAX_FIELDS_PARTITIONS:
                raise PartitionProvisioningError(
                    f"FIELDS mode is limited to {MAX_FIELDS_PARTITIONS} partitions "
                    f"(Atlas 64-index limit). Current count: {fields_count}. "
                    f"Switch to VIEWS mode for additional partitions.",
                    details={"current": fields_count, "max": MAX_FIELDS_PARTITIONS},
                )

        # Build a preliminary PartitionInfo for backend operations
        preliminary = PartitionInfo(
            name=name,
            filter_value=filter_value,
            filter_expression=filter_expression,
            index_location=index_location,
        )

        # Track state for rollback
        storage_created = False
        index_created = False
        config_modified = False
        created_partition: Optional[PartitionInfo] = None

        try:
            # 1. Create storage (backend decides how based on mode)
            storage_result = await self.backend.create_partition_storage(
                preliminary, self.config
            )
            storage_created = True
            logger.info(
                f"Created {storage_result.storage_type} storage: "
                f"{storage_result.storage_name}"
            )

            # Update preliminary with storage result for index creation
            preliminary.view_name = storage_result.view_name
            preliminary.embedding_field = storage_result.embedding_field
            preliminary.search_collection = storage_result.search_collection

            # 2. Create index (backend decides how based on mode)
            index_name = await self.backend.create_partition_index(
                preliminary, self.config
            )
            index_created = True

            # 3. Get document count
            if index_location == IndexLocation.FIELDS and storage_result.embedding_field:
                doc_count = await self.backend.count_documents(
                    source_collection,
                    {storage_result.embedding_field: {"$exists": True}},
                )
            elif storage_result.storage_type == "view" and storage_result.view_name:
                doc_count = await self.backend.count_documents(
                    storage_result.view_name
                )
            else:
                partition_field = self.config.partitioning.field
                if filter_expression:
                    doc_count = await self.backend.count_documents(
                        source_collection, filter_expression
                    )
                else:
                    doc_count = await self.backend.count_documents(
                        source_collection, {partition_field: filter_value}
                    )

            # 4. Create final partition info
            partition = PartitionInfo(
                name=name,
                view_name=storage_result.view_name,
                index_name=index_name,
                filter_value=filter_value,
                filter_expression=filter_expression,
                created_at=datetime.utcnow(),
                document_count=doc_count,
                status=PartitionStatus.ACTIVE,
                search_collection=storage_result.search_collection,
                index_location=index_location,
                embedding_field=storage_result.embedding_field,
            )
            created_partition = partition

            # 5. Register partition
            self.config.partitions.registry[name] = partition
            config_modified = True

            # Save config if auto-save enabled
            if self.auto_save_config:
                save_config(self.config)

            logger.info(
                f"Created partition '{name}' ({index_location.value.upper()} mode) "
                f"with {doc_count:,} documents"
            )

            await self._emit_event(
                "partition.created",
                name,
                index_location=index_location.value,
                document_count=doc_count,
            )

            return partition

        except Exception as e:
            logger.warning(f"Partition creation failed for '{name}', rolling back: {e}")
            await self._rollback_partition(
                name, preliminary, created_partition,
                storage_created, index_created, config_modified,
            )
            raise PartitionProvisioningError(
                f"Failed to create partition '{name}': {e}",
                details={"name": name, "error": str(e), "rollback_attempted": True}
            )

    async def _rollback_partition(
        self,
        name: str,
        preliminary: PartitionInfo,
        created_partition: Optional[PartitionInfo],
        storage_created: bool,
        index_created: bool,
        config_modified: bool,
    ) -> None:
        """Best-effort cleanup of partially created partition resources."""
        if config_modified and name in self.config.partitions.registry:
            del self.config.partitions.registry[name]
            logger.warning(f"Rollback: removed '{name}' from config registry")

        # Use the created partition if available, otherwise the preliminary
        partition = created_partition or preliminary

        if index_created:
            try:
                await self.backend.delete_partition_index(partition)
                logger.warning(f"Rollback: deleted index for '{name}'")
            except Exception as idx_err:
                logger.error(f"Rollback failed: could not delete index: {idx_err}")

        if storage_created:
            try:
                await self.backend.delete_partition_storage(partition)
                logger.warning(f"Rollback: deleted storage for '{name}'")
            except Exception as store_err:
                logger.error(f"Rollback failed: could not delete storage: {store_err}")

    async def create_partitions_batch(
        self,
        partition_values: list[Any],
        skip_existing: bool = True,
    ) -> dict[str, PartitionInfo]:
        """Create multiple partitions.

        Suppresses individual config saves during the batch and saves
        once at the end for atomicity and performance.

        Args:
            partition_values: List of partition values to create.
            skip_existing: Whether to skip existing partitions.

        Returns:
            Dictionary of created partitions.
        """
        created = {}
        errors = []

        # Suppress individual saves during batch
        original_auto_save = self.auto_save_config
        self.auto_save_config = False

        try:
            for value in partition_values:
                name = str(value)
                try:
                    partition = await self.create_partition(
                        name=name,
                        filter_value=value,
                        skip_if_exists=skip_existing,
                    )
                    created[name] = partition
                except Exception as e:
                    errors.append({"name": name, "error": str(e)})
                    logger.error(f"Error creating partition '{name}': {e}")
        finally:
            self.auto_save_config = original_auto_save

        if errors:
            logger.warning(f"Failed to create {len(errors)} partitions")

        # Save config once at the end
        if created and self.auto_save_config:
            save_config(self.config)

        return created

    async def delete_partition(
        self,
        name: str,
        delete_view: bool = True,
        delete_index: bool = True,
    ) -> None:
        """Delete a partition.

        Deregisters the partition from config first, then cleans up
        backend resources via abstract operations.

        Args:
            name: Partition name.
            delete_view: Whether to delete storage (view/table).
            delete_index: Whether to delete the vector search index.

        Raises:
            PartitionNotFoundError: If partition doesn't exist.
            PartitionProvisioningError: If config save fails.
        """
        partition = self.config.partitions.registry.get(name)
        if partition is None:
            from semantic_vector_router.exceptions import PartitionNotFoundError
            raise PartitionNotFoundError(f"Partition '{name}' not found")

        # Step 1: Remove from config first (authoritative state)
        del self.config.partitions.registry[name]

        if self.auto_save_config:
            try:
                save_config(self.config)
            except Exception as e:
                # Restore registry entry on save failure
                self.config.partitions.registry[name] = partition
                raise PartitionProvisioningError(
                    f"Failed to save config while deleting partition '{name}': {e}"
                )

        # Step 2: Best-effort cleanup of backend resources
        if delete_index:
            try:
                await self.backend.delete_partition_index(partition)
            except Exception as e:
                logger.warning(
                    f"Failed to delete index for partition '{name}': {e}. "
                    f"Orphaned index can be cleaned up manually."
                )

        if delete_view:
            try:
                await self.backend.delete_partition_storage(partition)
            except Exception as e:
                logger.warning(
                    f"Failed to delete storage for partition '{name}': {e}. "
                    f"Orphaned storage can be cleaned up manually."
                )

        logger.info(f"Deleted partition '{name}'")
        await self._emit_event("partition.deleted", name)

    async def update_partition_count(self, name: str) -> int:
        """Update document count for a partition.

        Args:
            name: Partition name.

        Returns:
            Updated document count.
        """
        partition = self.config.partitions.registry.get(name)
        if partition is None:
            from semantic_vector_router.exceptions import PartitionNotFoundError
            raise PartitionNotFoundError(f"Partition '{name}' not found")

        if partition.index_location == IndexLocation.FIELDS:
            assert partition.embedding_field is not None
            source = self.config.database.source_collection
            count = await self.backend.count_documents(
                source, {partition.embedding_field: {"$exists": True}}
            )
        elif partition.view_name:
            count = await self.backend.count_documents(partition.view_name)
        else:
            source = self.config.database.source_collection
            partition_field = self.config.partitioning.field
            if partition.filter_expression:
                count = await self.backend.count_documents(
                    source, partition.filter_expression
                )
            else:
                count = await self.backend.count_documents(
                    source, {partition_field: partition.filter_value}
                )
        partition.document_count = count
        partition.last_count_update = datetime.utcnow()

        if self.auto_save_config:
            save_config(self.config)

        return count

    async def update_all_partition_counts(self) -> dict[str, int]:
        """Update document counts for all partitions.

        Returns:
            Dictionary mapping partition names to counts.
        """
        original_auto_save = self.auto_save_config
        self.auto_save_config = False

        try:
            counts = {}
            for name in self.config.partitions.registry:
                counts[name] = await self.update_partition_count(name)
        finally:
            self.auto_save_config = original_auto_save

        if counts and self.auto_save_config:
            save_config(self.config)

        return counts

    async def verify_partition(self, name: str) -> dict[str, Any]:
        """Verify a partition's storage and index exist and are healthy.

        Args:
            name: Partition name.

        Returns:
            Verification status dictionary.
        """
        partition = self.config.partitions.registry.get(name)
        if partition is None:
            return {"name": name, "exists": False, "status": "not_registered"}

        storage_exists = await self.backend.partition_storage_exists(partition)
        index_status = await self.backend.get_partition_index_status(partition)

        return {
            "name": name,
            "exists": True,
            "view_exists": storage_exists,
            "index_status": index_status.value,
            "document_count": partition.document_count,
            "status": partition.status.value,
        }

    async def verify_all_partitions(self) -> list[dict[str, Any]]:
        """Verify all registered partitions.

        Returns:
            List of verification status dictionaries.
        """
        results = []
        for name in self.config.partitions.registry:
            status = await self.verify_partition(name)
            results.append(status)
        return results
