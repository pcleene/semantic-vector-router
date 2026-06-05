"""Repartition workflow engine for the Semantic Vector Router."""

import asyncio
from datetime import datetime
from typing import Any

from semantic_vector_router.backends.metadata import MetadataStore
from semantic_vector_router.exceptions import RepartitionError
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.models import IndexLocation, PartitionStatus, SVRConfig
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class RepartitionEngine:
    """Orchestrates multi-step repartitioning operations.

    Supports step-by-step execution with resume capability,
    rollback on failure, and multiple splitting strategies.
    """

    def __init__(self, backend, metadata: MetadataStore, config: SVRConfig,
                 event_bus: Any = None):
        self.backend = backend
        self.metadata = metadata
        self.config = config
        self.provisioner = PartitionProvisioner(backend, config, auto_save_config=False)
        self._event_bus = event_bus

    async def _emit_event(
        self, event_type_str: str, partition: str, **details: Any
    ) -> None:
        """Emit a repartition lifecycle event."""
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

    async def execute_operation(self, op_id: str) -> bool:
        """Execute a repartition operation step-by-step.

        Supports resume by skipping steps with status="done".

        Returns:
            True if completed successfully, False otherwise.
        """
        op = await self.metadata.get_operation(op_id)
        if not op:
            raise RepartitionError(f"Operation {op_id} not found")

        logger.info(f"Executing repartition operation {op_id}")

        parent_name = op.get("target_partition", "unknown")
        await self._emit_event("repartition.started", parent_name, operation_id=op_id)

        step_handlers = {
            "create_children": self._step_create_children,
            "build_indexes": self._step_build_indexes,
            "wait_indexes": self._step_wait_indexes,
            "switch_routing": self._step_switch_routing,
            "compute_centroids": self._step_compute_centroids,
            "cleanup_parent": self._step_cleanup_parent,
        }

        try:
            for step in op.get("steps", []):
                step_action = step["action"]
                step_status = step.get("status")

                if step_status == "done":
                    logger.info(f"Step {step_action} already done, skipping")
                    continue

                handler = step_handlers.get(step_action)
                if not handler:
                    raise RepartitionError(f"Unknown step: {step_action}")

                logger.info(f"Executing step: {step_action}")

                # Mark step as in_progress
                await self.metadata.update_operation_step(
                    op_id, step_action, "in_progress"
                )

                # Execute step handler
                await handler(op)

                # Mark step as done
                await self.metadata.update_operation_step(
                    op_id, step_action, "done"
                )

                logger.info(f"Step {step_action} completed successfully")

            # Mark operation as done
            await self.metadata.update_operation_status(
                op_id, "done", completed_at=datetime.utcnow()
            )

            logger.info(f"Repartition operation {op_id} completed successfully")
            await self._emit_event(
                "repartition.completed", parent_name, operation_id=op_id
            )
            return True

        except Exception as e:
            logger.error(f"Repartition operation {op_id} failed: {e}", exc_info=True)
            await self._emit_event(
                "repartition.failed", parent_name,
                operation_id=op_id, error=str(e),
            )

            # Mark current running step as failed
            for step in op.get("steps", []):
                if step.get("status") == "in_progress":
                    await self.metadata.update_operation_step(
                        op_id, step["action"], "failed", error=str(e)
                    )

            # Mark operation as failed
            await self.metadata.update_operation_status(
                op_id, "failed", error=str(e), failed_at=datetime.utcnow()
            )

            return False

    async def rollback_operation(self, op_id: str) -> None:
        """Rollback a repartition operation.

        Deletes child partitions, resets parent to ACTIVE, marks op as failed.
        """
        logger.info(f"Rolling back repartition operation {op_id}")

        op = await self.metadata.get_operation(op_id)
        if not op:
            raise RepartitionError(f"Operation {op_id} not found")

        parent_name = op.get("target_partition")
        if not parent_name:
            raise RepartitionError(f"Operation {op_id} has no target_partition")

        parent = await self.metadata.get_partition(parent_name)
        if not parent:
            raise RepartitionError(f"Parent partition {parent_name} not found")

        # Delete child partitions
        for child_name in parent.child_partitions:
            try:
                logger.info(f"Deleting child partition {child_name}")
                await self.provisioner.delete_partition(child_name)
            except Exception as e:
                logger.warning(f"Failed to delete child partition {child_name}: {e}")

        # Reset parent to ACTIVE
        parent.status = PartitionStatus.ACTIVE
        parent.child_partitions = []
        await self.metadata.save_partition(parent)

        logger.info(f"Parent partition {parent_name} reset to ACTIVE")

        # Mark operation as failed
        await self.metadata.update_operation_status(
            op_id, "failed", error="Rolled back by user", failed_at=datetime.utcnow()
        )

        logger.info(f"Repartition operation {op_id} rolled back successfully")
        await self._emit_event(
            "repartition.rolled_back", parent_name, operation_id=op_id
        )

    async def _step_create_children(self, op: dict) -> None:
        """Step 1: Create child partitions."""
        parent_name = op.get("target_partition")
        strategy = op.get("strategy")
        strategy_config = op.get("strategy_config", {})

        logger.info(f"Creating children for parent {parent_name} using strategy {strategy}")

        parent = await self.metadata.get_partition(parent_name)
        if not parent:
            raise RepartitionError(f"Parent partition {parent_name} not found")

        # Mark parent as SPLITTING
        parent.status = PartitionStatus.SPLITTING
        await self.metadata.save_partition(parent)

        child_names = []

        if strategy == "secondary_field":
            secondary_field = strategy_config.get("secondary_field")
            if not secondary_field:
                raise RepartitionError(
                    "secondary_field strategy requires 'secondary_field' in config"
                )

            # Build filter from parent's partition value
            filter_expression = None
            if parent.filter_value is not None:
                filter_expression = {
                    self.config.partitioning.field: parent.filter_value
                }

            # Get distinct values for the secondary field
            logger.info(f"Getting distinct values for field {secondary_field}")
            distinct_values = await self.backend.get_distinct_values(
                secondary_field, filter_expression
            )

            logger.info(f"Found {len(distinct_values)} distinct values for {secondary_field}")

            for value in distinct_values:
                child_name = f"{parent_name}_{secondary_field}_{value}"

                logger.info(f"Creating child partition {child_name}")
                await self.provisioner.create_partition(
                    name=child_name,
                    filter_value=value,
                )

                # Save child to metadata with parent reference
                child = await self.metadata.get_partition(child_name)
                if child:
                    child.parent_partition = parent_name
                    await self.metadata.save_partition(child)

                child_names.append(child_name)
        else:
            raise RepartitionError(f"Unknown splitting strategy: {strategy}")

        # Update parent with child references
        parent.child_partitions = child_names
        await self.metadata.save_partition(parent)

        logger.info(f"Created {len(child_names)} child partitions for {parent_name}")

    async def _step_build_indexes(self, op: dict) -> None:
        """Step 2: Verify all child indexes exist."""
        parent_name = op.get("target_partition")

        parent = await self.metadata.get_partition(parent_name)
        if not parent:
            raise RepartitionError(f"Parent partition {parent_name} not found")

        for child_name in parent.child_partitions:
            child = await self.metadata.get_partition(child_name)
            if not child:
                raise RepartitionError(f"Child partition {child_name} not found")

            collection_name = child.search_collection or child.view_name
            index_name = child.index_name

            if not collection_name or not index_name:
                raise RepartitionError(
                    f"Child partition {child_name} missing collection or index name"
                )

            logger.info(f"Checking index {index_name} on {collection_name}")
            status = await self.backend.get_index_status(collection_name, index_name)

            if not status:
                raise RepartitionError(
                    f"Index {index_name} not found on {collection_name}"
                )

        logger.info(f"All child indexes verified for {parent_name}")

    async def _step_wait_indexes(self, op: dict) -> None:
        """Step 3: Wait for all child indexes to become queryable."""
        parent_name = op.get("target_partition")
        timeout = self.config.lifecycle.repartition.index_wait_timeout_s
        poll_interval = self.config.lifecycle.repartition.index_poll_interval_s

        parent = await self.metadata.get_partition(parent_name)
        if not parent:
            raise RepartitionError(f"Parent partition {parent_name} not found")

        child_names = parent.child_partitions
        ready_children: set[str] = set()
        start_time = asyncio.get_event_loop().time()

        while len(ready_children) < len(child_names):
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise RepartitionError(
                    f"Timeout waiting for indexes to become queryable after {timeout}s"
                )

            for child_name in child_names:
                if child_name in ready_children:
                    continue

                child = await self.metadata.get_partition(child_name)
                if not child:
                    raise RepartitionError(f"Child partition {child_name} not found")

                collection_name = child.search_collection or child.view_name
                index_name = child.index_name

                status = await self.backend.get_index_status(collection_name, index_name)
                if not status:
                    raise RepartitionError(
                        f"Index {index_name} disappeared from {collection_name}"
                    )

                if status.get("queryable"):
                    logger.info(f"Index {index_name} on {collection_name} is queryable")
                    ready_children.add(child_name)

            if len(ready_children) < len(child_names):
                logger.info(
                    f"Waiting for {len(child_names) - len(ready_children)} indexes "
                    f"to become queryable (elapsed: {elapsed:.1f}s)"
                )
                await asyncio.sleep(poll_interval)

        logger.info(f"All child indexes of {parent_name} are queryable")

    async def _step_switch_routing(self, op: dict) -> None:
        """Step 4: Switch routing from parent to children."""
        parent_name = op.get("target_partition")

        parent = await self.metadata.get_partition(parent_name)
        if not parent:
            raise RepartitionError(f"Parent partition {parent_name} not found")

        # Mark parent as RETIRED
        parent.status = PartitionStatus.RETIRED
        await self.metadata.save_partition(parent)
        logger.info(f"Parent partition {parent_name} marked as RETIRED")

        # Mark all children as ACTIVE
        for child_name in parent.child_partitions:
            child = await self.metadata.get_partition(child_name)
            if not child:
                raise RepartitionError(f"Child partition {child_name} not found")

            child.status = PartitionStatus.ACTIVE
            await self.metadata.save_partition(child)
            logger.info(f"Child partition {child_name} marked as ACTIVE")

        logger.info(f"Routing switched for {parent_name}")

    async def _step_compute_centroids(self, op: dict) -> None:
        """Step 6: Compute centroid embeddings for new child partitions.

        Samples stored embedding vectors from each child partition and
        computes a normalized mean vector. Zero API calls.
        """
        parent_name = op.get("target_partition")

        parent = await self.metadata.get_partition(parent_name)
        if not parent:
            raise RepartitionError(f"Parent partition {parent_name} not found")

        # Centroid computation requires MongoDB aggregation pipeline.
        # Non-MongoDB backends skip centroid computation during repartition.
        if not hasattr(self.backend, "db"):
            logger.warning(
                "Centroid computation skipped during repartition: "
                "backend does not support MongoDB-style aggregation"
            )
            return

        source_collection = self.config.database.source_collection
        collection = self.backend.db[source_collection]
        embedding_field = self.config.vector_search.embedding_field
        partition_field = self.config.partitioning.field
        sample_size = self.config.routing.centroid_routing.sample_size

        from semantic_vector_router.routing.centroid import (
            compute_partition_centroid,
        )

        computed = 0
        for child_name in parent.child_partitions:
            child = await self.metadata.get_partition(child_name)
            if not child or child.status != PartitionStatus.ACTIVE:
                continue

            field_path = child.embedding_field or embedding_field

            partition_filter = None
            if child.filter_value is not None:
                partition_filter = {partition_field: child.filter_value}

            centroid = await compute_partition_centroid(
                collection=collection,
                embedding_field=field_path,
                partition_filter=partition_filter,
                sample_size=sample_size,
            )

            if centroid:
                await self.metadata.update_centroid(child_name, centroid)
                logger.info(f"Computed centroid for child partition {child_name}")
                await self._emit_event("centroid.computed", child_name)
                computed += 1
            else:
                logger.warning(
                    f"No vectors found for child partition {child_name}, "
                    f"skipping centroid"
                )

        logger.info(
            f"Computed {computed} centroids for children of {parent_name}"
        )

    async def _step_cleanup_parent(self, op: dict) -> None:
        """Step 5: Best-effort cleanup of parent partition resources."""
        parent_name = op.get("target_partition")

        if not self.config.lifecycle.repartition.auto_cleanup_retired:
            logger.info(f"Auto-cleanup disabled, skipping cleanup of {parent_name}")
            return

        logger.info(f"Cleaning up parent partition {parent_name}")

        try:
            parent = await self.metadata.get_partition(parent_name)
            if not parent:
                logger.warning(f"Parent partition {parent_name} not found, skipping cleanup")
                return

            # Best-effort delete index
            if parent.index_name:
                collection_name = parent.search_collection or parent.view_name
                if collection_name:
                    try:
                        await self.backend.delete_index(collection_name, parent.index_name)
                    except Exception as e:
                        logger.warning(f"Failed to delete index {parent.index_name}: {e}")

            # Best-effort delete view
            if parent.view_name and self.config.vector_storage.index_on == IndexLocation.VIEWS:
                try:
                    await self.backend.delete_view(parent.view_name)
                except Exception as e:
                    logger.warning(f"Failed to delete view {parent.view_name}: {e}")

            logger.info(f"Parent partition {parent_name} cleanup completed")

        except Exception as e:
            logger.warning(f"Non-fatal error during cleanup of {parent_name}: {e}")
