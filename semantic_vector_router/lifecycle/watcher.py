"""Change stream watcher for detecting new partition values."""

import asyncio
import random
from datetime import datetime
from typing import Any, Callable, Optional

from semantic_vector_router.backends.base import BaseBackend, ChangeStreamCapable
from semantic_vector_router.config import save_config
from semantic_vector_router.exceptions import ChangeStreamError
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.models import SVRConfig, WatcherStatus
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class PartitionWatcher:
    """Watches for new partition values via MongoDB change streams.

    Monitors the source collection for inserts and updates that
    introduce new values for the partition field, and optionally
    auto-provisions new partitions.
    """

    def __init__(
        self,
        backend: BaseBackend,
        config: SVRConfig,
        provisioner: Optional[PartitionProvisioner] = None,
        on_new_value: Optional[Callable[[Any], None]] = None,
    ):
        """Initialize the watcher.

        Args:
            backend: Database backend.
            config: SVR configuration.
            provisioner: Optional provisioner for auto-creating partitions.
            on_new_value: Optional callback when new value is detected.
        """
        self.backend = backend
        self.config = config
        self.provisioner = provisioner
        self.on_new_value = on_new_value

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._known_values: set[str] = set()
        # Load persisted pending partitions from config
        self._pending_partitions: list[Any] = list(
            self.config.lifecycle.pending_partitions
        )
        self._partitions_created = 0
        self._last_event: Optional[datetime] = None
        self._errors: list[str] = []

    @property
    def running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def get_status(self) -> WatcherStatus:
        """Get current watcher status.

        Returns:
            WatcherStatus with current state.
        """
        return WatcherStatus(
            running=self._running,
            last_event=self._last_event,
            partitions_created=self._partitions_created,
            errors=self._errors[-10:],  # Last 10 errors
        )

    async def start(self) -> None:
        """Start watching for new partition values.

        Raises:
            ChangeStreamError: If the backend doesn't support change streams.
        """
        if self._running:
            logger.warning("Watcher is already running")
            return

        if not isinstance(self.backend, ChangeStreamCapable):
            raise ChangeStreamError(
                "Backend does not support change streams. "
                "The watcher requires a ChangeStreamCapable backend (e.g., MongoDB)."
            )

        # Initialize known values from registry
        self._known_values = set(self.config.partitions.registry.keys())

        # Also get current values from data
        partition_field = self.config.partitioning.field
        current_values = await self.backend.get_distinct_values(partition_field)
        self._known_values.update(str(v) for v in current_values)

        # Merge persisted pending with in-memory (deduplicate)
        persisted = self.config.lifecycle.pending_partitions
        for val in persisted:
            if val not in self._pending_partitions:
                self._pending_partitions.append(val)

        logger.info(
            f"Starting watcher with {len(self._known_values)} known partition values"
        )

        self._running = True
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop the watcher."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Watcher stopped")

    async def _watch_loop(self) -> None:
        """Main watch loop with auto-reconnect on failure."""
        partition_field = self.config.partitioning.field

        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    f"fullDocument.{partition_field}": {"$exists": True}
                }
            }
        ]

        resilience = self.config.resilience
        attempt = 0

        while self._running and attempt < resilience.watcher_max_retries:
            try:
                async for change in self.backend.watch_collection(pipeline):  # type: ignore[attr-defined]
                    if not self._running:
                        break
                    await self._handle_change(change, partition_field)
                # Clean exit (stream ended or stopped)
                attempt = 0
                break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                attempt += 1
                error_msg = f"Change stream error (attempt {attempt}): {e}"
                logger.warning(error_msg)
                self._errors.append(error_msg)

                if attempt >= resilience.watcher_max_retries:
                    logger.error(f"Watcher giving up after {attempt} attempts")
                    self._running = False
                    raise ChangeStreamError(
                        f"Watcher failed after {attempt} reconnection attempts"
                    )

                # Exponential backoff with jitter
                delay = min(
                    resilience.watcher_base_delay * (2 ** (attempt - 1)),
                    resilience.watcher_max_delay,
                )
                jitter = delay * 0.25 * (2 * random.random() - 1)
                await asyncio.sleep(delay + jitter)
                logger.info(f"Watcher reconnecting (attempt {attempt})...")

    async def _handle_change(
        self, change: dict[str, Any], partition_field: str
    ) -> None:
        """Handle a change stream event.

        Args:
            change: Change event document.
            partition_field: Name of the partition field.
        """
        self._last_event = datetime.utcnow()

        full_document = change.get("fullDocument", {})
        partition_value = full_document.get(partition_field)

        if partition_value is None:
            return

        partition_name = str(partition_value)

        # Check if this is a new value
        if partition_name in self._known_values:
            return

        logger.info(f"Detected new partition value: {partition_name}")
        self._known_values.add(partition_name)

        # Call callback if provided
        if self.on_new_value:
            try:
                self.on_new_value(partition_value)
            except Exception as e:
                logger.error(f"Error in on_new_value callback: {e}")

        # Auto-provision if configured
        if self.config.lifecycle.auto_provision:
            if self.config.lifecycle.confirmation_required:
                # Store for manual confirmation
                self._pending_partitions.append(partition_value)
                self._persist_pending()
                logger.info(
                    f"Partition '{partition_name}' added to pending list "
                    "(confirmation required)"
                )
            else:
                # Auto-provision immediately
                await self._auto_provision(partition_value)

    async def _auto_provision(self, partition_value: Any) -> None:
        """Auto-provision a new partition.

        Args:
            partition_value: Value for the new partition.
        """
        if self.provisioner is None:
            logger.warning("No provisioner configured for auto-provisioning")
            return

        partition_name = str(partition_value)

        try:
            await self.provisioner.create_partition(
                name=partition_name,
                filter_value=partition_value,
                skip_if_exists=True,
            )
            self._partitions_created += 1
            logger.info(f"Auto-provisioned partition: {partition_name}")

        except Exception as e:
            error_msg = f"Failed to auto-provision '{partition_name}': {e}"
            logger.error(error_msg)
            self._errors.append(error_msg)

    def get_pending_partitions(self) -> list[Any]:
        """Get list of partition values pending confirmation.

        Returns:
            List of pending partition values.
        """
        return list(self._pending_partitions)

    async def confirm_partition(self, partition_value: Any) -> bool:
        """Confirm and provision a pending partition.

        Args:
            partition_value: Value to confirm.

        Returns:
            True if provisioned successfully.
        """
        if partition_value not in self._pending_partitions:
            logger.warning(f"Partition value not in pending list: {partition_value}")
            return False

        self._pending_partitions.remove(partition_value)
        self._persist_pending()
        await self._auto_provision(partition_value)
        return True

    async def confirm_all_pending(self) -> int:
        """Confirm and provision all pending partitions.

        Returns:
            Number of partitions provisioned.
        """
        count = 0
        pending = list(self._pending_partitions)

        for value in pending:
            if await self.confirm_partition(value):
                count += 1

        return count

    def reject_partition(self, partition_value: Any) -> bool:
        """Reject a pending partition (don't provision).

        Args:
            partition_value: Value to reject.

        Returns:
            True if removed from pending list.
        """
        if partition_value in self._pending_partitions:
            self._pending_partitions.remove(partition_value)
            self._persist_pending()
            logger.info(f"Rejected pending partition: {partition_value}")
            return True
        return False

    def _persist_pending(self) -> None:
        """Persist pending partitions to config file."""
        self.config.lifecycle.pending_partitions = [
            str(v) for v in self._pending_partitions
        ]
        try:
            save_config(self.config)
        except Exception as e:
            logger.error(f"Failed to persist pending partitions: {e}")
