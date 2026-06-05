"""Partition splitter for handling overgrown partitions."""

import warnings
from datetime import datetime, timezone
from typing import Any, cast

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.config import save_config
from semantic_vector_router.exceptions import SplitError
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.models import (
    PartitionInfo,
    PartitionStatus,
    SplitStrategy,
    SVRConfig,
)
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class PartitionSplitter:
    """Handles splitting of overgrown partitions.

    Supports split strategies that produce semantically meaningful child partitions:
    - secondary_field: Split by a secondary field (e.g., subcategory)
    - time: Split by time buckets
    - alert_only: Just notify, don't auto-split

    The hash strategy is deprecated. Hash-based sharding distributes documents
    randomly, which means every query must fan out to all child partitions —
    defeating the purpose of partitioned vector search. Use secondary_field
    with a meaningful metadata field instead. If no natural field exists,
    consider generating one via LLM classification or clustering externally.
    """

    def __init__(
        self,
        backend: BaseBackend,
        config: SVRConfig,
        provisioner: PartitionProvisioner,
    ):
        """Initialize the splitter.

        Args:
            backend: Database backend.
            config: SVR configuration.
            provisioner: Partition provisioner.
        """
        self.backend = backend
        self.config = config
        self.provisioner = provisioner

    async def check_and_split(self) -> list[str]:
        """Check all partitions and split those exceeding threshold.

        Returns:
            List of partition names that were split.
        """
        if not self.config.lifecycle.auto_split:
            return []

        if not self.config.lifecycle.auto_split.enabled:
            return []

        threshold = self.config.lifecycle.auto_split.threshold_vectors
        split_partitions = []

        for name, partition in list(self.config.partitions.registry.items()):
            # Skip already split or disabled partitions
            if partition.status in [PartitionStatus.SPLIT, PartitionStatus.DISABLED]:
                continue

            # Check document count
            count = await self.backend.count_documents(partition.view_name)

            if count > threshold:
                logger.warning(
                    f"Partition '{name}' exceeds threshold: "
                    f"{count:,} vectors (threshold: {threshold:,})"
                )

                if self._is_within_schedule():
                    await self.execute_split(name)
                    split_partitions.append(name)
                else:
                    await self._mark_pending_split(name)

        return split_partitions

    async def execute_split(self, partition_name: str) -> list[str]:
        """Execute split for a specific partition.

        Args:
            partition_name: Name of partition to split.

        Returns:
            List of child partition names created.

        Raises:
            SplitError: If split fails.
        """
        partition = self.config.partitions.registry.get(partition_name)
        if partition is None:
            raise SplitError(f"Partition '{partition_name}' not found")

        assert self.config.lifecycle.auto_split is not None
        strategy = self.config.lifecycle.auto_split.split_strategy

        try:
            if strategy == SplitStrategy.SECONDARY_FIELD:
                children = await self._split_by_secondary_field(partition)
            elif strategy == SplitStrategy.HASH:
                warnings.warn(
                    "Hash split strategy is deprecated. Hash-based sharding does not "
                    "produce semantically meaningful partitions — every query must fan "
                    "out to all child partitions, defeating the purpose of partitioned "
                    "vector search. Use 'secondary_field' with a meaningful metadata "
                    "field instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                logger.warning(
                    "Hash split is deprecated: results in query fan-out to all children. "
                    "Consider using secondary_field strategy instead."
                )
                children = await self._split_by_hash(partition)
            elif strategy == SplitStrategy.TIME:
                children = await self._split_by_time(
                    partition,
                    time_field=cast(str, self.config.lifecycle.auto_split.time_field),
                    bucket=self.config.lifecycle.auto_split.bucket,
                )
            else:  # ALERT_ONLY
                logger.info(f"Alert only: Partition '{partition_name}' needs splitting")
                return []

            # Mark original partition as split
            partition.status = PartitionStatus.SPLIT
            partition.child_partitions = children

            # Save config
            save_config(self.config)

            logger.info(
                f"Split partition '{partition_name}' into {len(children)} children"
            )
            return children

        except Exception as e:
            raise SplitError(
                f"Failed to split partition '{partition_name}': {e}",
                details={"partition": partition_name, "strategy": strategy.value}
            )

    async def _split_by_secondary_field(
        self, partition: PartitionInfo
    ) -> list[str]:
        """Split partition by secondary field values.

        Args:
            partition: Partition to split.

        Returns:
            List of child partition names.
        """
        assert self.config.lifecycle.auto_split is not None
        secondary_field = self.config.lifecycle.auto_split.secondary_field
        if not secondary_field:
            raise SplitError("Secondary field not configured for split")

        # Get distinct values of secondary field scoped to this partition only.
        # Build a filter to restrict to the partition being split.
        partition_filter = None
        if partition.filter_expression:
            partition_filter = partition.filter_expression
        elif partition.filter_value is not None:
            partition_filter = {self.config.partitioning.field: partition.filter_value}

        values = await self.backend.get_distinct_values(
            secondary_field, filter_expression=partition_filter
        )

        children = []
        partition_field = self.config.partitioning.field

        for value in values:
            child_name = f"{partition.name}__{value}"

            # Create filter for child partition
            filter_expr = {
                partition_field: partition.filter_value,
                secondary_field: value,
            }

            await self.provisioner.create_partition(
                name=child_name,
                filter_expression=filter_expr,
                skip_if_exists=True,
            )

            # Set parent reference
            child = self.config.partitions.registry.get(child_name)
            if child:
                child.parent_partition = partition.name

            children.append(child_name)

        return children

    async def _split_by_hash(self, partition: PartitionInfo) -> list[str]:
        """Split partition into hash-based shards.

        DEPRECATED: Hash splits don't produce semantically meaningful partitions.
        Use secondary_field strategy instead.

        Args:
            partition: Partition to split.

        Returns:
            List of child partition names.
        """
        assert self.config.lifecycle.auto_split is not None
        num_shards = self.config.lifecycle.auto_split.num_shards
        children = []
        partition_field = self.config.partitioning.field

        for shard_num in range(num_shards):
            child_name = f"{partition.name}__shard_{shard_num}"

            # Distribute documents across shards using last hex chars of _id.
            # Uses $substr on stringified _id to extract last 2 hex characters,
            # converts to integer, and applies $mod for bucket assignment.
            filter_expr = {
                partition_field: partition.filter_value,
                "$expr": {
                    "$eq": [
                        {
                            "$mod": [
                                {"$toInt": {"$substr": [{"$toString": "$_id"}, -2, 2]}},
                                num_shards,
                            ]
                        },
                        shard_num,
                    ]
                },
            }

            await self.provisioner.create_partition(
                name=child_name,
                filter_expression=filter_expr,
                skip_if_exists=True,
            )

            child = self.config.partitions.registry.get(child_name)
            if child:
                child.parent_partition = partition.name

            children.append(child_name)

        return children

    async def _split_by_time(
        self,
        partition: PartitionInfo,
        time_field: str,
        bucket: str,
    ) -> list[str]:
        """Split partition by time boundaries.

        1. Aggregate $min/$max of time_field within partition
        2. Generate bucket boundaries based on strategy
        3. Create child partitions with filter_expression for each bucket

        Args:
            partition: Parent partition to split.
            time_field: Field containing datetime values.
            bucket: Bucketing strategy ("monthly", "quarterly", "yearly").

        Returns:
            List of created child partition names.

        Raises:
            SplitError: If time_field doesn't exist or has no values.
        """
        if not time_field:
            raise SplitError("Time field not configured for split")

        # Build match filter scoped to parent partition
        match_filter: dict[str, Any] = {}
        if partition.filter_expression:
            match_filter = dict(partition.filter_expression)
        elif partition.filter_value is not None:
            match_filter = {self.config.partitioning.field: partition.filter_value}

        # Aggregate min/max of time_field
        pipeline = [
            {"$match": match_filter},
            {"$group": {
                "_id": None,
                "min_date": {"$min": f"${time_field}"},
                "max_date": {"$max": f"${time_field}"},
                "count": {"$sum": 1},
            }},
        ]

        collection = self.backend.get_collection()  # type: ignore[attr-defined]
        cursor = await collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)

        if not results or results[0].get("min_date") is None:
            raise SplitError(
                f"No documents with time field '{time_field}' found in partition "
                f"'{partition.name}'"
            )

        result = results[0]
        min_date = result["min_date"]
        max_date = result["max_date"]

        # Ensure timezone-aware UTC datetimes
        if min_date.tzinfo is None:
            min_date = min_date.replace(tzinfo=timezone.utc)
        if max_date.tzinfo is None:
            max_date = max_date.replace(tzinfo=timezone.utc)

        buckets = self._generate_time_buckets(min_date, max_date, bucket)

        children: list[str] = []

        for start, end, label in buckets:
            child_name = f"{partition.name}_{label}"

            filter_expr: dict[str, Any] = {}
            if partition.filter_expression:
                filter_expr.update(partition.filter_expression)
            elif partition.filter_value is not None:
                filter_expr[self.config.partitioning.field] = partition.filter_value

            filter_expr[time_field] = {
                "$gte": start,
                "$lt": end,
            }

            await self.provisioner.create_partition(
                name=child_name,
                filter_expression=filter_expr,
                skip_if_exists=True,
            )

            child = self.config.partitions.registry.get(child_name)
            if child:
                child.parent_partition = partition.name

            children.append(child_name)

        return children

    @staticmethod
    def _generate_time_buckets(
        min_date: datetime,
        max_date: datetime,
        bucket: str,
    ) -> list[tuple[datetime, datetime, str]]:
        """Generate (start, end, label) tuples for time buckets.

        All datetimes are UTC. Buckets span [start, end).

        Args:
            min_date: Earliest document timestamp.
            max_date: Latest document timestamp.
            bucket: Bucketing strategy ("monthly", "quarterly", "yearly").

        Returns:
            List of (start, end, label) tuples.
        """
        buckets: list[tuple[datetime, datetime, str]] = []

        if bucket == "yearly":
            year = min_date.year
            while year <= max_date.year:
                start = datetime(year, 1, 1, tzinfo=timezone.utc)
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                label = str(year)
                buckets.append((start, end, label))
                year += 1

        elif bucket == "quarterly":
            # Start from the quarter containing min_date
            year = min_date.year
            quarter = (min_date.month - 1) // 3 + 1
            while True:
                q_start_month = (quarter - 1) * 3 + 1
                start = datetime(year, q_start_month, 1, tzinfo=timezone.utc)
                # Next quarter
                if quarter == 4:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end = datetime(year, q_start_month + 3, 1, tzinfo=timezone.utc)
                label = f"{year}_Q{quarter}"
                buckets.append((start, end, label))
                if end > max_date:
                    break
                quarter += 1
                if quarter > 4:
                    quarter = 1
                    year += 1

        elif bucket == "monthly":
            year = min_date.year
            month = min_date.month
            while True:
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                # Next month
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
                label = f"{year}_{month:02d}"
                buckets.append((start, end, label))
                if end > max_date:
                    break
                month += 1
                if month > 12:
                    month = 1
                    year += 1

        return buckets

    async def _mark_pending_split(self, partition_name: str) -> None:
        """Mark a partition as pending split.

        Args:
            partition_name: Name of partition to mark.
        """
        partition = self.config.partitions.registry.get(partition_name)
        if partition:
            partition.status = PartitionStatus.PENDING_SPLIT
            save_config(self.config)
            logger.info(f"Marked partition '{partition_name}' as pending split")

    def _is_within_schedule(self) -> bool:
        """Check if current time is within allowed split schedule.

        Returns:
            True if within schedule, False otherwise.
        """
        if not self.config.lifecycle.auto_split:
            return True

        schedule = self.config.lifecycle.auto_split.schedule
        if not schedule:
            return True

        now = datetime.now(timezone.utc)
        day_name = now.strftime("%A").lower()
        hour = now.hour

        # Check day
        if schedule.allowed_days and day_name not in [
            d.lower() for d in schedule.allowed_days
        ]:
            return False

        # Check hour
        if schedule.allowed_hours:
            start = schedule.allowed_hours.get("start", 0)
            end = schedule.allowed_hours.get("end", 24)
            if not (start <= hour < end):
                return False

        return True

    async def get_pending_splits(self) -> list[str]:
        """Get list of partitions pending split.

        Returns:
            List of partition names with pending_split status.
        """
        return [
            name for name, p in self.config.partitions.registry.items()
            if p.status == PartitionStatus.PENDING_SPLIT
        ]

    async def execute_pending_splits(self) -> list[str]:
        """Execute all pending splits.

        Returns:
            List of partition names that were split.
        """
        pending = await self.get_pending_splits()
        split_partitions = []

        for name in pending:
            try:
                children = await self.execute_split(name)
                if children:
                    split_partitions.append(name)
            except Exception as e:
                logger.error(f"Failed to execute pending split for '{name}': {e}")

        return split_partitions
