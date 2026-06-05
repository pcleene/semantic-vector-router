"""Partition monitoring for health and size tracking."""

from datetime import datetime
from typing import Any

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.models import (
    PartitionHealthStatus,
    PartitionStatus,
    SVRConfig,
)
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class PartitionMonitor:
    """Monitors partition health and size thresholds.

    Tracks:
    - Document counts per partition
    - Threshold breaches for auto-splitting
    - Index health and query readiness
    - Overall partition utilization
    """

    def __init__(self, backend: BaseBackend, config: SVRConfig):
        """Initialize the monitor.

        Args:
            backend: Database backend.
            config: SVR configuration.
        """
        self.backend = backend
        self.config = config

    async def check_partition_health(self, name: str) -> PartitionHealthStatus:
        """Check health status of a single partition.

        Args:
            name: Partition name.

        Returns:
            PartitionHealthStatus with current metrics.
        """
        partition = self.config.partitions.registry.get(name)
        if partition is None:
            return PartitionHealthStatus(
                partition=name,
                vector_count=0,
                threshold=self._get_threshold(),
                utilization=0.0,
                status="not_found",
            )

        # Get current document count
        count = await self.backend.count_documents(partition.view_name)

        # Calculate utilization against threshold
        threshold = self._get_threshold()
        utilization = count / threshold if threshold > 0 else 0.0

        # Determine status
        if count > threshold:
            status = "critical"
        elif utilization > 0.8:
            status = "warning"
        else:
            status = "healthy"

        return PartitionHealthStatus(
            partition=name,
            vector_count=count,
            threshold=threshold,
            utilization=utilization,
            status=status,
        )

    async def check_all_partitions(self) -> list[PartitionHealthStatus]:
        """Check health of all partitions.

        Returns:
            List of PartitionHealthStatus for all partitions.
        """
        results = []
        for name in self.config.partitions.registry:
            status = await self.check_partition_health(name)
            results.append(status)

        # Sort by utilization descending
        results.sort(key=lambda x: x.utilization, reverse=True)
        return results

    async def get_critical_partitions(self) -> list[PartitionHealthStatus]:
        """Get partitions that have exceeded the threshold.

        Returns:
            List of critical partition statuses.
        """
        all_status = await self.check_all_partitions()
        return [s for s in all_status if s.status == "critical"]

    async def get_warning_partitions(self) -> list[PartitionHealthStatus]:
        """Get partitions approaching the threshold (>80%).

        Returns:
            List of warning partition statuses.
        """
        all_status = await self.check_all_partitions()
        return [s for s in all_status if s.status == "warning"]

    async def get_partition_summary(self) -> dict[str, Any]:
        """Get summary of all partition health.

        Returns:
            Summary dictionary with counts and details.
        """
        all_status = await self.check_all_partitions()

        total_vectors = sum(s.vector_count for s in all_status)
        critical_count = len([s for s in all_status if s.status == "critical"])
        warning_count = len([s for s in all_status if s.status == "warning"])
        healthy_count = len([s for s in all_status if s.status == "healthy"])

        return {
            "total_partitions": len(all_status),
            "total_vectors": total_vectors,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "healthy_count": healthy_count,
            "threshold": self._get_threshold(),
            "partitions": [
                {
                    "name": s.partition,
                    "vectors": s.vector_count,
                    "utilization": f"{s.utilization * 100:.1f}%",
                    "status": s.status,
                }
                for s in all_status
            ],
        }

    async def check_index_health(self, name: str) -> dict[str, Any]:
        """Check index health for a partition.

        Args:
            name: Partition name.

        Returns:
            Index health status dictionary.
        """
        partition = self.config.partitions.registry.get(name)
        if partition is None:
            return {"name": name, "status": "partition_not_found"}

        assert partition.view_name is not None
        assert partition.index_name is not None
        index_status = await self.backend.get_index_status(
            partition.view_name,
            partition.index_name,
        )

        return {
            "name": name,
            "view_name": partition.view_name,
            "index_name": partition.index_name,
            "index_status": index_status.get("status", "unknown"),
            "queryable": index_status.get("queryable", False),
        }

    async def check_all_index_health(self) -> list[dict[str, Any]]:
        """Check index health for all partitions.

        Returns:
            List of index health statuses.
        """
        results = []
        for name in self.config.partitions.registry:
            status = await self.check_index_health(name)
            results.append(status)
        return results

    async def update_partition_counts(self) -> dict[str, int]:
        """Update document counts for all partitions in config.

        Returns:
            Dictionary mapping partition names to updated counts.
        """
        counts = {}
        for name, partition in self.config.partitions.registry.items():
            if partition.status == PartitionStatus.DISABLED:
                continue

            try:
                count = await self.backend.count_documents(partition.view_name)
                partition.document_count = count
                partition.last_count_update = datetime.utcnow()
                counts[name] = count
            except Exception as e:
                logger.error(f"Error updating count for {name}: {e}")
                counts[name] = -1

        return counts

    def _get_threshold(self) -> int:
        """Get the vector count threshold from config.

        Returns:
            Threshold value, or default if not configured.
        """
        if self.config.lifecycle.auto_split:
            return self.config.lifecycle.auto_split.threshold_vectors
        return 10_000_000  # Default 10M

    async def needs_attention(self) -> dict[str, list[str]]:
        """Check which partitions need attention.

        Returns:
            Dictionary with 'critical', 'warning', and 'unhealthy_indexes' lists.
        """
        all_status = await self.check_all_partitions()
        all_indexes = await self.check_all_index_health()

        return {
            "critical": [s.partition for s in all_status if s.status == "critical"],
            "warning": [s.partition for s in all_status if s.status == "warning"],
            "unhealthy_indexes": [
                i["name"] for i in all_indexes if not i.get("queryable", False)
            ],
        }
