"""Detection pipeline for identifying partition health issues."""

import os
import socket
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from semantic_vector_router.backends.metadata import MetadataStore
from semantic_vector_router.models import DetectionSignal, PartitionInfo, SVRConfig
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Result of a detection check."""

    signal: DetectionSignal
    partition: str
    details: dict
    auto_executable: bool
    suggested_action: str


class PartitionDetector:
    """Detects partition health issues and creates remediation operations."""

    def __init__(self, backend, metadata: MetadataStore, config: SVRConfig,
                 event_bus: Any = None):
        """Initialize detector.

        Args:
            backend: Backend instance for counting documents
            metadata: MetadataStore for health history and operations
            config: SVRConfig with lifecycle detection settings
            event_bus: Optional EventBus for emitting health alert events
        """
        self.backend = backend
        self.metadata = metadata
        self.config = config
        self.logger = get_logger(__name__)
        self._event_bus = event_bus

    async def _emit_event(
        self, event_type_str: str, partition: str, **details: Any
    ) -> None:
        """Emit a health event."""
        if self._event_bus is None:
            return
        try:
            from semantic_vector_router.events.models import SVREvent, SVREventType
            event = SVREvent(
                event_type=SVREventType(event_type_str),
                partition=partition,
                details=details,
                severity="warning" if "alert" in event_type_str or "breach" in event_type_str else "info",
            )
            await self._event_bus.emit(event)
        except Exception as e:
            self.logger.warning(f"Failed to emit event: {e}")

    async def run_detection(self) -> list[DetectionResult]:
        """Run full detection pipeline: COLLECT → STORE → ANALYZE → DECIDE.

        Returns:
            List of DetectionResult objects
        """
        self.logger.info("Starting detection pipeline")

        # Get all partitions
        partitions = await self.metadata.list_partitions()
        if not partitions:
            self.logger.info("No partitions found, skipping detection")
            return []

        # COLLECT: Count documents per partition
        self.logger.info(f"Collecting counts for {len(partitions)} partitions")
        counts = await self._collect_counts(partitions)

        # STORE: Save to health history
        self.logger.info("Storing health data")
        await self._store_health_data(counts)

        # ANALYZE: Run all detection checks
        self.logger.info("Running detection checks")
        results = []

        # Check for threshold breaches
        breach_results = await self._check_threshold_breaches(partitions, counts)
        results.extend(breach_results)

        # Check for approaching thresholds
        approach_results = await self._check_approaching_thresholds(partitions)
        results.extend(approach_results)

        # Check for skew
        skew_results = await self._check_skew(partitions, counts)
        results.extend(skew_results)

        # Check for underpopulated partitions
        underpop_results = await self._check_underpopulated(partitions, counts)
        results.extend(underpop_results)

        # Check for stale partitions
        stale_results = await self._check_stale(partitions)
        results.extend(stale_results)

        self.logger.info(f"Detection complete: {len(results)} issues found")

        # Emit events for each detection result
        signal_to_event = {
            DetectionSignal.THRESHOLD_BREACH: "health.threshold_breach",
            DetectionSignal.APPROACHING_THRESHOLD: "health.approaching_threshold",
            DetectionSignal.SEVERE_SKEW: "health.skew_detected",
        }
        for r in results:
            event_type = signal_to_event.get(r.signal, "health.alert")
            await self._emit_event(
                event_type,
                r.partition,
                signal=r.signal.value,
                suggested_action=r.suggested_action,
                **r.details,
            )

        # DECIDE: Create operations for auto-executable results
        await self._create_operations(results)

        return results

    async def run_detection_with_lock(self) -> Optional[list[DetectionResult]]:
        """Acquire distributed lock, run detection, release.

        Returns:
            Detection results if lock acquired, None otherwise
        """
        holder = f"{socket.gethostname()}-{os.getpid()}"
        lock_id = "monitor"

        self.logger.info(f"Attempting to acquire lock '{lock_id}' as {holder}")

        # Try to acquire lock
        acquired = await self.metadata.acquire_lock(lock_id, holder)
        if not acquired:
            self.logger.info(f"Could not acquire lock '{lock_id}', skipping detection")
            return None

        try:
            self.logger.info(f"Lock '{lock_id}' acquired, running detection")
            results = await self.run_detection()
            return results
        finally:
            # Always release lock
            await self.metadata.release_lock(lock_id, holder)
            self.logger.info(f"Lock '{lock_id}' released")

    async def _collect_counts(self, partitions: list[PartitionInfo]) -> dict[str, int]:
        """Count documents per partition.

        Args:
            partitions: List of PartitionInfo objects

        Returns:
            Dict mapping partition name to document count
        """
        counts = {}

        for partition in partitions:
            try:
                collection_name = partition.search_collection or partition.view_name
                filter_expr = self._build_filter(partition)
                count = await self.backend.count_documents(collection_name, filter_expr)
                counts[partition.name] = count
                self.logger.debug(f"Partition '{partition.name}': {count} documents")
            except Exception as e:
                self.logger.error(f"Error counting partition '{partition.name}': {e}")
                counts[partition.name] = 0

        return counts

    def _build_filter(self, partition: PartitionInfo) -> dict:
        """Build filter expression for counting partition documents.

        Args:
            partition: PartitionInfo object

        Returns:
            Filter dict for count_documents
        """
        # For VIEWS mode, the view already filters, so use empty filter
        from semantic_vector_router.models import IndexLocation
        if self.config.vector_storage.index_on == IndexLocation.VIEWS:
            return {}

        # For SOURCE/FIELDS modes, filter by partition field
        if partition.filter_value is not None:
            return {self.config.partitioning.field: partition.filter_value}

        # Root partition or no filter
        return {}

    async def _store_health_data(self, counts: dict[str, int]) -> None:
        """Append counts to health history.

        Args:
            counts: Dict mapping partition name to count
        """
        for partition_name, count in counts.items():
            await self.metadata.append_health_history(
                partition_name,
                count
            )

    async def _check_threshold_breaches(
        self,
        partitions: list[PartitionInfo],
        counts: dict[str, int]
    ) -> list[DetectionResult]:
        """Check if any partition exceeds threshold.

        Args:
            partitions: List of PartitionInfo objects
            counts: Dict mapping partition name to count

        Returns:
            List of DetectionResult for breached partitions
        """
        results = []
        threshold = self.config.lifecycle.detection.threshold_vectors

        for partition in partitions:
            count = counts.get(partition.name, 0)
            if count > threshold:
                self.logger.warning(
                    f"Partition '{partition.name}' breached threshold: "
                    f"{count} > {threshold}"
                )
                results.append(DetectionResult(
                    signal=DetectionSignal.THRESHOLD_BREACH,
                    partition=partition.name,
                    details={
                        "count": count,
                        "threshold": threshold,
                        "overage": count - threshold
                    },
                    auto_executable=True,
                    suggested_action=f"split-{partition.name}"
                ))

        return results

    async def _check_approaching_thresholds(
        self,
        partitions: list[PartitionInfo]
    ) -> list[DetectionResult]:
        """Check if any partition is approaching threshold based on trend.

        Args:
            partitions: List of PartitionInfo objects

        Returns:
            List of DetectionResult for partitions approaching threshold
        """
        results = []
        threshold = self.config.lifecycle.detection.threshold_vectors
        trend_window_days = self.config.lifecycle.detection.trend_window_days

        for partition in partitions:
            # Get health history
            history = await self.metadata.get_health_history(partition.name)

            # Need at least 3 data points for trend analysis
            if len(history) < 3:
                continue

            # Calculate trend slope (vectors per second)
            slope_per_second = self._calculate_trend_slope(history)

            # Convert to per-day
            slope_per_day = slope_per_second * 86400

            # Skip if not growing
            if slope_per_day <= 0:
                continue

            # Get current count
            current_count = history[-1]["count"]

            # Calculate days to breach
            vectors_to_breach = threshold - current_count
            if vectors_to_breach <= 0:
                continue  # Already breached, handled by _check_threshold_breaches

            days_to_breach = vectors_to_breach / slope_per_day

            # Check if breach is within trend window
            if days_to_breach <= trend_window_days:
                self.logger.info(
                    f"Partition '{partition.name}' approaching threshold: "
                    f"will breach in {days_to_breach:.1f} days"
                )
                results.append(DetectionResult(
                    signal=DetectionSignal.APPROACHING_THRESHOLD,
                    partition=partition.name,
                    details={
                        "current_count": current_count,
                        "threshold": threshold,
                        "days_to_breach": days_to_breach,
                        "growth_rate_per_day": slope_per_day
                    },
                    auto_executable=False,
                    suggested_action=f"prepare-split-{partition.name}"
                ))

        return results

    async def _check_skew(
        self,
        partitions: list[PartitionInfo],
        counts: dict[str, int]
    ) -> list[DetectionResult]:
        """Check for skew among sibling partitions.

        Args:
            partitions: List of PartitionInfo objects
            counts: Dict mapping partition name to count

        Returns:
            List of DetectionResult for skewed partition groups
        """
        results = []
        skew_ratio = self.config.lifecycle.detection.skew_ratio

        # Group partitions by parent
        groups: dict[str, list[PartitionInfo]] = {}
        for partition in partitions:
            parent = partition.parent_partition or "_root"
            if parent not in groups:
                groups[parent] = []
            groups[parent].append(partition)

        # Check each group with 2+ siblings
        for parent, siblings in groups.items():
            if len(siblings) < 2:
                continue

            # Get counts for siblings
            sibling_counts = [counts.get(s.name, 0) for s in siblings]

            # Skip if any count is 0
            if any(c == 0 for c in sibling_counts):
                continue

            # Calculate max/avg ratio
            max_count = max(sibling_counts)
            avg_count = sum(sibling_counts) / len(sibling_counts)
            ratio = max_count / avg_count if avg_count > 0 else 0

            if ratio > skew_ratio:
                # Find the skewed partition
                max_partition = siblings[sibling_counts.index(max_count)]

                self.logger.warning(
                    f"Skew detected in group with parent '{parent}': "
                    f"max/avg ratio = {ratio:.2f}"
                )
                results.append(DetectionResult(
                    signal=DetectionSignal.SEVERE_SKEW,
                    partition=max_partition.name,
                    details={
                        "max_count": max_count,
                        "avg_count": avg_count,
                        "ratio": ratio,
                        "threshold_ratio": skew_ratio,
                        "parent": parent,
                        "sibling_counts": dict(zip([s.name for s in siblings], sibling_counts))
                    },
                    auto_executable=False,
                    suggested_action=f"rebalance-{parent}"
                ))

        return results

    async def _check_underpopulated(
        self,
        partitions: list[PartitionInfo],
        counts: dict[str, int]
    ) -> list[DetectionResult]:
        """Check for underpopulated partitions.

        Args:
            partitions: List of PartitionInfo objects
            counts: Dict mapping partition name to count

        Returns:
            List of DetectionResult for underpopulated partitions
        """
        results = []
        min_threshold = self.config.lifecycle.detection.min_threshold_vectors

        for partition in partitions:
            count = counts.get(partition.name, 0)

            # Skip root partition or partitions without children
            # (we only care about leaf partitions that could be merged)
            has_children = any(
                p.parent_partition == partition.name for p in partitions
            )
            if has_children:
                continue

            if count < min_threshold:
                self.logger.info(
                    f"Partition '{partition.name}' is underpopulated: "
                    f"{count} < {min_threshold}"
                )
                results.append(DetectionResult(
                    signal=DetectionSignal.UNDERPOPULATED,
                    partition=partition.name,
                    details={
                        "count": count,
                        "min_threshold": min_threshold,
                        "shortfall": min_threshold - count
                    },
                    auto_executable=False,
                    suggested_action=f"merge-{partition.name}"
                ))

        return results

    async def _check_stale(self, partitions: list[PartitionInfo]) -> list[DetectionResult]:
        """Check for stale partitions (no growth).

        Args:
            partitions: List of PartitionInfo objects

        Returns:
            List of DetectionResult for stale partitions
        """
        results = []

        for partition in partitions:
            # Get health history
            history = await self.metadata.get_health_history(partition.name)

            # Need at least 10 data points
            if len(history) < 10:
                continue

            # Get recent counts
            recent_counts = [h["count"] for h in history[-10:]]

            # Check if all counts are the same
            if len(set(recent_counts)) == 1:
                self.logger.info(
                    f"Partition '{partition.name}' is stale: "
                    f"no growth in last {len(recent_counts)} measurements"
                )
                results.append(DetectionResult(
                    signal=DetectionSignal.STALE,
                    partition=partition.name,
                    details={
                        "count": recent_counts[0],
                        "measurements": len(recent_counts),
                        "history_points": len(history)
                    },
                    auto_executable=False,
                    suggested_action=f"archive-{partition.name}"
                ))

        return results

    def _calculate_trend_slope(self, history: list[dict]) -> float:
        """Calculate trend slope using linear regression.

        Args:
            history: List of dicts with 'ts' (datetime) and 'count' (int)

        Returns:
            Slope (vectors per second)
        """
        # Convert timestamps to seconds since first measurement
        first_ts = history[0]["ts"]
        x_values = [(h["ts"] - first_ts).total_seconds() for h in history]
        y_values = [h["count"] for h in history]

        # Try numpy first
        try:
            import numpy as np
            x = np.array(x_values)
            y = np.array(y_values)
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return float(slope)
        except ImportError:
            pass

        # Fallback to manual least squares
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    async def _create_operations(self, results: list[DetectionResult]) -> None:
        """Create operation documents for auto-executable results.

        Args:
            results: List of DetectionResult objects
        """
        for result in results:
            if not result.auto_executable:
                continue

            timestamp = datetime.utcnow().isoformat()
            operation_id = f"op:split-{result.partition}-{timestamp}"

            # Build operation document
            operation = {
                "_id": operation_id,
                "type": "operation",
                "action": "split",
                "partition": result.partition,
                "signal": result.signal.value,
                "created_at": timestamp,
                "status": "pending",
                "details": result.details,
                "steps": [
                    {
                        "step": 1,
                        "action": "lock_partition",
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None
                    },
                    {
                        "step": 2,
                        "action": "compute_split_strategy",
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None
                    },
                    {
                        "step": 3,
                        "action": "create_child_partitions",
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None
                    },
                    {
                        "step": 4,
                        "action": "create_indexes",
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None
                    },
                    {
                        "step": 5,
                        "action": "mark_parent_readonly",
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None
                    }
                ]
            }

            try:
                await self.metadata.create_operation(operation)
                self.logger.info(f"Created operation '{operation_id}' for partition '{result.partition}'")
            except Exception as e:
                self.logger.error(f"Error creating operation for '{result.partition}': {e}")
