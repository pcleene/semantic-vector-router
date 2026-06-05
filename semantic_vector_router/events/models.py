"""Event models for the SVR event system."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol


class SVREventType(str, Enum):
    """Types of events emitted by SVR."""

    # Partition lifecycle
    PARTITION_CREATED = "partition.created"
    PARTITION_DELETED = "partition.deleted"
    PARTITION_SPLIT_STARTED = "partition.split.started"
    PARTITION_SPLIT_COMPLETED = "partition.split.completed"
    PARTITION_RETIRED = "partition.retired"

    # Health
    HEALTH_ALERT = "health.alert"
    HEALTH_THRESHOLD_BREACH = "health.threshold_breach"
    HEALTH_APPROACHING_THRESHOLD = "health.approaching_threshold"
    HEALTH_SKEW_DETECTED = "health.skew_detected"

    # Operations
    REPARTITION_STARTED = "repartition.started"
    REPARTITION_COMPLETED = "repartition.completed"
    REPARTITION_FAILED = "repartition.failed"
    REPARTITION_ROLLED_BACK = "repartition.rolled_back"

    # Centroid
    CENTROID_COMPUTED = "centroid.computed"
    CENTROID_REFRESH_COMPLETED = "centroid.refresh.completed"

    # Ingestion
    INGEST_COMPLETED = "ingest.completed"
    INGEST_FAILED = "ingest.failed"

    # Scheduler
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_SKIPPED = "job.skipped"


class SVREvent:
    """Structured event emitted by SVR.

    Uses __slots__ for efficiency since many events may be created.
    """

    __slots__ = (
        "event_type", "timestamp", "partition", "operation_id",
        "job_id", "details", "severity", "worker_id",
    )

    def __init__(
        self,
        event_type: SVREventType,
        timestamp: Optional[datetime] = None,
        partition: Optional[str] = None,
        operation_id: Optional[str] = None,
        job_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        severity: str = "info",
        worker_id: Optional[str] = None,
    ):
        self.event_type = event_type
        self.timestamp = timestamp or datetime.utcnow()
        self.partition = partition
        self.operation_id = operation_id
        self.job_id = job_id
        self.details = details or {}
        self.severity = severity
        self.worker_id = worker_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary (for webhook payload)."""
        d: dict[str, Any] = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat() + "Z" if self.timestamp else None,
            "severity": self.severity,
        }
        if self.partition is not None:
            d["partition"] = self.partition
        if self.operation_id is not None:
            d["operation_id"] = self.operation_id
        if self.job_id is not None:
            d["job_id"] = self.job_id
        if self.details:
            d["details"] = self.details
        if self.worker_id is not None:
            d["worker_id"] = self.worker_id
        return d


class EventHandler(Protocol):
    """Protocol for event subscribers."""

    async def handle_event(self, event: SVREvent) -> None: ...
