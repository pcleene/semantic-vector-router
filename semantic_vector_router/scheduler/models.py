"""Models for the job scheduler."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobType(str, Enum):
    """Types of scheduled jobs."""
    DETECTION = "detection"
    CENTROID_REFRESH = "centroid_refresh"
    PARTITION_COUNT_UPDATE = "partition_count_update"
    REPARTITION = "repartition"
    INDEX_HEALTH_CHECK = "index_health_check"
    CUSTOM = "custom"


class MaintenanceWindow(BaseModel):
    """Time window when operations are allowed to run."""
    allowed_days: list[str] = Field(
        default_factory=lambda: [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        ]
    )
    allowed_hours: dict[str, int] = Field(
        default_factory=lambda: {"start": 0, "end": 24}
    )
    timezone: str = "UTC"


class JobConfig(BaseModel):
    """Configuration for a scheduled job."""
    job_id: str
    job_type: JobType
    interval: str  # "1h", "6h", "daily", "weekly"
    enabled: bool = True
    maintenance_window: Optional[MaintenanceWindow] = None
    lock_id: Optional[str] = None
    lock_ttl_seconds: int = 600
    timeout_seconds: int = 300
    retry_on_failure: bool = True
    max_retries: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobStatus(str, Enum):
    """Status of a job run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobRun(BaseModel):
    """Record of a single job execution."""
    job_id: str
    job_type: str
    status: JobStatus
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    result: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    skipped_reason: Optional[str] = None
    worker_id: Optional[str] = None


class JobState(BaseModel):
    """Persistent state for a job (survives restarts)."""
    job_id: str
    last_run_at: Optional[datetime] = None
    last_status: Optional[JobStatus] = None
    next_due_at: Optional[datetime] = None
    consecutive_failures: int = 0


class SchedulerStatus(BaseModel):
    """Overall scheduler status."""
    running: bool
    worker_id: str
    jobs_registered: int
    tick_interval_seconds: int
    last_tick_at: Optional[datetime] = None
    jobs: list[dict[str, Any]] = Field(default_factory=list)
