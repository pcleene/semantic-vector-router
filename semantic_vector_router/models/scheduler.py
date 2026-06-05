"""Scheduler and events configuration models."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from semantic_vector_router.models.config import MaintenanceWindow


class WebhookConfig(BaseModel):
    """Webhook endpoint configuration."""

    url: str
    events: list[str] = Field(default_factory=list)
    secret: Optional[str] = None
    timeout_seconds: int = 10
    retry_count: int = 3
    retry_delay_seconds: float = 5.0
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class SchedulerConfig(BaseModel):
    """Job scheduler configuration."""

    enabled: bool = False
    tick_interval_seconds: int = 30
    worker_id: Optional[str] = None
    maintenance_window: Optional[MaintenanceWindow] = None
    detection_interval: Optional[str] = "1h"
    centroid_refresh_interval: Optional[str] = "6h"
    count_update_interval: Optional[str] = "1h"
    repartition_check_interval: Optional[str] = "30m"
    index_health_interval: Optional[str] = "6h"
    custom_jobs: list[dict[str, Any]] = Field(default_factory=list)


class EventsConfig(BaseModel):
    """Event system configuration."""

    enabled: bool = True
    webhooks: list[WebhookConfig] = Field(default_factory=list)
    log_events: bool = False
    store_events: bool = False
    event_retention_days: int = 30
