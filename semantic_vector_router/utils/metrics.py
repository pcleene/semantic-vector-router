"""Metrics hooks for the Semantic Vector Router SDK.

Provides a lightweight, extensible metrics collection system that can be
wired into any observability backend (Datadog, Prometheus, OTLP, etc.)
via the ``MetricsHandler`` protocol.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class MetricType(str, Enum):
    """Types of metrics the SDK emits."""

    SEARCH_LATENCY = "search_latency"
    SEARCH_PARTITION_LATENCY = "search_partition_latency"
    EMBEDDING_LATENCY = "embedding_latency"
    RERANKING_LATENCY = "reranking_latency"
    SEARCH_RESULTS = "search_results"
    SEARCH_CANDIDATES = "search_candidates"
    PARTITION_COUNT = "partition_count"
    DETECTION_RUN = "detection_run"
    INDEX_BUILD_TIME = "index_build_time"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR = "error"

    # Ingestion metrics
    INGEST_LATENCY = "ingest_latency"
    INGEST_DOCUMENTS = "ingest_documents"
    INGEST_ERRORS = "ingest_errors"
    INGEST_EMBED_LATENCY = "ingest_embed_latency"

    # Rate limiting metrics
    RATE_LIMIT_WAIT = "rate_limit_wait"
    RATE_LIMIT_ACQUIRED = "rate_limit_acquired"

    # Centroid routing metrics
    CENTROID_ROUTE_LATENCY = "centroid_route_latency"
    CENTROID_ROUTE_PARTITIONS = "centroid_route_partitions"


@dataclass(frozen=True)
class MetricEvent:
    """A single metric data point.

    Attributes:
        metric_type: The type of metric.
        value: The metric value (duration in ms, count, etc.).
        tags: Dimensional tags for filtering/grouping.
        timestamp: Unix timestamp (seconds). Defaults to now.
    """

    metric_type: MetricType
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsHandler(Protocol):
    """Protocol for metrics handler implementations."""

    def handle(self, event: MetricEvent) -> None: ...


class MetricsCollector:
    """Collects and dispatches metric events to registered handlers.

    Thread-safe and async-safe. Handlers are called synchronously
    (fire-and-forget) to avoid blocking the hot path.
    """

    def __init__(self) -> None:
        self._handlers: list[MetricsHandler] = []

    def add_handler(self, handler: MetricsHandler) -> None:
        """Register a metrics handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: MetricsHandler) -> None:
        """Unregister a metrics handler."""
        self._handlers.remove(handler)

    def emit(self, event: MetricEvent) -> None:
        """Emit a metric event to all registered handlers.

        Catches and logs exceptions from handlers to prevent
        metrics collection from breaking the hot path.
        """
        for handler in self._handlers:
            try:
                handler.handle(event)
            except Exception as e:
                logging.getLogger("semantic_vector_router.metrics").warning(
                    f"Metrics handler {type(handler).__name__} failed: {e}"
                )

    def emit_timing(
        self,
        metric_type: MetricType,
        duration_ms: float,
        **tags: str,
    ) -> None:
        """Convenience: emit a timing metric.

        Args:
            metric_type: The type of timing metric.
            duration_ms: Duration in milliseconds.
            **tags: Dimensional tags for filtering/grouping.
        """
        self.emit(MetricEvent(
            metric_type=metric_type,
            value=duration_ms,
            tags=tags,
        ))

    def emit_count(
        self,
        metric_type: MetricType,
        count: float = 1,
        **tags: str,
    ) -> None:
        """Convenience: emit a count metric.

        Args:
            metric_type: The type of count metric.
            count: The count value. Defaults to 1.
            **tags: Dimensional tags for filtering/grouping.
        """
        self.emit(MetricEvent(
            metric_type=metric_type,
            value=count,
            tags=tags,
        ))


class NoOpCollector(MetricsCollector):
    """No-op collector that discards all events. Used when no handler is configured."""

    def emit(self, event: MetricEvent) -> None:
        """Discard the event."""
        pass
