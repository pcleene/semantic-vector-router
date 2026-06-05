"""Unit tests for metrics hooks (utils/metrics.py)."""

import logging
import time

import pytest

from semantic_vector_router.utils.metrics import (
    MetricEvent,
    MetricType,
    MetricsCollector,
    MetricsHandler,
    NoOpCollector,
)


# ---------------------------------------------------------------------------
# Test handler
# ---------------------------------------------------------------------------


class RecordingHandler:
    """Test handler that records all received events."""

    def __init__(self):
        self.events: list[MetricEvent] = []

    def handle(self, event: MetricEvent) -> None:
        self.events.append(event)


class FailingHandler:
    """Test handler that raises on every event."""

    def handle(self, event: MetricEvent) -> None:
        raise RuntimeError("handler crash")


# ---------------------------------------------------------------------------
# MetricEvent
# ---------------------------------------------------------------------------


class TestMetricEvent:
    def test_is_frozen(self):
        event = MetricEvent(
            metric_type=MetricType.SEARCH_LATENCY,
            value=42.0,
        )
        with pytest.raises(AttributeError):
            event.value = 99.0

    def test_default_timestamp(self):
        before = time.time()
        event = MetricEvent(
            metric_type=MetricType.CACHE_HIT,
            value=1.0,
        )
        after = time.time()
        assert before <= event.timestamp <= after

    def test_default_empty_tags(self):
        event = MetricEvent(
            metric_type=MetricType.ERROR,
            value=1.0,
        )
        assert event.tags == {}

    def test_custom_tags(self):
        event = MetricEvent(
            metric_type=MetricType.EMBEDDING_LATENCY,
            value=150.0,
            tags={"provider": "voyage", "model": "voyage-4-lite"},
        )
        assert event.tags["provider"] == "voyage"


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_dispatch_to_handler(self):
        collector = MetricsCollector()
        handler = RecordingHandler()
        collector.add_handler(handler)

        event = MetricEvent(
            metric_type=MetricType.SEARCH_LATENCY,
            value=42.0,
        )
        collector.emit(event)

        assert len(handler.events) == 1
        assert handler.events[0] is event

    def test_dispatch_to_multiple_handlers(self):
        collector = MetricsCollector()
        h1 = RecordingHandler()
        h2 = RecordingHandler()
        collector.add_handler(h1)
        collector.add_handler(h2)

        event = MetricEvent(metric_type=MetricType.CACHE_MISS, value=1.0)
        collector.emit(event)

        assert len(h1.events) == 1
        assert len(h2.events) == 1
        assert h1.events[0] is h2.events[0]

    def test_no_handlers_no_crash(self):
        collector = MetricsCollector()
        event = MetricEvent(metric_type=MetricType.ERROR, value=1.0)
        collector.emit(event)  # Should not raise

    def test_handler_exception_swallowed(self):
        collector = MetricsCollector()
        failing = FailingHandler()
        recording = RecordingHandler()
        collector.add_handler(failing)
        collector.add_handler(recording)

        event = MetricEvent(metric_type=MetricType.SEARCH_LATENCY, value=10.0)

        # Emit should not raise despite failing handler
        collector.emit(event)

        # Failing handler doesn't prevent recording handler from receiving event
        assert len(recording.events) == 1

    def test_remove_handler(self):
        collector = MetricsCollector()
        handler = RecordingHandler()
        collector.add_handler(handler)
        collector.remove_handler(handler)

        event = MetricEvent(metric_type=MetricType.CACHE_HIT, value=1.0)
        collector.emit(event)

        assert len(handler.events) == 0


# ---------------------------------------------------------------------------
# emit_timing / emit_count convenience methods
# ---------------------------------------------------------------------------


class TestEmitConvenience:
    def test_emit_timing(self):
        collector = MetricsCollector()
        handler = RecordingHandler()
        collector.add_handler(handler)

        collector.emit_timing(
            MetricType.SEARCH_LATENCY,
            42.5,
            partitions="3",
            reranked="true",
        )

        assert len(handler.events) == 1
        event = handler.events[0]
        assert event.metric_type == MetricType.SEARCH_LATENCY
        assert event.value == 42.5
        assert event.tags == {"partitions": "3", "reranked": "true"}

    def test_emit_count_default(self):
        collector = MetricsCollector()
        handler = RecordingHandler()
        collector.add_handler(handler)

        collector.emit_count(MetricType.CACHE_HIT, provider="voyage")

        assert len(handler.events) == 1
        event = handler.events[0]
        assert event.metric_type == MetricType.CACHE_HIT
        assert event.value == 1
        assert event.tags == {"provider": "voyage"}

    def test_emit_count_custom_value(self):
        collector = MetricsCollector()
        handler = RecordingHandler()
        collector.add_handler(handler)

        collector.emit_count(MetricType.SEARCH_RESULTS, 25)

        assert handler.events[0].value == 25


# ---------------------------------------------------------------------------
# NoOpCollector
# ---------------------------------------------------------------------------


class TestNoOpCollector:
    def test_discards_events(self):
        collector = NoOpCollector()
        handler = RecordingHandler()
        collector.add_handler(handler)

        event = MetricEvent(metric_type=MetricType.SEARCH_LATENCY, value=42.0)
        collector.emit(event)

        # NoOpCollector overrides emit to do nothing, handler should not receive
        assert len(handler.events) == 0


# ---------------------------------------------------------------------------
# MetricType enum
# ---------------------------------------------------------------------------


class TestMetricType:
    def test_all_types_are_strings(self):
        for mt in MetricType:
            assert isinstance(mt.value, str)

    def test_expected_types_exist(self):
        expected = {
            "search_latency", "search_partition_latency", "embedding_latency",
            "reranking_latency", "search_results", "search_candidates",
            "partition_count", "detection_run", "index_build_time",
            "cache_hit", "cache_miss", "error",
            "ingest_latency", "ingest_documents", "ingest_errors",
            "ingest_embed_latency", "rate_limit_wait", "rate_limit_acquired",
            "centroid_route_latency", "centroid_route_partitions",
        }
        actual = {mt.value for mt in MetricType}
        assert expected == actual
