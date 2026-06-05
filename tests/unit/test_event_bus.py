"""Comprehensive unit tests for the EventBus."""

import asyncio
import logging
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from semantic_vector_router.events.bus import EventBus
from semantic_vector_router.events.models import SVREvent, SVREventType, EventHandler


# ---------------------------------------------------------------------------
# Mock handlers
# ---------------------------------------------------------------------------

class MockHandler:
    """Handler that records every event it receives."""

    def __init__(self) -> None:
        self.events: list[SVREvent] = []

    async def handle_event(self, event: SVREvent) -> None:
        self.events.append(event)


class FailingHandler:
    """Handler that always raises."""

    async def handle_event(self, event: SVREvent) -> None:
        raise RuntimeError("handler error")


class SlowHandler:
    """Handler that takes a while to complete."""

    def __init__(self) -> None:
        self.events: list[SVREvent] = []

    async def handle_event(self, event: SVREvent) -> None:
        await asyncio.sleep(0.02)
        self.events.append(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: SVREventType = SVREventType.PARTITION_CREATED,
    **kwargs,
) -> SVREvent:
    return SVREvent(event_type=event_type, **kwargs)


SETTLE = 0.05  # seconds — let fire-and-forget tasks land


# ---------------------------------------------------------------------------
# 1. Subscribe to specific type, emit matching -> handler called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscribe_specific_type_matching_event():
    bus = EventBus()
    handler = MockHandler()
    bus.subscribe(SVREventType.PARTITION_CREATED, handler)

    event = _make_event(SVREventType.PARTITION_CREATED)
    await bus.emit(event)
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 1
    assert handler.events[0] is event


# ---------------------------------------------------------------------------
# 2. Subscribe to specific type, emit different -> handler NOT called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscribe_specific_type_non_matching_event():
    bus = EventBus()
    handler = MockHandler()
    bus.subscribe(SVREventType.PARTITION_CREATED, handler)

    await bus.emit(_make_event(SVREventType.PARTITION_DELETED))
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 0


# ---------------------------------------------------------------------------
# 3. Subscribe all -> handler called for any event type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscribe_all_receives_every_event_type():
    bus = EventBus()
    handler = MockHandler()
    bus.subscribe_all(handler)

    types = [SVREventType.PARTITION_CREATED, SVREventType.JOB_COMPLETED, SVREventType.HEALTH_ALERT]
    for t in types:
        await bus.emit(_make_event(t))
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 3
    received_types = [e.event_type for e in handler.events]
    assert received_types == types


# ---------------------------------------------------------------------------
# 4. Multiple handlers on same event -> all called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_handlers_same_event_type():
    bus = EventBus()
    h1, h2, h3 = MockHandler(), MockHandler(), MockHandler()
    bus.subscribe(SVREventType.JOB_COMPLETED, h1)
    bus.subscribe(SVREventType.JOB_COMPLETED, h2)
    bus.subscribe(SVREventType.JOB_COMPLETED, h3)

    event = _make_event(SVREventType.JOB_COMPLETED)
    await bus.emit(event)
    await asyncio.sleep(SETTLE)

    for h in (h1, h2, h3):
        assert len(h.events) == 1
        assert h.events[0] is event


# ---------------------------------------------------------------------------
# 5. Unsubscribe -> handler no longer called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsubscribe_removes_handler():
    bus = EventBus()
    handler = MockHandler()
    bus.subscribe(SVREventType.HEALTH_ALERT, handler)

    # First emit — handler should see it
    await bus.emit(_make_event(SVREventType.HEALTH_ALERT))
    await asyncio.sleep(SETTLE)
    assert len(handler.events) == 1

    # Unsubscribe, second emit — handler should NOT see it
    bus.unsubscribe(SVREventType.HEALTH_ALERT, handler)
    await bus.emit(_make_event(SVREventType.HEALTH_ALERT))
    await asyncio.sleep(SETTLE)
    assert len(handler.events) == 1


# ---------------------------------------------------------------------------
# 6. Unsubscribe_all -> handler removed from everything
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsubscribe_all_removes_from_global_and_specific():
    bus = EventBus()
    handler = MockHandler()
    bus.subscribe_all(handler)
    bus.subscribe(SVREventType.PARTITION_CREATED, handler)
    bus.subscribe(SVREventType.JOB_FAILED, handler)

    bus.unsubscribe_all(handler)

    await bus.emit(_make_event(SVREventType.PARTITION_CREATED))
    await bus.emit(_make_event(SVREventType.JOB_FAILED))
    await bus.emit(_make_event(SVREventType.HEALTH_ALERT))
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 0


# ---------------------------------------------------------------------------
# 7. Handler exception is caught and logged, doesn't propagate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_exception_caught_and_logged(caplog):
    bus = EventBus()
    failing = FailingHandler()
    bus.subscribe(SVREventType.INGEST_FAILED, failing)

    with caplog.at_level(logging.WARNING, logger="semantic_vector_router.events"):
        await bus.emit(_make_event(SVREventType.INGEST_FAILED))
        await asyncio.sleep(SETTLE)

    # No exception propagated (test would fail if it did)
    assert any("FailingHandler" in rec.message and "handler error" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 8. Handler exception doesn't prevent other handlers from being called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_exception_does_not_block_other_handlers():
    bus = EventBus()
    before = MockHandler()
    failing = FailingHandler()
    after = MockHandler()

    bus.subscribe(SVREventType.REPARTITION_STARTED, before)
    bus.subscribe(SVREventType.REPARTITION_STARTED, failing)
    bus.subscribe(SVREventType.REPARTITION_STARTED, after)

    await bus.emit(_make_event(SVREventType.REPARTITION_STARTED))
    await asyncio.sleep(SETTLE)

    assert len(before.events) == 1
    assert len(after.events) == 1


# ---------------------------------------------------------------------------
# 9. Duplicate subscribe is idempotent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_subscribe_is_idempotent():
    bus = EventBus()
    handler = MockHandler()

    bus.subscribe(SVREventType.PARTITION_CREATED, handler)
    bus.subscribe(SVREventType.PARTITION_CREATED, handler)
    bus.subscribe(SVREventType.PARTITION_CREATED, handler)

    await bus.emit(_make_event(SVREventType.PARTITION_CREATED))
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 1


@pytest.mark.asyncio
async def test_duplicate_subscribe_all_is_idempotent():
    bus = EventBus()
    handler = MockHandler()

    bus.subscribe_all(handler)
    bus.subscribe_all(handler)

    await bus.emit(_make_event(SVREventType.JOB_STARTED))
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 1


# ---------------------------------------------------------------------------
# 10. Emit with no handlers -> no error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_no_handlers_no_error():
    bus = EventBus()
    # Should not raise
    await bus.emit(_make_event(SVREventType.PARTITION_DELETED))
    await asyncio.sleep(SETTLE)


# ---------------------------------------------------------------------------
# 11. Event to_dict() serialization includes all fields
# ---------------------------------------------------------------------------

def test_event_to_dict_includes_all_fields():
    ts = datetime(2025, 5, 1, 12, 0, 0)
    event = SVREvent(
        event_type=SVREventType.HEALTH_ALERT,
        timestamp=ts,
        partition="my-part",
        operation_id="op-123",
        job_id="job-456",
        details={"count": 42},
        severity="warning",
        worker_id="w-1",
    )
    d = event.to_dict()

    assert d["event_type"] == "health.alert"
    assert d["timestamp"] == "2025-05-01T12:00:00Z"
    assert d["partition"] == "my-part"
    assert d["operation_id"] == "op-123"
    assert d["job_id"] == "job-456"
    assert d["details"] == {"count": 42}
    assert d["severity"] == "warning"
    assert d["worker_id"] == "w-1"


# ---------------------------------------------------------------------------
# 12. Event to_dict() omits None fields
# ---------------------------------------------------------------------------

def test_event_to_dict_omits_none_fields():
    event = SVREvent(event_type=SVREventType.JOB_COMPLETED)
    d = event.to_dict()

    assert "partition" not in d
    assert "operation_id" not in d
    assert "job_id" not in d
    assert "worker_id" not in d
    # details should be omitted when empty dict
    assert "details" not in d
    # These are always present
    assert "event_type" in d
    assert "timestamp" in d
    assert "severity" in d


# ---------------------------------------------------------------------------
# 13. SVREvent default timestamp is set
# ---------------------------------------------------------------------------

def test_event_default_timestamp_is_set():
    before = datetime.utcnow()
    event = SVREvent(event_type=SVREventType.PARTITION_CREATED)
    after = datetime.utcnow()

    assert event.timestamp is not None
    assert before <= event.timestamp <= after


# ---------------------------------------------------------------------------
# 14. SVREvent severity defaults to "info"
# ---------------------------------------------------------------------------

def test_event_severity_defaults_to_info():
    event = SVREvent(event_type=SVREventType.PARTITION_CREATED)
    assert event.severity == "info"


# ---------------------------------------------------------------------------
# 15. SVREvent details defaults to empty dict
# ---------------------------------------------------------------------------

def test_event_details_defaults_to_empty_dict():
    event = SVREvent(event_type=SVREventType.PARTITION_CREATED)
    assert event.details == {}


# ---------------------------------------------------------------------------
# 16. Unsubscribe handler that was never subscribed -> no error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsubscribe_unknown_handler_no_error():
    bus = EventBus()
    handler = MockHandler()
    # Should not raise
    bus.unsubscribe(SVREventType.PARTITION_CREATED, handler)


@pytest.mark.asyncio
async def test_unsubscribe_all_unknown_handler_no_error():
    bus = EventBus()
    handler = MockHandler()
    # Should not raise
    bus.unsubscribe_all(handler)


# ---------------------------------------------------------------------------
# 17. Global handler AND specific handler both receive event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_global_and_specific_handlers_both_called():
    bus = EventBus()
    specific = MockHandler()
    global_h = MockHandler()

    bus.subscribe(SVREventType.INGEST_COMPLETED, specific)
    bus.subscribe_all(global_h)

    event = _make_event(SVREventType.INGEST_COMPLETED)
    await bus.emit(event)
    await asyncio.sleep(SETTLE)

    assert len(specific.events) == 1
    assert len(global_h.events) == 1
    assert specific.events[0] is event
    assert global_h.events[0] is event


# ---------------------------------------------------------------------------
# 18. Same handler subscribed to specific AND global -> called twice
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_handler_specific_and_global_called_twice():
    bus = EventBus()
    handler = MockHandler()

    bus.subscribe(SVREventType.PARTITION_CREATED, handler)
    bus.subscribe_all(handler)

    await bus.emit(_make_event(SVREventType.PARTITION_CREATED))
    await asyncio.sleep(SETTLE)

    # Called once via specific, once via global
    assert len(handler.events) == 2


# ---------------------------------------------------------------------------
# 19. Multiple event types with separate handlers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_separate_handlers_for_different_types():
    bus = EventBus()
    h_create = MockHandler()
    h_delete = MockHandler()

    bus.subscribe(SVREventType.PARTITION_CREATED, h_create)
    bus.subscribe(SVREventType.PARTITION_DELETED, h_delete)

    await bus.emit(_make_event(SVREventType.PARTITION_CREATED))
    await bus.emit(_make_event(SVREventType.PARTITION_DELETED))
    await asyncio.sleep(SETTLE)

    assert len(h_create.events) == 1
    assert h_create.events[0].event_type == SVREventType.PARTITION_CREATED
    assert len(h_delete.events) == 1
    assert h_delete.events[0].event_type == SVREventType.PARTITION_DELETED


# ---------------------------------------------------------------------------
# 20. Emit multiple events in succession
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_multiple_events_in_succession():
    bus = EventBus()
    handler = MockHandler()
    bus.subscribe(SVREventType.JOB_STARTED, handler)

    for i in range(10):
        await bus.emit(_make_event(SVREventType.JOB_STARTED, details={"i": i}))
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 10
    for i, ev in enumerate(handler.events):
        assert ev.details["i"] == i


# ---------------------------------------------------------------------------
# 21. Fire-and-forget: emit does not wait for slow handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_does_not_block_on_slow_handler():
    bus = EventBus()
    slow = SlowHandler()
    bus.subscribe(SVREventType.PARTITION_CREATED, slow)

    t0 = asyncio.get_event_loop().time()
    await bus.emit(_make_event(SVREventType.PARTITION_CREATED))
    elapsed = asyncio.get_event_loop().time() - t0

    # emit should return almost instantly (< 10ms), not wait the 20ms of SlowHandler
    assert elapsed < 0.01

    # But after settling, the handler should have received the event
    await asyncio.sleep(SETTLE)
    assert len(slow.events) == 1


# ---------------------------------------------------------------------------
# 22. SVREventType enum values are strings
# ---------------------------------------------------------------------------

def test_event_type_enum_values_are_strings():
    for member in SVREventType:
        assert isinstance(member.value, str)
        assert "." in member.value  # All follow "category.action" pattern


# ---------------------------------------------------------------------------
# 23. SVREvent stores custom severity
# ---------------------------------------------------------------------------

def test_event_custom_severity():
    event = SVREvent(event_type=SVREventType.HEALTH_ALERT, severity="critical")
    assert event.severity == "critical"
    assert event.to_dict()["severity"] == "critical"


# ---------------------------------------------------------------------------
# 24. to_dict timestamp ends with Z suffix
# ---------------------------------------------------------------------------

def test_to_dict_timestamp_ends_with_z():
    event = SVREvent(
        event_type=SVREventType.PARTITION_CREATED,
        timestamp=datetime(2025, 1, 15, 8, 30, 0),
    )
    d = event.to_dict()
    assert d["timestamp"].endswith("Z")


# ---------------------------------------------------------------------------
# 25. Unsubscribe from specific type leaves other types intact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsubscribe_specific_type_leaves_other_types():
    bus = EventBus()
    handler = MockHandler()
    bus.subscribe(SVREventType.PARTITION_CREATED, handler)
    bus.subscribe(SVREventType.PARTITION_DELETED, handler)

    bus.unsubscribe(SVREventType.PARTITION_CREATED, handler)

    await bus.emit(_make_event(SVREventType.PARTITION_CREATED))
    await bus.emit(_make_event(SVREventType.PARTITION_DELETED))
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 1
    assert handler.events[0].event_type == SVREventType.PARTITION_DELETED


# ---------------------------------------------------------------------------
# 26. EventBus starts with empty handler state
# ---------------------------------------------------------------------------

def test_event_bus_starts_empty():
    bus = EventBus()
    assert bus._handlers == {}
    assert bus._global_handlers == []


# ---------------------------------------------------------------------------
# 27. SVREvent stores all constructor arguments
# ---------------------------------------------------------------------------

def test_event_stores_all_fields():
    ts = datetime(2025, 6, 1)
    event = SVREvent(
        event_type=SVREventType.REPARTITION_COMPLETED,
        timestamp=ts,
        partition="p1",
        operation_id="op-1",
        job_id="j-1",
        details={"key": "val"},
        severity="warning",
        worker_id="w-5",
    )
    assert event.event_type == SVREventType.REPARTITION_COMPLETED
    assert event.timestamp is ts
    assert event.partition == "p1"
    assert event.operation_id == "op-1"
    assert event.job_id == "j-1"
    assert event.details == {"key": "val"}
    assert event.severity == "warning"
    assert event.worker_id == "w-5"


# ---------------------------------------------------------------------------
# 28. Handler subscribed to multiple types receives only matching events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_multiple_subscriptions():
    bus = EventBus()
    handler = MockHandler()

    bus.subscribe(SVREventType.JOB_STARTED, handler)
    bus.subscribe(SVREventType.JOB_COMPLETED, handler)

    await bus.emit(_make_event(SVREventType.JOB_STARTED))
    await bus.emit(_make_event(SVREventType.JOB_COMPLETED))
    await bus.emit(_make_event(SVREventType.JOB_FAILED))  # not subscribed
    await asyncio.sleep(SETTLE)

    assert len(handler.events) == 2
    types = {e.event_type for e in handler.events}
    assert types == {SVREventType.JOB_STARTED, SVREventType.JOB_COMPLETED}


# ---------------------------------------------------------------------------
# 29. _safe_handle logs with exc_info on failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_handle_logs_with_exc_info(caplog):
    bus = EventBus()
    failing = FailingHandler()

    with caplog.at_level(logging.WARNING, logger="semantic_vector_router.events"):
        await bus._safe_handle(failing, _make_event(SVREventType.HEALTH_ALERT))

    assert len(caplog.records) >= 1
    record = caplog.records[0]
    assert record.exc_info is not None
    assert "FailingHandler" in record.message
    assert "health.alert" in record.message


# ---------------------------------------------------------------------------
# 30. Unsubscribe_all removes from global but handler still registered for
#     specific types is also removed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsubscribe_all_is_thorough():
    bus = EventBus()
    handler = MockHandler()

    bus.subscribe_all(handler)
    bus.subscribe(SVREventType.CENTROID_COMPUTED, handler)
    bus.subscribe(SVREventType.INGEST_COMPLETED, handler)

    bus.unsubscribe_all(handler)

    # Verify internal state is clean
    assert handler not in bus._global_handlers
    for handlers_list in bus._handlers.values():
        assert handler not in handlers_list


# ---------------------------------------------------------------------------
# 31. SVREvent with explicit empty details -> to_dict omits details
# ---------------------------------------------------------------------------

def test_to_dict_omits_empty_details():
    event = SVREvent(event_type=SVREventType.JOB_SKIPPED, details={})
    d = event.to_dict()
    assert "details" not in d


# ---------------------------------------------------------------------------
# 32. SVREvent with None details defaults to empty dict
# ---------------------------------------------------------------------------

def test_event_none_details_becomes_empty_dict():
    event = SVREvent(event_type=SVREventType.JOB_SKIPPED, details=None)
    assert event.details == {}


# ---------------------------------------------------------------------------
# 33. to_dict with non-empty details includes them
# ---------------------------------------------------------------------------

def test_to_dict_includes_non_empty_details():
    event = SVREvent(
        event_type=SVREventType.INGEST_COMPLETED,
        details={"docs": 100, "ms": 42},
    )
    d = event.to_dict()
    assert d["details"] == {"docs": 100, "ms": 42}
