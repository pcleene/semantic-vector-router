"""Comprehensive unit tests for PartitionDetector."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.lifecycle.detector import PartitionDetector, DetectionResult
from semantic_vector_router.models import (
    DetectionSignal,
    IndexLocation,
    PartitionInfo,
    PartitionStatus,
    SVRConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metadata():
    """Create a fully mocked MetadataStore."""
    metadata = AsyncMock()
    metadata.list_partitions = AsyncMock(return_value=[])
    metadata.get_health_history = AsyncMock(return_value=[])
    metadata.append_health_history = AsyncMock()
    metadata.acquire_lock = AsyncMock(return_value=True)
    metadata.release_lock = AsyncMock(return_value=True)
    metadata.create_operation = AsyncMock()
    return metadata


@pytest.fixture
def sample_partitions():
    """Two active leaf partitions with different counts."""
    return [
        PartitionInfo(
            name="part_a",
            index_name="idx_a",
            filter_value="a",
            document_count=5000,
            status=PartitionStatus.ACTIVE,
        ),
        PartitionInfo(
            name="part_b",
            index_name="idx_b",
            filter_value="b",
            document_count=3000,
            status=PartitionStatus.ACTIVE,
        ),
    ]


@pytest.fixture
def detector(sample_config, mock_backend, mock_metadata):
    """Create a PartitionDetector with default sample_config."""
    return PartitionDetector(
        backend=mock_backend,
        metadata=mock_metadata,
        config=sample_config,
    )


# ---------------------------------------------------------------------------
# 1. Threshold breach detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_threshold_breach_detected(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """Partition count exceeding threshold produces THRESHOLD_BREACH result."""
    # Set a low threshold so our count exceeds it
    sample_config.lifecycle.detection.threshold_vectors = 10_000

    mock_metadata.list_partitions.return_value = sample_partitions
    # part_a returns 15000 (over threshold), part_b returns 5000 (under)
    mock_backend.count_documents = AsyncMock(side_effect=[15000, 5000])

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    breach_results = [
        r for r in results if r.signal == DetectionSignal.THRESHOLD_BREACH
    ]
    assert len(breach_results) == 1
    assert breach_results[0].partition == "part_a"
    assert breach_results[0].details["count"] == 15000
    assert breach_results[0].details["threshold"] == 10_000
    assert breach_results[0].details["overage"] == 5000
    assert breach_results[0].auto_executable is True
    assert "split" in breach_results[0].suggested_action


@pytest.mark.asyncio
async def test_no_breach_under_threshold(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """No THRESHOLD_BREACH when all counts are below the threshold."""
    sample_config.lifecycle.detection.threshold_vectors = 100_000

    mock_metadata.list_partitions.return_value = sample_partitions
    # Both partitions well under threshold
    mock_backend.count_documents = AsyncMock(side_effect=[5000, 3000])

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    breach_results = [
        r for r in results if r.signal == DetectionSignal.THRESHOLD_BREACH
    ]
    assert len(breach_results) == 0


# ---------------------------------------------------------------------------
# 2. Approaching threshold detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approaching_threshold_detected(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """Linear growth trend that predicts breach within trend_window triggers APPROACHING_THRESHOLD."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000
    sample_config.lifecycle.detection.trend_window_days = 30

    mock_metadata.list_partitions.return_value = [sample_partitions[0]]  # just part_a
    # Current count below threshold — no breach
    mock_backend.count_documents = AsyncMock(return_value=9000)

    # Build linear growth history: 5000 → 7000 → 9000 over 10 days
    now = datetime.utcnow()
    history = [
        {"ts": now - timedelta(days=10), "count": 5000},
        {"ts": now - timedelta(days=5), "count": 7000},
        {"ts": now, "count": 9000},
    ]
    mock_metadata.get_health_history.return_value = history

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    approach_results = [
        r for r in results if r.signal == DetectionSignal.APPROACHING_THRESHOLD
    ]
    assert len(approach_results) == 1
    r = approach_results[0]
    assert r.partition == "part_a"
    assert r.details["current_count"] == 9000
    assert r.details["threshold"] == 10_000
    assert r.details["days_to_breach"] > 0
    assert r.details["days_to_breach"] <= 30  # within trend window
    assert r.details["growth_rate_per_day"] > 0
    assert r.auto_executable is False
    assert "prepare-split" in r.suggested_action


@pytest.mark.asyncio
async def test_no_approaching_when_stable(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """Stable (flat) counts do not trigger APPROACHING_THRESHOLD."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000
    sample_config.lifecycle.detection.trend_window_days = 30

    mock_metadata.list_partitions.return_value = [sample_partitions[0]]
    mock_backend.count_documents = AsyncMock(return_value=5000)

    # Flat history — slope is zero
    now = datetime.utcnow()
    history = [
        {"ts": now - timedelta(days=10), "count": 5000},
        {"ts": now - timedelta(days=5), "count": 5000},
        {"ts": now, "count": 5000},
    ]
    mock_metadata.get_health_history.return_value = history

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    approach_results = [
        r for r in results if r.signal == DetectionSignal.APPROACHING_THRESHOLD
    ]
    assert len(approach_results) == 0


@pytest.mark.asyncio
async def test_approaching_needs_min_history(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """Fewer than 3 data points skips approaching threshold analysis."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000

    mock_metadata.list_partitions.return_value = [sample_partitions[0]]
    mock_backend.count_documents = AsyncMock(return_value=9000)

    # Only 2 history points — not enough
    now = datetime.utcnow()
    history = [
        {"ts": now - timedelta(days=5), "count": 7000},
        {"ts": now, "count": 9000},
    ]
    mock_metadata.get_health_history.return_value = history

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    approach_results = [
        r for r in results if r.signal == DetectionSignal.APPROACHING_THRESHOLD
    ]
    assert len(approach_results) == 0


# ---------------------------------------------------------------------------
# 3. Skew detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_severe_skew_detected(
    sample_config, mock_backend, mock_metadata
):
    """Sibling partitions with large max/avg ratio trigger SEVERE_SKEW."""
    sample_config.lifecycle.detection.skew_ratio = 3.0
    # Make threshold very high so we don't get breach results
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    # Set min_threshold low to avoid underpopulated results
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    # Siblings share the same parent_partition
    siblings = [
        PartitionInfo(
            name="child_a",
            index_name="idx_ca",
            filter_value="ca",
            document_count=90000,
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
        PartitionInfo(
            name="child_b",
            index_name="idx_cb",
            filter_value="cb",
            document_count=10000,
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
    ]
    mock_metadata.list_partitions.return_value = siblings
    # child_a=90000, child_b=10000 → avg=50000, max/avg=1.8 (not enough with ratio 3)
    # Increase skew: child_a=90000, child_b=5000 → avg=47500, max/avg≈1.89 (still not)
    # Use very skewed counts: child_a=100000, child_b=1000 → avg=50500, ratio≈1.98 ... still < 3
    # Actually: max/avg for 2 elements: if child_a=X, child_b=Y, avg=(X+Y)/2, ratio=X/avg=2X/(X+Y)
    # For ratio > 3: 2X/(X+Y)>3 → 2X>3X+3Y → -X>3Y ... impossible for positive values with 2 elements.
    # Need 3+ siblings to get ratio > 3:
    siblings = [
        PartitionInfo(
            name="child_a",
            index_name="idx_ca",
            filter_value="ca",
            document_count=0,
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
        PartitionInfo(
            name="child_b",
            index_name="idx_cb",
            filter_value="cb",
            document_count=0,
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
        PartitionInfo(
            name="child_c",
            index_name="idx_cc",
            filter_value="cc",
            document_count=0,
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
    ]
    mock_metadata.list_partitions.return_value = siblings
    # counts: child_a=90000, child_b=1000, child_c=1000
    # avg=30666.67, max/avg=90000/30666.67≈2.93 — still under 3
    # child_a=100000, child_b=1000, child_c=1000 → avg=34000, ratio≈2.94
    # child_a=100000, child_b=100, child_c=100 → avg=33400, ratio≈2.99
    # Set skew_ratio to 2.0 to be safe with 3 siblings
    sample_config.lifecycle.detection.skew_ratio = 2.0
    # counts: 90000, 1000, 1000 → avg=30666.67, max/avg=2.93 > 2.0
    mock_backend.count_documents = AsyncMock(side_effect=[90000, 1000, 1000])

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    skew_results = [r for r in results if r.signal == DetectionSignal.SEVERE_SKEW]
    assert len(skew_results) == 1
    r = skew_results[0]
    assert r.partition == "child_a"
    assert r.details["ratio"] > 2.0
    assert r.details["parent"] == "parent_x"
    assert "sibling_counts" in r.details
    assert r.auto_executable is False
    assert "rebalance" in r.suggested_action


@pytest.mark.asyncio
async def test_no_skew_when_balanced(
    sample_config, mock_backend, mock_metadata
):
    """Balanced sibling partitions produce no SEVERE_SKEW result."""
    sample_config.lifecycle.detection.skew_ratio = 5.0
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    siblings = [
        PartitionInfo(
            name="child_a",
            index_name="idx_ca",
            filter_value="ca",
            document_count=0,
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
        PartitionInfo(
            name="child_b",
            index_name="idx_cb",
            filter_value="cb",
            document_count=0,
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
    ]
    mock_metadata.list_partitions.return_value = siblings
    # Nearly equal counts: ratio ≈ 1.0
    mock_backend.count_documents = AsyncMock(side_effect=[5000, 5100])

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    skew_results = [r for r in results if r.signal == DetectionSignal.SEVERE_SKEW]
    assert len(skew_results) == 0


# ---------------------------------------------------------------------------
# 4. Underpopulated detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_underpopulated_detected(
    sample_config, mock_backend, mock_metadata
):
    """Partition count below min_threshold triggers UNDERPOPULATED."""
    sample_config.lifecycle.detection.min_threshold_vectors = 500
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000

    partition = PartitionInfo(
        name="tiny_part",
        index_name="idx_tiny",
        filter_value="tiny",
        document_count=0,
        status=PartitionStatus.ACTIVE,
    )
    mock_metadata.list_partitions.return_value = [partition]
    mock_backend.count_documents = AsyncMock(return_value=100)

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    underpop = [r for r in results if r.signal == DetectionSignal.UNDERPOPULATED]
    assert len(underpop) == 1
    r = underpop[0]
    assert r.partition == "tiny_part"
    assert r.details["count"] == 100
    assert r.details["min_threshold"] == 500
    assert r.details["shortfall"] == 400
    assert r.auto_executable is False
    assert "merge" in r.suggested_action


@pytest.mark.asyncio
async def test_parent_partitions_not_flagged(
    sample_config, mock_backend, mock_metadata
):
    """Partitions that have children are skipped for underpopulated check."""
    sample_config.lifecycle.detection.min_threshold_vectors = 5000
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000

    parent = PartitionInfo(
        name="parent_part",
        index_name="idx_parent",
        filter_value="parent",
        document_count=0,
        status=PartitionStatus.ACTIVE,
    )
    child = PartitionInfo(
        name="child_part",
        index_name="idx_child",
        filter_value="child",
        document_count=0,
        status=PartitionStatus.ACTIVE,
        parent_partition="parent_part",
    )
    mock_metadata.list_partitions.return_value = [parent, child]
    # parent has only 100 docs (would be underpopulated), child has 100 too
    mock_backend.count_documents = AsyncMock(side_effect=[100, 100])

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    underpop = [r for r in results if r.signal == DetectionSignal.UNDERPOPULATED]
    # Only child_part should be flagged, not parent_part (parent has children)
    assert len(underpop) == 1
    assert underpop[0].partition == "child_part"


# ---------------------------------------------------------------------------
# 5. Stale detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_detected(
    sample_config, mock_backend, mock_metadata
):
    """10+ identical counts in history triggers STALE."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    partition = PartitionInfo(
        name="frozen_part",
        index_name="idx_frozen",
        filter_value="frozen",
        document_count=0,
        status=PartitionStatus.ACTIVE,
    )
    mock_metadata.list_partitions.return_value = [partition]
    mock_backend.count_documents = AsyncMock(return_value=5000)

    # 10 identical history entries
    now = datetime.utcnow()
    stale_history = [
        {"ts": now - timedelta(days=10 - i), "count": 5000}
        for i in range(10)
    ]
    mock_metadata.get_health_history.return_value = stale_history

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    stale_results = [r for r in results if r.signal == DetectionSignal.STALE]
    assert len(stale_results) == 1
    r = stale_results[0]
    assert r.partition == "frozen_part"
    assert r.details["count"] == 5000
    assert r.details["measurements"] == 10
    assert r.auto_executable is False
    assert "archive" in r.suggested_action


@pytest.mark.asyncio
async def test_not_stale_with_growth(
    sample_config, mock_backend, mock_metadata
):
    """Changing counts in history do not trigger STALE."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    partition = PartitionInfo(
        name="growing_part",
        index_name="idx_growing",
        filter_value="growing",
        document_count=0,
        status=PartitionStatus.ACTIVE,
    )
    mock_metadata.list_partitions.return_value = [partition]
    mock_backend.count_documents = AsyncMock(return_value=6000)

    # 10 history entries with incremental growth
    now = datetime.utcnow()
    growing_history = [
        {"ts": now - timedelta(days=10 - i), "count": 5000 + i * 100}
        for i in range(10)
    ]
    mock_metadata.get_health_history.return_value = growing_history

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    stale_results = [r for r in results if r.signal == DetectionSignal.STALE]
    assert len(stale_results) == 0


# ---------------------------------------------------------------------------
# 6. run_detection pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_detection_empty(
    sample_config, mock_backend, mock_metadata
):
    """Empty partition list returns empty results."""
    mock_metadata.list_partitions.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    assert results == []
    # Should not have attempted counts or health history
    mock_backend.count_documents.assert_not_called()
    mock_metadata.append_health_history.assert_not_called()


@pytest.mark.asyncio
async def test_run_detection_full_pipeline(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """Verify COLLECT -> STORE -> ANALYZE -> DECIDE stages run in order."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    mock_metadata.list_partitions.return_value = sample_partitions
    # part_a: 15000 (breach), part_b: 5000 (ok)
    mock_backend.count_documents = AsyncMock(side_effect=[15000, 5000])
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    # COLLECT: count_documents called for each partition
    assert mock_backend.count_documents.call_count == 2

    # STORE: append_health_history called for each partition
    assert mock_metadata.append_health_history.call_count == 2
    mock_metadata.append_health_history.assert_any_call("part_a", 15000)
    mock_metadata.append_health_history.assert_any_call("part_b", 5000)

    # ANALYZE: at least one breach found
    breach_results = [
        r for r in results if r.signal == DetectionSignal.THRESHOLD_BREACH
    ]
    assert len(breach_results) == 1

    # DECIDE: create_operation called for auto_executable (breach)
    mock_metadata.create_operation.assert_called_once()
    op_doc = mock_metadata.create_operation.call_args[0][0]
    assert op_doc["action"] == "split"
    assert op_doc["partition"] == "part_a"
    assert op_doc["status"] == "pending"
    assert len(op_doc["steps"]) == 5


# ---------------------------------------------------------------------------
# 7. run_detection_with_lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detection_with_lock_acquired(
    sample_config, mock_backend, mock_metadata
):
    """When lock is acquired, detection runs and results are returned."""
    mock_metadata.acquire_lock.return_value = True
    mock_metadata.list_partitions.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection_with_lock()

    assert results is not None
    assert results == []
    mock_metadata.acquire_lock.assert_called_once()
    mock_metadata.release_lock.assert_called_once()


@pytest.mark.asyncio
async def test_detection_with_lock_not_acquired(
    sample_config, mock_backend, mock_metadata
):
    """When lock is not acquired, returns None without running detection."""
    mock_metadata.acquire_lock.return_value = False

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection_with_lock()

    assert results is None
    mock_metadata.acquire_lock.assert_called_once()
    # Detection not run, so list_partitions never called
    mock_metadata.list_partitions.assert_not_called()
    # Lock not acquired, so release not called
    mock_metadata.release_lock.assert_not_called()


@pytest.mark.asyncio
async def test_detection_with_lock_released_on_exception(
    sample_config, mock_backend, mock_metadata
):
    """Lock is released even when detection raises an exception."""
    mock_metadata.acquire_lock.return_value = True
    mock_metadata.list_partitions.side_effect = RuntimeError("boom")

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)

    with pytest.raises(RuntimeError, match="boom"):
        await detector.run_detection_with_lock()

    # Lock should still be released via finally block
    mock_metadata.release_lock.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Trend calculation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculate_trend_slope_positive(sample_config, mock_backend, mock_metadata):
    """Increasing counts produce a positive trend slope."""
    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)

    now = datetime.utcnow()
    history = [
        {"ts": now - timedelta(days=10), "count": 1000},
        {"ts": now - timedelta(days=5), "count": 2000},
        {"ts": now, "count": 3000},
    ]

    slope = detector._calculate_trend_slope(history)
    assert slope > 0

    # slope is per second; convert to per day
    slope_per_day = slope * 86400
    # Growth is ~200 per day (2000 over 10 days)
    assert 150 < slope_per_day < 250


@pytest.mark.asyncio
async def test_calculate_trend_slope_zero(sample_config, mock_backend, mock_metadata):
    """Flat counts produce a zero trend slope."""
    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)

    now = datetime.utcnow()
    history = [
        {"ts": now - timedelta(days=10), "count": 5000},
        {"ts": now - timedelta(days=5), "count": 5000},
        {"ts": now, "count": 5000},
    ]

    slope = detector._calculate_trend_slope(history)
    assert slope == pytest.approx(0.0, abs=1e-10)


@pytest.mark.asyncio
async def test_calculate_trend_slope_negative(sample_config, mock_backend, mock_metadata):
    """Decreasing counts produce a negative trend slope."""
    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)

    now = datetime.utcnow()
    history = [
        {"ts": now - timedelta(days=10), "count": 5000},
        {"ts": now - timedelta(days=5), "count": 3000},
        {"ts": now, "count": 1000},
    ]

    slope = detector._calculate_trend_slope(history)
    assert slope < 0


# ---------------------------------------------------------------------------
# 9. Operation creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creates_operations_for_auto_executable(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """THRESHOLD_BREACH (auto_executable=True) creates a pending operation."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000

    mock_metadata.list_partitions.return_value = [sample_partitions[0]]
    mock_backend.count_documents = AsyncMock(return_value=15000)
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    # Operation created for the breach
    mock_metadata.create_operation.assert_called_once()
    op_doc = mock_metadata.create_operation.call_args[0][0]
    assert op_doc["action"] == "split"
    assert op_doc["partition"] == "part_a"
    assert op_doc["signal"] == DetectionSignal.THRESHOLD_BREACH.value
    assert op_doc["status"] == "pending"
    assert "_id" in op_doc
    assert op_doc["_id"].startswith("op:split-part_a-")
    # Check steps structure
    assert len(op_doc["steps"]) == 5
    step_actions = [s["action"] for s in op_doc["steps"]]
    assert step_actions == [
        "lock_partition",
        "compute_split_strategy",
        "create_child_partitions",
        "create_indexes",
        "mark_parent_readonly",
    ]
    for step in op_doc["steps"]:
        assert step["status"] == "pending"


@pytest.mark.asyncio
async def test_skips_non_auto_executable(
    sample_config, mock_backend, mock_metadata
):
    """SEVERE_SKEW (auto_executable=False) does not create an operation."""
    sample_config.lifecycle.detection.skew_ratio = 2.0
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    siblings = [
        PartitionInfo(
            name="child_a",
            index_name="idx_ca",
            filter_value="ca",
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
        PartitionInfo(
            name="child_b",
            index_name="idx_cb",
            filter_value="cb",
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
        PartitionInfo(
            name="child_c",
            index_name="idx_cc",
            filter_value="cc",
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_x",
        ),
    ]
    mock_metadata.list_partitions.return_value = siblings
    # Highly skewed: child_a=90000, child_b=1000, child_c=1000
    mock_backend.count_documents = AsyncMock(side_effect=[90000, 1000, 1000])
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    # Should have skew results
    skew_results = [r for r in results if r.signal == DetectionSignal.SEVERE_SKEW]
    assert len(skew_results) >= 1
    assert skew_results[0].auto_executable is False

    # No operation created (skew is not auto_executable)
    mock_metadata.create_operation.assert_not_called()


@pytest.mark.asyncio
async def test_underpopulated_does_not_create_operation(
    sample_config, mock_backend, mock_metadata
):
    """UNDERPOPULATED (auto_executable=False) does not create an operation."""
    sample_config.lifecycle.detection.min_threshold_vectors = 500
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000

    partition = PartitionInfo(
        name="small_part",
        index_name="idx_small",
        filter_value="small",
        status=PartitionStatus.ACTIVE,
    )
    mock_metadata.list_partitions.return_value = [partition]
    mock_backend.count_documents = AsyncMock(return_value=50)
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    underpop = [r for r in results if r.signal == DetectionSignal.UNDERPOPULATED]
    assert len(underpop) == 1
    mock_metadata.create_operation.assert_not_called()


# ---------------------------------------------------------------------------
# 10. _collect_counts edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_counts_handles_error(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """When count_documents raises, that partition defaults to 0."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    mock_metadata.list_partitions.return_value = sample_partitions
    # First partition fails, second succeeds
    mock_backend.count_documents = AsyncMock(
        side_effect=[Exception("connection lost"), 5000]
    )
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    # health history stored 0 for part_a due to error
    mock_metadata.append_health_history.assert_any_call("part_a", 0)
    mock_metadata.append_health_history.assert_any_call("part_b", 5000)


# ---------------------------------------------------------------------------
# 11. _build_filter logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_filter_views_mode(sample_config, mock_backend, mock_metadata):
    """In VIEWS mode, filter is empty dict (view handles filtering)."""
    sample_config.vector_storage.index_on = IndexLocation.VIEWS

    partition = PartitionInfo(
        name="part_x",
        index_name="idx_x",
        filter_value="x",
        status=PartitionStatus.ACTIVE,
    )

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    result = detector._build_filter(partition)
    assert result == {}


@pytest.mark.asyncio
async def test_build_filter_source_mode(sample_config, mock_backend, mock_metadata):
    """In SOURCE mode, filter uses partitioning field and filter_value."""
    sample_config.vector_storage.index_on = IndexLocation.SOURCE

    partition = PartitionInfo(
        name="part_x",
        index_name="idx_x",
        filter_value="electronics",
        status=PartitionStatus.ACTIVE,
    )

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    result = detector._build_filter(partition)
    assert result == {"category": "electronics"}


@pytest.mark.asyncio
async def test_build_filter_no_filter_value(sample_config, mock_backend, mock_metadata):
    """Partition without filter_value returns empty dict."""
    sample_config.vector_storage.index_on = IndexLocation.SOURCE

    partition = PartitionInfo(
        name="root",
        index_name="idx_root",
        filter_value=None,
        status=PartitionStatus.ACTIVE,
    )

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    result = detector._build_filter(partition)
    assert result == {}


# ---------------------------------------------------------------------------
# 12. Multiple signals from same detection run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_signals_in_single_run(
    sample_config, mock_backend, mock_metadata
):
    """A single run can produce multiple signal types simultaneously."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000
    sample_config.lifecycle.detection.min_threshold_vectors = 500
    sample_config.lifecycle.detection.skew_ratio = 2.0

    # part_a: breaches threshold, part_b: underpopulated, both are siblings
    partitions = [
        PartitionInfo(
            name="part_a",
            index_name="idx_a",
            filter_value="a",
            status=PartitionStatus.ACTIVE,
            parent_partition="root",
        ),
        PartitionInfo(
            name="part_b",
            index_name="idx_b",
            filter_value="b",
            status=PartitionStatus.ACTIVE,
            parent_partition="root",
        ),
        PartitionInfo(
            name="part_c",
            index_name="idx_c",
            filter_value="c",
            status=PartitionStatus.ACTIVE,
            parent_partition="root",
        ),
    ]
    mock_metadata.list_partitions.return_value = partitions
    # part_a=50000 (breach), part_b=100 (underpopulated), part_c=1000
    # skew: max=50000, avg≈17033, ratio≈2.93 > 2.0
    mock_backend.count_documents = AsyncMock(side_effect=[50000, 100, 1000])
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    signal_types = {r.signal for r in results}
    # Should have at least breach and underpopulated
    assert DetectionSignal.THRESHOLD_BREACH in signal_types
    assert DetectionSignal.UNDERPOPULATED in signal_types
    # Skew should also be detected (ratio ≈ 2.93 > 2.0)
    assert DetectionSignal.SEVERE_SKEW in signal_types


# ---------------------------------------------------------------------------
# 13. Approaching threshold — already breached is skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approaching_skips_already_breached(
    sample_config, mock_backend, mock_metadata
):
    """Partitions already over threshold are not flagged as 'approaching'."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000
    sample_config.lifecycle.detection.trend_window_days = 30

    partition = PartitionInfo(
        name="over_part",
        index_name="idx_over",
        filter_value="over",
        status=PartitionStatus.ACTIVE,
    )
    mock_metadata.list_partitions.return_value = [partition]
    mock_backend.count_documents = AsyncMock(return_value=12000)

    # History shows growth trend, but current count already over threshold
    now = datetime.utcnow()
    history = [
        {"ts": now - timedelta(days=10), "count": 8000},
        {"ts": now - timedelta(days=5), "count": 10000},
        {"ts": now, "count": 12000},
    ]
    mock_metadata.get_health_history.return_value = history

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    approach_results = [
        r for r in results if r.signal == DetectionSignal.APPROACHING_THRESHOLD
    ]
    assert len(approach_results) == 0

    # But should have a breach
    breach_results = [
        r for r in results if r.signal == DetectionSignal.THRESHOLD_BREACH
    ]
    assert len(breach_results) == 1


# ---------------------------------------------------------------------------
# 14. Stale detection — fewer than 10 history points
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_not_triggered_with_few_history_points(
    sample_config, mock_backend, mock_metadata
):
    """Fewer than 10 history points does not trigger stale detection."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    partition = PartitionInfo(
        name="few_points",
        index_name="idx_few",
        filter_value="few",
        status=PartitionStatus.ACTIVE,
    )
    mock_metadata.list_partitions.return_value = [partition]
    mock_backend.count_documents = AsyncMock(return_value=5000)

    # Only 5 identical points — not enough for stale detection
    now = datetime.utcnow()
    short_history = [
        {"ts": now - timedelta(days=5 - i), "count": 5000}
        for i in range(5)
    ]
    mock_metadata.get_health_history.return_value = short_history

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    stale_results = [r for r in results if r.signal == DetectionSignal.STALE]
    assert len(stale_results) == 0


# ---------------------------------------------------------------------------
# 15. Skew detection — single sibling (no pair) skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skew_skipped_for_single_child(
    sample_config, mock_backend, mock_metadata
):
    """A parent with only one child does not trigger skew detection."""
    sample_config.lifecycle.detection.skew_ratio = 2.0
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 1

    single = PartitionInfo(
        name="only_child",
        index_name="idx_only",
        filter_value="only",
        status=PartitionStatus.ACTIVE,
        parent_partition="parent_y",
    )
    mock_metadata.list_partitions.return_value = [single]
    mock_backend.count_documents = AsyncMock(return_value=5000)
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    skew_results = [r for r in results if r.signal == DetectionSignal.SEVERE_SKEW]
    assert len(skew_results) == 0


# ---------------------------------------------------------------------------
# 16. Skew detection — zero count siblings skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skew_skipped_when_sibling_has_zero_count(
    sample_config, mock_backend, mock_metadata
):
    """Siblings with zero count are excluded from skew check."""
    sample_config.lifecycle.detection.skew_ratio = 2.0
    sample_config.lifecycle.detection.threshold_vectors = 10_000_000
    sample_config.lifecycle.detection.min_threshold_vectors = 0

    siblings = [
        PartitionInfo(
            name="child_a",
            index_name="idx_ca",
            filter_value="ca",
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_z",
        ),
        PartitionInfo(
            name="child_b",
            index_name="idx_cb",
            filter_value="cb",
            status=PartitionStatus.ACTIVE,
            parent_partition="parent_z",
        ),
    ]
    mock_metadata.list_partitions.return_value = siblings
    # One sibling has zero docs
    mock_backend.count_documents = AsyncMock(side_effect=[5000, 0])
    mock_metadata.get_health_history.return_value = []

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    results = await detector.run_detection()

    skew_results = [r for r in results if r.signal == DetectionSignal.SEVERE_SKEW]
    assert len(skew_results) == 0


# ---------------------------------------------------------------------------
# 17. DetectionResult dataclass
# ---------------------------------------------------------------------------


def test_detection_result_dataclass():
    """DetectionResult stores all fields correctly."""
    r = DetectionResult(
        signal=DetectionSignal.THRESHOLD_BREACH,
        partition="my_part",
        details={"count": 15000, "threshold": 10000},
        auto_executable=True,
        suggested_action="split-my_part",
    )
    assert r.signal == DetectionSignal.THRESHOLD_BREACH
    assert r.partition == "my_part"
    assert r.details["count"] == 15000
    assert r.auto_executable is True
    assert r.suggested_action == "split-my_part"


# ---------------------------------------------------------------------------
# 18. Operation creation error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_operation_error_does_not_crash(
    sample_config, mock_backend, mock_metadata, sample_partitions
):
    """If create_operation raises, the pipeline still returns results."""
    sample_config.lifecycle.detection.threshold_vectors = 10_000

    mock_metadata.list_partitions.return_value = [sample_partitions[0]]
    mock_backend.count_documents = AsyncMock(return_value=15000)
    mock_metadata.get_health_history.return_value = []
    mock_metadata.create_operation.side_effect = Exception("DB write failed")

    detector = PartitionDetector(mock_backend, mock_metadata, sample_config)
    # Should not raise
    results = await detector.run_detection()

    # Results still contain the breach even though operation creation failed
    breach_results = [
        r for r in results if r.signal == DetectionSignal.THRESHOLD_BREACH
    ]
    assert len(breach_results) == 1
