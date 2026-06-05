"""Unit tests for PartitionMonitor."""

from unittest.mock import AsyncMock

import pytest

from semantic_vector_router.lifecycle.monitor import PartitionMonitor
from semantic_vector_router.models import AutoSplitConfig, PartitionHealthStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor_config(sample_config_with_partitions):
    """Config with auto_split threshold set to 1M vectors."""
    sample_config_with_partitions.lifecycle.auto_split = AutoSplitConfig(
        enabled=True, threshold_vectors=1_000_000
    )
    return sample_config_with_partitions


# ---------------------------------------------------------------------------
# Health threshold tests
# ---------------------------------------------------------------------------


async def test_health_thresholds_healthy(mock_backend, monitor_config):
    """A partition well below the threshold (< 80%) is reported as 'healthy'."""
    # 200K out of 1M threshold = 20% utilization
    mock_backend.count_documents = AsyncMock(return_value=200_000)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    status = await monitor.check_partition_health("electronics")

    assert isinstance(status, PartitionHealthStatus)
    assert status.status == "healthy"
    assert status.partition == "electronics"
    assert status.vector_count == 200_000
    assert status.threshold == 1_000_000
    assert status.utilization == pytest.approx(0.2)


async def test_health_thresholds_warning(mock_backend, monitor_config):
    """A partition at 85% of the threshold is reported as 'warning'."""
    # 850K out of 1M threshold = 85% utilization
    mock_backend.count_documents = AsyncMock(return_value=850_000)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    status = await monitor.check_partition_health("electronics")

    assert status.status == "warning"
    assert status.utilization == pytest.approx(0.85)


async def test_health_thresholds_critical(mock_backend, monitor_config):
    """A partition above the threshold is reported as 'critical'."""
    # 1.2M out of 1M threshold = 120% utilization
    mock_backend.count_documents = AsyncMock(return_value=1_200_000)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    status = await monitor.check_partition_health("electronics")

    assert status.status == "critical"
    assert status.vector_count == 1_200_000
    assert status.utilization == pytest.approx(1.2)


async def test_health_boundary_at_80_percent(mock_backend, monitor_config):
    """A partition at exactly 80% utilization is still 'healthy' (threshold
    is strictly > 0.8)."""
    mock_backend.count_documents = AsyncMock(return_value=800_000)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    status = await monitor.check_partition_health("electronics")

    assert status.status == "healthy"
    assert status.utilization == pytest.approx(0.8)


async def test_health_boundary_at_exact_threshold(mock_backend, monitor_config):
    """A partition at exactly the threshold (count == threshold) is NOT critical
    because the check is strictly > threshold."""
    mock_backend.count_documents = AsyncMock(return_value=1_000_000)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    status = await monitor.check_partition_health("electronics")

    # count == threshold: not > threshold so not critical, but 1.0 > 0.8 so "warning"
    assert status.status == "warning"
    assert status.utilization == pytest.approx(1.0)


async def test_health_not_found_partition(mock_backend, monitor_config):
    """Checking health for a non-existent partition returns 'not_found'."""
    monitor = PartitionMonitor(mock_backend, monitor_config)
    status = await monitor.check_partition_health("nonexistent")

    assert status.status == "not_found"
    assert status.vector_count == 0
    assert status.utilization == 0.0


# ---------------------------------------------------------------------------
# Default threshold fallback
# ---------------------------------------------------------------------------


async def test_default_threshold_without_auto_split(
    mock_backend, sample_config_with_partitions
):
    """When auto_split is not configured, the default threshold of 10M is used."""
    # Ensure no auto_split config
    sample_config_with_partitions.lifecycle.auto_split = None

    mock_backend.count_documents = AsyncMock(return_value=5_000_000)

    monitor = PartitionMonitor(mock_backend, sample_config_with_partitions)
    status = await monitor.check_partition_health("electronics")

    assert status.threshold == 10_000_000
    assert status.status == "healthy"
    assert status.utilization == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Partition summary
# ---------------------------------------------------------------------------


async def test_partition_summary_aggregation(mock_backend, monitor_config):
    """get_partition_summary returns correct structure and aggregated counts."""
    # Set up per-partition counts:
    #   electronics: 1.2M (critical)
    #   furniture:   900K (warning, 90%)
    #   clothing:    300K (healthy, 30%)
    call_count = 0
    counts_by_view = {
        "svr_test_partition_electronics": 1_200_000,
        "svr_test_partition_furniture": 900_000,
        "svr_test_partition_clothing": 300_000,
    }

    async def count_side_effect(view_name, *args, **kwargs):
        return counts_by_view.get(view_name, 0)

    mock_backend.count_documents = AsyncMock(side_effect=count_side_effect)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    summary = await monitor.get_partition_summary()

    assert summary["total_partitions"] == 3
    assert summary["total_vectors"] == 1_200_000 + 900_000 + 300_000
    assert summary["critical_count"] == 1
    assert summary["warning_count"] == 1
    assert summary["healthy_count"] == 1
    assert summary["threshold"] == 1_000_000

    # Verify partitions list structure
    assert len(summary["partitions"]) == 3
    partition_entry = summary["partitions"][0]
    assert "name" in partition_entry
    assert "vectors" in partition_entry
    assert "utilization" in partition_entry
    assert "status" in partition_entry

    # Sorted by utilization descending: electronics first
    assert summary["partitions"][0]["name"] == "electronics"
    assert summary["partitions"][0]["status"] == "critical"


# ---------------------------------------------------------------------------
# needs_attention
# ---------------------------------------------------------------------------


async def test_needs_attention_detection(mock_backend, monitor_config):
    """needs_attention returns critical, warning, and unhealthy_indexes lists."""
    # Partition health counts
    counts_by_view = {
        "svr_test_partition_electronics": 1_200_000,  # critical
        "svr_test_partition_furniture": 850_000,       # warning
        "svr_test_partition_clothing": 200_000,        # healthy
    }

    async def count_side_effect(view_name, *args, **kwargs):
        return counts_by_view.get(view_name, 0)

    mock_backend.count_documents = AsyncMock(side_effect=count_side_effect)

    # Index health: furniture has an unhealthy index
    index_statuses = {
        ("svr_test_partition_electronics", "svr_test_idx_electronics"): {
            "status": "READY",
            "queryable": True,
        },
        ("svr_test_partition_furniture", "svr_test_idx_furniture"): {
            "status": "FAILED",
            "queryable": False,
        },
        ("svr_test_partition_clothing", "svr_test_idx_clothing"): {
            "status": "READY",
            "queryable": True,
        },
    }

    async def index_side_effect(view_name, index_name, *args, **kwargs):
        return index_statuses.get((view_name, index_name), {"status": "unknown", "queryable": False})

    mock_backend.get_index_status = AsyncMock(side_effect=index_side_effect)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    result = await monitor.needs_attention()

    assert "critical" in result
    assert "warning" in result
    assert "unhealthy_indexes" in result

    assert "electronics" in result["critical"]
    assert "furniture" in result["warning"]
    assert "furniture" in result["unhealthy_indexes"]

    # Healthy partition should not appear in any attention list
    assert "clothing" not in result["critical"]
    assert "clothing" not in result["warning"]
    assert "clothing" not in result["unhealthy_indexes"]


async def test_needs_attention_all_healthy(mock_backend, monitor_config):
    """When all partitions are healthy with good indexes, needs_attention
    returns empty lists."""
    mock_backend.count_documents = AsyncMock(return_value=100_000)
    mock_backend.get_index_status = AsyncMock(
        return_value={"status": "READY", "queryable": True}
    )

    monitor = PartitionMonitor(mock_backend, monitor_config)
    result = await monitor.needs_attention()

    assert result["critical"] == []
    assert result["warning"] == []
    assert result["unhealthy_indexes"] == []


# ---------------------------------------------------------------------------
# check_all_partitions ordering
# ---------------------------------------------------------------------------


async def test_check_all_partitions_sorted_by_utilization(mock_backend, monitor_config):
    """check_all_partitions returns results sorted by utilization descending."""
    counts_by_view = {
        "svr_test_partition_electronics": 100_000,
        "svr_test_partition_furniture": 900_000,
        "svr_test_partition_clothing": 500_000,
    }

    async def count_side_effect(view_name, *args, **kwargs):
        return counts_by_view.get(view_name, 0)

    mock_backend.count_documents = AsyncMock(side_effect=count_side_effect)

    monitor = PartitionMonitor(mock_backend, monitor_config)
    results = await monitor.check_all_partitions()

    assert len(results) == 3
    utilizations = [r.utilization for r in results]
    assert utilizations == sorted(utilizations, reverse=True)
    # Highest utilization first: furniture (0.9), clothing (0.5), electronics (0.1)
    assert results[0].partition == "furniture"
    assert results[1].partition == "clothing"
    assert results[2].partition == "electronics"


# ---------------------------------------------------------------------------
# Index health
# ---------------------------------------------------------------------------


async def test_check_index_health_partition_not_found(mock_backend, monitor_config):
    """check_index_health returns 'partition_not_found' for unknown partitions."""
    monitor = PartitionMonitor(mock_backend, monitor_config)
    result = await monitor.check_index_health("nonexistent")

    assert result["name"] == "nonexistent"
    assert result["status"] == "partition_not_found"
