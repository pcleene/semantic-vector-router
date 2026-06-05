"""Unit tests for PartitionScanner."""

import pytest
from unittest.mock import AsyncMock

from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError

from semantic_vector_router.exceptions import ScanError
from semantic_vector_router.lifecycle.scanner import PartitionScanner


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


async def test_scan_partition_values_happy_path(mock_backend, sample_config):
    """scan_partition_values returns a dict sorted by count descending."""
    mock_backend.get_partition_document_counts = AsyncMock(
        return_value={"furniture": 85000, "electronics": 150000, "clothing": 234000}
    )
    scanner = PartitionScanner(mock_backend, sample_config)

    result = await scanner.scan_partition_values()

    assert isinstance(result, dict)
    keys = list(result.keys())
    assert keys == ["clothing", "electronics", "furniture"]
    assert list(result.values()) == [234000, 150000, 85000]
    mock_backend.get_partition_document_counts.assert_awaited_once_with("category")


async def test_scan_partition_values_empty_collection(mock_backend, sample_config):
    """scan_partition_values returns an empty dict when no documents exist."""
    mock_backend.get_partition_document_counts = AsyncMock(return_value={})
    scanner = PartitionScanner(mock_backend, sample_config)

    result = await scanner.scan_partition_values()

    assert result == {}


async def test_scan_partition_values_with_limit(mock_backend, sample_config):
    """scan_partition_values truncates to the requested limit."""
    mock_backend.get_partition_document_counts = AsyncMock(
        return_value={
            "a": 500,
            "b": 300,
            "c": 200,
            "d": 100,
            "e": 50,
        }
    )
    scanner = PartitionScanner(mock_backend, sample_config)

    result = await scanner.scan_partition_values(limit=3)

    assert len(result) == 3
    # Top-3 by count: a(500), b(300), c(200)
    assert list(result.keys()) == ["a", "b", "c"]


async def test_scan_partition_values_limit_not_exceeded(mock_backend, sample_config):
    """scan_partition_values returns all items when limit exceeds count."""
    mock_backend.get_partition_document_counts = AsyncMock(
        return_value={"x": 10, "y": 5}
    )
    scanner = PartitionScanner(mock_backend, sample_config)

    result = await scanner.scan_partition_values(limit=100)

    assert len(result) == 2


# ---------------------------------------------------------------------------
# get_new_partition_values
# ---------------------------------------------------------------------------


async def test_get_new_partition_values_finds_orphans(
    mock_backend, sample_config_with_partitions
):
    """get_new_partition_values returns values present in data but not in the registry."""
    mock_backend.get_distinct_values = AsyncMock(
        return_value=["electronics", "furniture", "clothing", "sports", "toys"]
    )
    scanner = PartitionScanner(mock_backend, sample_config_with_partitions)

    new_values = await scanner.get_new_partition_values()

    assert set(new_values) == {"sports", "toys"}
    mock_backend.get_distinct_values.assert_awaited_once_with("category")


async def test_get_new_partition_values_returns_empty_when_all_known(
    mock_backend, sample_config_with_partitions
):
    """get_new_partition_values returns an empty list when every value is registered."""
    mock_backend.get_distinct_values = AsyncMock(
        return_value=["electronics", "furniture", "clothing"]
    )
    scanner = PartitionScanner(mock_backend, sample_config_with_partitions)

    new_values = await scanner.get_new_partition_values()

    assert new_values == []


# ---------------------------------------------------------------------------
# get_partition_stats
# ---------------------------------------------------------------------------


async def test_get_partition_stats_structure(
    mock_backend, sample_config_with_partitions
):
    """get_partition_stats returns dicts with the expected keys and correct status mapping."""
    mock_backend.get_partition_document_counts = AsyncMock(
        return_value={
            "electronics": 150000,
            "furniture": 85000,
            "clothing": 234000,
            "unknown_cat": 42,
        }
    )
    scanner = PartitionScanner(mock_backend, sample_config_with_partitions)

    stats = await scanner.get_partition_stats()

    expected_keys = {
        "value",
        "name",
        "document_count",
        "has_partition",
        "view_name",
        "index_name",
        "status",
    }
    for stat in stats:
        assert set(stat.keys()) == expected_keys

    # Sorted descending by document_count
    counts = [s["document_count"] for s in stats]
    assert counts == sorted(counts, reverse=True)

    # Known partition should have status from PartitionStatus enum
    electronics = next(s for s in stats if s["name"] == "electronics")
    assert electronics["has_partition"] is True
    assert electronics["status"] == "active"
    assert electronics["view_name"] == "svr_test_partition_electronics"
    assert electronics["index_name"] == "svr_test_idx_electronics"

    # Unknown partition should report as not_provisioned
    unknown = next(s for s in stats if s["name"] == "unknown_cat")
    assert unknown["has_partition"] is False
    assert unknown["status"] == "not_provisioned"
    assert unknown["view_name"] is None
    assert unknown["index_name"] is None


# ---------------------------------------------------------------------------
# validate_partitions
# ---------------------------------------------------------------------------


async def test_validate_partitions_categorization(
    mock_backend, sample_config_with_partitions
):
    """validate_partitions correctly categorises missing, orphaned, and valid partitions."""
    # Registry has: electronics, furniture, clothing
    # Data has:     electronics, furniture, sports
    mock_backend.get_distinct_values = AsyncMock(
        return_value=["electronics", "furniture", "sports"]
    )
    scanner = PartitionScanner(mock_backend, sample_config_with_partitions)

    result = await scanner.validate_partitions()

    assert set(result.keys()) == {"missing", "orphaned", "valid"}
    assert set(result["valid"]) == {"electronics", "furniture"}
    assert set(result["missing"]) == {"clothing"}  # registered but not in data
    assert set(result["orphaned"]) == {"sports"}  # in data but not registered


async def test_validate_partitions_all_valid(
    mock_backend, sample_config_with_partitions
):
    """validate_partitions returns empty missing/orphaned when everything matches."""
    mock_backend.get_distinct_values = AsyncMock(
        return_value=["electronics", "furniture", "clothing"]
    )
    scanner = PartitionScanner(mock_backend, sample_config_with_partitions)

    result = await scanner.validate_partitions()

    assert result["missing"] == []
    assert result["orphaned"] == []
    assert set(result["valid"]) == {"electronics", "furniture", "clothing"}


# ---------------------------------------------------------------------------
# Phase 1.3 regression: connection / operation error handling
# ---------------------------------------------------------------------------

_MONGO_ERRORS = [
    pytest.param(ConnectionFailure("connection reset"), id="ConnectionFailure"),
    pytest.param(
        ServerSelectionTimeoutError("timed out"),
        id="ServerSelectionTimeoutError",
    ),
    pytest.param(OperationFailure("unauthorized"), id="OperationFailure"),
]


class TestScanPartitionValuesErrors:
    """scan_partition_values wraps PyMongo errors in ScanError."""

    @pytest.mark.parametrize("error", _MONGO_ERRORS)
    async def test_raises_scan_error(self, mock_backend, sample_config, error):
        mock_backend.get_partition_document_counts = AsyncMock(side_effect=error)
        scanner = PartitionScanner(mock_backend, sample_config)

        with pytest.raises(ScanError) as exc_info:
            await scanner.scan_partition_values()

        assert "scan partition values" in exc_info.value.message.lower()
        assert exc_info.value.details["field"] == "category"
        assert str(error) in exc_info.value.details["error"]


class TestGetNewPartitionValuesErrors:
    """get_new_partition_values wraps PyMongo errors in ScanError."""

    @pytest.mark.parametrize("error", _MONGO_ERRORS)
    async def test_raises_scan_error(self, mock_backend, sample_config, error):
        mock_backend.get_distinct_values = AsyncMock(side_effect=error)
        scanner = PartitionScanner(mock_backend, sample_config)

        with pytest.raises(ScanError) as exc_info:
            await scanner.get_new_partition_values()

        assert "distinct values" in exc_info.value.message.lower()
        assert exc_info.value.details["field"] == "category"
        assert str(error) in exc_info.value.details["error"]


class TestGetPartitionStatsErrors:
    """get_partition_stats wraps PyMongo errors in ScanError."""

    @pytest.mark.parametrize("error", _MONGO_ERRORS)
    async def test_raises_scan_error(self, mock_backend, sample_config, error):
        mock_backend.get_partition_document_counts = AsyncMock(side_effect=error)
        scanner = PartitionScanner(mock_backend, sample_config)

        with pytest.raises(ScanError) as exc_info:
            await scanner.get_partition_stats()

        assert "partition stats" in exc_info.value.message.lower()
        assert exc_info.value.details["field"] == "category"
        assert str(error) in exc_info.value.details["error"]


class TestValidatePartitionsErrors:
    """validate_partitions wraps PyMongo errors in ScanError."""

    @pytest.mark.parametrize("error", _MONGO_ERRORS)
    async def test_raises_scan_error(self, mock_backend, sample_config, error):
        mock_backend.get_distinct_values = AsyncMock(side_effect=error)
        scanner = PartitionScanner(mock_backend, sample_config)

        with pytest.raises(ScanError) as exc_info:
            await scanner.validate_partitions()

        assert "validate partitions" in exc_info.value.message.lower()
        assert exc_info.value.details["field"] == "category"
        assert str(error) in exc_info.value.details["error"]
