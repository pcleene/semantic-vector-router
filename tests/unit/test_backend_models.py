"""Unit tests for universal backend models (Phase 12)."""

import pytest

from semantic_vector_router.models.backend import IndexStatus, PartitionStorageResult


class TestIndexStatus:
    """Tests for the IndexStatus enum."""

    def test_has_four_values(self):
        assert len(IndexStatus) == 4

    def test_ready(self):
        assert IndexStatus.READY == "ready"

    def test_building(self):
        assert IndexStatus.BUILDING == "building"

    def test_error(self):
        assert IndexStatus.ERROR == "error"

    def test_not_found(self):
        assert IndexStatus.NOT_FOUND == "not_found"

    def test_string_conversion(self):
        assert str(IndexStatus.READY) == "IndexStatus.READY"

    def test_from_value(self):
        assert IndexStatus("ready") == IndexStatus.READY
        assert IndexStatus("not_found") == IndexStatus.NOT_FOUND


class TestPartitionStorageResult:
    """Tests for PartitionStorageResult."""

    def test_basic_creation(self):
        result = PartitionStorageResult(
            storage_name="svr_partition_electronics",
            storage_type="view",
        )
        assert result.storage_name == "svr_partition_electronics"
        assert result.storage_type == "view"
        assert result.metadata == {}

    def test_with_metadata(self):
        result = PartitionStorageResult(
            storage_name="products_electronics",
            storage_type="table",
            metadata={"schema": "public", "partitioned": True},
        )
        assert result.metadata["schema"] == "public"
        assert result.metadata["partitioned"] is True

    def test_serialization(self):
        result = PartitionStorageResult(
            storage_name="test",
            storage_type="namespace",
            metadata={"key": "value"},
        )
        dumped = result.model_dump()
        assert dumped == {
            "storage_name": "test",
            "storage_type": "namespace",
            "view_name": None,
            "search_collection": None,
            "embedding_field": None,
            "metadata": {"key": "value"},
        }

    def test_json_round_trip(self):
        result = PartitionStorageResult(
            storage_name="test",
            storage_type="field",
        )
        json_str = result.model_dump_json()
        restored = PartitionStorageResult.model_validate_json(json_str)
        assert restored == result


class TestModelImports:
    """Test that new models are importable from models/__init__.py."""

    def test_import_from_models_package(self):
        from semantic_vector_router.models import IndexStatus, PartitionStorageResult

        assert IndexStatus.READY.value == "ready"
        assert PartitionStorageResult(
            storage_name="x", storage_type="y"
        ).storage_name == "x"

    def test_import_from_top_level(self):
        """Ensure IndexStatus and PartitionStorageResult are in models.__all__."""
        from semantic_vector_router import models

        assert "IndexStatus" in models.__all__
        assert "PartitionStorageResult" in models.__all__
