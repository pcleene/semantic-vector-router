"""Unit tests for BaseBackend default implementations (Phase 12)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from semantic_vector_router.backends.base import (
    AutoEmbeddingCapable,
    BaseBackend,
    ChangeStreamCapable,
)
from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.models import (
    DatabaseConfig,
    PartitioningConfig,
    SVRConfig,
)
from semantic_vector_router.models.backend import IndexStatus


def _make_config() -> SVRConfig:
    return SVRConfig(
        database=DatabaseConfig(
            database="test_db",
            source_collection="products",
        ),
        partitioning=PartitioningConfig(field="category"),
    )


class TestBaseBackendDefaults:
    """Tests for BaseBackend default method implementations."""

    @pytest.mark.asyncio
    async def test_health_check_delegates_to_is_connected(self):
        """Default health_check() should call is_connected()."""
        config = _make_config()
        backend = MongoDBBackend(config)
        backend._client = None  # Not connected

        result = await backend.health_check()
        # Without a client, is_connected returns False
        assert result is False

    def test_translate_filters_passthrough(self):
        """Default translate_filters should return filters unchanged."""
        config = _make_config()
        backend = MongoDBBackend(config)

        filters = {"field": "value", "$and": [{"a": 1}]}
        assert backend.translate_filters(filters) is filters

    def test_translate_filters_none(self):
        """translate_filters(None) should return None."""
        config = _make_config()
        backend = MongoDBBackend(config)
        assert backend.translate_filters(None) is None


class TestWaitForIndexReady:
    """Tests for BaseBackend.wait_for_index_ready() default implementation."""

    @pytest.mark.asyncio
    async def test_returns_true_when_ready(self):
        config = _make_config()
        backend = MongoDBBackend(config)

        from semantic_vector_router.models import PartitionInfo, IndexLocation

        partition = PartitionInfo(
            name="test",
            index_name="test_idx",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        # Mock get_partition_index_status to return READY immediately
        backend.get_partition_index_status = AsyncMock(return_value=IndexStatus.READY)

        result = await backend.wait_for_index_ready(
            partition, timeout_s=1.0, poll_interval_s=0.1
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self):
        config = _make_config()
        backend = MongoDBBackend(config)

        from semantic_vector_router.models import PartitionInfo, IndexLocation

        partition = PartitionInfo(
            name="test",
            index_name="test_idx",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        backend.get_partition_index_status = AsyncMock(return_value=IndexStatus.ERROR)

        result = await backend.wait_for_index_ready(
            partition, timeout_s=1.0, poll_interval_s=0.1
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        config = _make_config()
        backend = MongoDBBackend(config)

        from semantic_vector_router.models import PartitionInfo, IndexLocation

        partition = PartitionInfo(
            name="test",
            index_name="test_idx",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        backend.get_partition_index_status = AsyncMock(
            return_value=IndexStatus.BUILDING
        )

        result = await backend.wait_for_index_ready(
            partition, timeout_s=0.3, poll_interval_s=0.1
        )
        assert result is False


class TestCapabilityProtocols:
    """Tests for runtime_checkable Protocol mixins."""

    def test_mongodb_is_change_stream_capable(self):
        assert issubclass(MongoDBBackend, ChangeStreamCapable)

    def test_mongodb_is_auto_embedding_capable(self):
        assert issubclass(MongoDBBackend, AutoEmbeddingCapable)

    def test_isinstance_check_works(self):
        config = _make_config()
        backend = MongoDBBackend(config)
        assert isinstance(backend, ChangeStreamCapable)
        assert isinstance(backend, AutoEmbeddingCapable)

    def test_base_backend_interface_methods(self):
        """Verify new abstract methods exist on BaseBackend."""
        abstract_methods = BaseBackend.__abstractmethods__
        assert "create_partition_storage" in abstract_methods
        assert "delete_partition_storage" in abstract_methods
        assert "partition_storage_exists" in abstract_methods
        assert "create_partition_index" in abstract_methods
        assert "delete_partition_index" in abstract_methods
        assert "get_partition_index_status" in abstract_methods
