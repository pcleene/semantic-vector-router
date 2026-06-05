"""Unit tests for BackendFactory (Phase 12)."""

import pytest

from semantic_vector_router.backends.factory import create_backend
from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.exceptions import ConfigurationError
from semantic_vector_router.models import (
    BackendType,
    DatabaseConfig,
    PartitioningConfig,
    SVRConfig,
)


def _make_config(backend: BackendType = BackendType.MONGODB) -> SVRConfig:
    return SVRConfig(
        database=DatabaseConfig(
            backend=backend,
            database="test_db",
            source_collection="products",
        ),
        partitioning=PartitioningConfig(field="category"),
    )


class TestCreateBackend:
    """Tests for create_backend() factory function."""

    def test_creates_mongodb_backend(self):
        config = _make_config()
        backend = create_backend(config)
        assert isinstance(backend, MongoDBBackend)

    def test_mongodb_backend_has_config(self):
        config = _make_config()
        backend = create_backend(config)
        assert backend.config is config

    def test_unknown_backend_raises(self):
        """Unknown backend type should raise ConfigurationError."""
        config = _make_config()
        # Monkey-patch to simulate unknown backend
        config.database.backend = "redis"  # type: ignore[assignment]
        with pytest.raises(ConfigurationError, match="Unknown backend"):
            create_backend(config)

    def test_default_backend_is_mongodb(self):
        """Default BackendType should be MONGODB."""
        config = SVRConfig(
            database=DatabaseConfig(
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
        )
        assert config.database.backend == BackendType.MONGODB
        backend = create_backend(config)
        assert isinstance(backend, MongoDBBackend)


class TestFactoryImports:
    """Test that factory is importable from expected paths."""

    def test_import_from_backends_factory(self):
        from semantic_vector_router.backends.factory import create_backend

        assert callable(create_backend)

    def test_import_from_client(self):
        from semantic_vector_router.client import create_backend

        assert callable(create_backend)
