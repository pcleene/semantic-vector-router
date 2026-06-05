"""Unit tests for BackendFactory PostgreSQL path."""

import pytest

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.backends.factory import create_backend
from semantic_vector_router.backends.postgres.backend import PostgresBackend
from semantic_vector_router.exceptions import ConfigurationError
from semantic_vector_router.models.svr_config import SVRConfig


def _make_config(backend="postgres"):
    """Build a minimal SVRConfig."""
    return SVRConfig(
        database={
            "backend": backend,
            "database": "testdb",
            "source_collection": "docs",
        },
        partitioning={"field": "category"},
    )


class TestCreateBackendPostgres:
    """Test that create_backend returns PostgresBackend for postgres config."""

    def test_creates_postgres_backend(self):
        config = _make_config("postgres")
        backend = create_backend(config)
        assert isinstance(backend, PostgresBackend)
        assert isinstance(backend, BaseBackend)

    def test_creates_mongodb_backend_still_works(self):
        from semantic_vector_router.backends.mongodb import MongoDBBackend

        config = _make_config("mongodb")
        backend = create_backend(config)
        assert isinstance(backend, MongoDBBackend)

    def test_unknown_backend_error_includes_postgres(self):
        """Error message should list postgres as a supported backend."""
        config = _make_config("postgres")
        # Monkey-patch to simulate unknown backend
        config.database.backend = "redis"  # type: ignore[assignment]

        with pytest.raises(ConfigurationError, match="postgres"):
            create_backend(config)

    def test_postgres_backend_has_correct_config(self):
        config = _make_config("postgres")
        backend = create_backend(config)
        assert backend.config is config

    def test_postgres_backend_not_connected(self):
        config = _make_config("postgres")
        backend = create_backend(config)
        assert backend._pool is None
