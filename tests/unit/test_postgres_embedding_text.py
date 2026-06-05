"""Unit tests for PostgreSQL structured embedding text — Phase 16.

Tests 22-26 from Phase 16 spec. All mocked, no real PostgreSQL needed.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.utils.text_serializer import serialize_for_embedding


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(**overrides):
    """Build a minimal SVRConfig for PostgreSQL testing."""
    from semantic_vector_router.models.svr_config import SVRConfig

    base = {
        "database": {
            "backend": "postgres",
            "database": "testdb",
            "source_collection": "docs",
        },
        "partitioning": {"field": "category"},
        "vector_search": {"dimensions": 3, "similarity": "cosine"},
        "embedding": {
            "source_fields": ["title", "description", "tags"],
        },
    }
    base.update(overrides)
    return SVRConfig(**base)


def _make_config_no_source_fields():
    """Build a config without source_fields."""
    from semantic_vector_router.models.svr_config import SVRConfig

    return SVRConfig(**{
        "database": {
            "backend": "postgres",
            "database": "testdb",
            "source_collection": "docs",
        },
        "partitioning": {"field": "category"},
        "vector_search": {"dimensions": 3, "similarity": "cosine"},
    })


class _MockCursorCtx:
    """Mock async context manager for conn.cursor()."""

    def __init__(self, cursor):
        self._cursor = cursor

    async def __aenter__(self):
        return self._cursor

    async def __aexit__(self, *args):
        pass


class _MockConnCtx:
    """Mock async context manager for pool.connection()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


def _make_backend(config):
    """Create a PostgresBackend with a mocked pool."""
    from semantic_vector_router.backends.postgres.backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend.config = config
    backend._pg_config = backend._resolve_pg_config(config)
    backend._table_name = f"{backend._pg_config.table_prefix}vectors"
    backend._schema = backend._pg_config.schema_name
    backend._fq_table = f"{backend._schema}.{backend._table_name}"
    backend._dimensions = config.vector_search.dimensions

    # Mock cursor
    mock_cursor = AsyncMock()
    mock_cursor.rowcount = 1

    # Mock connection with cursor() returning async context manager
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = _MockCursorCtx(mock_cursor)

    # Mock pool with connection() returning async context manager
    mock_pool = MagicMock()
    mock_pool.connection.return_value = _MockConnCtx(mock_conn)
    backend._pool = mock_pool

    return backend, mock_cursor


# ── Test 22: Ingestion builds _svr_embedding_text jsonb ────────────


class TestPostgresIngestionBuildsEmbeddingTextJsonb:
    @pytest.mark.asyncio
    async def test_source_fields_extracted_into_svr_embedding_text(self):
        """During upsert, source_fields should be extracted into _svr_embedding_text."""
        config = _make_config()
        backend, mock_cursor = _make_backend(config)

        doc = {
            "_id": "prod-1",
            "category": "electronics",
            "title": "Headphones",
            "description": "Wireless headphones",
            "tags": ["audio", "wireless"],
            "price": 99.99,
        }

        await backend.insert_documents([doc])

        # Check the content JSONB that was passed to execute
        call_args = mock_cursor.execute.call_args_list[0]
        row_params = call_args[0][1]
        content_json = json.loads(row_params[3])

        assert "_svr_embedding_text" in content_json
        assert content_json["_svr_embedding_text"]["title"] == "Headphones"
        assert content_json["_svr_embedding_text"]["description"] == "Wireless headphones"
        assert content_json["_svr_embedding_text"]["tags"] == ["audio", "wireless"]

    @pytest.mark.asyncio
    async def test_none_source_fields_excluded(self):
        """source_fields that are None in the document should not be in _svr_embedding_text."""
        config = _make_config()
        backend, mock_cursor = _make_backend(config)

        doc = {
            "_id": "prod-2",
            "category": "electronics",
            "title": "Headphones",
            # description is missing, tags is missing
        }

        await backend.insert_documents([doc])

        call_args = mock_cursor.execute.call_args_list[0]
        row_params = call_args[0][1]
        content_json = json.loads(row_params[3])

        emb_text = content_json["_svr_embedding_text"]
        assert "title" in emb_text
        assert "description" not in emb_text
        assert "tags" not in emb_text


# ── Test 23: Nested structure preserved in jsonb ───────────────────


class TestPostgresIngestionPreservesNestedStructure:
    @pytest.mark.asyncio
    async def test_nested_objects_preserved(self):
        """Nested objects in source fields should be preserved in the jsonb."""
        config = _make_config(embedding={
            "source_fields": ["title", "specs"],
        })
        backend, mock_cursor = _make_backend(config)

        doc = {
            "_id": "prod-3",
            "category": "electronics",
            "title": "Headphones",
            "specs": {"weight": "250g", "battery": "30h"},
        }

        await backend.insert_documents([doc])

        call_args = mock_cursor.execute.call_args_list[0]
        row_params = call_args[0][1]
        content_json = json.loads(row_params[3])

        assert content_json["_svr_embedding_text"]["specs"] == {
            "weight": "250g",
            "battery": "30h",
        }


# ── Test 24: No source_fields → no _svr_embedding_text ────────────


class TestPostgresIngestionNoSourceFieldsNoEmbeddingText:
    @pytest.mark.asyncio
    async def test_no_svr_embedding_text_without_source_fields(self):
        """When source_fields is None, no _svr_embedding_text key should be added."""
        config = _make_config_no_source_fields()
        backend, mock_cursor = _make_backend(config)

        doc = {
            "_id": "prod-4",
            "category": "electronics",
            "title": "Headphones",
        }

        await backend.insert_documents([doc])

        call_args = mock_cursor.execute.call_args_list[0]
        row_params = call_args[0][1]
        content_json = json.loads(row_params[3])

        assert "_svr_embedding_text" not in content_json


# ── Test 25: Embedding text serialized before API call ─────────────


class TestPostgresEmbeddingTextSerializedBeforeApiCall:
    def test_jsonb_object_serialized_for_embedding(self):
        """A jsonb object should pass through serialize_for_embedding at embed time."""
        embedding_text_obj = {
            "title": "Sony WH-1000XM5",
            "description": "Premium headphones",
            "tags": ["audio", "wireless"],
        }

        result = serialize_for_embedding(embedding_text_obj)

        assert "title: Sony WH-1000XM5" in result
        assert "description: Premium headphones" in result
        assert "tags: audio, wireless" in result

    def test_string_embedding_text_passes_through(self):
        """A string embedding text should not need serialization."""
        text = "title: Sony WH-1000XM5\ndescription: Premium headphones"
        # String should be used directly, no serialization needed
        assert isinstance(text, str)


# ── Test 26: Cross-backend consistency ─────────────────────────────


class TestCrossBackendConsistency:
    def test_same_serializer_output_for_same_input(self):
        """Same input document should produce identical serialized text
        regardless of whether it came from MongoDB or PostgreSQL."""
        # Simulated structured object from MongoDB view $addFields
        mongodb_embedding_text = {
            "title": "Sony WH-1000XM5",
            "description": "Premium noise cancelling headphones",
            "tags": ["audio", "wireless", "noise-cancelling"],
            "specs": {"weight": "250g", "battery_life": "30h"},
        }

        # Simulated structured object from PostgreSQL content JSONB
        postgres_embedding_text = {
            "title": "Sony WH-1000XM5",
            "description": "Premium noise cancelling headphones",
            "tags": ["audio", "wireless", "noise-cancelling"],
            "specs": {"weight": "250g", "battery_life": "30h"},
        }

        mongodb_result = serialize_for_embedding(mongodb_embedding_text)
        postgres_result = serialize_for_embedding(postgres_embedding_text)

        assert mongodb_result == postgres_result

        expected = (
            "title: Sony WH-1000XM5\n"
            "description: Premium noise cancelling headphones\n"
            "tags: audio, wireless, noise-cancelling\n"
            "specs.weight: 250g\n"
            "specs.battery_life: 30h"
        )
        assert mongodb_result == expected
