"""Unit tests for PostgresBackend with mocked psycopg pool.

All tests mock the psycopg connection pool and cursor — no real
PostgreSQL connection is needed. Tests verify SQL generation,
parameter passing, and method contracts.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.backends.postgres.backend import PostgresBackend
from semantic_vector_router.backends.postgres.config import (
    PgDistanceMetric,
    PgIndexType,
    PostgresBackendConfig,
)
from semantic_vector_router.exceptions import ConnectionError, SearchError
from semantic_vector_router.models.backend import IndexStatus, PartitionStorageResult
from semantic_vector_router.models.partition import PartitionInfo



# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(**overrides):
    """Build a minimal SVRConfig for testing."""
    from semantic_vector_router.models.svr_config import SVRConfig

    base = {
        "database": {
            "backend": "postgres",
            "database": "testdb",
            "source_collection": "docs",
        },
        "partitioning": {"field": "category"},
        "vector_search": {"dimensions": 3, "similarity": "cosine"},
    }
    base.update(overrides)
    return SVRConfig(**base)


def _make_partition(name="electronics", filter_value=None, **kwargs):
    """Build a PartitionInfo for testing."""
    return PartitionInfo(
        name=name,
        filter_value=filter_value or name,
        **kwargs,
    )


def _mock_cursor(rows=None, fetchone_result=None):
    """Create a mock cursor with configurable results."""
    cursor = AsyncMock()
    if rows is not None:
        cursor.fetchall = AsyncMock(return_value=rows)
    if fetchone_result is not None:
        cursor.fetchone = AsyncMock(return_value=fetchone_result)
    else:
        cursor.fetchone = AsyncMock(return_value=None)
    return cursor


def _mock_conn(cursor=None):
    """Create a mock async connection."""
    conn = AsyncMock()
    if cursor:
        conn.execute = AsyncMock(return_value=cursor)
    else:
        conn.execute = AsyncMock(return_value=_mock_cursor())
    return conn


class MockConnectionCtx:
    """Mock async context manager for pool.connection()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


# ── Connection lifecycle ─────────────────────────────────────────────


class TestConnect:
    """Test connect/disconnect/is_connected."""

    @patch.dict("os.environ", {"POSTGRES_URI": "postgresql://<user>:<password>@<host>:5432/<db>"})
    @patch("semantic_vector_router.backends.postgres.backend.AsyncConnectionPool")
    async def test_connect_creates_pool(self, mock_pool_cls):
        config = _make_config()
        backend = PostgresBackend(config)

        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_conn = _mock_conn()
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        mock_pool_cls.return_value = mock_pool

        await backend.connect()

        mock_pool_cls.assert_called_once()
        mock_pool.open.assert_awaited_once()
        assert backend._pool is mock_pool

    @patch.dict("os.environ", {}, clear=True)
    async def test_connect_raises_without_env_var(self):
        config = _make_config()
        backend = PostgresBackend(config)

        with pytest.raises(ConnectionError, match="not set or empty"):
            await backend.connect()

    async def test_disconnect_closes_pool(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = AsyncMock()
        backend._pool = mock_pool

        await backend.disconnect()

        mock_pool.close.assert_awaited_once()
        assert backend._pool is None

    async def test_disconnect_when_no_pool(self):
        config = _make_config()
        backend = PostgresBackend(config)
        backend._pool = None

        await backend.disconnect()  # Should not raise

    async def test_is_connected_true(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        mock_conn = _mock_conn()
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        result = await backend.is_connected()
        assert result is True

    async def test_is_connected_false_when_no_pool(self):
        config = _make_config()
        backend = PostgresBackend(config)
        backend._pool = None

        result = await backend.is_connected()
        assert result is False

    async def test_is_connected_false_on_error(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("connection lost"))
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        result = await backend.is_connected()
        assert result is False


class TestHealthCheck:
    """Test health_check method."""

    async def test_health_check_true(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        mock_conn = _mock_conn()
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        result = await backend.health_check()
        assert result is True

    async def test_health_check_false_when_disconnected(self):
        config = _make_config()
        backend = PostgresBackend(config)
        backend._pool = None

        result = await backend.health_check()
        assert result is False


# ── Partition storage ────────────────────────────────────────────────


class TestPartitionStorage:
    """Test create/delete/exists partition storage."""

    async def test_create_partition_storage(self):
        config = _make_config()
        backend = PostgresBackend(config)

        partition = _make_partition("electronics")
        result = await backend.create_partition_storage(partition, config)

        assert isinstance(result, PartitionStorageResult)
        assert result.storage_name == "svr_vectors"
        assert result.storage_type == "table"
        assert result.view_name is None
        assert result.embedding_field is None
        assert result.metadata["partition_column"] == "partition_name"
        assert result.metadata["partition_value"] == "electronics"

    async def test_create_partition_storage_custom_filter_value(self):
        config = _make_config()
        backend = PostgresBackend(config)

        partition = _make_partition("elec", filter_value="electronics_category")
        result = await backend.create_partition_storage(partition, config)

        assert result.metadata["partition_value"] == "electronics_category"

    async def test_delete_partition_storage(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        mock_conn = _mock_conn()
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition("electronics")
        await backend.delete_partition_storage(partition)

        mock_conn.execute.assert_awaited()
        call_args = mock_conn.execute.call_args
        assert "DELETE FROM" in str(call_args[0][0])
        assert call_args[0][1] == ("electronics",)

    async def test_partition_storage_exists_true(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=(True,))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        result = await backend.partition_storage_exists(partition)
        assert result is True

    async def test_partition_storage_exists_false(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=(False,))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        result = await backend.partition_storage_exists(partition)
        assert result is False


# ── Index lifecycle ──────────────────────────────────────────────────


class TestIndexLifecycle:
    """Test index creation, deletion, and status."""

    async def test_create_partition_index_hnsw(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        # First call: check if index exists (returns None = no)
        cursor_check = _mock_cursor(fetchone_result=None)
        # Second call: CREATE INDEX
        cursor_create = _mock_cursor()

        call_count = 0
        mock_conn = AsyncMock()

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return cursor_check
            return cursor_create

        mock_conn.execute = AsyncMock(side_effect=side_effect)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        idx_name = await backend.create_partition_index(partition, config)

        assert idx_name == "svr_vectors_embedding_idx"

    async def test_create_partition_index_already_exists(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        # Index exists
        cursor = _mock_cursor(fetchone_result=(1,))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        idx_name = await backend.create_partition_index(partition, config)
        assert idx_name == "svr_vectors_embedding_idx"

    async def test_create_partition_index_ivfflat(self):
        pg_config = PostgresBackendConfig(index_type=PgIndexType.IVFFLAT)
        config = _make_config(postgres=pg_config.model_dump())
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        cursor_check = _mock_cursor(fetchone_result=None)
        cursor_create = _mock_cursor()

        call_count = 0
        mock_conn = AsyncMock()

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return cursor_check
            return cursor_create

        mock_conn.execute = AsyncMock(side_effect=side_effect)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        idx_name = await backend.create_partition_index(partition, config)
        assert idx_name == "svr_vectors_embedding_idx"

        # Verify IVFFlat was used
        create_call = mock_conn.execute.call_args_list[1]
        assert "ivfflat" in str(create_call[0][0]).lower()

    async def test_delete_partition_index_is_noop(self):
        config = _make_config()
        backend = PostgresBackend(config)

        partition = _make_partition()
        await backend.delete_partition_index(partition)  # Should not raise

    async def test_get_partition_index_status_ready(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=(1,))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        status = await backend.get_partition_index_status(partition)
        assert status == IndexStatus.READY

    async def test_get_partition_index_status_not_found(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=None)
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        status = await backend.get_partition_index_status(partition)
        assert status == IndexStatus.NOT_FOUND

    async def test_wait_for_index_ready_returns_immediately(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=(1,))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        result = await backend.wait_for_index_ready(partition)
        assert result is True

    async def test_wait_for_index_ready_false_when_not_found(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=None)
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        result = await backend.wait_for_index_ready(partition)
        assert result is False


# ── Search ───────────────────────────────────────────────────────────


class TestSearch:
    """Test execute_search and search_partitions."""

    async def test_execute_search_basic(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        # Simulate search results
        rows = [
            ("doc1", {"title": "Laptop"}, "electronics", 0.95),
            ("doc2", {"title": "Phone"}, "electronics", 0.88),
        ]
        cursor = _mock_cursor(rows=rows)
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        results = await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        assert len(results) == 2
        assert results[0]["_id"] == "doc1"
        assert results[0]["_svr_score"] == 0.95
        assert results[0]["_svr_partition"] == "electronics"
        assert results[0]["title"] == "Laptop"

    async def test_execute_search_with_filters(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        rows = [("doc1", {"brand": "apple"}, "electronics", 0.9)]
        cursor = _mock_cursor(rows=rows)
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        results = await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=5,
            num_candidates=50,
            filters={"brand": "apple"},
        )

        assert len(results) == 1
        # Verify the SQL contains filter clause
        execute_calls = mock_conn.execute.call_args_list
        sql = str(execute_calls[-1][0][0])
        assert "content->>'brand' = %s" in sql

    async def test_execute_search_sets_hnsw_config(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        cursor = _mock_cursor(rows=[])
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partition = _make_partition()
        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        # Verify SET hnsw.ef_search was called
        calls = mock_conn.execute.call_args_list
        set_call = str(calls[0][0][0])
        assert "hnsw.ef_search" in set_call

    async def test_search_partitions_fan_out(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        rows = [("doc1", {"title": "Item"}, "p1", 0.9)]
        cursor = _mock_cursor(rows=rows)
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partitions = [
            _make_partition("electronics"),
            _make_partition("clothing"),
        ]
        results = await backend.search_partitions(
            partitions=partitions,
            limit=10,
            query_vector=[1.0, 0.0, 0.0],
        )

        # Should get results from both partitions
        assert len(results) == 2

    async def test_search_partitions_requires_vector(self):
        config = _make_config()
        backend = PostgresBackend(config)

        partitions = [_make_partition("electronics")]
        with pytest.raises(SearchError, match="requires query_vector"):
            await backend.search_partitions(
                partitions=partitions,
                limit=10,
                query_vector=None,
                query="wireless headphones",
            )

    async def test_search_partitions_empty_list(self):
        config = _make_config()
        backend = PostgresBackend(config)

        results = await backend.search_partitions(
            partitions=[],
            limit=10,
            query_vector=[1.0, 0.0, 0.0],
        )
        assert results == []

    async def test_search_partitions_handles_errors(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("db error"))
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        partitions = [_make_partition("electronics")]
        # Should not raise — errors are logged, empty results returned
        results = await backend.search_partitions(
            partitions=partitions,
            limit=10,
            query_vector=[1.0, 0.0, 0.0],
        )
        assert results == []


# ── Data operations ──────────────────────────────────────────────────


class TestDataOperations:
    """Test count_documents, get_distinct_values, etc."""

    async def test_count_documents_no_filter(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=(42,))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        count = await backend.count_documents()
        assert count == 42

    async def test_count_documents_with_filter(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=(10,))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        count = await backend.count_documents(
            filter_expression={"status": "active"}
        )
        assert count == 10

        sql = str(mock_conn.execute.call_args[0][0])
        assert "WHERE" in sql
        assert "content->>'status' = %s" in sql

    async def test_get_distinct_values_partition_name(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(rows=[("electronics",), ("clothing",)])
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        values = await backend.get_distinct_values("partition_name")
        assert values == ["electronics", "clothing"]

    async def test_get_distinct_values_jsonb_field(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(rows=[("apple",), ("samsung",)])
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        values = await backend.get_distinct_values("brand")
        assert values == ["apple", "samsung"]

        sql = str(mock_conn.execute.call_args[0][0])
        assert "content->>'brand'" in sql

    async def test_get_distinct_values_filters_none(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(rows=[(None,), ("apple",)])
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        values = await backend.get_distinct_values("brand")
        assert values == ["apple"]  # None is filtered out

    async def test_get_partition_document_counts(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(rows=[("electronics", 100), ("clothing", 50)])
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        counts = await backend.get_partition_document_counts("category")
        assert counts == {"electronics": 100, "clothing": 50}

    async def test_get_collection_stats(self):
        config = _make_config()
        backend = PostgresBackend(config)
        mock_pool = MagicMock()
        cursor = _mock_cursor(fetchone_result=(42, 8192))
        mock_conn = _mock_conn(cursor)
        mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
        backend._pool = mock_pool

        stats = await backend.get_collection_stats()
        assert stats["count"] == 42
        assert stats["size"] == 8192
        assert stats["backend"] == "postgres"


# ── Configuration ────────────────────────────────────────────────────


class TestConfiguration:
    """Test config resolution and helper methods."""

    def test_default_config_resolution(self):
        config = _make_config()
        backend = PostgresBackend(config)
        assert backend._pg_config.index_type == PgIndexType.HNSW
        assert backend._pg_config.distance_metric == PgDistanceMetric.COSINE
        assert backend._schema == "public"
        assert backend._table_name == "svr_vectors"

    def test_custom_config_from_svr_config(self):
        pg_cfg = PostgresBackendConfig(
            index_type=PgIndexType.IVFFLAT,
            distance_metric=PgDistanceMetric.L2,
            table_prefix="custom_",
        )
        config = _make_config(postgres=pg_cfg.model_dump())
        backend = PostgresBackend(config)
        assert backend._pg_config.index_type == PgIndexType.IVFFLAT
        assert backend._pg_config.distance_metric == PgDistanceMetric.L2
        assert backend._table_name == "custom_vectors"

    def test_dimensions_from_vector_search_config(self):
        config = _make_config()
        backend = PostgresBackend(config)
        assert backend._dimensions == 3  # From vector_search.dimensions

    def test_dimensions_override_from_pg_config(self):
        pg_cfg = PostgresBackendConfig(vector_dimensions=1024)
        config = _make_config(postgres=pg_cfg.model_dump())
        backend = PostgresBackend(config)
        assert backend._dimensions == 1024

    def test_ops_class_cosine(self):
        config = _make_config()
        backend = PostgresBackend(config)
        assert backend._get_ops_class() == "vector_cosine_ops"

    def test_ops_class_l2(self):
        pg_cfg = PostgresBackendConfig(distance_metric=PgDistanceMetric.L2)
        config = _make_config(postgres=pg_cfg.model_dump())
        backend = PostgresBackend(config)
        assert backend._get_ops_class() == "vector_l2_ops"

    def test_ops_class_ip(self):
        pg_cfg = PostgresBackendConfig(
            distance_metric=PgDistanceMetric.INNER_PRODUCT
        )
        config = _make_config(postgres=pg_cfg.model_dump())
        backend = PostgresBackend(config)
        assert backend._get_ops_class() == "vector_ip_ops"

    def test_distance_operator_cosine(self):
        config = _make_config()
        backend = PostgresBackend(config)
        assert backend._get_distance_operator() == "<=>"

    def test_distance_operator_l2(self):
        pg_cfg = PostgresBackendConfig(distance_metric=PgDistanceMetric.L2)
        config = _make_config(postgres=pg_cfg.model_dump())
        backend = PostgresBackend(config)
        assert backend._get_distance_operator() == "<->"

    def test_search_config_sql_hnsw(self):
        config = _make_config()
        backend = PostgresBackend(config)
        result = backend._get_search_config_sql()
        rendered = result.as_string(None)
        assert "SET hnsw.ef_search = 40" in rendered

    def test_search_config_sql_ivfflat(self):
        pg_cfg = PostgresBackendConfig(index_type=PgIndexType.IVFFLAT)
        config = _make_config(postgres=pg_cfg.model_dump())
        backend = PostgresBackend(config)
        result = backend._get_search_config_sql()
        rendered = result.as_string(None)
        assert "SET ivfflat.probes = 10" in rendered

    def test_is_subclass_of_base_backend(self):
        from semantic_vector_router.backends.base import BaseBackend

        assert issubclass(PostgresBackend, BaseBackend)

    def test_not_change_stream_capable(self):
        from semantic_vector_router.backends.base import ChangeStreamCapable

        config = _make_config()
        backend = PostgresBackend(config)
        assert not isinstance(backend, ChangeStreamCapable)

    def test_not_auto_embedding_capable(self):
        from semantic_vector_router.backends.base import AutoEmbeddingCapable

        config = _make_config()
        backend = PostgresBackend(config)
        assert not isinstance(backend, AutoEmbeddingCapable)


# ── Filter translation ──────────────────────────────────────────────


class TestTranslateFilters:
    """Test the translate_filters method on the backend."""

    def test_translate_filters_none(self):
        config = _make_config()
        backend = PostgresBackend(config)
        assert backend.translate_filters(None) is None

    def test_translate_filters_dict(self):
        config = _make_config()
        backend = PostgresBackend(config)
        result = backend.translate_filters({"name": "test"})
        assert result == ("content->>'name' = %s", ["test"])
