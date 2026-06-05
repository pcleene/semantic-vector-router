"""Phase 14 unit tests.

Covers:
1. PostgresBackend score expressions per distance metric
2. Field name validation (validate_field_name, _column_ref)
3. Postgres config validators (HnswConfig, IvfflatConfig, PostgresBackendConfig)
4. PostgresBackend.insert_documents
5. MongoDBBackend.insert_documents
6. BaseBackend abstract methods (cannot instantiate, 16 abstract methods)
7. IngestPipeline._write_batch dispatch (MongoDB vs generic)
8. .db access guards (field_analyzer)
9. PostgresBackend.get_partition_document_counts field handling
"""

import inspect
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from psycopg import sql

import pytest
from pydantic import ValidationError

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.backends.mongodb.backend import MongoDBBackend
from semantic_vector_router.backends.mongodb.indexes import MongoDBIndexOps
from semantic_vector_router.backends.mongodb.views import MongoDBViewOps
from semantic_vector_router.backends.postgres.backend import PostgresBackend
from semantic_vector_router.backends.postgres.config import (
    HnswConfig,
    IvfflatConfig,
    PgDistanceMetric,
    PgIndexType,
    PostgresBackendConfig,
)
from semantic_vector_router.backends.postgres.filters import (
    COLUMN_FIELDS,
    _column_ref,
    validate_field_name,
)
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    PartitioningConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_svr_config(**overrides) -> SVRConfig:
    """Build a minimal SVRConfig for unit tests."""
    defaults = dict(
        database=DatabaseConfig(
            backend="postgres",
            database="testdb",
            source_collection="docs",
        ),
        partitioning=PartitioningConfig(field="category"),
        vector_search=VectorSearchConfig(dimensions=3, similarity="cosine"),
        vector_storage=VectorStorageConfig(
            storage_format=VectorStorageFormat.ARRAY,
        ),
        embedding=EmbeddingConfig(mode=EmbeddingMode.BYOM),
    )
    defaults.update(overrides)
    return SVRConfig(**defaults)


def _make_pg_backend(
    distance_metric: PgDistanceMetric = PgDistanceMetric.COSINE,
    index_type: PgIndexType = PgIndexType.HNSW,
) -> PostgresBackend:
    """Create a PostgresBackend with custom metric, without connection."""
    pg_config = PostgresBackendConfig(
        distance_metric=distance_metric,
        index_type=index_type,
    )
    config = _make_svr_config(postgres=pg_config)
    backend = PostgresBackend(config)
    return backend


def _make_mongo_backend() -> MongoDBBackend:
    """Create a MongoDBBackend without calling __init__ (no connection)."""
    config = SVRConfig(
        database=DatabaseConfig(
            backend="mongodb",
            database="test_db",
            source_collection="products",
        ),
        partitioning=PartitioningConfig(field="category"),
        vector_search=VectorSearchConfig(dimensions=1536),
        embedding=EmbeddingConfig(mode=EmbeddingMode.BYOM),
    )
    be = MongoDBBackend.__new__(MongoDBBackend)
    be.config = config
    be._client = None
    be.__db = None
    be._server_version = None
    be._last_health_check = None
    be._views = MongoDBViewOps(config)
    be._indexes = MongoDBIndexOps()
    be._source_index_ensured = False
    return be


# ===================================================================
# 1. Score expressions per distance metric
# ===================================================================

class TestScoreExpressions:
    """Test _get_score_expression() for each PgDistanceMetric."""

    def test_cosine_score_expression(self):
        backend = _make_pg_backend(PgDistanceMetric.COSINE)
        expr = backend._get_score_expression()
        assert expr == "1 - (embedding <=> %s::vector)"

    def test_l2_score_expression(self):
        backend = _make_pg_backend(PgDistanceMetric.L2)
        expr = backend._get_score_expression()
        assert expr == "1.0 / (1.0 + (embedding <-> %s::vector))"

    def test_inner_product_score_expression(self):
        backend = _make_pg_backend(PgDistanceMetric.INNER_PRODUCT)
        expr = backend._get_score_expression()
        assert expr == "-(embedding <#> %s::vector)"

    def test_cosine_distance_operator(self):
        backend = _make_pg_backend(PgDistanceMetric.COSINE)
        assert backend._get_distance_operator() == "<=>"

    def test_l2_distance_operator(self):
        backend = _make_pg_backend(PgDistanceMetric.L2)
        assert backend._get_distance_operator() == "<->"

    def test_ip_distance_operator(self):
        backend = _make_pg_backend(PgDistanceMetric.INNER_PRODUCT)
        assert backend._get_distance_operator() == "<#>"

    def test_cosine_ops_class(self):
        backend = _make_pg_backend(PgDistanceMetric.COSINE)
        assert backend._get_ops_class() == "vector_cosine_ops"

    def test_l2_ops_class(self):
        backend = _make_pg_backend(PgDistanceMetric.L2)
        assert backend._get_ops_class() == "vector_l2_ops"

    def test_ip_ops_class(self):
        backend = _make_pg_backend(PgDistanceMetric.INNER_PRODUCT)
        assert backend._get_ops_class() == "vector_ip_ops"


# ===================================================================
# 2. Field name validation
# ===================================================================

class TestFieldNameValidation:
    """Test validate_field_name() and _column_ref()."""

    # Valid names
    @pytest.mark.parametrize("name", [
        "category",
        "sub_type",
        "data.nested",
        "_private",
        "A",
        "field123",
        "my_field.sub_field",
    ])
    def test_valid_field_names(self, name):
        validate_field_name(name)  # Should not raise

    # Invalid names
    @pytest.mark.parametrize("name", [
        "'; DROP TABLE",
        "field name",
        "123abc",
        "",
        "hello world",
        "field@name",
        "field-name",
        "field$",
        " leading_space",
    ])
    def test_invalid_field_names(self, name):
        with pytest.raises(ValueError, match="Invalid field name"):
            validate_field_name(name)

    # _column_ref for column fields
    def test_column_ref_for_known_columns(self):
        for col in COLUMN_FIELDS:
            assert _column_ref(col) == col

    # _column_ref for JSONB fields
    def test_column_ref_for_jsonb_field(self):
        assert _column_ref("category") == "content->>'category'"

    def test_column_ref_with_nested_field(self):
        assert _column_ref("data.nested") == "content->>'data.nested'"

    # _column_ref raises for invalid field
    def test_column_ref_raises_for_invalid_field(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            _column_ref("'; DROP TABLE users;")

    def test_column_ref_raises_for_empty(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            _column_ref("")


# ===================================================================
# 3. Config validators
# ===================================================================

class TestHnswConfig:
    """HnswConfig validation: m (2-100), ef_construction (10-2000), ef_search (10-1000)."""

    def test_defaults(self):
        cfg = HnswConfig()
        assert cfg.m == 16
        assert cfg.ef_construction == 64
        assert cfg.ef_search == 40

    def test_valid_boundaries(self):
        cfg = HnswConfig(m=2, ef_construction=10, ef_search=10)
        assert cfg.m == 2
        assert cfg.ef_construction == 10
        assert cfg.ef_search == 10

    def test_valid_upper_boundaries(self):
        cfg = HnswConfig(m=100, ef_construction=2000, ef_search=1000)
        assert cfg.m == 100
        assert cfg.ef_construction == 2000
        assert cfg.ef_search == 1000

    def test_m_too_low(self):
        with pytest.raises(ValidationError):
            HnswConfig(m=1)

    def test_m_too_high(self):
        with pytest.raises(ValidationError):
            HnswConfig(m=101)

    def test_ef_construction_too_low(self):
        with pytest.raises(ValidationError):
            HnswConfig(ef_construction=9)

    def test_ef_construction_too_high(self):
        with pytest.raises(ValidationError):
            HnswConfig(ef_construction=2001)

    def test_ef_search_too_low(self):
        with pytest.raises(ValidationError):
            HnswConfig(ef_search=9)

    def test_ef_search_too_high(self):
        with pytest.raises(ValidationError):
            HnswConfig(ef_search=1001)


class TestIvfflatConfig:
    """IvfflatConfig validation: lists (1-10000), probes (1-500)."""

    def test_defaults(self):
        cfg = IvfflatConfig()
        assert cfg.lists == 100
        assert cfg.probes == 10

    def test_valid_boundaries(self):
        cfg = IvfflatConfig(lists=1, probes=1)
        assert cfg.lists == 1
        assert cfg.probes == 1

    def test_valid_upper_boundaries(self):
        cfg = IvfflatConfig(lists=10000, probes=500)
        assert cfg.lists == 10000
        assert cfg.probes == 500

    def test_lists_too_low(self):
        with pytest.raises(ValidationError):
            IvfflatConfig(lists=0)

    def test_lists_too_high(self):
        with pytest.raises(ValidationError):
            IvfflatConfig(lists=10001)

    def test_probes_too_low(self):
        with pytest.raises(ValidationError):
            IvfflatConfig(probes=0)

    def test_probes_too_high(self):
        with pytest.raises(ValidationError):
            IvfflatConfig(probes=501)


class TestPostgresBackendConfig:
    """PostgresBackendConfig field validation."""

    def test_defaults(self):
        cfg = PostgresBackendConfig()
        assert cfg.pool_min_size == 5
        assert cfg.pool_max_size == 20
        assert cfg.schema_name == "public"
        assert cfg.table_prefix == "svr_"
        assert cfg.statement_timeout_ms == 30_000

    # pool_min_size
    def test_pool_min_size_lower_bound(self):
        cfg = PostgresBackendConfig(pool_min_size=0)
        assert cfg.pool_min_size == 0

    def test_pool_min_size_upper_bound(self):
        cfg = PostgresBackendConfig(pool_min_size=100, pool_max_size=100)
        assert cfg.pool_min_size == 100

    def test_pool_min_size_too_low(self):
        with pytest.raises(ValidationError):
            PostgresBackendConfig(pool_min_size=-1)

    def test_pool_min_size_too_high(self):
        with pytest.raises(ValidationError):
            PostgresBackendConfig(pool_min_size=101)

    # pool_max_size
    def test_pool_max_size_lower_bound(self):
        cfg = PostgresBackendConfig(pool_min_size=0, pool_max_size=1)
        assert cfg.pool_max_size == 1

    def test_pool_max_size_upper_bound(self):
        cfg = PostgresBackendConfig(pool_max_size=500)
        assert cfg.pool_max_size == 500

    def test_pool_max_size_too_low(self):
        with pytest.raises(ValidationError):
            PostgresBackendConfig(pool_max_size=0)

    def test_pool_max_size_too_high(self):
        with pytest.raises(ValidationError):
            PostgresBackendConfig(pool_max_size=501)

    def test_pool_max_size_less_than_min(self):
        with pytest.raises(ValidationError, match="pool_max_size"):
            PostgresBackendConfig(pool_min_size=10, pool_max_size=5)

    # schema_name validation
    def test_valid_schema_name(self):
        cfg = PostgresBackendConfig(schema="my_schema")
        assert cfg.schema_name == "my_schema"

    def test_invalid_schema_name(self):
        with pytest.raises(ValidationError, match="Invalid schema name"):
            PostgresBackendConfig(schema="bad schema!")

    def test_invalid_schema_numeric_start(self):
        with pytest.raises(ValidationError, match="Invalid schema name"):
            PostgresBackendConfig(schema="123schema")

    # table_prefix validation
    def test_valid_table_prefix(self):
        cfg = PostgresBackendConfig(table_prefix="myapp_")
        assert cfg.table_prefix == "myapp_"

    def test_invalid_table_prefix(self):
        with pytest.raises(ValidationError, match="Invalid table_prefix"):
            PostgresBackendConfig(table_prefix="bad prefix!")

    def test_empty_table_prefix_is_valid(self):
        # Empty prefix is allowed (the validator checks v.rstrip("_"))
        cfg = PostgresBackendConfig(table_prefix="")
        assert cfg.table_prefix == ""

    # statement_timeout_ms
    def test_statement_timeout_lower_bound(self):
        cfg = PostgresBackendConfig(statement_timeout_ms=1000)
        assert cfg.statement_timeout_ms == 1000

    def test_statement_timeout_upper_bound(self):
        cfg = PostgresBackendConfig(statement_timeout_ms=600_000)
        assert cfg.statement_timeout_ms == 600_000

    def test_statement_timeout_too_low(self):
        with pytest.raises(ValidationError):
            PostgresBackendConfig(statement_timeout_ms=999)

    def test_statement_timeout_too_high(self):
        with pytest.raises(ValidationError):
            PostgresBackendConfig(statement_timeout_ms=600_001)


# ===================================================================
# 4. PostgresBackend.insert_documents
# ===================================================================

@pytest.mark.asyncio
class TestPostgresInsertDocuments:
    """Test PostgresBackend.insert_documents with mocked pool."""

    def _mock_pool(self):
        """Create a mock pool with connection/cursor context managers.

        psycopg_pool's AsyncConnectionPool.connection() returns a sync
        context-manager-like object whose __aenter__/__aexit__ are async.
        Similarly, conn.cursor() returns a sync CM with async enter/exit.
        """
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.execute = AsyncMock()

        # cursor() returns a sync CM with async __aenter__/__aexit__
        cursor_cm = MagicMock()
        cursor_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_cm.__aexit__ = AsyncMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor_cm
        mock_conn.execute = AsyncMock()

        # connection() returns a sync CM with async __aenter__/__aexit__
        conn_cm = MagicMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.connection.return_value = conn_cm

        return mock_pool, mock_conn, mock_cursor

    async def test_empty_list_returns_zero(self):
        backend = _make_pg_backend()
        result = await backend.insert_documents([])
        assert result == 0

    async def test_basic_insert(self):
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [
            {"_id": "doc1", "partition_name": "electronics", "embedding": [0.1, 0.2, 0.3], "title": "Test"}
        ]
        count = await backend.insert_documents(docs)
        assert count == 1
        mock_cursor.execute.assert_called_once()

    async def test_missing_id_gets_uuid(self):
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [{"partition_name": "electronics", "title": "No ID"}]
        await backend.insert_documents(docs)

        # Verify the execute call was mParts Distributor (the id should be a UUID)
        call_args = mock_cursor.execute.call_args
        row = call_args[0][1]
        doc_id = row[0]
        # Verify it looks like a UUID
        uuid.UUID(doc_id)  # Should not raise

    async def test_embedding_conversion_to_vector_literal(self):
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [{"_id": "d1", "partition_name": "cat", "embedding": [1.0, 2.5, 3.0]}]
        await backend.insert_documents(docs)

        call_args = mock_cursor.execute.call_args
        row = call_args[0][1]
        vector_literal = row[2]
        assert vector_literal == "[1.0,2.5,3.0]"

    async def test_partition_name_extraction(self):
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [{"_id": "d1", "partition_name": "electronics", "title": "T"}]
        await backend.insert_documents(docs)

        call_args = mock_cursor.execute.call_args
        row = call_args[0][1]
        assert row[1] == "electronics"

    async def test_partition_falls_back_to_partition_field(self):
        """When no partition_name key, uses the config partitioning field."""
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        # Config partitioning field is "category"
        docs = [{"_id": "d1", "category": "books", "title": "T"}]
        await backend.insert_documents(docs)

        call_args = mock_cursor.execute.call_args
        row = call_args[0][1]
        assert row[1] == "books"

    async def test_default_partition_when_missing(self):
        """When no partition_name and no partitioning field, defaults to 'default'."""
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [{"_id": "d1", "title": "T"}]
        await backend.insert_documents(docs)

        call_args = mock_cursor.execute.call_args
        row = call_args[0][1]
        assert row[1] == "default"

    async def test_content_json_excludes_extracted_fields(self):
        """Remaining fields after _id/partition_name/embedding go to content JSONB."""
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [{"_id": "d1", "partition_name": "cat", "embedding": [1, 2, 3], "title": "T", "price": 10}]
        await backend.insert_documents(docs)

        call_args = mock_cursor.execute.call_args
        row = call_args[0][1]
        content = json.loads(row[3])
        assert "title" in content
        assert "price" in content
        assert "_id" not in content
        assert "partition_name" not in content
        assert "embedding" not in content

    async def test_multiple_documents(self):
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [
            {"_id": "d1", "partition_name": "a", "title": "one"},
            {"_id": "d2", "partition_name": "b", "title": "two"},
            {"_id": "d3", "partition_name": "c", "title": "three"},
        ]
        count = await backend.insert_documents(docs)
        assert count == 3
        assert mock_cursor.execute.call_count == 3

    async def test_no_embedding_sets_none(self):
        backend = _make_pg_backend()
        mock_pool, mock_conn, mock_cursor = self._mock_pool()
        backend._pool = mock_pool

        docs = [{"_id": "d1", "partition_name": "cat", "title": "no vec"}]
        await backend.insert_documents(docs)

        call_args = mock_cursor.execute.call_args
        row = call_args[0][1]
        assert row[2] is None  # vector_literal is None


# ===================================================================
# 5. MongoDBBackend.insert_documents
# ===================================================================

@pytest.mark.asyncio
class TestMongoDBInsertDocuments:
    """Test MongoDBBackend.insert_documents with mocked db."""

    async def test_empty_list_returns_zero(self):
        be = _make_mongo_backend()
        result = await be.insert_documents([])
        assert result == 0

    async def test_basic_insert(self):
        be = _make_mongo_backend()

        mock_result = MagicMock()
        mock_result.inserted_ids = ["id1", "id2"]

        mock_collection = AsyncMock()
        mock_collection.insert_many = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        # Set up the _db property
        be._MongoDBBackend__db = mock_db
        be._views._db = mock_db
        be._indexes._db = mock_db

        # Mock _with_retry to just call the function directly
        async def pass_through(func, *args, **kwargs):
            return await func()

        be._with_retry = pass_through

        docs = [{"_id": "id1", "text": "hello"}, {"_id": "id2", "text": "world"}]
        count = await be.insert_documents(docs)
        assert count == 2
        mock_collection.insert_many.assert_called_once_with(docs, ordered=False)

    async def test_bulk_write_error_returns_partial_count(self):
        be = _make_mongo_backend()

        from pymongo.errors import BulkWriteError

        bulk_error = BulkWriteError({"nInserted": 3, "writeErrors": [{"index": 0, "errmsg": "dup"}]})

        mock_collection = AsyncMock()
        mock_collection.insert_many = AsyncMock(side_effect=bulk_error)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        be._MongoDBBackend__db = mock_db
        be._views._db = mock_db
        be._indexes._db = mock_db

        async def pass_through(func, *args, **kwargs):
            return await func()

        be._with_retry = pass_through

        docs = [{"_id": f"id{i}"} for i in range(5)]
        count = await be.insert_documents(docs)
        assert count == 3

    async def test_insert_uses_source_collection_by_default(self):
        be = _make_mongo_backend()

        mock_result = MagicMock()
        mock_result.inserted_ids = ["id1"]

        mock_collection = AsyncMock()
        mock_collection.insert_many = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        be._MongoDBBackend__db = mock_db
        be._views._db = mock_db
        be._indexes._db = mock_db

        async def pass_through(func, *args, **kwargs):
            return await func()

        be._with_retry = pass_through

        await be.insert_documents([{"_id": "id1"}])
        mock_db.__getitem__.assert_called_with("products")

    async def test_insert_with_custom_collection(self):
        be = _make_mongo_backend()

        mock_result = MagicMock()
        mock_result.inserted_ids = ["id1"]

        mock_collection = AsyncMock()
        mock_collection.insert_many = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        be._MongoDBBackend__db = mock_db
        be._views._db = mock_db
        be._indexes._db = mock_db

        async def pass_through(func, *args, **kwargs):
            return await func()

        be._with_retry = pass_through

        await be.insert_documents([{"_id": "id1"}], collection_name="other_coll")
        mock_db.__getitem__.assert_called_with("other_coll")


# ===================================================================
# 6. BaseBackend abstract methods
# ===================================================================

class TestBaseBackendAbstract:
    """Verify BaseBackend cannot be instantiated and has 16 abstract methods."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="abstract method"):
            BaseBackend(_make_svr_config())

    def test_has_16_abstract_methods(self):
        abstract_methods = set()
        for name, method in inspect.getmembers(BaseBackend):
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)
        assert len(abstract_methods) == 16, (
            f"Expected 16 abstract methods, found {len(abstract_methods)}: {sorted(abstract_methods)}"
        )

    def test_expected_abstract_method_names(self):
        expected = {
            "connect",
            "disconnect",
            "is_connected",
            "create_partition_storage",
            "delete_partition_storage",
            "partition_storage_exists",
            "create_partition_index",
            "delete_partition_index",
            "get_partition_index_status",
            "execute_search",
            "search_partitions",
            "insert_documents",
            "get_distinct_values",
            "count_documents",
            "get_collection_stats",
            "get_partition_document_counts",
        }
        abstract_methods = {
            name
            for name, method in inspect.getmembers(BaseBackend)
            if getattr(method, "__isabstractmethod__", False)
        }
        assert abstract_methods == expected


# ===================================================================
# 7. Ingestion pipeline dispatch
# ===================================================================

@pytest.mark.asyncio
class TestIngestionWriteBatchDispatch:
    """Test _write_batch dispatches correctly between MongoDB and generic."""

    def _make_pipeline(self, backend):
        """Create an IngestPipeline with a given backend."""
        from semantic_vector_router.ingestion import IngestPipeline
        from semantic_vector_router.models import IngestConfig

        config = _make_svr_config()
        embedder = MagicMock()
        metrics = MagicMock()
        return IngestPipeline(backend, config, embedder, metrics)

    async def test_dispatch_to_mongodb_when_backend_has_db(self):
        """When backend has .db attribute, dispatches to _write_batch_mongodb."""
        mock_backend = MagicMock()
        mock_backend.db = MagicMock()  # has .db

        pipeline = self._make_pipeline(mock_backend)

        with patch.object(pipeline, "_write_batch_mongodb", new_callable=AsyncMock, return_value=(2, [])) as mock_mongo:
            with patch.object(pipeline, "_write_batch_generic", new_callable=AsyncMock) as mock_generic:
                from semantic_vector_router.models import IngestMode
                result = await pipeline._write_batch(
                    [{"_id": "1"}, {"_id": "2"}], [0, 1], IngestMode.INSERT
                )
                mock_mongo.assert_called_once()
                mock_generic.assert_not_called()
                assert result == (2, [])

    async def test_dispatch_to_generic_when_no_db(self):
        """When backend lacks .db attribute, dispatches to _write_batch_generic."""
        mock_backend = MagicMock(spec=[])  # No .db attribute

        pipeline = self._make_pipeline(mock_backend)

        with patch.object(pipeline, "_write_batch_mongodb", new_callable=AsyncMock) as mock_mongo:
            with patch.object(pipeline, "_write_batch_generic", new_callable=AsyncMock, return_value=(2, [])) as mock_generic:
                from semantic_vector_router.models import IngestMode
                result = await pipeline._write_batch(
                    [{"_id": "1"}, {"_id": "2"}], [0, 1], IngestMode.INSERT
                )
                mock_generic.assert_called_once()
                mock_mongo.assert_not_called()
                assert result == (2, [])

    async def test_write_batch_generic_success(self):
        """_write_batch_generic calls insert_documents and returns count."""
        mock_backend = MagicMock(spec=[])
        mock_backend.insert_documents = AsyncMock(return_value=3)

        pipeline = self._make_pipeline(mock_backend)

        count, errors = await pipeline._write_batch_generic(
            [{"_id": "1"}, {"_id": "2"}, {"_id": "3"}], [0, 1, 2]
        )
        assert count == 3
        assert errors == []
        mock_backend.insert_documents.assert_called_once()

    async def test_write_batch_generic_failure(self):
        """_write_batch_generic on failure returns 0 and error for each doc."""
        mock_backend = MagicMock(spec=[])
        mock_backend.insert_documents = AsyncMock(side_effect=RuntimeError("DB down"))

        pipeline = self._make_pipeline(mock_backend)

        count, errors = await pipeline._write_batch_generic(
            [{"_id": "1"}, {"_id": "2"}], [5, 6]
        )
        assert count == 0
        assert len(errors) == 2
        assert errors[0][0] == 5
        assert errors[1][0] == 6
        assert "DB down" in errors[0][1]


# ===================================================================
# 8. .db access guards
# ===================================================================

@pytest.mark.asyncio
class TestDbAccessGuards:
    """Test that non-MongoDB backends are gracefully handled."""

    async def test_field_analyzer_returns_empty_for_non_mongo_backend(self):
        """analyze_fields returns [] when backend has no .db attribute."""
        from semantic_vector_router.utils.field_analyzer import analyze_fields

        mock_backend = AsyncMock(spec=[])
        # Provide count_documents so the function can get total_docs
        mock_backend.count_documents = AsyncMock(return_value=100)

        config = _make_svr_config()
        result = await analyze_fields(mock_backend, config)
        assert result == []

    async def test_field_analyzer_returns_empty_for_empty_collection(self):
        """analyze_fields returns [] for collections with 0 docs."""
        from semantic_vector_router.utils.field_analyzer import analyze_fields

        mock_backend = AsyncMock()
        mock_backend.count_documents = AsyncMock(return_value=0)

        config = _make_svr_config()
        result = await analyze_fields(mock_backend, config)
        assert result == []


# ===================================================================
# 9. get_partition_document_counts field fix
# ===================================================================

@pytest.mark.asyncio
class TestGetPartitionDocumentCounts:
    """Test get_partition_document_counts with different field values."""

    def _mock_pool_with_rows(self, rows):
        """Create a mock pool that returns specified rows."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=rows)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        conn_cm = MagicMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.connection.return_value = conn_cm

        return mock_pool, mock_conn

    async def test_partition_name_field_uses_column_directly(self):
        """When field='partition_name', the SQL uses the column directly.

        We verify that the composed SQL object contains 'partition_name'
        as a raw SQL fragment (not wrapped in content->>).
        """
        backend = _make_pg_backend()
        rows = [("electronics", 10), ("books", 5)]
        mock_pool, mock_conn = self._mock_pool_with_rows(rows)
        backend._pool = mock_pool

        result = await backend.get_partition_document_counts("partition_name")
        assert result == {"electronics": 10, "books": 5}

        # Verify that execute was called and the composed query contains
        # 'partition_name' as a SQL fragment
        call_args = mock_conn.execute.call_args
        query_composed = call_args[0][0]
        # The composed SQL object has ._obj list of SQL fragments
        sql_parts = [str(part) for part in query_composed._obj]
        full_sql = "".join(sql_parts)
        assert "partition_name" in full_sql
        # Should NOT contain content->> for partition_name
        assert "content->>'partition_name'" not in full_sql

    async def test_custom_field_uses_jsonb_expression(self):
        """When field='category', the SQL uses content->>'category'."""
        backend = _make_pg_backend()
        rows = [("electronics", 10), ("books", 5)]
        mock_pool, mock_conn = self._mock_pool_with_rows(rows)
        backend._pool = mock_pool

        result = await backend.get_partition_document_counts("category")
        assert result == {"electronics": 10, "books": 5}

        # Verify that the composed query contains the JSONB expression
        call_args = mock_conn.execute.call_args
        query_composed = call_args[0][0]
        sql_parts = [str(part) for part in query_composed._obj]
        full_sql = "".join(sql_parts)
        assert "content->>'category'" in full_sql

    async def test_none_values_are_excluded(self):
        """Rows with None keys are excluded from the result."""
        backend = _make_pg_backend()
        rows = [("electronics", 10), (None, 3), ("books", 5)]
        mock_pool, mock_conn = self._mock_pool_with_rows(rows)
        backend._pool = mock_pool

        result = await backend.get_partition_document_counts("partition_name")
        assert result == {"electronics": 10, "books": 5}
        assert None not in result

    async def test_invalid_field_raises_valueerror(self):
        """get_partition_document_counts with invalid field name raises ValueError."""
        backend = _make_pg_backend()
        backend._pool = MagicMock()

        with pytest.raises(ValueError, match="Invalid field name"):
            await backend.get_partition_document_counts("'; DROP TABLE")


# ===================================================================
# Additional: search_config_sql per index type
# ===================================================================

class TestSearchConfigSql:
    """Test _get_search_config_sql for HNSW and IVFFlat."""

    def test_hnsw_config(self):
        backend = _make_pg_backend(index_type=PgIndexType.HNSW)
        result = backend._get_search_config_sql()
        assert result is not None
        rendered = result.as_string(None)
        assert "hnsw.ef_search" in rendered

    def test_ivfflat_config(self):
        backend = _make_pg_backend(index_type=PgIndexType.IVFFLAT)
        result = backend._get_search_config_sql()
        assert result is not None
        rendered = result.as_string(None)
        assert "ivfflat.probes" in rendered


# ===================================================================
# Additional: PostgresBackend initialization
# ===================================================================

class TestPostgresBackendInit:
    """Test PostgresBackend initialization and config resolution."""

    def test_default_pg_config(self):
        """When no postgres config on SVRConfig, defaults are used."""
        config = _make_svr_config()
        backend = PostgresBackend(config)
        assert backend._pg_config.pool_min_size == 5
        assert backend._pg_config.pool_max_size == 20

    def test_custom_pg_config(self):
        """When SVRConfig has postgres config, it is used."""
        pg_config = PostgresBackendConfig(pool_min_size=2, pool_max_size=10)
        config = _make_svr_config(postgres=pg_config)
        backend = PostgresBackend(config)
        assert backend._pg_config.pool_min_size == 2
        assert backend._pg_config.pool_max_size == 10

    def test_dimensions_from_vector_search(self):
        """Dimensions come from vector_search config when pg_config has None."""
        config = _make_svr_config()
        backend = PostgresBackend(config)
        assert backend._dimensions == 3  # from vector_search.dimensions

    def test_dimensions_from_pg_config_override(self):
        """Dimensions from pg_config.vector_dimensions take precedence."""
        pg_config = PostgresBackendConfig(vector_dimensions=128)
        config = _make_svr_config(postgres=pg_config)
        backend = PostgresBackend(config)
        assert backend._dimensions == 128

    def test_table_name(self):
        backend = _make_pg_backend()
        assert backend._table_name == "svr_vectors"

    def test_fq_table(self):
        backend = _make_pg_backend()
        assert backend._fq_table == "public.svr_vectors"


# ===================================================================
# Phase 14.5 — Bug fixes and hardening
# ===================================================================

@pytest.mark.asyncio
class TestPhase14_5_UpsertFixes:
    """14.5.1 + 14.5.2: Upsert no-_id fallback and success counting."""

    def _make_pipeline(self, backend):
        from semantic_vector_router.ingestion import IngestPipeline

        config = _make_svr_config()
        embedder = MagicMock()
        metrics = MagicMock()
        return IngestPipeline(backend, config, embedder, metrics)

    async def test_write_batch_mongodb_upsert_no_id(self):
        """Documents without _id get a UUID assigned, not the entire doc as _id."""
        from semantic_vector_router.models import IngestMode

        mock_backend = MagicMock()
        mock_backend.db = MagicMock()

        mock_result = MagicMock()
        mock_result.matched_count = 0
        mock_result.upserted_count = 2

        mock_collection = AsyncMock()
        mock_collection.bulk_write = AsyncMock(return_value=mock_result)
        mock_backend.db.__getitem__ = MagicMock(return_value=mock_collection)

        pipeline = self._make_pipeline(mock_backend)

        docs = [{"title": "no id 1"}, {"title": "no id 2"}]
        success, errors = await pipeline._write_batch_mongodb(
            docs, [0, 1], IngestMode.UPSERT
        )

        # Verify UUIDs were generated
        for doc in docs:
            assert "_id" in doc
            uuid.UUID(doc["_id"])  # Should not raise

        # Verify the UpdateOne filter uses the UUID, not the doc dict
        call_args = mock_collection.bulk_write.call_args[0][0]
        for op in call_args:
            filter_val = op._filter["_id"]
            assert isinstance(filter_val, str)
            uuid.UUID(filter_val)  # Should not raise

    async def test_write_batch_mongodb_upsert_with_id(self):
        """Documents with _id use their existing _id (regression guard)."""
        from semantic_vector_router.models import IngestMode

        mock_backend = MagicMock()
        mock_backend.db = MagicMock()

        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.upserted_count = 0

        mock_collection = AsyncMock()
        mock_collection.bulk_write = AsyncMock(return_value=mock_result)
        mock_backend.db.__getitem__ = MagicMock(return_value=mock_collection)

        pipeline = self._make_pipeline(mock_backend)

        docs = [{"_id": "existing-id", "title": "has id"}]
        success, errors = await pipeline._write_batch_mongodb(
            docs, [0], IngestMode.UPSERT
        )

        call_args = mock_collection.bulk_write.call_args[0][0]
        assert call_args[0]._filter["_id"] == "existing-id"

    async def test_write_batch_mongodb_upsert_returns_actual_count(self):
        """Upsert returns matched_count + upserted_count, not len(operations)."""
        from semantic_vector_router.models import IngestMode

        mock_backend = MagicMock()
        mock_backend.db = MagicMock()

        mock_result = MagicMock()
        mock_result.matched_count = 2
        mock_result.upserted_count = 1
        # 3 docs sent, but only 3 accounted for (2 matched + 1 upserted)

        mock_collection = AsyncMock()
        mock_collection.bulk_write = AsyncMock(return_value=mock_result)
        mock_backend.db.__getitem__ = MagicMock(return_value=mock_collection)

        pipeline = self._make_pipeline(mock_backend)

        docs = [
            {"_id": "a", "t": "1"},
            {"_id": "b", "t": "2"},
            {"_id": "c", "t": "3"},
            {"_id": "d", "t": "4"},
            {"_id": "e", "t": "5"},
        ]
        success, errors = await pipeline._write_batch_mongodb(
            docs, [0, 1, 2, 3, 4], IngestMode.UPSERT
        )

        # Should be 3 (matched + upserted), NOT 5 (len(operations))
        assert success == 3
        assert errors == []


@pytest.mark.asyncio
class TestPhase14_5_ConvertVector:
    """14.5.3: _convert_vector guard for non-MongoDB backends."""

    def _make_pipeline(self, backend):
        from semantic_vector_router.ingestion import IngestPipeline

        config = _make_svr_config()
        embedder = MagicMock()
        metrics = MagicMock()
        return IngestPipeline(backend, config, embedder, metrics)

    def test_convert_vector_non_mongodb_returns_raw(self):
        """Backend without .db returns vector unchanged."""
        mock_backend = MagicMock(spec=[])  # No .db attribute
        pipeline = self._make_pipeline(mock_backend)

        vector = [0.1, 0.2, 0.3]
        result = pipeline._convert_vector(vector)
        assert result is vector  # Same object, not converted

    @patch("semantic_vector_router.ingestion.vector_to_bindata")
    def test_convert_vector_mongodb_calls_bindata(self, mock_bindata):
        """Backend with .db calls vector_to_bindata."""
        mock_backend = MagicMock()
        mock_backend.db = MagicMock()  # Has .db
        pipeline = self._make_pipeline(mock_backend)

        vector = [0.1, 0.2, 0.3]
        mock_bindata.return_value = b"binary"
        result = pipeline._convert_vector(vector)

        mock_bindata.assert_called_once_with(vector, pipeline._config.vector_storage.storage_format)
        assert result == b"binary"


class TestPhase14_5_SetParameterization:
    """14.5.4: _get_search_config_sql uses sql.Literal instead of f-strings."""

    def test_hnsw_returns_composed(self):
        """HNSW config returns a sql.Composed instance."""
        backend = _make_pg_backend(index_type=PgIndexType.HNSW)
        result = backend._get_search_config_sql()
        assert result is not None
        assert isinstance(result, sql.Composed)
        rendered = result.as_string(None)
        assert "hnsw.ef_search" in rendered
        assert str(backend._pg_config.hnsw.ef_search) in rendered

    def test_ivfflat_returns_composed(self):
        """IVFFlat config returns a sql.Composed instance."""
        backend = _make_pg_backend(index_type=PgIndexType.IVFFLAT)
        result = backend._get_search_config_sql()
        assert result is not None
        assert isinstance(result, sql.Composed)
        rendered = result.as_string(None)
        assert "ivfflat.probes" in rendered
        assert str(backend._pg_config.ivfflat.probes) in rendered

    def test_hnsw_literal_value_matches_config(self):
        """The rendered SQL contains the actual ef_search value from config."""
        pg_config = PostgresBackendConfig(index_type=PgIndexType.HNSW, hnsw=HnswConfig(ef_search=100))
        config = _make_svr_config(postgres=pg_config)
        backend = PostgresBackend(config)
        result = backend._get_search_config_sql()
        rendered = result.as_string(None)
        assert "100" in rendered
