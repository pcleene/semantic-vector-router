"""Unit tests for MongoDBBackend.

Tests the synchronous / non-connection-dependent parts of MongoDBBackend:
server version property, supports_search_index_on_views, and
_build_vector_search_pipeline for SOURCE, VIEWS, and FIELDS modes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pymongo.errors import OperationFailure, PyMongoError

from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    IndexLocation,
    MongoDBIndexQuantization,
    PartitionInfo,
    PartitioningConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
    VectorStorageMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> SVRConfig:
    """Create a minimal SVRConfig for testing pipeline construction."""
    return SVRConfig(
        database=DatabaseConfig(
            database="test_db",
            source_collection="products",
        ),
        partitioning=PartitioningConfig(
            field="category",
        ),
        vector_storage=VectorStorageConfig(
            storage_format=VectorStorageFormat.BINDATA_FLOAT32,
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=1536,
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
        ),
    )


@pytest.fixture
def backend(sample_config: SVRConfig) -> MongoDBBackend:
    """Create a MongoDBBackend without establishing a real connection."""
    be = MongoDBBackend.__new__(MongoDBBackend)
    be.config = sample_config
    be._client = None
    be._db = None
    be._server_version = None
    be._last_health_check = None
    from semantic_vector_router.backends.mongodb_views import MongoDBViewOps
    from semantic_vector_router.backends.mongodb_indexes import MongoDBIndexOps
    be._views = MongoDBViewOps(sample_config)
    be._indexes = MongoDBIndexOps()
    return be


def _make_query_vector(dim: int = 1536) -> list[float]:
    """Create a dummy query vector."""
    return [0.1] * dim


# ===========================================================================
# Server version properties
# ===========================================================================


class TestServerVersionProperty:
    """Tests for the server_version property."""

    def test_server_version_returns_set_value(self, backend: MongoDBBackend):
        """server_version returns the tuple set via _server_version."""
        backend._server_version = (8, 0, 19)
        assert backend.server_version == (8, 0, 19)

    def test_server_version_defaults_to_zero(self, backend: MongoDBBackend):
        """server_version returns (0, 0, 0) when not yet connected."""
        backend._server_version = None
        assert backend.server_version == (0, 0, 0)

    def test_server_version_five_digit(self, backend: MongoDBBackend):
        """server_version handles unusual version tuples."""
        backend._server_version = (7, 0, 5)
        assert backend.server_version == (7, 0, 5)


class TestSupportsSearchIndexOnViews:
    """Tests for the supports_search_index_on_views property."""

    def test_supports_search_index_on_views_81_plus(self, backend: MongoDBBackend):
        """Version >= (8, 1, 0) supports search indexes on views."""
        backend._server_version = (8, 1, 0)
        assert backend.supports_search_index_on_views is True

    def test_supports_search_index_on_views_82(self, backend: MongoDBBackend):
        """Version (8, 2, 0) also supports search indexes on views."""
        backend._server_version = (8, 2, 0)
        assert backend.supports_search_index_on_views is True

    def test_does_not_support_search_index_on_views_80(self, backend: MongoDBBackend):
        """Version (8, 0, 19) does NOT support search indexes on views."""
        backend._server_version = (8, 0, 19)
        assert backend.supports_search_index_on_views is False

    def test_does_not_support_search_index_on_views_70(self, backend: MongoDBBackend):
        """Version (7, 0, 0) does NOT support search indexes on views."""
        backend._server_version = (7, 0, 0)
        assert backend.supports_search_index_on_views is False

    def test_does_not_support_when_not_connected(self, backend: MongoDBBackend):
        """Not connected (version None) defaults to False."""
        backend._server_version = None
        assert backend.supports_search_index_on_views is False


# ===========================================================================
# _build_vector_search_pipeline — SOURCE mode
# ===========================================================================


class TestBuildPipelineSourceMode:
    """Tests for _build_vector_search_pipeline with IndexLocation.SOURCE."""

    def test_source_mode_adds_partition_filter(self, backend: MongoDBBackend):
        """SOURCE partition adds a partition field filter to the $vectorSearch stage."""
        partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert vs_stage["index"] == "svr_vector_idx_electronics"
        assert vs_stage["path"] == "embedding"
        assert vs_stage["limit"] == 10
        assert vs_stage["numCandidates"] == 100

        # Partition filter must be present
        assert "filter" in vs_stage
        assert vs_stage["filter"]["category"] == "electronics"

    def test_source_mode_with_custom_filter_expression(self, backend: MongoDBBackend):
        """SOURCE partition with filter_expression uses that expression as filter."""
        custom_filter = {"category": {"$in": ["electronics", "gadgets"]}}
        partition = PartitionInfo(
            name="electronics_gadgets",
            view_name="svr_partition_electronics_gadgets",
            index_name="svr_vector_idx_eg",
            filter_expression=custom_filter,
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=5,
            num_candidates=50,
            query_vector=_make_query_vector(),
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert "filter" in vs_stage
        assert vs_stage["filter"]["category"] == {"$in": ["electronics", "gadgets"]}

    def test_source_mode_adds_user_filters(self, backend: MongoDBBackend):
        """User-provided filters are merged into the $vectorSearch filter."""
        partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
            filters={"brand": "Sony"},
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert vs_stage["filter"]["category"] == "electronics"
        assert vs_stage["filter"]["brand"] == "Sony"


# ===========================================================================
# _build_vector_search_pipeline — VIEWS mode (pre-8.1)
# ===========================================================================


class TestBuildPipelineViewsPre81:
    """Tests for VIEWS partition where search_collection is the source collection.

    When the server is < 8.1, the index lives on the source collection, so a
    partition filter is still needed even in VIEWS mode.
    """

    def test_views_pre81_adds_filter(self, backend: MongoDBBackend):
        """VIEWS partition with search_collection == source adds partition filter."""
        backend._server_version = (8, 0, 19)

        partition = PartitionInfo(
            name="clothing",
            view_name="svr_partition_clothing",
            index_name="svr_vector_idx_clothing",
            filter_value="clothing",
            index_location=IndexLocation.VIEWS,
            search_collection="products",  # same as source_collection
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert "filter" in vs_stage
        assert vs_stage["filter"]["category"] == "clothing"


# ===========================================================================
# _build_vector_search_pipeline — VIEWS mode (8.1+)
# ===========================================================================


class TestBuildPipelineViews81Plus:
    """Tests for VIEWS partition where search_collection is the view itself.

    On MongoDB 8.1+, the index lives directly on the view, so the view
    already scopes data — no partition filter is needed.
    """

    def test_views_81_plus_no_filter(self, backend: MongoDBBackend):
        """VIEWS partition with search_collection == view_name does NOT add filter."""
        backend._server_version = (8, 1, 0)

        partition = PartitionInfo(
            name="clothing",
            view_name="svr_partition_clothing",
            index_name="svr_vector_idx_clothing",
            filter_value="clothing",
            index_location=IndexLocation.VIEWS,
            search_collection="svr_partition_clothing",  # the view itself
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        # No partition filter because the view scopes the data
        assert "filter" not in vs_stage

    def test_views_81_plus_user_filter_still_applied(self, backend: MongoDBBackend):
        """Even on 8.1+ VIEWS, user-provided filters are still included."""
        backend._server_version = (8, 1, 0)

        partition = PartitionInfo(
            name="clothing",
            view_name="svr_partition_clothing",
            index_name="svr_vector_idx_clothing",
            filter_value="clothing",
            index_location=IndexLocation.VIEWS,
            search_collection="svr_partition_clothing",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
            filters={"price": {"$lt": 100}},
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert "filter" in vs_stage
        # Only the user filter; no partition filter
        assert "category" not in vs_stage["filter"]
        assert vs_stage["filter"]["price"] == {"$lt": 100}


# ===========================================================================
# _build_vector_search_pipeline — FIELDS mode
# ===========================================================================


class TestBuildPipelineFieldsMode:
    """Tests for FIELDS partition which uses a partition-specific embedding field."""

    def test_fields_mode_uses_partition_embedding_field(self, backend: MongoDBBackend):
        """FIELDS partition uses partition.embedding_field as the vectorSearch path."""
        partition = PartitionInfo(
            name="electronics",
            view_name=None,  # FIELDS mode doesn't use views
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.FIELDS,
            search_collection="products",
            embedding_field="embedding_electronics",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert vs_stage["path"] == "embedding_electronics"
        assert vs_stage["index"] == "svr_vector_idx_electronics"

    def test_fields_mode_no_partition_filter(self, backend: MongoDBBackend):
        """FIELDS mode does not add a partition field filter.

        Each partition has its own dedicated embedding field and index,
        so filtering by partition value is unnecessary.
        """
        partition = PartitionInfo(
            name="clothing",
            view_name=None,
            index_name="svr_vector_idx_clothing",
            filter_value="clothing",
            index_location=IndexLocation.FIELDS,
            search_collection="products",
            embedding_field="embedding_clothing",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        # FIELDS mode: no partition filter
        assert "filter" not in vs_stage

    def test_fields_mode_with_user_filters(self, backend: MongoDBBackend):
        """FIELDS mode still applies user-provided filters."""
        partition = PartitionInfo(
            name="electronics",
            view_name=None,
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.FIELDS,
            search_collection="products",
            embedding_field="embedding_electronics",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=5,
            num_candidates=50,
            query_vector=_make_query_vector(),
            filters={"brand": "Apple"},
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert "filter" in vs_stage
        assert vs_stage["filter"]["brand"] == "Apple"
        # No partition filter
        assert "category" not in vs_stage["filter"]


# ===========================================================================
# _build_vector_search_pipeline — pipeline structure checks
# ===========================================================================


class TestBuildPipelineStructure:
    """Tests for the overall structure of the generated pipeline."""

    def test_pipeline_has_two_stages(self, backend: MongoDBBackend):
        """Pipeline always has $vectorSearch followed by $addFields."""
        partition = PartitionInfo(
            name="test",
            view_name="svr_partition_test",
            index_name="svr_vector_idx_test",
            filter_value="test",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
        )

        assert len(pipeline) == 2
        assert "$vectorSearch" in pipeline[0]
        assert "$addFields" in pipeline[1]

    def test_add_fields_includes_partition_and_score(self, backend: MongoDBBackend):
        """The $addFields stage adds _svr_partition and _svr_score."""
        partition = PartitionInfo(
            name="mypartition",
            view_name="svr_partition_mypartition",
            index_name="svr_vector_idx_mypartition",
            filter_value="mypartition",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=_make_query_vector(),
        )

        add_fields = pipeline[1]["$addFields"]
        assert add_fields["_svr_partition"] == "mypartition"
        assert add_fields["_svr_score"] == {"$meta": "vectorSearchScore"}

    def test_query_vector_is_set_in_byom_mode(self, backend: MongoDBBackend):
        """In BYOM mode, queryVector is set from the provided query_vector."""
        partition = PartitionInfo(
            name="test",
            view_name="svr_partition_test",
            index_name="svr_vector_idx_test",
            filter_value="test",
            index_location=IndexLocation.VIEWS,
            search_collection="svr_partition_test",
        )

        qv = _make_query_vector()
        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=qv,
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert vs_stage["queryVector"] == qv
        assert "queryString" not in vs_stage

    def test_auto_mode_uses_query_string(self, backend: MongoDBBackend, sample_config: SVRConfig):
        """In AUTO mode, queryString is set from the query parameter."""
        # Switch to AUTO embedding mode
        sample_config.embedding.mode = EmbeddingMode.AUTO

        partition = PartitionInfo(
            name="test",
            view_name="svr_partition_test",
            index_name="svr_vector_idx_test",
            filter_value="test",
            index_location=IndexLocation.VIEWS,
            search_collection="svr_partition_test",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_string="what are the best headphones",
        )

        vs_stage = pipeline[0]["$vectorSearch"]
        assert vs_stage["queryString"] == "what are the best headphones"
        assert "queryVector" not in vs_stage


# ===========================================================================
# execute_search uses correct collection
# ===========================================================================


class TestExecuteSearchCollection:
    """Tests verifying which collection is used for each mode.

    Since execute_search requires an actual DB connection, we verify the
    search_collection logic by inspecting the PartitionInfo attributes
    that determine collection selection in execute_search.
    """

    def test_source_mode_search_collection_is_source(self):
        """SOURCE mode sets search_collection to the source collection name."""
        partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        # execute_search uses: partition.search_collection or partition.view_name
        effective = partition.search_collection or partition.view_name
        assert effective == "products"

    def test_views_mode_search_collection_is_view(self):
        """VIEWS mode (8.1+) sets search_collection to the partition view."""
        partition = PartitionInfo(
            name="clothing",
            view_name="svr_partition_clothing",
            index_name="svr_vector_idx_clothing",
            filter_value="clothing",
            index_location=IndexLocation.VIEWS,
            search_collection="svr_partition_clothing",
        )

        effective = partition.search_collection or partition.view_name
        assert effective == "svr_partition_clothing"

    def test_views_mode_pre81_search_collection_is_source(self):
        """VIEWS mode (pre-8.1) sets search_collection to source collection."""
        partition = PartitionInfo(
            name="clothing",
            view_name="svr_partition_clothing",
            index_name="svr_vector_idx_clothing",
            filter_value="clothing",
            index_location=IndexLocation.VIEWS,
            search_collection="products",
        )

        effective = partition.search_collection or partition.view_name
        assert effective == "products"

    def test_fields_mode_search_collection_is_source(self):
        """FIELDS mode sets search_collection to the source collection."""
        partition = PartitionInfo(
            name="electronics",
            view_name=None,
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.FIELDS,
            search_collection="products",
            embedding_field="embedding_electronics",
        )

        effective = partition.search_collection or partition.view_name
        assert effective == "products"

    def test_fallback_to_view_name_when_search_collection_none(self):
        """When search_collection is None, falls back to view_name."""
        partition = PartitionInfo(
            name="clothing",
            view_name="svr_partition_clothing",
            index_name="svr_vector_idx_clothing",
            filter_value="clothing",
            index_location=IndexLocation.VIEWS,
            search_collection=None,
        )

        effective = partition.search_collection or partition.view_name
        assert effective == "svr_partition_clothing"


# ===========================================================================
# Connection management
# ===========================================================================


class TestConnectionProperties:
    def test_client_property_raises_when_not_connected(self, backend):
        from semantic_vector_router.exceptions import ConnectionError
        with pytest.raises(ConnectionError, match="Not connected"):
            _ = backend.client

    def test_db_property_raises_when_not_connected(self, backend):
        from semantic_vector_router.exceptions import ConnectionError
        with pytest.raises(ConnectionError, match="Not connected"):
            _ = backend.db

    def test_client_property_returns_client(self, backend):
        backend._client = MagicMock()
        assert backend.client is backend._client

    def test_db_property_returns_db(self, backend):
        backend._db = MagicMock()
        assert backend.db is backend._db


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self, sample_config):
        backend = MongoDBBackend(sample_config)
        mock_client = AsyncMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_client.server_info = AsyncMock(return_value={"version": "8.0.19"})
        mock_db = AsyncMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)

        with patch('semantic_vector_router.backends.mongodb.backend.AsyncMongoClient', return_value=mock_client), \
             patch('semantic_vector_router.backends.mongodb.backend.get_connection_string', return_value="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<db>"):
            await backend.connect()

        assert backend._server_version == (8, 0, 19)
        assert backend._client is mock_client
        mock_client.admin.command.assert_awaited_once_with("ping")

    @pytest.mark.asyncio
    async def test_connect_failure(self, sample_config):
        from semantic_vector_router.exceptions import ConnectionError as SVRConnectionError
        backend = MongoDBBackend(sample_config)

        with patch('semantic_vector_router.backends.mongodb.backend.AsyncMongoClient', side_effect=PyMongoError("fail")), \
             patch('semantic_vector_router.backends.mongodb.backend.get_connection_string', return_value="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<db>"):
            with pytest.raises(SVRConnectionError, match="Failed to connect"):
                await backend.connect()


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self, backend):
        mock_client = AsyncMock()
        backend._client = mock_client
        backend._db = AsyncMock()

        await backend.disconnect()

        mock_client.close.assert_awaited_once()
        assert backend._client is None
        assert backend._db is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, backend):
        backend._client = None
        await backend.disconnect()  # Should not raise


class TestIsConnected:
    @pytest.mark.asyncio
    async def test_is_connected_true(self, backend):
        mock_client = AsyncMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        backend._client = mock_client
        assert await backend.is_connected() is True

    @pytest.mark.asyncio
    async def test_is_connected_false_no_client(self, backend):
        backend._client = None
        assert await backend.is_connected() is False

    @pytest.mark.asyncio
    async def test_is_connected_false_ping_fails(self, backend):
        mock_client = AsyncMock()
        mock_client.admin.command = AsyncMock(side_effect=PyMongoError("timeout"))
        backend._client = mock_client
        assert await backend.is_connected() is False


# ===========================================================================
# View management
# ===========================================================================


class TestBuildPartitionViewPipeline:
    def test_simple_filter(self, backend):
        pipeline = backend._build_partition_view_pipeline("electronics", "electronics")
        assert pipeline[0] == {"$match": {"category": "electronics"}}

    def test_custom_filter_expression(self, backend):
        custom = {"category": {"$in": ["a", "b"]}}
        pipeline = backend._build_partition_view_pipeline("test", "test", filter_expression=custom)
        assert pipeline[0] == {"$match": custom}

    def test_with_source_fields(self, backend, sample_config):
        sample_config.embedding.source_fields = ["title", "description"]
        sample_config.embedding.computed_field = "embedding_text"
        pipeline = backend._build_partition_view_pipeline("test", "test")
        assert len(pipeline) == 2
        assert "$addFields" in pipeline[1]
        assert "embedding_text" in pipeline[1]["$addFields"]

    def test_separate_storage_adds_lookup(self, backend, sample_config):
        sample_config.vector_storage.mode = "separate"
        sample_config.vector_storage.embeddings_collection = "embeddings"
        sample_config.vector_storage.reference_field = "doc_id"
        pipeline = backend._build_partition_view_pipeline("test", "test")
        stage_types = [list(s.keys())[0] for s in pipeline]
        assert "$lookup" in stage_types
        assert "$unwind" in stage_types


class TestBuildConcatExpression:
    def test_no_source_fields(self, backend, sample_config):
        sample_config.embedding.source_fields = None
        result = backend._build_concat_expression()
        assert result == {"$literal": ""}

    def test_with_separator(self, backend, sample_config):
        """BYOM mode (default) now produces an object projection, not $concat."""
        sample_config.embedding.source_fields = ["title", "description"]
        sample_config.embedding.template = None
        sample_config.embedding.separator = " | "
        result = backend._build_concat_expression()
        # BYOM default is object projection (Phase 16)
        assert "title" in result
        assert "description" in result

    def test_with_template(self, backend, sample_config):
        sample_config.embedding.source_fields = ["title", "description"]
        sample_config.embedding.template = "{title}\n{description}"
        result = backend._build_concat_expression()
        assert "$concat" in result


class TestBuildTemplateExpression:
    def test_simple_template(self, backend):
        result = backend._build_template_expression("{title}: {desc}", ["title", "desc"])
        parts = result["$concat"]
        assert any(p == ": " for p in parts if isinstance(p, str))

    def test_template_with_trailing_text(self, backend):
        result = backend._build_template_expression("{title}!", ["title"])
        parts = result["$concat"]
        assert parts[-1] == "!"


class TestCreatePartitionView:
    @pytest.mark.asyncio
    async def test_create_view_success(self, backend, sample_config):
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(return_value=[])
        mock_db.command = AsyncMock(return_value={"ok": 1})
        backend._db = mock_db

        view_name = await backend.create_partition_view("electronics", "electronics")
        assert view_name == f"{sample_config.partitioning.view_prefix}electronics"
        mock_db.command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_view_drops_existing(self, backend, sample_config):
        view_name = f"{sample_config.partitioning.view_prefix}electronics"
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(return_value=[view_name])
        mock_db.drop_collection = AsyncMock()
        mock_db.command = AsyncMock(return_value={"ok": 1})
        backend._db = mock_db

        await backend.create_partition_view("electronics", "electronics")
        mock_db.drop_collection.assert_awaited_once_with(view_name)

    @pytest.mark.asyncio
    async def test_create_view_operation_failure(self, backend):
        from semantic_vector_router.exceptions import ViewCreationError
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(return_value=[])
        mock_db.command = AsyncMock(side_effect=OperationFailure("fail"))
        backend._db = mock_db

        with pytest.raises(ViewCreationError):
            await backend.create_partition_view("test", "test")


class TestDeletePartitionView:
    @pytest.mark.asyncio
    async def test_delete_view_success(self, backend):
        mock_db = AsyncMock()
        mock_db.drop_collection = AsyncMock()
        backend._db = mock_db
        await backend.delete_partition_view("test_view")
        mock_db.drop_collection.assert_awaited_once_with("test_view")

    @pytest.mark.asyncio
    async def test_delete_view_failure(self, backend):
        from semantic_vector_router.exceptions import SVRException
        mock_db = AsyncMock()
        mock_db.drop_collection = AsyncMock(side_effect=PyMongoError("fail"))
        backend._db = mock_db
        with pytest.raises(SVRException):
            await backend.delete_partition_view("test_view")


class TestViewExists:
    @pytest.mark.asyncio
    async def test_view_exists_true(self, backend):
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(return_value=["my_view"])
        backend._db = mock_db
        assert await backend.view_exists("my_view") is True

    @pytest.mark.asyncio
    async def test_view_exists_false(self, backend):
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(return_value=["other"])
        backend._db = mock_db
        assert await backend.view_exists("my_view") is False

    @pytest.mark.asyncio
    async def test_view_exists_error_returns_false(self, backend):
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(side_effect=PyMongoError("err"))
        backend._db = mock_db
        assert await backend.view_exists("my_view") is False


# ===========================================================================
# Index management
# ===========================================================================


class TestCreateVectorSearchIndex:
    @pytest.mark.asyncio
    async def test_create_index_basic(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.create_search_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        await backend.create_vector_search_index(
            "products", "idx_test", "embedding", 1536, "cosine"
        )
        mock_collection.create_search_index.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_index_with_quantization(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.create_search_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        await backend.create_vector_search_index(
            "products", "idx_test", "embedding", 1536, "cosine",
            quantization=MongoDBIndexQuantization.SCALAR,
        )
        call_args = mock_collection.create_search_index.call_args[0][0]
        fields = call_args["definition"]["fields"]
        assert fields[0].get("quantization") == "scalar"

    @pytest.mark.asyncio
    async def test_create_index_with_filter_fields(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.create_search_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        await backend.create_vector_search_index(
            "products", "idx_test", "embedding", 1536, "cosine",
            filter_fields=["category", "brand"],
        )
        call_args = mock_collection.create_search_index.call_args[0][0]
        fields = call_args["definition"]["fields"]
        filter_paths = [f["path"] for f in fields if f["type"] == "filter"]
        assert "category" in filter_paths
        assert "brand" in filter_paths

    @pytest.mark.asyncio
    async def test_create_index_already_exists(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.create_search_index = AsyncMock(
            side_effect=OperationFailure("index already exists", code=68)
        )
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        # Should not raise - gracefully handled
        await backend.create_vector_search_index(
            "products", "idx_test", "embedding", 1536, "cosine"
        )

    @pytest.mark.asyncio
    async def test_create_index_other_failure(self, backend):
        from semantic_vector_router.exceptions import IndexCreationError
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.create_search_index = AsyncMock(
            side_effect=OperationFailure("some other error", code=999)
        )
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        with pytest.raises(IndexCreationError):
            await backend.create_vector_search_index(
                "products", "idx_test", "embedding", 1536, "cosine"
            )


class TestDeleteVectorSearchIndex:
    @pytest.mark.asyncio
    async def test_delete_index_success(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.drop_search_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        await backend.delete_vector_search_index("products", "idx_test")
        mock_collection.drop_search_index.assert_awaited_once_with("idx_test")

    @pytest.mark.asyncio
    async def test_delete_index_failure(self, backend):
        from semantic_vector_router.exceptions import SVRException
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.drop_search_index = AsyncMock(side_effect=PyMongoError("fail"))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        with pytest.raises(SVRException):
            await backend.delete_vector_search_index("products", "idx_test")


class TestIndexExists:
    @pytest.mark.asyncio
    async def test_index_exists_true(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[{"name": "idx_test"}])
        mock_collection.list_search_indexes = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        assert await backend.index_exists("products", "idx_test") is True

    @pytest.mark.asyncio
    async def test_index_exists_false(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[{"name": "other_idx"}])
        mock_collection.list_search_indexes = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        assert await backend.index_exists("products", "idx_test") is False

    @pytest.mark.asyncio
    async def test_index_exists_error(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.list_search_indexes = AsyncMock(side_effect=PyMongoError("err"))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        assert await backend.index_exists("products", "idx_test") is False


class TestGetIndexStatus:
    @pytest.mark.asyncio
    async def test_get_index_status_found(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"name": "idx_test", "status": "READY", "type": "vectorSearch", "queryable": True}
        ])
        mock_collection.list_search_indexes = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        status = await backend.get_index_status("products", "idx_test")
        assert status["status"] == "READY"
        assert status["queryable"] is True

    @pytest.mark.asyncio
    async def test_get_index_status_not_found(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.list_search_indexes = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        status = await backend.get_index_status("products", "idx_test")
        assert status["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_index_status_error(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.list_search_indexes = AsyncMock(side_effect=PyMongoError("err"))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        status = await backend.get_index_status("products", "idx_test")
        assert status["status"] == "error"


# ===========================================================================
# Search operations
# ===========================================================================


class TestExecuteSearch:
    @pytest.mark.asyncio
    async def test_execute_search_success(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": "doc1", "_svr_score": 0.95, "_svr_partition": "electronics"}
        ])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        results = await backend.execute_search(
            partition, _make_query_vector(), limit=10, num_candidates=100
        )
        assert len(results) == 1
        assert results[0]["_svr_score"] == 0.95

    @pytest.mark.asyncio
    async def test_execute_search_uses_search_collection(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        partition = PartitionInfo(
            name="test", view_name="view_test", index_name="idx",
            filter_value="test", search_collection="my_collection",
        )
        await backend.execute_search(partition, _make_query_vector(), 10, 100)
        mock_db.__getitem__.assert_called_with("my_collection")

    @pytest.mark.asyncio
    async def test_execute_search_falls_back_to_view_name(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        partition = PartitionInfo(
            name="test", view_name="view_test", index_name="idx",
            filter_value="test", search_collection=None,
        )
        await backend.execute_search(partition, _make_query_vector(), 10, 100)
        mock_db.__getitem__.assert_called_with("view_test")

    @pytest.mark.asyncio
    async def test_execute_search_operation_failure(self, backend):
        from semantic_vector_router.exceptions import SearchError
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(side_effect=OperationFailure("fail", code=1))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        partition = PartitionInfo(
            name="test", view_name="view_test", index_name="idx",
            filter_value="test", search_collection="products",
        )
        with pytest.raises(SearchError):
            await backend.execute_search(partition, _make_query_vector(), 10, 100)

    @pytest.mark.asyncio
    async def test_build_pipeline_missing_query_vector_byom(self, backend):
        from semantic_vector_router.exceptions import SearchError
        partition = PartitionInfo(
            name="test", view_name="v", index_name="idx",
            filter_value="test", search_collection="products",
        )
        with pytest.raises(SearchError, match="query_vector or query_string must be provided"):
            backend._build_vector_search_pipeline(partition, 10, 100)


class TestSearchPartitions:
    @pytest.mark.asyncio
    async def test_search_partitions_empty(self, backend):
        result = await backend.search_partitions([], 10, query_vector=[0.1]*1536)
        assert result == []

    @pytest.mark.asyncio
    async def test_search_partitions_combines_results(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": "doc1", "_svr_score": 0.9}
        ])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        p1 = PartitionInfo(name="a", view_name="va", index_name="idx", filter_value="a", search_collection="products")
        p2 = PartitionInfo(name="b", view_name="vb", index_name="idx", filter_value="b", search_collection="products")
        results = await backend.search_partitions([p1, p2], 10, query_vector=_make_query_vector())
        assert len(results) == 2  # one from each partition

    @pytest.mark.asyncio
    async def test_search_partitions_handles_exceptions(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(side_effect=OperationFailure("fail", code=1))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        p1 = PartitionInfo(name="a", view_name="va", index_name="idx", filter_value="a", search_collection="products")
        results = await backend.search_partitions([p1], 10, query_vector=_make_query_vector())
        assert results == []  # error logged, not raised


# ===========================================================================
# Collection operations
# ===========================================================================


class TestGetDistinctValues:
    @pytest.mark.asyncio
    async def test_get_distinct_no_filter(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(return_value=["a", "b"])
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        result = await backend.get_distinct_values("category")
        assert result == ["a", "b"]
        mock_collection.distinct.assert_awaited_once_with("category")

    @pytest.mark.asyncio
    async def test_get_distinct_with_filter(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(return_value=["a"])
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        filt = {"brand": "Sony"}
        result = await backend.get_distinct_values("category", filter_expression=filt)
        mock_collection.distinct.assert_awaited_once_with("category", filt)

    @pytest.mark.asyncio
    async def test_get_distinct_error(self, backend):
        from semantic_vector_router.exceptions import SVRException
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(side_effect=PyMongoError("err"))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        with pytest.raises(SVRException):
            await backend.get_distinct_values("category")


class TestCountDocuments:
    @pytest.mark.asyncio
    async def test_count_default_collection(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.count_documents = AsyncMock(return_value=42)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        count = await backend.count_documents()
        assert count == 42

    @pytest.mark.asyncio
    async def test_count_with_filter(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.count_documents = AsyncMock(return_value=10)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        count = await backend.count_documents(filter_expression={"status": "active"})
        assert count == 10
        mock_collection.count_documents.assert_awaited_once_with({"status": "active"})

    @pytest.mark.asyncio
    async def test_count_custom_collection(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.count_documents = AsyncMock(return_value=5)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        await backend.count_documents(collection_name="custom_coll")
        mock_db.__getitem__.assert_called_with("custom_coll")

    @pytest.mark.asyncio
    async def test_count_error(self, backend):
        from semantic_vector_router.exceptions import SVRException
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.count_documents = AsyncMock(side_effect=PyMongoError("err"))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        with pytest.raises(SVRException):
            await backend.count_documents()


class TestGetPartitionDocumentCounts:
    @pytest.mark.asyncio
    async def test_get_counts(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": "electronics", "count": 100},
            {"_id": "clothing", "count": 50},
            {"_id": None, "count": 5},
        ])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        counts = await backend.get_partition_document_counts("category")
        assert counts == {"electronics": 100, "clothing": 50}
        assert None not in counts  # None filtered out

    @pytest.mark.asyncio
    async def test_get_counts_error(self, backend):
        from semantic_vector_router.exceptions import SVRException
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(side_effect=PyMongoError("err"))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        with pytest.raises(SVRException):
            await backend.get_partition_document_counts("category")


class TestGetCollectionStats:
    @pytest.mark.asyncio
    async def test_get_stats_success(self, backend):
        mock_db = AsyncMock()
        mock_db.command = AsyncMock(return_value={
            "count": 100, "size": 1024, "avgObjSize": 10,
            "storageSize": 2048, "nindexes": 3,
        })
        backend._db = mock_db

        stats = await backend.get_collection_stats()
        assert stats["count"] == 100
        assert stats["indexes"] == 3

    @pytest.mark.asyncio
    async def test_get_stats_error(self, backend):
        mock_db = AsyncMock()
        mock_db.command = AsyncMock(side_effect=PyMongoError("err"))
        backend._db = mock_db

        stats = await backend.get_collection_stats()
        assert "error" in stats


# ===========================================================================
# Utility methods
# ===========================================================================


class TestListViews:
    @pytest.mark.asyncio
    async def test_list_views_success(self, backend):
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(return_value=["view1", "view2"])
        backend._db = mock_db

        views = await backend.list_views()
        assert views == ["view1", "view2"]

    @pytest.mark.asyncio
    async def test_list_views_error(self, backend):
        from semantic_vector_router.exceptions import SVRException
        mock_db = AsyncMock()
        mock_db.list_collection_names = AsyncMock(side_effect=PyMongoError("err"))
        backend._db = mock_db

        with pytest.raises(SVRException):
            await backend.list_views()


class TestListPartitionViews:
    @pytest.mark.asyncio
    async def test_list_partition_views(self, backend, sample_config):
        mock_db = AsyncMock()
        prefix = sample_config.partitioning.view_prefix
        mock_db.list_collection_names = AsyncMock(
            return_value=[f"{prefix}electronics", f"{prefix}clothing", "other_view"]
        )
        backend._db = mock_db

        views = await backend.list_partition_views()
        assert len(views) == 2
        assert all(v.startswith(prefix) for v in views)


class TestGetSampleDocument:
    @pytest.mark.asyncio
    async def test_get_sample_success(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value={"_id": 1, "title": "Test"})
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        doc = await backend.get_sample_document()
        assert doc["title"] == "Test"

    @pytest.mark.asyncio
    async def test_get_sample_empty(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        doc = await backend.get_sample_document()
        assert doc is None

    @pytest.mark.asyncio
    async def test_get_sample_error(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(side_effect=PyMongoError("err"))
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        doc = await backend.get_sample_document()
        assert doc is None


class TestGetFieldNames:
    @pytest.mark.asyncio
    async def test_get_field_names_success(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value={"_id": 1, "title": "T", "category": "C"})
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        fields = await backend.get_field_names()
        assert "_id" in fields
        assert "title" in fields

    @pytest.mark.asyncio
    async def test_get_field_names_empty(self, backend):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        fields = await backend.get_field_names()
        assert fields == []


# ===========================================================================
# Phase 3 — Health check
# ===========================================================================


class TestHealthCheck:
    """Tests for the health_check method with staleness caching."""

    @pytest.mark.asyncio
    async def test_health_check_fresh_ping(self, backend):
        """Client exists, no previous check -> pings, returns True, sets _last_health_check."""
        import time

        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        backend._client = mock_client
        backend._last_health_check = None

        result = await backend.health_check()

        assert result is True
        assert backend._last_health_check is not None
        assert backend._last_health_check <= time.monotonic()
        mock_client.admin.command.assert_awaited_once_with("ping")

    @pytest.mark.asyncio
    async def test_health_check_cached_skip(self, backend):
        """_last_health_check was set recently (within interval) -> returns True WITHOUT calling ping."""
        import time

        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock()
        backend._client = mock_client
        # Set _last_health_check to just now (well within the default 30s interval)
        backend._last_health_check = time.monotonic()

        result = await backend.health_check()

        assert result is True
        mock_client.admin.command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_health_check_stale_repings(self, backend):
        """_last_health_check was set long ago -> pings again."""
        import time

        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        backend._client = mock_client
        # Set _last_health_check far in the past (beyond the default 30s interval)
        backend._last_health_check = time.monotonic() - 60

        result = await backend.health_check()

        assert result is True
        mock_client.admin.command.assert_awaited_once_with("ping")
        # _last_health_check should have been refreshed
        assert backend._last_health_check > time.monotonic() - 5

    @pytest.mark.asyncio
    async def test_health_check_failed(self, backend):
        """Ping raises PyMongoError -> returns False, clears _last_health_check."""
        import time

        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(side_effect=PyMongoError("timeout"))
        backend._client = mock_client
        backend._last_health_check = time.monotonic() - 60  # stale, will trigger ping

        result = await backend.health_check()

        assert result is False
        assert backend._last_health_check is None

    @pytest.mark.asyncio
    async def test_health_check_no_client(self, backend):
        """_client is None -> returns False."""
        backend._client = None
        backend._last_health_check = None

        result = await backend.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_disconnect(self, backend):
        """After disconnect(), health_check returns False."""
        mock_client = AsyncMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        backend._client = mock_client
        backend._db = AsyncMock()
        backend._last_health_check = 12345.0

        await backend.disconnect()

        # After disconnect, _client is None
        result = await backend.health_check()
        assert result is False


# ===========================================================================
# Phase 3 — Connection timeouts
# ===========================================================================


class TestConnectionTimeouts:
    """Tests verifying connect() passes timeout config to AsyncMongoClient."""

    @pytest.mark.asyncio
    async def test_connect_passes_timeout_config(self, sample_config):
        """Verify AsyncMongoClient is called with connectTimeoutMS and serverSelectionTimeoutMS."""
        backend = MongoDBBackend(sample_config)
        mock_client = AsyncMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_client.server_info = AsyncMock(return_value={"version": "8.0.19"})
        mock_db = AsyncMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)

        with patch('semantic_vector_router.backends.mongodb.backend.AsyncMongoClient', return_value=mock_client) as mock_ctor, \
             patch('semantic_vector_router.backends.mongodb.backend.get_connection_string', return_value="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<db>"):
            await backend.connect()

        expected_connect_ms = sample_config.resilience.connection_timeout_ms
        expected_server_ms = sample_config.resilience.server_selection_timeout_ms

        mock_ctor.assert_called_once_with(
            "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<db>",
            connectTimeoutMS=expected_connect_ms,
            serverSelectionTimeoutMS=expected_server_ms,
            maxPoolSize=sample_config.database.max_pool_size,
            minPoolSize=sample_config.database.min_pool_size,
            maxIdleTimeMS=sample_config.database.max_idle_time_ms or None,
            waitQueueTimeoutMS=sample_config.database.wait_queue_timeout_ms or None,
        )


# ===========================================================================
# Phase 3 — Search timeout
# ===========================================================================


class TestSearchTimeout:
    """Tests verifying execute_search passes maxTimeMS from resilience config."""

    @pytest.mark.asyncio
    async def test_execute_search_passes_max_time_ms(self, backend):
        """Verify collection.aggregate is called with maxTimeMS parameter from config."""
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        partition = PartitionInfo(
            name="test",
            view_name="svr_partition_test",
            index_name="svr_vector_idx_test",
            filter_value="test",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        await backend.execute_search(
            partition, _make_query_vector(), limit=10, num_candidates=100
        )

        expected_timeout = backend.config.resilience.search_timeout_ms
        # Verify aggregate was called with the pipeline and maxTimeMS kwarg
        mock_collection.aggregate.assert_awaited_once()
        call_kwargs = mock_collection.aggregate.call_args
        assert call_kwargs.kwargs.get("maxTimeMS") == expected_timeout or \
               call_kwargs[1].get("maxTimeMS") == expected_timeout


# ===========================================================================
# Phase 3 — Retry on search
# ===========================================================================


class TestRetryOnSearch:
    """Tests verifying execute_search retries on transient errors."""

    @pytest.mark.asyncio
    async def test_search_retries_on_auto_reconnect(self, backend):
        """execute_search retries when AutoReconnect is raised, succeeds on second attempt."""
        from pymongo.errors import AutoReconnect

        mock_db = AsyncMock()
        mock_collection = AsyncMock()

        # First call: raises AutoReconnect, second call: succeeds
        mock_cursor_success = AsyncMock()
        mock_cursor_success.to_list = AsyncMock(return_value=[
            {"_id": "doc1", "_svr_score": 0.85, "_svr_partition": "test"}
        ])
        mock_collection.aggregate = AsyncMock(
            side_effect=[AutoReconnect("connection lost"), mock_cursor_success]
        )
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend._db = mock_db

        partition = PartitionInfo(
            name="test",
            view_name="svr_partition_test",
            index_name="svr_vector_idx_test",
            filter_value="test",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        results = await backend.execute_search(
            partition, _make_query_vector(), limit=10, num_candidates=100
        )

        assert len(results) == 1
        assert results[0]["_svr_score"] == 0.85
        # aggregate was called twice: first raised AutoReconnect, second succeeded
        assert mock_collection.aggregate.await_count == 2
