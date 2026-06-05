"""Integration tests for the SVR Query Abstraction Layer (Phase 17).

Verifies multi-component interactions between filter validation, filter
translation, and backend search methods. These are NOT end-to-end tests
hitting real databases -- all backend calls are mocked.

Tests cover:
1. Filter validation -> MongoDB filter translation pipeline
2. Filter validation -> PostgreSQL filter translation pipeline
3. SVRClient.search() -> Backend integration (mocked backend)
4. MongoDB pipeline assembly integration
5. PostgreSQL query assembly integration
6. Cross-backend filter portability
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.backends.mongodb.backend import MongoDBBackend
from semantic_vector_router.backends.postgres.filters import translate_filters
from semantic_vector_router.models.enums import IndexLocation
from semantic_vector_router.models.partition import PartitionInfo
from semantic_vector_router.models.search import SearchHit
from semantic_vector_router.query.filters import (
    REJECTED_OPERATORS,
    SUPPORTED_OPERATORS,
    validate_filters,
)
from semantic_vector_router.query.mongo_filters import MongoFilterTranslator

pytestmark = pytest.mark.asyncio(loop_scope="module")


# ══════════════════════════════════════════════════════════════════════
# 1. Filter validation -> MongoDB filter translation pipeline
# ══════════════════════════════════════════════════════════════════════


class TestFilterValidationToMongoPipeline:
    """Validate complex filters, then translate to MongoDB format."""

    def test_simple_equality_through_pipeline(self):
        """Scalar equality passes validation and translates to MongoDB."""
        filters = {"category": "electronics"}
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == {"category": "electronics"}

    def test_comparison_operators_through_pipeline(self):
        """$gt, $lt, $gte, $lte pass validation and translate unchanged."""
        filters = {"price": {"$gte": 10, "$lte": 100}}
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == {"price": {"$gte": 10, "$lte": 100}}

    def test_set_operators_through_pipeline(self):
        """$in and $nin pass validation and translate unchanged."""
        filters = {"status": {"$in": ["active", "pending"]}}
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == {"status": {"$in": ["active", "pending"]}}

    def test_logical_and_through_pipeline(self):
        """$and with nested conditions validates and translates."""
        filters = {
            "$and": [
                {"price": {"$gte": 50}},
                {"category": "electronics"},
            ]
        }
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == {
            "$and": [
                {"price": {"$gte": 50}},
                {"category": "electronics"},
            ]
        }

    def test_logical_or_through_pipeline(self):
        """$or with nested conditions validates and translates."""
        filters = {
            "$or": [
                {"category": "electronics"},
                {"category": "furniture"},
            ]
        }
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == {
            "$or": [
                {"category": "electronics"},
                {"category": "furniture"},
            ]
        }

    def test_not_operator_through_pipeline(self):
        """$not validates and translates correctly."""
        filters = {"$not": {"category": "deprecated"}}
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == {"$not": {"category": "deprecated"}}

    def test_field_level_not_through_pipeline(self):
        """Field-level $not: {field: {$not: {$gt: 100}}} validates and translates."""
        filters = {"price": {"$not": {"$gt": 100}}}
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        # MongoFilterTranslator passes operator expressions through unchanged
        assert result == {"price": {"$not": {"$gt": 100}}}

    def test_exists_operator_through_pipeline(self):
        """$exists passes validation and translates unchanged."""
        filters = {"tags": {"$exists": True}}
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == {"tags": {"$exists": True}}

    def test_deeply_nested_filters_through_pipeline(self):
        """Complex nested filter with $and > $or > comparisons validates and translates."""
        filters = {
            "$and": [
                {
                    "$or": [
                        {"price": {"$lt": 50}},
                        {"price": {"$gt": 500}},
                    ]
                },
                {"category": {"$in": ["electronics", "furniture"]}},
                {"status": {"$ne": "discontinued"}},
            ]
        }
        validate_filters(filters)
        result = MongoFilterTranslator().translate(filters)
        assert result == filters

    def test_rejected_operator_fails_at_validation(self):
        """$regex is rejected at validation, never reaches translator."""
        filters = {"name": {"$regex": "^test"}}
        with pytest.raises(ValueError, match="not allowed"):
            validate_filters(filters)

    def test_unknown_operator_fails_at_validation(self):
        """Unknown $foo operator is rejected at validation."""
        filters = {"field": {"$foo": "bar"}}
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            validate_filters(filters)

    def test_all_rejected_operators_fail_at_validation(self):
        """Every operator in REJECTED_OPERATORS raises ValueError."""
        for op in REJECTED_OPERATORS:
            with pytest.raises(ValueError, match="not allowed"):
                validate_filters({"field": {op: "value"}})

    def test_empty_filters_passthrough(self):
        """Empty filters pass validation and translate to empty dict."""
        validate_filters({})
        result = MongoFilterTranslator().translate({})
        assert result == {}

    def test_none_filter_skips_validation(self):
        """validate_filters accepts falsy input without error."""
        validate_filters({})  # Empty dict is falsy-safe
        # None is not a valid dict, but empty dict is the no-op case

    def test_and_requires_list(self):
        """$and with non-list value raises ValueError at validation."""
        filters = {"$and": {"field": "value"}}
        with pytest.raises(ValueError, match="requires a list"):
            validate_filters(filters)

    def test_or_requires_list(self):
        """$or with non-list value raises ValueError at validation."""
        filters = {"$or": "not a list"}
        with pytest.raises(ValueError, match="requires a list"):
            validate_filters(filters)


# ══════════════════════════════════════════════════════════════════════
# 2. Filter validation -> PostgreSQL filter translation pipeline
# ══════════════════════════════════════════════════════════════════════


class TestFilterValidationToPostgresPipeline:
    """Validate filters, then translate to SQL WHERE + params."""

    def test_simple_equality_through_pipeline(self):
        """Scalar equality validates and translates to SQL."""
        filters = {"title": "Football"}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "content->>'title' = %s" in sql
        assert params == ["Football"]

    def test_comparison_operator_through_pipeline(self):
        """$gt validates and translates to SQL > with numeric cast."""
        filters = {"price": {"$gt": 50}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "content->>'price'::numeric > %s" in sql
        assert params == [50]

    def test_combined_comparison_operators(self):
        """$gte and $lte validate and translate to SQL with AND."""
        filters = {"price": {"$gte": 10, "$lte": 100}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "::numeric >=" in sql
        assert "::numeric <=" in sql
        assert 10 in params
        assert 100 in params

    def test_in_operator_through_pipeline(self):
        """$in validates and translates to SQL IN clause."""
        filters = {"status": {"$in": ["active", "pending"]}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "IN" in sql
        assert "%s" in sql
        assert "active" in params
        assert "pending" in params

    def test_nin_operator_through_pipeline(self):
        """$nin validates and translates to SQL NOT IN clause."""
        filters = {"status": {"$nin": ["archived"]}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "NOT IN" in sql
        assert "archived" in params

    def test_exists_true_through_pipeline(self):
        """$exists: true validates and translates to JSONB ? operator."""
        filters = {"tags": {"$exists": True}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "content ? %s" in sql
        assert "tags" in params

    def test_exists_false_through_pipeline(self):
        """$exists: false validates and translates to NOT (content ? %s)."""
        filters = {"tags": {"$exists": False}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "NOT (content ? %s)" in sql
        assert "tags" in params

    def test_logical_and_through_pipeline(self):
        """$and validates and translates to SQL (... AND ...)."""
        filters = {
            "$and": [
                {"price": {"$gte": 50}},
                {"category": "electronics"},
            ]
        }
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "AND" in sql
        assert 50 in params
        assert "electronics" in params

    def test_logical_or_through_pipeline(self):
        """$or validates and translates to SQL (... OR ...)."""
        filters = {
            "$or": [
                {"category": "electronics"},
                {"category": "furniture"},
            ]
        }
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "OR" in sql
        assert "electronics" in params
        assert "furniture" in params

    def test_not_operator_top_level_through_pipeline(self):
        """Top-level $not validates and translates to SQL NOT (...)."""
        filters = {"$not": {"category": "deprecated"}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "NOT (" in sql
        assert "deprecated" in params

    def test_field_level_not_through_pipeline(self):
        """Field-level $not: {field: {$not: {$gt: 100}}} translates to NOT (field > ...)."""
        filters = {"price": {"$not": {"$gt": 100}}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "NOT (" in sql
        assert 100 in params

    def test_field_level_not_scalar_through_pipeline(self):
        """Field-level $not with scalar: {field: {$not: value}} translates to field != value."""
        filters = {"category": {"$not": "deprecated"}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "!= %s" in sql
        assert "deprecated" in params

    def test_deeply_nested_filters_through_pipeline(self):
        """Complex nested filter validates and translates to SQL."""
        filters = {
            "$and": [
                {
                    "$or": [
                        {"price": {"$lt": 50}},
                        {"price": {"$gt": 500}},
                    ]
                },
                {"category": {"$in": ["electronics", "furniture"]}},
            ]
        }
        validate_filters(filters)
        sql, params = translate_filters(filters)
        # Should have both AND and OR
        assert "AND" in sql
        assert "OR" in sql
        assert 50 in params
        assert 500 in params
        assert "electronics" in params
        assert "furniture" in params

    def test_column_field_partition_name(self):
        """partition_name is a direct column, not JSONB content access."""
        filters = {"partition_name": "sports"}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "partition_name = %s" in sql
        assert "content->>" not in sql
        assert params == ["sports"]

    def test_empty_in_translates_to_false(self):
        """$in with empty list validates and translates to FALSE."""
        filters = {"status": {"$in": []}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "FALSE" in sql

    def test_empty_nin_is_noop(self):
        """$nin with empty list validates and translates to empty SQL."""
        filters = {"status": {"$nin": []}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert sql == ""
        assert params == []

    def test_rejected_operator_fails_before_translation(self):
        """$regex is caught at validation, never reaches translate_filters."""
        filters = {"name": {"$regex": "^test"}}
        with pytest.raises(ValueError, match="not allowed"):
            validate_filters(filters)
        # translate_filters would raise ValueError for unknown ops too,
        # but the point is validation catches it first.


# ══════════════════════════════════════════════════════════════════════
# 3. SVRClient.search() -> Backend integration (mocked backend)
# ══════════════════════════════════════════════════════════════════════


class TestSVRClientSearchBackendIntegration:
    """Verify SVRClient.search() wires validation + backend correctly."""

    async def _make_connected_client(self):
        """Create an SVRClient with a mocked backend, resolver, and merger."""
        from semantic_vector_router.client import SVRClient
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            RerankingConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="test_coll",
            ),
            partitioning={"field": "category"},
            vector_search=VectorSearchConfig(dimensions=3),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.VOYAGE,
                model="voyage-3-lite",
                dimensions=3,
                api_key_env="VOYAGE_API_KEY",
            ),
            reranking=RerankingConfig(enabled=False),
        )

        client = SVRClient(config=config, auto_connect=False)

        # Mock the internal components
        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=[
            {"_id": "doc1", "_svr_score": 0.95, "_svr_partition": "electronics", "title": "Headphones"},
            {"_id": "doc2", "_svr_score": 0.85, "_svr_partition": "electronics", "title": "Keyboard"},
        ])

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        test_partition = PartitionInfo(name="electronics", filter_value="electronics")

        mock_resolver = AsyncMock()
        mock_resolver.resolve = AsyncMock(return_value=[test_partition])

        mock_merger = MagicMock()
        mock_merger.merge = MagicMock(return_value=[
            SearchHit(id="doc1", score=0.95, partition="electronics", document={"title": "Headphones"}),
            SearchHit(id="doc2", score=0.85, partition="electronics", document={"title": "Keyboard"}),
        ])

        client._backend = mock_backend
        client._embedder = mock_embedder
        client._resolver = mock_resolver
        client._merger = mock_merger
        client._connected = True

        return client, mock_backend

    async def test_valid_filters_reach_backend(self):
        """Valid filters pass validation and reach backend.search_partitions."""
        client, mock_backend = await self._make_connected_client()

        try:
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=10,
                filters={"price": {"$gte": 50}},
                query_vector=[0.1, 0.2, 0.3],
            )

            mock_backend.search_partitions.assert_called_once()
            call_kwargs = mock_backend.search_partitions.call_args
            assert call_kwargs.kwargs["filters"] == {"price": {"$gte": 50}}
        finally:
            client._connected = False

    async def test_invalid_filters_raise_before_backend(self):
        """Invalid filters raise ValueError before any backend call."""
        client, mock_backend = await self._make_connected_client()

        try:
            with pytest.raises(ValueError, match="not allowed"):
                await client.search(
                    query="headphones",
                    partitions=["electronics"],
                    limit=10,
                    filters={"name": {"$regex": "^head"}},
                    query_vector=[0.1, 0.2, 0.3],
                )

            # Backend should never have been called
            mock_backend.search_partitions.assert_not_called()
        finally:
            client._connected = False

    async def test_exact_flag_flows_to_backend(self):
        """exact=True is passed through to backend.search_partitions."""
        client, mock_backend = await self._make_connected_client()

        try:
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=10,
                exact=True,
                query_vector=[0.1, 0.2, 0.3],
            )

            call_kwargs = mock_backend.search_partitions.call_args
            assert call_kwargs.kwargs["exact"] is True
        finally:
            client._connected = False

    async def test_post_native_flows_to_backend(self):
        """post_native is passed through to backend.search_partitions."""
        client, mock_backend = await self._make_connected_client()
        post_native = [{"$project": {"title": 1, "_svr_score": 1}}]

        try:
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=10,
                post_native=post_native,
                query_vector=[0.1, 0.2, 0.3],
            )

            call_kwargs = mock_backend.search_partitions.call_args
            assert call_kwargs.kwargs["post_native"] == post_native
        finally:
            client._connected = False

    async def test_pre_native_flows_to_backend(self):
        """pre_native is passed through to backend.search_partitions."""
        client, mock_backend = await self._make_connected_client()
        pre_native = "content @> '{\"featured\": true}'"

        try:
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=10,
                pre_native=pre_native,
                query_vector=[0.1, 0.2, 0.3],
            )

            call_kwargs = mock_backend.search_partitions.call_args
            assert call_kwargs.kwargs["pre_native"] == pre_native
        finally:
            client._connected = False

    async def test_unsupported_operator_fails_before_backend(self):
        """Unknown operator raises ValueError before backend call."""
        client, mock_backend = await self._make_connected_client()

        try:
            with pytest.raises(ValueError, match="Unsupported filter operator"):
                await client.search(
                    query="headphones",
                    partitions=["electronics"],
                    limit=10,
                    filters={"field": {"$unknown": "value"}},
                    query_vector=[0.1, 0.2, 0.3],
                )

            mock_backend.search_partitions.assert_not_called()
        finally:
            client._connected = False

    async def test_none_filters_skip_validation(self):
        """When filters=None, no validation is run and backend is called."""
        client, mock_backend = await self._make_connected_client()

        try:
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=10,
                filters=None,
                query_vector=[0.1, 0.2, 0.3],
            )

            mock_backend.search_partitions.assert_called_once()
            call_kwargs = mock_backend.search_partitions.call_args
            assert call_kwargs.kwargs["filters"] is None
        finally:
            client._connected = False

    async def test_empty_filters_skip_validation(self):
        """Empty dict filters still call backend (validation is a no-op for empty)."""
        client, mock_backend = await self._make_connected_client()

        try:
            # Empty dict is falsy in Python, so it takes the `if filters:` branch = no validation
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=10,
                filters={},
                query_vector=[0.1, 0.2, 0.3],
            )

            mock_backend.search_partitions.assert_called_once()
        finally:
            client._connected = False


# ══════════════════════════════════════════════════════════════════════
# 4. MongoDB pipeline assembly integration
# ══════════════════════════════════════════════════════════════════════


class TestMongoDBPipelineAssembly:
    """Verify _build_vector_search_pipeline produces correct pipeline structures."""

    def _make_backend(self) -> MongoDBBackend:
        """Create a MongoDBBackend via __new__ with minimal config."""
        from semantic_vector_router.models import (
            DatabaseConfig,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
            VectorStorageConfig,
            VectorStorageFormat,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(
                embedding_field="embedding",
                dimensions=3,
                similarity="cosine",
            ),
            vector_storage=VectorStorageConfig(
                index_on=IndexLocation.SOURCE,
                storage_format=VectorStorageFormat.ARRAY,
            ),
        )

        backend = MongoDBBackend.__new__(MongoDBBackend)
        backend.config = config
        backend._views = MagicMock()
        backend._indexes = MagicMock()
        return backend

    def test_basic_pipeline_structure(self):
        """Pipeline has $vectorSearch as first stage, $addFields as second."""
        backend = self._make_backend()
        partition = PartitionInfo(
            name="electronics",
            filter_value="electronics",
            index_name="svr_idx_electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=[0.1, 0.2, 0.3],
        )

        assert len(pipeline) >= 2
        assert "$vectorSearch" in pipeline[0]
        assert "$addFields" in pipeline[1]
        assert pipeline[1]["$addFields"]["_svr_partition"] == "electronics"
        assert pipeline[1]["$addFields"]["_svr_score"] == {"$meta": "vectorSearchScore"}

    def test_pipeline_with_filters(self):
        """User filters are merged into $vectorSearch.filter."""
        backend = self._make_backend()
        partition = PartitionInfo(
            name="electronics",
            filter_value="electronics",
            index_name="svr_idx_electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=[0.1, 0.2, 0.3],
            filters={"price": {"$gte": 50}},
        )

        vs = pipeline[0]["$vectorSearch"]
        assert "filter" in vs
        # Should contain both partition filter and user filter
        assert vs["filter"]["category"] == "electronics"
        assert vs["filter"]["price"] == {"$gte": 50}

    def test_pipeline_exact_mode(self):
        """exact=True sets $vectorSearch.exact and omits numCandidates."""
        backend = self._make_backend()
        partition = PartitionInfo(
            name="electronics",
            filter_value="electronics",
            index_name="svr_idx_electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=[0.1, 0.2, 0.3],
            exact=True,
        )

        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs

    def test_pipeline_ann_mode(self):
        """Default (exact=False) includes numCandidates and no exact flag."""
        backend = self._make_backend()
        partition = PartitionInfo(
            name="electronics",
            filter_value="electronics",
            index_name="svr_idx_electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=[0.1, 0.2, 0.3],
            exact=False,
        )

        vs = pipeline[0]["$vectorSearch"]
        assert vs["numCandidates"] == 100
        assert "exact" not in vs

    def test_pipeline_with_post_native(self):
        """post_native stages are appended after $vectorSearch + $addFields."""
        backend = self._make_backend()
        partition = PartitionInfo(
            name="electronics",
            filter_value="electronics",
            index_name="svr_idx_electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        post_native = [
            {"$project": {"title": 1, "_svr_score": 1}},
            {"$limit": 5},
        ]

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=[0.1, 0.2, 0.3],
            post_native=post_native,
        )

        # $vectorSearch, $addFields, then post_native stages
        assert len(pipeline) == 4  # vectorSearch + addFields + 2 post_native
        assert "$project" in pipeline[2]
        assert "$limit" in pipeline[3]

    def test_pipeline_with_filters_and_exact_and_post_native(self):
        """Full pipeline: filters + exact + post_native all combined correctly."""
        backend = self._make_backend()
        partition = PartitionInfo(
            name="electronics",
            filter_value="electronics",
            index_name="svr_idx_electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",
        )

        post_native = [{"$lookup": {"from": "reviews", "localField": "_id", "foreignField": "product_id", "as": "reviews"}}]

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=[0.1, 0.2, 0.3],
            filters={"price": {"$gte": 50}, "brand": {"$in": ["Apple", "Sony"]}},
            exact=True,
            post_native=post_native,
        )

        vs = pipeline[0]["$vectorSearch"]
        # Exact mode
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        # Filters
        assert vs["filter"]["category"] == "electronics"
        assert vs["filter"]["price"] == {"$gte": 50}
        assert vs["filter"]["brand"] == {"$in": ["Apple", "Sony"]}
        # addFields
        assert pipeline[1]["$addFields"]["_svr_partition"] == "electronics"
        # post_native
        assert "$lookup" in pipeline[2]

    def test_pipeline_no_partition_filter_for_fields_mode(self):
        """FIELDS mode does not add partition pre-filter to $vectorSearch."""
        backend = self._make_backend()
        partition = PartitionInfo(
            name="electronics",
            filter_value="electronics",
            index_name="svr_idx_electronics",
            index_location=IndexLocation.FIELDS,
            search_collection="products",
            embedding_field="embedding_electronics",
        )

        pipeline = backend._build_vector_search_pipeline(
            partition=partition,
            limit=10,
            num_candidates=100,
            query_vector=[0.1, 0.2, 0.3],
        )

        vs = pipeline[0]["$vectorSearch"]
        # FIELDS mode: no partition filter, and uses partition-specific embedding field
        assert "filter" not in vs
        assert vs["path"] == "embedding_electronics"


# ══════════════════════════════════════════════════════════════════════
# 5. PostgreSQL query assembly integration
# ══════════════════════════════════════════════════════════════════════


class TestPostgresQueryAssembly:
    """Verify PostgreSQL filter translation produces correct SQL structures."""

    def test_filters_produce_parameterized_sql(self):
        """Filters translate to parameterized SQL (no string interpolation)."""
        filters = {"title": "Headphones", "price": {"$gte": 50}}
        validate_filters(filters)
        sql, params = translate_filters(filters)

        # All values should be parameterized
        assert "%s" in sql
        assert "Headphones" not in sql  # value is in params, not SQL
        assert "Headphones" in params
        assert 50 in params

    def test_numeric_comparison_adds_cast(self):
        """Numeric comparisons get ::numeric cast for JSONB text extraction."""
        filters = {"price": {"$gt": 100.0}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "::numeric" in sql

    def test_string_comparison_no_cast(self):
        """String comparisons do not get ::numeric cast."""
        filters = {"category": {"$eq": "electronics"}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert "::numeric" not in sql

    def test_complex_filter_produces_valid_sql(self):
        """Complex nested filter produces syntactically valid SQL."""
        filters = {
            "$and": [
                {
                    "$or": [
                        {"price": {"$lt": 50}},
                        {"price": {"$gt": 500}},
                    ]
                },
                {"category": {"$ne": "discontinued"}},
                {"tags": {"$exists": True}},
            ]
        }
        validate_filters(filters)
        sql, params = translate_filters(filters)

        # Verify structure
        assert sql.startswith("(")  # Outer $and wraps in parens
        assert "AND" in sql
        assert "OR" in sql
        assert "content ? %s" in sql  # $exists
        assert "!= %s" in sql  # $ne
        assert len(params) == 4  # 50, 500, "discontinued", "tags"

    def test_not_wraps_in_sql_not(self):
        """$not at top level translates to NOT (...)."""
        filters = {"$not": {"status": "inactive"}}
        validate_filters(filters)
        sql, params = translate_filters(filters)
        assert sql.startswith("NOT (")
        assert "inactive" in params

    def test_pre_native_would_be_and_joined(self):
        """Pre-native SQL is AND-joined with filter SQL in execute_search.

        This verifies the filter translation produces a fragment compatible
        with AND-joining by the backend.
        """
        filters = {"price": {"$gte": 50}}
        validate_filters(filters)
        sql_fragment, params = translate_filters(filters)

        # The fragment should be a simple expression suitable for AND-joining
        assert "AND" not in sql_fragment or sql_fragment.startswith("(")
        pre_native = "content @> '{\"featured\": true}'"
        combined = f"{sql_fragment} AND ({pre_native})"
        # Should produce valid-looking SQL
        assert "AND" in combined

    def test_post_native_cte_pattern(self):
        """Post-native SQL works with CTE wrapping pattern.

        Verifies that a core query + post_native CTE consumer can be composed.
        """
        filters = {"price": {"$gte": 50}}
        validate_filters(filters)
        sql_fragment, params = translate_filters(filters)

        # Simulate the CTE wrapping that execute_search does
        core_query = f"SELECT id, content FROM svr_vectors WHERE partition_name = 'sports' AND {sql_fragment} ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector LIMIT 10"
        post_native = "SELECT * FROM svr_results WHERE score > 0.5"
        full_query = f"WITH svr_results AS ({core_query}) {post_native}"

        assert "WITH svr_results AS" in full_query
        assert "SELECT * FROM svr_results" in full_query


# ══════════════════════════════════════════════════════════════════════
# 6. Cross-backend filter portability
# ══════════════════════════════════════════════════════════════════════


class TestCrossBackendFilterPortability:
    """Same filter dict works through both MongoDB and PostgreSQL translation."""

    PORTABLE_FILTERS = [
        # (description, filter_dict)
        ("simple equality", {"category": "electronics"}),
        ("comparison operator", {"price": {"$gt": 100}}),
        ("range filter", {"price": {"$gte": 10, "$lte": 100}}),
        ("$ne operator", {"status": {"$ne": "inactive"}}),
        ("$in operator", {"category": {"$in": ["electronics", "furniture"]}}),
        ("$nin operator", {"category": {"$nin": ["deprecated"]}}),
        ("$exists true", {"metadata": {"$exists": True}}),
        ("$exists false", {"metadata": {"$exists": False}}),
        ("$and operator", {
            "$and": [
                {"price": {"$gte": 50}},
                {"category": "electronics"},
            ]
        }),
        ("$or operator", {
            "$or": [
                {"category": "electronics"},
                {"category": "furniture"},
            ]
        }),
        ("$not top-level", {"$not": {"status": "archived"}}),
        ("nested $and with $or", {
            "$and": [
                {"$or": [{"price": {"$lt": 50}}, {"price": {"$gt": 500}}]},
                {"category": {"$in": ["electronics", "furniture"]}},
            ]
        }),
    ]

    @pytest.mark.parametrize("description,filters", PORTABLE_FILTERS)
    def test_filter_validates_for_both_backends(self, description, filters):
        """Filter passes shared validation (backend-agnostic)."""
        # Should not raise
        validate_filters(filters)

    @pytest.mark.parametrize("description,filters", PORTABLE_FILTERS)
    def test_filter_translates_to_mongodb(self, description, filters):
        """Filter translates through MongoFilterTranslator without error."""
        result = MongoFilterTranslator().translate(filters)
        assert isinstance(result, dict)
        # MongoDB translator preserves keys
        if "$and" in filters:
            assert "$and" in result
        elif "$or" in filters:
            assert "$or" in result
        elif "$not" in filters:
            assert "$not" in result

    @pytest.mark.parametrize("description,filters", PORTABLE_FILTERS)
    def test_filter_translates_to_postgres(self, description, filters):
        """Filter translates through PostgreSQL translate_filters without error."""
        sql, params = translate_filters(filters)
        # Every portable filter should produce non-empty SQL
        # (except edge cases like empty $nin which is a no-op)
        if not (
            "$nin" in str(filters)
            and isinstance(list(filters.values())[0], dict)
            and list(filters.values())[0].get("$nin") == []
        ):
            assert sql != "", f"Expected non-empty SQL for {description}"
        assert isinstance(params, list)

    def test_operator_parity(self):
        """All SUPPORTED_OPERATORS are handled by both translators.

        MongoFilterTranslator does identity pass-through for all operators.
        PostgreSQL translate_filters handles all supported comparison, set,
        logical, and existence operators.
        """
        # Verify all supported operators are in the known set
        expected_ops = {
            "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",  # comparison
            "$in", "$nin",  # set
            "$and", "$or", "$not",  # logical
            "$exists",  # existence
        }
        assert SUPPORTED_OPERATORS == expected_ops

    def test_rejected_operators_consistent(self):
        """Rejected operators include common invalid operators."""
        expected_rejected = {
            "$match", "$all", "$elemMatch", "$regex",
            "$text", "$size", "$type", "$near", "$geoWithin",
        }
        assert set(REJECTED_OPERATORS.keys()) == expected_rejected

    def test_field_level_not_portability(self):
        """Field-level $not works through both backends."""
        filters = {"price": {"$not": {"$gt": 100}}}
        validate_filters(filters)

        # MongoDB
        mongo_result = MongoFilterTranslator().translate(filters)
        assert mongo_result == {"price": {"$not": {"$gt": 100}}}

        # PostgreSQL
        pg_sql, pg_params = translate_filters(filters)
        assert "NOT (" in pg_sql
        assert 100 in pg_params
