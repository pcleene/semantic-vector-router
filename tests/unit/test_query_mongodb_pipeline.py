"""Unit tests for MongoDBBackend._build_vector_search_pipeline — exact mode and post_native.

Tests the exact search flag and post_native pipeline injection features
introduced in the pipeline builder. This file complements the existing
test_mongodb_backend.py without modifying it.
"""

import pytest
from unittest.mock import MagicMock

from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.backends.mongodb.views import MongoDBViewOps
from semantic_vector_router.backends.mongodb.indexes import MongoDBIndexOps
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    IndexLocation,
    PartitionInfo,
    PartitioningConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> SVRConfig:
    """Create a minimal SVRConfig for pipeline construction tests."""
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
    be._MongoDBBackend__db = None  # name-mangled _db setter won't fire without _views
    be._server_version = None
    be._last_health_check = None
    be._views = MongoDBViewOps(sample_config)
    be._indexes = MongoDBIndexOps()
    be._source_index_ensured = False
    return be


@pytest.fixture
def source_partition() -> PartitionInfo:
    """A SOURCE-mode partition for pipeline tests."""
    return PartitionInfo(
        name="electronics",
        view_name="svr_partition_electronics",
        index_name="svr_vector_idx",
        filter_value="electronics",
        index_location=IndexLocation.SOURCE,
        search_collection="products",
    )


@pytest.fixture
def fields_partition() -> PartitionInfo:
    """A FIELDS-mode partition with a custom embedding field."""
    return PartitionInfo(
        name="clothing",
        index_name="svr_vector_idx_clothing",
        filter_value="clothing",
        index_location=IndexLocation.FIELDS,
        search_collection="products",
        embedding_field="embedding_clothing",
    )


@pytest.fixture
def views_partition() -> PartitionInfo:
    """A VIEWS-mode partition searching on a view (8.1+ path)."""
    return PartitionInfo(
        name="books",
        view_name="svr_partition_books",
        index_name="svr_vector_idx_books",
        filter_value="books",
        index_location=IndexLocation.VIEWS,
        search_collection="svr_partition_books",
    )


def _qv(dim: int = 1536) -> list[float]:
    """Create a dummy query vector."""
    return [0.1] * dim


# ===========================================================================
# exact mode tests
# ===========================================================================


class TestExactModeDefault:
    """exact=False (default): ANN search with numCandidates."""

    def test_default_has_num_candidates(self, backend, source_partition):
        """Default (exact=False) pipeline includes numCandidates."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(),
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["numCandidates"] == 100

    def test_default_has_no_exact_key(self, backend, source_partition):
        """Default pipeline does NOT contain an 'exact' key."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(),
        )
        vs = pipeline[0]["$vectorSearch"]
        assert "exact" not in vs

    def test_default_explicit_false(self, backend, source_partition):
        """Passing exact=False explicitly behaves the same as default."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), exact=False,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["numCandidates"] == 100
        assert "exact" not in vs


class TestExactModeTrue:
    """exact=True: brute-force search without numCandidates."""

    def test_exact_true_sets_exact_flag(self, backend, source_partition):
        """exact=True sets exact: true in $vectorSearch."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True

    def test_exact_true_omits_num_candidates(self, backend, source_partition):
        """exact=True must NOT include numCandidates."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=200,
            query_vector=_qv(), exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert "numCandidates" not in vs

    def test_exact_true_still_has_limit(self, backend, source_partition):
        """exact=True still includes the limit field."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=25, num_candidates=200,
            query_vector=_qv(), exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["limit"] == 25


class TestExactModeWithQueryString:
    """exact=True with query_string instead of query_vector."""

    def test_exact_with_query_string(self, backend, source_partition):
        """exact=True with query_string: has exact, has queryString, no numCandidates."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_string="wireless headphones", exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert vs["queryString"] == "wireless headphones"
        assert "numCandidates" not in vs
        assert "queryVector" not in vs

    def test_ann_with_query_string(self, backend, source_partition):
        """Default ANN mode with query_string: has numCandidates, no exact."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_string="wireless headphones",
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["numCandidates"] == 100
        assert vs["queryString"] == "wireless headphones"
        assert "exact" not in vs


class TestExactModeWithFilters:
    """exact=True combined with filters."""

    def test_exact_with_filters_source_mode(self, backend, source_partition):
        """exact=True with filters: both exact and filter present."""
        user_filter = {"price": {"$gte": 100}}
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), exact=True, filters=user_filter,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        # SOURCE mode adds partition filter + user filter
        assert "filter" in vs
        combined = vs["filter"]
        assert combined["category"] == "electronics"
        assert combined["price"] == {"$gte": 100}

    def test_exact_with_filters_fields_mode(self, backend, fields_partition):
        """exact=True with filters on FIELDS mode: no partition filter, user filter only."""
        user_filter = {"brand": "Nike"}
        pipeline = backend._build_vector_search_pipeline(
            fields_partition, limit=5, num_candidates=50,
            query_vector=_qv(), exact=True, filters=user_filter,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        assert vs["filter"] == {"brand": "Nike"}

    def test_exact_no_filters_fields_mode(self, backend, fields_partition):
        """exact=True on FIELDS mode without extra filters: no filter key."""
        pipeline = backend._build_vector_search_pipeline(
            fields_partition, limit=5, num_candidates=50,
            query_vector=_qv(), exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "filter" not in vs


# ===========================================================================
# post_native tests
# ===========================================================================


class TestPostNativeNone:
    """No post_native: pipeline has exactly 2 stages."""

    def test_no_post_native_two_stages(self, backend, source_partition):
        """Pipeline with post_native=None has exactly $vectorSearch + $addFields."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(),
        )
        assert len(pipeline) == 2

    def test_no_post_native_default_param(self, backend, source_partition):
        """Omitting post_native entirely gives same result as None."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=None,
        )
        assert len(pipeline) == 2
        assert "$vectorSearch" in pipeline[0]
        assert "$addFields" in pipeline[1]


class TestPostNativeSingleStage:
    """post_native with a single stage appended."""

    def test_single_stage_appended(self, backend, source_partition):
        """A single post_native stage appears as the 3rd pipeline stage."""
        post = [{"$limit": 5}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 3
        assert pipeline[2] == {"$limit": 5}

    def test_single_stage_does_not_affect_first_two(self, backend, source_partition):
        """Adding post_native does not change $vectorSearch or $addFields."""
        post = [{"$limit": 5}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert "$vectorSearch" in pipeline[0]
        assert "$addFields" in pipeline[1]


class TestPostNativeMultipleStages:
    """post_native with multiple stages."""

    def test_multiple_stages_appended_in_order(self, backend, source_partition):
        """Multiple post_native stages appended after $addFields in order."""
        post = [
            {"$project": {"name": 1, "price": 1}},
            {"$limit": 3},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 4
        assert pipeline[2] == {"$project": {"name": 1, "price": 1}}
        assert pipeline[3] == {"$limit": 3}

    def test_three_stages_appended(self, backend, source_partition):
        """Three post_native stages all present and in order."""
        post = [
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=20, num_candidates=200,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 5
        assert pipeline[2] == {"$unwind": "$tags"}
        assert pipeline[3] == {"$group": {"_id": "$tags", "count": {"$sum": 1}}}
        assert pipeline[4] == {"$sort": {"count": -1}}


class TestPostNativeLookup:
    """post_native with a $lookup stage (real-world join example)."""

    def test_lookup_stage_appended(self, backend, source_partition):
        """$lookup stage is appended verbatim after $addFields."""
        lookup = {
            "$lookup": {
                "from": "reviews",
                "localField": "_id",
                "foreignField": "product_id",
                "as": "reviews",
            }
        }
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=[lookup],
        )
        assert len(pipeline) == 3
        assert pipeline[2] == lookup

    def test_lookup_with_unwind(self, backend, source_partition):
        """$lookup + $unwind is a common pattern that should work."""
        post = [
            {
                "$lookup": {
                    "from": "reviews",
                    "localField": "_id",
                    "foreignField": "product_id",
                    "as": "reviews",
                }
            },
            {"$unwind": {"path": "$reviews", "preserveNullAndEmptyArrays": True}},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 4
        assert "$lookup" in pipeline[2]
        assert "$unwind" in pipeline[3]


class TestPostNativeMatchSort:
    """post_native with $match + $sort stages."""

    def test_match_and_sort_appended(self, backend, source_partition):
        """$match + $sort stages appended correctly."""
        post = [
            {"$match": {"in_stock": True}},
            {"$sort": {"price": 1}},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 4
        assert pipeline[2] == {"$match": {"in_stock": True}}
        assert pipeline[3] == {"$sort": {"price": 1}}

    def test_post_native_match_does_not_affect_vector_search_filter(
        self, backend, source_partition
    ):
        """Post-native $match is separate from $vectorSearch filter."""
        user_filter = {"brand": "Sony"}
        post = [{"$match": {"rating": {"$gte": 4}}}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), filters=user_filter, post_native=post,
        )
        vs_filter = pipeline[0]["$vectorSearch"]["filter"]
        # $vectorSearch filter has partition + user filter
        assert vs_filter["brand"] == "Sony"
        assert vs_filter["category"] == "electronics"
        # rating filter is NOT in $vectorSearch, it is in post_native $match
        assert "rating" not in vs_filter
        assert pipeline[2] == {"$match": {"rating": {"$gte": 4}}}


class TestPostNativeWithExact:
    """post_native combined with exact=True."""

    def test_post_native_with_exact(self, backend, source_partition):
        """exact=True + post_native: exact flag set, no numCandidates, stages appended."""
        post = [{"$project": {"name": 1, "_svr_score": 1}}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), exact=True, post_native=post,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        assert len(pipeline) == 3
        assert pipeline[2] == {"$project": {"name": 1, "_svr_score": 1}}

    def test_post_native_with_exact_and_filters(self, backend, source_partition):
        """exact=True + filters + post_native: all three features combined."""
        user_filter = {"price": {"$lt": 500}}
        post = [
            {"$match": {"available": True}},
            {"$sort": {"_svr_score": -1}},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=5, num_candidates=50,
            query_vector=_qv(), exact=True, filters=user_filter, post_native=post,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        assert vs["filter"]["price"] == {"$lt": 500}
        assert vs["filter"]["category"] == "electronics"
        assert len(pipeline) == 4
        assert pipeline[2] == {"$match": {"available": True}}
        assert pipeline[3] == {"$sort": {"_svr_score": -1}}


class TestPostNativeWithFilters:
    """post_native combined with user filters on $vectorSearch."""

    def test_post_native_with_vector_search_filters(self, backend, source_partition):
        """User filters go into $vectorSearch, post_native stages go after $addFields."""
        user_filter = {"color": "red"}
        post = [{"$limit": 3}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), filters=user_filter, post_native=post,
        )
        # Filter in $vectorSearch
        vs = pipeline[0]["$vectorSearch"]
        assert vs["filter"]["color"] == "red"
        # post_native after $addFields
        assert pipeline[2] == {"$limit": 3}


class TestPostNativeMetadataIntegrity:
    """post_native stages do not interfere with $addFields metadata."""

    def test_svr_score_present_with_post_native(self, backend, source_partition):
        """_svr_score is in $addFields even when post_native stages are added."""
        post = [{"$project": {"name": 1}}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        add_fields = pipeline[1]["$addFields"]
        assert "_svr_score" in add_fields
        assert add_fields["_svr_score"] == {"$meta": "vectorSearchScore"}

    def test_svr_partition_present_with_post_native(self, backend, source_partition):
        """_svr_partition is in $addFields even when post_native stages are added."""
        post = [{"$project": {"name": 1}}, {"$limit": 5}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        add_fields = pipeline[1]["$addFields"]
        assert "_svr_partition" in add_fields
        assert add_fields["_svr_partition"] == "electronics"

    def test_metadata_unchanged_with_multiple_post_native(self, backend, source_partition):
        """$addFields metadata is identical regardless of post_native content."""
        pipeline_without = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(),
        )
        pipeline_with = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(),
            post_native=[{"$match": {"x": 1}}, {"$sort": {"y": -1}}, {"$limit": 2}],
        )
        assert pipeline_without[1] == pipeline_with[1]

    def test_metadata_with_different_partitions(self, backend, fields_partition):
        """_svr_partition reflects the partition name, even with post_native."""
        post = [{"$limit": 1}]
        pipeline = backend._build_vector_search_pipeline(
            fields_partition, limit=5, num_candidates=50,
            query_vector=_qv(), post_native=post,
        )
        add_fields = pipeline[1]["$addFields"]
        assert add_fields["_svr_partition"] == "clothing"
        assert add_fields["_svr_score"] == {"$meta": "vectorSearchScore"}


# ===========================================================================
# Pipeline structure verification
# ===========================================================================


class TestPipelineStructureOrder:
    """Verify the ordering of pipeline stages."""

    def test_vector_search_always_first(self, backend, source_partition):
        """$vectorSearch is always the first stage, per Atlas requirement."""
        for post_native in [None, [], [{"$limit": 1}], [{"$match": {"x": 1}}, {"$sort": {"y": -1}}]]:
            pipeline = backend._build_vector_search_pipeline(
                source_partition, limit=10, num_candidates=100,
                query_vector=_qv(), post_native=post_native,
            )
            assert "$vectorSearch" in pipeline[0]

    def test_add_fields_always_second(self, backend, source_partition):
        """$addFields is always the second stage."""
        for post_native in [None, [], [{"$limit": 1}], [{"$match": {"x": 1}}, {"$sort": {"y": -1}}]]:
            pipeline = backend._build_vector_search_pipeline(
                source_partition, limit=10, num_candidates=100,
                query_vector=_qv(), post_native=post_native,
            )
            assert "$addFields" in pipeline[1]

    def test_post_native_stages_after_add_fields(self, backend, source_partition):
        """All post_native stages appear after position 1 (after $addFields)."""
        post = [
            {"$match": {"active": True}},
            {"$project": {"name": 1}},
            {"$limit": 5},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert pipeline[2:] == post

    def test_post_native_stages_preserve_order(self, backend, source_partition):
        """post_native stages maintain their original order in the pipeline."""
        post = [
            {"$addFields": {"custom_field": "hello"}},
            {"$match": {"custom_field": "hello"}},
            {"$project": {"custom_field": 1}},
            {"$sort": {"custom_field": 1}},
            {"$limit": 1},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        for i, stage in enumerate(post):
            assert pipeline[2 + i] == stage


class TestEmptyPostNative:
    """Empty post_native list behaves the same as None."""

    def test_empty_list_same_as_none(self, backend, source_partition):
        """post_native=[] produces the same pipeline as post_native=None."""
        pipeline_none = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=None,
        )
        pipeline_empty = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=[],
        )
        assert pipeline_none == pipeline_empty

    def test_empty_list_has_two_stages(self, backend, source_partition):
        """post_native=[] results in exactly 2 stages."""
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=[],
        )
        assert len(pipeline) == 2


# ===========================================================================
# Cross-cutting: exact + post_native + modes
# ===========================================================================


class TestExactWithViewsMode:
    """exact mode on VIEWS partitions."""

    def test_exact_on_views_partition_no_partition_filter(self, backend, views_partition):
        """VIEWS mode searching on the view: no partition filter needed, exact works."""
        pipeline = backend._build_vector_search_pipeline(
            views_partition, limit=10, num_candidates=100,
            query_vector=_qv(), exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        # View-scoped search: no partition filter
        assert "filter" not in vs

    def test_views_on_source_collection_has_partition_filter(self, backend):
        """VIEWS mode on source collection (pre-8.1): partition filter + exact."""
        partition = PartitionInfo(
            name="books",
            view_name="svr_partition_books",
            index_name="svr_vector_idx",
            filter_value="books",
            index_location=IndexLocation.VIEWS,
            search_collection="products",  # same as source_collection
        )
        pipeline = backend._build_vector_search_pipeline(
            partition, limit=10, num_candidates=100,
            query_vector=_qv(), exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        assert vs["filter"]["category"] == "books"


class TestExactWithFieldsMode:
    """exact mode on FIELDS partitions."""

    def test_exact_on_fields_uses_partition_embedding_field(self, backend, fields_partition):
        """FIELDS mode with exact=True uses the partition-specific embedding field."""
        pipeline = backend._build_vector_search_pipeline(
            fields_partition, limit=10, num_candidates=100,
            query_vector=_qv(), exact=True,
        )
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert vs["path"] == "embedding_clothing"
        assert "numCandidates" not in vs


class TestPostNativeAcrossModes:
    """post_native works identically across all index location modes."""

    def test_post_native_with_source_mode(self, backend, source_partition):
        """post_native stages appended in SOURCE mode."""
        post = [{"$limit": 3}]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 3
        assert pipeline[2] == {"$limit": 3}

    def test_post_native_with_fields_mode(self, backend, fields_partition):
        """post_native stages appended in FIELDS mode."""
        post = [{"$project": {"name": 1, "embedding_clothing": 0}}]
        pipeline = backend._build_vector_search_pipeline(
            fields_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 3
        assert pipeline[2] == {"$project": {"name": 1, "embedding_clothing": 0}}

    def test_post_native_with_views_mode(self, backend, views_partition):
        """post_native stages appended in VIEWS mode."""
        post = [
            {"$match": {"rating": {"$gte": 4.0}}},
            {"$sort": {"rating": -1}},
        ]
        pipeline = backend._build_vector_search_pipeline(
            views_partition, limit=10, num_candidates=100,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 4
        assert pipeline[2] == {"$match": {"rating": {"$gte": 4.0}}}
        assert pipeline[3] == {"$sort": {"rating": -1}}


class TestPostNativeComplexRealWorld:
    """Real-world complex post_native scenarios."""

    def test_faceted_search_pipeline(self, backend, source_partition):
        """post_native with $facet for faceted search results."""
        post = [
            {
                "$facet": {
                    "results": [{"$limit": 10}],
                    "total_count": [{"$count": "count"}],
                }
            }
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=100, num_candidates=500,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 3
        assert "$facet" in pipeline[2]

    def test_bucket_aggregation(self, backend, source_partition):
        """post_native with $bucket for price range grouping."""
        post = [
            {
                "$bucket": {
                    "groupBy": "$price",
                    "boundaries": [0, 50, 100, 500, 1000],
                    "default": "Other",
                    "output": {"count": {"$sum": 1}},
                }
            }
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=50, num_candidates=200,
            query_vector=_qv(), post_native=post,
        )
        assert len(pipeline) == 3
        assert "$bucket" in pipeline[2]

    def test_full_pipeline_exact_filters_post_native(self, backend, source_partition):
        """All features combined: exact + filters + multi-stage post_native."""
        user_filter = {"brand": "Apple", "in_stock": True}
        post = [
            {"$lookup": {"from": "inventory", "localField": "sku", "foreignField": "sku", "as": "stock"}},
            {"$unwind": "$stock"},
            {"$match": {"stock.quantity": {"$gt": 0}}},
            {"$project": {"name": 1, "price": 1, "_svr_score": 1, "_svr_partition": 1}},
            {"$sort": {"price": 1}},
            {"$limit": 5},
        ]
        pipeline = backend._build_vector_search_pipeline(
            source_partition, limit=20, num_candidates=200,
            query_vector=_qv(), exact=True, filters=user_filter, post_native=post,
        )
        # Structure checks
        assert len(pipeline) == 8  # $vectorSearch + $addFields + 6 post_native
        vs = pipeline[0]["$vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs
        assert vs["filter"]["brand"] == "Apple"
        assert vs["filter"]["in_stock"] is True
        assert vs["filter"]["category"] == "electronics"
        # $addFields metadata intact
        add_fields = pipeline[1]["$addFields"]
        assert add_fields["_svr_partition"] == "electronics"
        assert add_fields["_svr_score"] == {"$meta": "vectorSearchScore"}
        # post_native stages in order
        assert "$lookup" in pipeline[2]
        assert "$unwind" in pipeline[3]
        assert "$match" in pipeline[4]
        assert "$project" in pipeline[5]
        assert "$sort" in pipeline[6]
        assert "$limit" in pipeline[7]
