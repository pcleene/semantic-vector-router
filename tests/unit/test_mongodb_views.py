"""Unit tests for MongoDB view pipeline building — Phase 16 embedding text.

Tests 13-21 from Phase 16 spec. All mocked, no MongoDB connection needed.
"""

import pytest

from semantic_vector_router.backends.mongodb.views import MongoDBViewOps
from semantic_vector_router.models.enums import (
    EmbeddingMode,
    IndexLocation,
    VectorStorageMode,
)
from semantic_vector_router.models.svr_config import SVRConfig


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(
    source_fields=None,
    template=None,
    embedding_mode="byom",
    vector_storage_mode="embedded",
    index_on="views",
    **overrides,
):
    """Build a minimal SVRConfig for view pipeline testing."""
    embedding = {"mode": embedding_mode}
    if source_fields is not None:
        embedding["source_fields"] = source_fields
    if template is not None:
        embedding["template"] = template

    base = {
        "database": {"database": "testdb", "source_collection": "products"},
        "partitioning": {"field": "category"},
        "vector_search": {"dimensions": 1536, "similarity": "cosine"},
        "embedding": embedding,
        "vector_storage": {"mode": vector_storage_mode, "index_on": index_on},
    }
    base.update(overrides)
    return SVRConfig(**base)


# ── Test 13: BYOM default produces object projection ───────────────


class TestBuildEmbeddingFieldExpressionDefaultByom:
    def test_object_projection_not_concat(self):
        """BYOM mode without template should produce object projection, not $concat."""
        config = _make_config(source_fields=["title", "description", "tags"])
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        # Should NOT be a $concat expression
        assert "$concat" not in str(result)
        # Should be a dict with field projections
        assert "title" in result
        assert "description" in result
        assert "tags" in result

    def test_each_field_uses_ifnull(self):
        """Each field in the projection should have $ifNull for safety."""
        config = _make_config(source_fields=["title", "price"])
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        assert result["title"] == {"$ifNull": ["$title", None]}
        assert result["price"] == {"$ifNull": ["$price", None]}


# ── Test 14: Array source fields preserved ─────────────────────────


class TestBuildEmbeddingFieldExpressionPreservesArrays:
    def test_array_fields_projected_as_is(self):
        """Array fields should be projected via $ifNull, preserving them as arrays."""
        config = _make_config(source_fields=["tags"])
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        # The projection keeps the original field value (which may be an array)
        assert result["tags"] == {"$ifNull": ["$tags", None]}


# ── Test 15: Nested objects projected as-is ────────────────────────


class TestBuildEmbeddingFieldExpressionPreservesNestedObjects:
    def test_nested_doc_fields_projected(self):
        """Nested document fields should be projected via $ifNull, preserving structure."""
        config = _make_config(source_fields=["specs"])
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        assert result["specs"] == {"$ifNull": ["$specs", None]}


# ── Test 16: AUTO mode produces labeled $concat string ─────────────


class TestBuildEmbeddingFieldExpressionAutoMode:
    def test_auto_produces_labeled_concat(self):
        """AUTO mode should produce $concat with field labels."""
        config = _make_config(
            source_fields=["title", "description", "tags"],
            embedding_mode="auto",
        )
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        assert "$concat" in result
        concat_parts = result["$concat"]

        # Should contain field labels
        assert "title: " in concat_parts
        assert "description: " in concat_parts
        assert "tags: " in concat_parts

        # Should have newline separators between fields
        assert "\n" in concat_parts

    def test_auto_uses_toString_for_safety(self):
        """AUTO mode should wrap values in $toString for type safety."""
        config = _make_config(
            source_fields=["title"],
            embedding_mode="auto",
        )
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        concat_parts = result["$concat"]
        # Find the value expression (not the label)
        value_exprs = [p for p in concat_parts if isinstance(p, dict)]
        assert len(value_exprs) > 0
        # Should use $toString wrapping $ifNull
        assert any("$toString" in str(v) for v in value_exprs)


# ── Test 17: Template overrides default ────────────────────────────


class TestBuildEmbeddingFieldExpressionTemplateOverrides:
    def test_byom_with_template(self):
        """Template mode should produce $concat even in BYOM mode."""
        config = _make_config(
            source_fields=["title", "description"],
            template="Product: {title} - {description}",
        )
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        assert "$concat" in result
        concat_parts = result["$concat"]
        assert "Product: " in concat_parts

    def test_auto_with_template(self):
        """Template mode should work the same in AUTO mode."""
        config = _make_config(
            source_fields=["title", "description"],
            template="Product: {title} - {description}",
            embedding_mode="auto",
        )
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()

        assert "$concat" in result
        concat_parts = result["$concat"]
        assert "Product: " in concat_parts


# ── Test 18: End-to-end view pipeline BYOM ─────────────────────────


class TestEndToEndViewPipelineByom:
    def test_full_pipeline_has_object_addfields(self):
        """Full BYOM pipeline should contain $addFields with object projection."""
        config = _make_config(source_fields=["title", "description"])
        ops = MongoDBViewOps(config)
        pipeline = ops._build_partition_view_pipeline(
            "electronics", "electronics"
        )

        # Find the $addFields stage
        add_fields_stages = [s for s in pipeline if "$addFields" in s]
        assert len(add_fields_stages) >= 1

        embedding_text = add_fields_stages[0]["$addFields"]["embedding_text"]
        # Should be an object projection, not $concat
        assert isinstance(embedding_text, dict)
        assert "title" in embedding_text
        assert "description" in embedding_text
        assert "$concat" not in embedding_text


# ── Test 19: End-to-end view pipeline AUTO ─────────────────────────


class TestEndToEndViewPipelineAuto:
    def test_full_pipeline_has_labeled_string(self):
        """Full AUTO pipeline should contain $addFields with labeled $concat string."""
        config = _make_config(
            source_fields=["title", "description"],
            embedding_mode="auto",
        )
        ops = MongoDBViewOps(config)
        pipeline = ops._build_partition_view_pipeline(
            "electronics", "electronics"
        )

        add_fields_stages = [s for s in pipeline if "$addFields" in s]
        assert len(add_fields_stages) >= 1

        embedding_text = add_fields_stages[0]["$addFields"]["embedding_text"]
        assert "$concat" in embedding_text


# ── Test 20: SEPARATE mode unaffected ──────────────────────────────


class TestViewPipelineSeparateModeUnaffected:
    def test_separate_mode_has_lookup(self):
        """SEPARATE mode should still have $lookup after the embedding text stage."""
        config = _make_config(
            source_fields=["title", "description"],
            vector_storage_mode="separate",
        )
        config.vector_storage.embeddings_collection = "embeddings"
        config.vector_storage.reference_field = "source_id"
        ops = MongoDBViewOps(config)
        pipeline = ops._build_partition_view_pipeline(
            "electronics", "electronics"
        )

        # Should have $addFields for embedding text
        add_fields_stages = [s for s in pipeline if "$addFields" in s]
        assert len(add_fields_stages) >= 1

        # Should also have $lookup for SEPARATE mode
        lookup_stages = [s for s in pipeline if "$lookup" in s]
        assert len(lookup_stages) == 1

        # $addFields for embedding text should come BEFORE $lookup
        add_idx = pipeline.index(add_fields_stages[0])
        lookup_idx = pipeline.index(lookup_stages[0])
        assert add_idx < lookup_idx

    def test_separate_mode_byom_object_projection(self):
        """SEPARATE mode + BYOM should still use object projection for embedding text."""
        config = _make_config(
            source_fields=["title"],
            vector_storage_mode="separate",
        )
        config.vector_storage.embeddings_collection = "embeddings"
        ops = MongoDBViewOps(config)
        pipeline = ops._build_partition_view_pipeline(
            "electronics", "electronics"
        )

        add_fields = [s for s in pipeline if "$addFields" in s][0]
        embedding_text = add_fields["$addFields"]["embedding_text"]
        assert "title" in embedding_text
        assert "$concat" not in str(embedding_text)


# ── Test 21: No source_fields → no $addFields ─────────────────────


class TestViewPipelineNoSourceFieldsUnchanged:
    def test_no_addfields_without_source_fields(self):
        """When source_fields is None, no embedding text $addFields should be added."""
        config = _make_config(source_fields=None)
        ops = MongoDBViewOps(config)
        pipeline = ops._build_partition_view_pipeline(
            "electronics", "electronics"
        )

        # Should only have $match, no $addFields for embedding text
        add_fields_stages = [s for s in pipeline if "$addFields" in s]
        assert len(add_fields_stages) == 0


# ── Test: Legacy _build_concat_expression alias ────────────────────


class TestLegacyAlias:
    def test_build_concat_expression_still_works(self):
        """Legacy _build_concat_expression should delegate to new method."""
        config = _make_config(source_fields=["title", "description"])
        ops = MongoDBViewOps(config)
        legacy = ops._build_concat_expression()
        new = ops._build_embedding_field_expression()
        assert legacy == new

    def test_empty_source_fields_literal(self):
        """Empty source_fields should return $literal empty string."""
        config = _make_config(source_fields=[])
        ops = MongoDBViewOps(config)
        result = ops._build_embedding_field_expression()
        assert result == {"$literal": ""}
