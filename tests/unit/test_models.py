"""Tests for Pydantic models."""

from datetime import datetime

import pytest

from semantic_vector_router.models import (
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    PartitionInfo,
    PartitionStatus,
    SearchHit,
    SearchResult,
    SVRConfig,
    VectorStorageConfig,
)


class TestPartitionInfo:
    """Tests for PartitionInfo model."""

    def test_create_partition_info(self):
        """Test creating a partition info object."""
        partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            document_count=150000,
        )

        assert partition.name == "electronics"
        assert partition.view_name == "svr_partition_electronics"
        assert partition.status == PartitionStatus.ACTIVE
        assert partition.document_count == 150000

    def test_partition_datetime_parsing(self):
        """Test datetime parsing from ISO string."""
        partition = PartitionInfo(
            name="test",
            view_name="test_view",
            index_name="test_index",
            created_at="2025-02-01T10:00:00Z",
        )

        assert isinstance(partition.created_at, datetime)

    def test_partition_defaults(self):
        """Test default values."""
        partition = PartitionInfo(
            name="test",
            view_name="test_view",
            index_name="test_index",
        )

        assert partition.status == PartitionStatus.ACTIVE
        assert partition.document_count == 0
        assert partition.child_partitions == []
        assert partition.metadata == {}
        assert partition.index_location == IndexLocation.VIEWS
        assert partition.search_collection is None

    def test_partition_views_mode(self):
        """Test partition configured for VIEWS mode."""
        partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_vector_idx_electronics",
            index_location=IndexLocation.VIEWS,
            search_collection="svr_partition_electronics",
        )

        assert partition.index_location == IndexLocation.VIEWS
        assert partition.search_collection == "svr_partition_electronics"

    def test_partition_source_mode(self):
        """Test partition configured for SOURCE mode."""
        partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_vector_idx_source",
            filter_value="electronics",
            index_location=IndexLocation.SOURCE,
            search_collection="products",  # Source collection
        )

        assert partition.index_location == IndexLocation.SOURCE
        assert partition.search_collection == "products"
        assert partition.filter_value == "electronics"

    def test_partition_fields_mode(self):
        """Test partition configured for FIELDS mode."""
        partition = PartitionInfo(
            name="electronics",
            view_name=None,
            index_name="svr_vector_idx_electronics",
            filter_value="electronics",
            index_location=IndexLocation.FIELDS,
            search_collection="products",
            embedding_field="embedding_electronics",
        )

        assert partition.index_location == IndexLocation.FIELDS
        assert partition.embedding_field == "embedding_electronics"
        assert partition.view_name is None
        assert partition.search_collection == "products"

    def test_partition_fields_mode_defaults(self):
        """Test FIELDS mode partition defaults."""
        partition = PartitionInfo(
            name="test",
            view_name=None,
            index_name="svr_vector_idx_test",
        )

        assert partition.embedding_field is None  # Only set for FIELDS mode


class TestSearchHit:
    """Tests for SearchHit model."""

    def test_create_search_hit(self):
        """Test creating a search hit."""
        hit = SearchHit(
            id="doc123",
            score=0.95,
            partition="electronics",
            document={"name": "Test Product", "price": 99.99},
        )

        assert hit.id == "doc123"
        assert hit.score == 0.95
        assert hit.partition == "electronics"
        assert hit.rerank_score is None

    def test_search_hit_comparison(self):
        """Test SearchHit comparison for sorting."""
        hit1 = SearchHit(
            id="doc1", score=0.8, partition="test", document={}
        )
        hit2 = SearchHit(
            id="doc2", score=0.9, partition="test", document={}
        )

        assert hit1 < hit2  # Lower score is "less than"

    def test_search_hit_with_rerank_score(self):
        """Test SearchHit with rerank score."""
        hit = SearchHit(
            id="doc1",
            score=0.8,
            rerank_score=0.95,
            partition="test",
            document={},
        )

        assert hit.rerank_score == 0.95


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        hits = [
            SearchHit(id="1", score=0.9, partition="test", document={}),
            SearchHit(id="2", score=0.8, partition="test", document={}),
        ]

        result = SearchResult(
            hits=hits,
            query="test query",
            partitions_searched=["test"],
            total_candidates=10,
            reranked=False,
            latency_ms=50.5,
        )

        assert len(result.hits) == 2
        assert result.query == "test query"
        assert result.reranked is False
        assert result.latency_ms == 50.5


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig model."""

    def test_byom_config(self):
        """Test BYOM embedding config."""
        config = EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key_env="OPENAI_API_KEY",
            dimensions=1536,
        )

        assert config.mode == EmbeddingMode.BYOM
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.dimensions == 1536

    def test_auto_embedding_config(self):
        """Test auto-embedding config."""
        config = EmbeddingConfig(
            mode=EmbeddingMode.AUTO,
            provider=EmbeddingProvider.ATLAS_VOYAGE,
            model="voyage-3-large",
        )

        assert config.mode == EmbeddingMode.AUTO
        assert config.provider == EmbeddingProvider.ATLAS_VOYAGE

    def test_multi_field_config(self):
        """Test multi-field embedding config."""
        config = EmbeddingConfig(
            source_fields=["title", "description", "specs"],
            separator=" | ",
            computed_field="embedding_text",
        )

        assert config.source_fields == ["title", "description", "specs"]
        assert config.separator == " | "
        assert config.computed_field == "embedding_text"


class TestVectorStorageConfig:
    """Tests for VectorStorageConfig model."""

    def test_default_index_location(self):
        """Test default index location is VIEWS."""
        config = VectorStorageConfig()
        assert config.index_on == IndexLocation.VIEWS

    def test_source_index_location(self):
        """Test configuring SOURCE index location."""
        config = VectorStorageConfig(index_on=IndexLocation.SOURCE)
        assert config.index_on == IndexLocation.SOURCE

    def test_views_index_location(self):
        """Test configuring VIEWS index location."""
        config = VectorStorageConfig(index_on=IndexLocation.VIEWS)
        assert config.index_on == IndexLocation.VIEWS

    def test_fields_index_location(self):
        """Test configuring FIELDS index location."""
        config = VectorStorageConfig(index_on=IndexLocation.FIELDS)
        assert config.index_on == IndexLocation.FIELDS


class TestSVRConfig:
    """Tests for main SVRConfig model."""

    def test_minimal_config(self, sample_config):
        """Test minimal valid configuration."""
        assert sample_config.database.database == "test_db"
        assert sample_config.partitioning.field == "category"

    def test_config_defaults(self, sample_config):
        """Test default values are applied."""
        assert sample_config.routing.mode.value == "explicit"
        assert sample_config.routing.default_partitions == "all"
        assert sample_config.lifecycle.auto_provision is True

    def test_config_with_partitions(self, sample_config_with_partitions):
        """Test config with registered partitions."""
        partitions = sample_config_with_partitions.partitions.registry

        assert len(partitions) == 3
        assert "electronics" in partitions
        assert partitions["electronics"].document_count == 150000
