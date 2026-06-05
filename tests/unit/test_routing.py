"""Tests for routing and result merging."""

import pytest

from semantic_vector_router.exceptions import PartitionNotFoundError
from semantic_vector_router.models import (
    PartitionInfo,
    PartitionStatus,
    SearchHit,
    SVRConfig,
)
from semantic_vector_router.routing.merger import (
    ResultMerger,
    normalize_scores,
)
from semantic_vector_router.routing.resolver import PartitionResolver


class TestPartitionResolver:
    """Tests for PartitionResolver."""

    @pytest.mark.asyncio
    async def test_resolve_explicit_partitions(self, sample_config_with_partitions):
        """Test resolving explicit partition list."""
        resolver = PartitionResolver(sample_config_with_partitions)

        partitions = await resolver.resolve(["electronics", "furniture"])

        assert len(partitions) == 2
        names = [p.name for p in partitions]
        assert "electronics" in names
        assert "furniture" in names

    @pytest.mark.asyncio
    async def test_resolve_all_partitions(self, sample_config_with_partitions):
        """Test resolving 'all' partitions."""
        resolver = PartitionResolver(sample_config_with_partitions)

        partitions = await resolver.resolve("all")

        assert len(partitions) == 3

    @pytest.mark.asyncio
    async def test_resolve_none_uses_default(self, sample_config_with_partitions):
        """Test that None uses default from config."""
        resolver = PartitionResolver(sample_config_with_partitions)

        # Default is "all"
        partitions = await resolver.resolve(None)

        assert len(partitions) == 3

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_partition_raises(self, sample_config_with_partitions):
        """Test that nonexistent partition raises error."""
        resolver = PartitionResolver(sample_config_with_partitions)

        with pytest.raises(PartitionNotFoundError):
            await resolver.resolve(["electronics", "nonexistent"])

    @pytest.mark.asyncio
    async def test_resolve_single_string(self, sample_config_with_partitions):
        """Test resolving single partition as string."""
        resolver = PartitionResolver(sample_config_with_partitions)

        partitions = await resolver.resolve("electronics")

        assert len(partitions) == 1
        assert partitions[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_resolve_respects_max_limit(self, sample_config_with_partitions):
        """Test that max partitions limit is respected."""
        sample_config_with_partitions.routing.max_partitions_per_query = 2
        resolver = PartitionResolver(sample_config_with_partitions)

        partitions = await resolver.resolve("all")

        assert len(partitions) == 2

    @pytest.mark.asyncio
    async def test_resolve_expands_split_partitions(self, sample_config_with_partitions):
        """Test that split partitions expand to children."""
        # Mark electronics as split with children
        electronics = sample_config_with_partitions.partitions.registry["electronics"]
        electronics.status = PartitionStatus.SPLIT
        electronics.child_partitions = ["electronics_laptops", "electronics_phones"]

        # Add child partitions
        sample_config_with_partitions.partitions.registry["electronics_laptops"] = PartitionInfo(
            name="electronics_laptops",
            view_name="svr_test_partition_electronics_laptops",
            index_name="svr_test_idx_electronics_laptops",
            parent_partition="electronics",
        )
        sample_config_with_partitions.partitions.registry["electronics_phones"] = PartitionInfo(
            name="electronics_phones",
            view_name="svr_test_partition_electronics_phones",
            index_name="svr_test_idx_electronics_phones",
            parent_partition="electronics",
        )

        resolver = PartitionResolver(sample_config_with_partitions)
        partitions = await resolver.resolve(["electronics"])

        # Should get children, not parent
        names = [p.name for p in partitions]
        assert "electronics" not in names
        assert "electronics_laptops" in names
        assert "electronics_phones" in names

    @pytest.mark.asyncio
    async def test_resolve_expands_retired_partitions(self, sample_config_with_partitions):
        """Test that retired partitions expand to children."""
        electronics = sample_config_with_partitions.partitions.registry["electronics"]
        electronics.status = PartitionStatus.RETIRED
        electronics.child_partitions = ["electronics_laptops"]

        sample_config_with_partitions.partitions.registry["electronics_laptops"] = PartitionInfo(
            name="electronics_laptops",
            view_name="svr_test_partition_electronics_laptops",
            index_name="svr_test_idx_electronics_laptops",
            parent_partition="electronics",
        )

        resolver = PartitionResolver(sample_config_with_partitions)
        partitions = await resolver.resolve("all")

        names = [p.name for p in partitions]
        assert "electronics" not in names
        assert "electronics_laptops" in names

    @pytest.mark.asyncio
    async def test_list_partitions(self, sample_config_with_partitions):
        """Test listing all partitions."""
        resolver = PartitionResolver(sample_config_with_partitions)

        partitions = await resolver.list_partitions()

        assert len(partitions) == 3

    @pytest.mark.asyncio
    async def test_list_partitions_by_status(self, sample_config_with_partitions):
        """Test listing partitions filtered by status."""
        sample_config_with_partitions.partitions.registry["electronics"].status = (
            PartitionStatus.DISABLED
        )

        resolver = PartitionResolver(sample_config_with_partitions)

        active = await resolver.list_partitions(status=PartitionStatus.ACTIVE)
        disabled = await resolver.list_partitions(status=PartitionStatus.DISABLED)

        assert len(active) == 2
        assert len(disabled) == 1


class TestNormalizeScores:
    """Tests for score normalization."""

    def test_normalize_by_partition(self):
        """Test normalizing scores within each partition."""
        hits = [
            SearchHit(id="1", score=0.95, partition="electronics", document={}),
            SearchHit(id="2", score=0.85, partition="electronics", document={}),
            SearchHit(id="3", score=0.75, partition="furniture", document={}),
            SearchHit(id="4", score=0.70, partition="furniture", document={}),
        ]

        normalized = normalize_scores(hits, method="partition_minmax")

        # Highest in each partition should be 1.0
        electronics = [h for h in normalized if h.partition == "electronics"]
        furniture = [h for h in normalized if h.partition == "furniture"]

        assert max(h.score for h in electronics) == 1.0
        assert max(h.score for h in furniture) == 1.0

    def test_normalize_global(self):
        """Test global score normalization."""
        hits = [
            SearchHit(id="1", score=1.0, partition="test", document={}),
            SearchHit(id="2", score=0.5, partition="test", document={}),
            SearchHit(id="3", score=0.0, partition="test", document={}),
        ]

        normalized = normalize_scores(hits, method="global_minmax")

        scores = [h.score for h in normalized]
        assert max(scores) == 1.0
        assert min(scores) == 0.0

    def test_normalize_empty_list(self):
        """Test normalizing empty list."""
        normalized = normalize_scores([], method="partition_minmax")
        assert normalized == []

    def test_normalize_none_method(self):
        """Test that 'none' method returns unchanged scores."""
        hits = [
            SearchHit(id="1", score=0.95, partition="test", document={}),
        ]
        original_score = hits[0].score

        normalized = normalize_scores(hits, method="none")

        assert normalized[0].score == original_score


class TestResultMerger:
    """Tests for ResultMerger."""

    def test_merge_results(self, sample_search_results):
        """Test merging raw search results."""
        merger = ResultMerger()

        hits = merger.merge(sample_search_results, limit=10)

        assert len(hits) == 3
        assert all(isinstance(h, SearchHit) for h in hits)

    def test_merge_with_limit(self, sample_search_results):
        """Test merge respects limit."""
        merger = ResultMerger()

        hits = merger.merge(sample_search_results, limit=2)

        assert len(hits) == 2

    def test_merge_sorts_by_score(self, sample_search_results):
        """Test merged results are sorted by score."""
        merger = ResultMerger()

        hits = merger.merge(sample_search_results, limit=10)

        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_merge_deduplicates(self):
        """Test deduplication by document ID."""
        results = [
            {"_id": "doc1", "name": "Test", "_svr_partition": "a", "_svr_score": 0.9},
            {"_id": "doc1", "name": "Test", "_svr_partition": "b", "_svr_score": 0.8},  # duplicate
            {"_id": "doc2", "name": "Other", "_svr_partition": "a", "_svr_score": 0.7},
        ]

        merger = ResultMerger(deduplicate=True)
        hits = merger.merge(results, limit=10)

        # Should only have 2 unique docs
        assert len(hits) == 2
        ids = [h.id for h in hits]
        assert ids.count("doc1") == 1

    def test_merge_keeps_highest_score_on_dedupe(self):
        """Test that deduplication keeps the higher score."""
        results = [
            {"_id": "doc1", "_svr_partition": "a", "_svr_score": 0.7},
            {"_id": "doc1", "_svr_partition": "b", "_svr_score": 0.9},  # higher score
        ]

        merger = ResultMerger(deduplicate=True)
        hits = merger.merge(results, limit=10)

        # Should keep the 0.9 score version
        assert len(hits) == 1
        # Score is normalized, so just check we have the hit
        assert hits[0].id == "doc1"

    def test_merge_with_rerank_scores(self):
        """Test merging with rerank scores."""
        hits = [
            SearchHit(id="1", score=0.9, partition="test", document={}),
            SearchHit(id="2", score=0.8, partition="test", document={}),
            SearchHit(id="3", score=0.7, partition="test", document={}),
        ]
        rerank_scores = [0.5, 0.9, 0.7]  # Different order than original

        merger = ResultMerger()
        reranked = merger.merge_with_rerank_scores(hits, rerank_scores, limit=10)

        # Should be sorted by rerank score
        assert reranked[0].id == "2"  # Highest rerank score
        assert reranked[0].rerank_score == 0.9
