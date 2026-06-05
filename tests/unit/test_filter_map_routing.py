"""Tests for filter-map routing in PartitionResolver.

Tests the filter-map cascParts Distributor step: when resolve() is called with
partitions="all" and filters matching the partition field, the resolver
performs O(1) dict lookup to route to matching leaf partitions.
"""

import pytest

from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    PartitionInfo,
    PartitioningConfig,
    PartitionsRegistry,
    PartitionStatus,
    RerankingConfig,
    RerankerProvider,
    SVRConfig,
    VectorSearchConfig,
)
from semantic_vector_router.routing.resolver import PartitionResolver


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_partition(
    name: str,
    *,
    filter_value: object = None,
    status: PartitionStatus = PartitionStatus.ACTIVE,
    parent_partition: str | None = None,
    child_partitions: list[str] | None = None,
) -> PartitionInfo:
    """Create a PartitionInfo with sensible defaults for testing."""
    return PartitionInfo(
        name=name,
        view_name=f"svr_test_partition_{name}",
        index_name=f"svr_test_idx_{name}",
        filter_value=filter_value if filter_value is not None else name,
        status=status,
        parent_partition=parent_partition,
        child_partitions=child_partitions or [],
    )


def _make_config(
    registry: dict[str, PartitionInfo],
    partition_field: str = "category",
    max_partitions: int = 50,
) -> SVRConfig:
    """Build a minimal SVRConfig with the given partition registry."""
    cfg = SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(
            field=partition_field,
            view_prefix="svr_test_partition_",
            index_name_prefix="svr_test_idx_",
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=1536,
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key_env="OPENAI_API_KEY",
            dimensions=1536,
        ),
        reranking=RerankingConfig(
            enabled=True,
            provider=RerankerProvider.VOYAGE,
            model="rerank-2",
            api_key_env="VOYAGE_API_KEY",
            top_k_per_partition=20,
            final_top_k=10,
        ),
    )
    cfg.routing.max_partitions_per_query = max_partitions
    cfg.partitions = PartitionsRegistry(registry=registry)
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_partition_registry() -> dict[str, PartitionInfo]:
    """Three ACTIVE partitions with distinct filter_values."""
    return {
        "electronics": _make_partition("electronics", filter_value="electronics"),
        "furniture": _make_partition("furniture", filter_value="furniture"),
        "clothing": _make_partition("clothing", filter_value="clothing"),
    }


@pytest.fixture
def split_partition_registry() -> dict[str, PartitionInfo]:
    """Registry where 'electronics' is SPLIT into two children."""
    return {
        "electronics": _make_partition(
            "electronics",
            filter_value="electronics",
            status=PartitionStatus.SPLIT,
            child_partitions=["electronics_laptops", "electronics_phones"],
        ),
        "electronics_laptops": _make_partition(
            "electronics_laptops",
            filter_value="electronics_laptops",
            parent_partition="electronics",
        ),
        "electronics_phones": _make_partition(
            "electronics_phones",
            filter_value="electronics_phones",
            parent_partition="electronics",
        ),
        "furniture": _make_partition("furniture", filter_value="furniture"),
    }


@pytest.fixture
def nested_split_registry() -> dict[str, PartitionInfo]:
    """Grandparent SPLIT -> parent SPLIT -> leaf ACTIVE children."""
    return {
        "electronics": _make_partition(
            "electronics",
            filter_value="electronics",
            status=PartitionStatus.SPLIT,
            child_partitions=["electronics_computing"],
        ),
        "electronics_computing": _make_partition(
            "electronics_computing",
            filter_value="electronics_computing",
            status=PartitionStatus.SPLIT,
            parent_partition="electronics",
            child_partitions=["electronics_laptops", "electronics_desktops"],
        ),
        "electronics_laptops": _make_partition(
            "electronics_laptops",
            filter_value="electronics_laptops",
            parent_partition="electronics_computing",
        ),
        "electronics_desktops": _make_partition(
            "electronics_desktops",
            filter_value="electronics_desktops",
            parent_partition="electronics_computing",
        ),
        "furniture": _make_partition("furniture", filter_value="furniture"),
    }


# ===========================================================================
# Test class: _try_filter_map
# ===========================================================================


class TestTryFilterMap:
    """Tests for the _try_filter_map method directly."""

    @pytest.mark.asyncio
    async def test_filter_matches_partition_field(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter key == partition field resolves to correct partition."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"category": "electronics"})

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_filter_no_match_on_field(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter key != partition field returns None (fall through)."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"brand": "Sony"})

        assert result is None

    @pytest.mark.asyncio
    async def test_filter_value_not_in_map(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter value not matching any partition returns None."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"category": "automotive"})

        assert result is None

    @pytest.mark.asyncio
    async def test_filter_matches_split_partition(
        self, split_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter matching a SPLIT partition resolves to leaf children."""
        config = _make_config(split_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"category": "electronics"})

        assert result is not None
        names = [p.name for p in result]
        assert "electronics_laptops" in names
        assert "electronics_phones" in names
        assert "electronics" not in names

    @pytest.mark.asyncio
    async def test_filter_matches_retired_partition(self) -> None:
        """Filter matching a RETIRED partition resolves to leaf children."""
        registry = {
            "electronics": _make_partition(
                "electronics",
                filter_value="electronics",
                status=PartitionStatus.RETIRED,
                child_partitions=["electronics_v2"],
            ),
            "electronics_v2": _make_partition(
                "electronics_v2",
                filter_value="electronics_v2",
                parent_partition="electronics",
            ),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"category": "electronics"})

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "electronics_v2"

    @pytest.mark.asyncio
    async def test_filter_multiple_values_list(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter value as a list resolves all matching partitions."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map(
            {"category": ["electronics", "furniture"]},
        )

        assert result is not None
        names = [p.name for p in result]
        assert "electronics" in names
        assert "furniture" in names
        assert len(names) == 2

    @pytest.mark.asyncio
    async def test_filter_list_with_some_not_found(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter list with a mix of found/not-found values returns found only."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map(
            {"category": ["electronics", "automotive"]},
        )

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_filter_list_all_not_found(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter list where no values match returns None."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map(
            {"category": ["automotive", "garden"]},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_filter_with_non_string_value(self) -> None:
        """Non-string filter values are converted to str for lookup."""
        registry = {
            "tier_1": _make_partition("tier_1", filter_value=1),
            "tier_2": _make_partition("tier_2", filter_value=2),
        }
        config = _make_config(registry, partition_field="tier")
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"tier": 1})

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "tier_1"

    @pytest.mark.asyncio
    async def test_filter_with_non_string_list_values(self) -> None:
        """Non-string values in list are converted to str."""
        registry = {
            "tier_1": _make_partition("tier_1", filter_value=1),
            "tier_2": _make_partition("tier_2", filter_value=2),
        }
        config = _make_config(registry, partition_field="tier")
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"tier": [1, 2]})

        assert result is not None
        names = [p.name for p in result]
        assert "tier_1" in names
        assert "tier_2" in names

    @pytest.mark.asyncio
    async def test_disabled_partitions_excluded_from_filter_map(self) -> None:
        """DISABLED partitions do not appear in the filter map."""
        registry = {
            "electronics": _make_partition(
                "electronics",
                filter_value="electronics",
                status=PartitionStatus.DISABLED,
            ),
            "furniture": _make_partition("furniture", filter_value="furniture"),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"category": "electronics"})

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_registry(self) -> None:
        """Empty registry returns None for any filter."""
        config = _make_config({})
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"category": "electronics"})

        assert result is None

    @pytest.mark.asyncio
    async def test_nested_split_resolves_to_leaves(
        self, nested_split_registry: dict[str, PartitionInfo],
    ) -> None:
        """Grandparent SPLIT -> parent SPLIT -> resolves to ACTIVE leaves."""
        config = _make_config(nested_split_registry)
        resolver = PartitionResolver(config)

        result = await resolver._try_filter_map({"category": "electronics"})

        assert result is not None
        names = [p.name for p in result]
        assert "electronics_laptops" in names
        assert "electronics_desktops" in names
        assert "electronics" not in names
        assert "electronics_computing" not in names

    @pytest.mark.asyncio
    async def test_include_disabled_flag_passthrough(self) -> None:
        """include_disabled=True passes through to _resolve_explicit."""
        registry = {
            "electronics": _make_partition(
                "electronics",
                filter_value="electronics",
                status=PartitionStatus.DISABLED,
            ),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        # DISABLED partitions are excluded from the filter MAP itself,
        # so even with include_disabled the filter map won't contain them.
        result = await resolver._try_filter_map(
            {"category": "electronics"}, include_disabled=True,
        )
        assert result is None


# ===========================================================================
# Test class: _build_filter_map
# ===========================================================================


class TestBuildFilterMap:
    """Tests for _build_filter_map internals."""

    @pytest.mark.asyncio
    async def test_basic_map_structure(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Each ACTIVE partition maps filter_value -> [name]."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        fmap = resolver._build_filter_map(three_partition_registry)

        assert fmap["electronics"] == ["electronics"]
        assert fmap["furniture"] == ["furniture"]
        assert fmap["clothing"] == ["clothing"]

    @pytest.mark.asyncio
    async def test_split_partition_maps_to_leaves(
        self, split_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """SPLIT partition's filter_value maps to its ACTIVE leaf children."""
        config = _make_config(split_partition_registry)
        resolver = PartitionResolver(config)

        fmap = resolver._build_filter_map(split_partition_registry)

        assert "electronics_laptops" in fmap["electronics"]
        assert "electronics_phones" in fmap["electronics"]

    @pytest.mark.asyncio
    async def test_cache_returns_same_map(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Subsequent calls with same registry size return cached map."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        fmap1 = resolver._build_filter_map(three_partition_registry)
        fmap2 = resolver._build_filter_map(three_partition_registry)

        assert fmap1 is fmap2

    @pytest.mark.asyncio
    async def test_cache_invalidated_on_registry_size_change(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Adding a partition to registry triggers rebuild."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        fmap1 = resolver._build_filter_map(three_partition_registry)
        assert "electronics" in fmap1

        # Add a new partition
        three_partition_registry["toys"] = _make_partition(
            "toys", filter_value="toys",
        )
        fmap2 = resolver._build_filter_map(three_partition_registry)

        assert fmap2 is not fmap1
        assert "toys" in fmap2

    @pytest.mark.asyncio
    async def test_disabled_excluded(self) -> None:
        """DISABLED partitions are excluded from the map."""
        registry = {
            "active_one": _make_partition("active_one", filter_value="a"),
            "disabled_one": _make_partition(
                "disabled_one",
                filter_value="d",
                status=PartitionStatus.DISABLED,
            ),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        fmap = resolver._build_filter_map(registry)

        assert "a" in fmap
        assert "d" not in fmap

    @pytest.mark.asyncio
    async def test_duplicate_filter_values(self) -> None:
        """Multiple ACTIVE partitions with the same filter_value are grouped."""
        registry = {
            "electronics_us": _make_partition(
                "electronics_us", filter_value="electronics",
            ),
            "electronics_eu": _make_partition(
                "electronics_eu", filter_value="electronics",
            ),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        fmap = resolver._build_filter_map(registry)

        assert len(fmap["electronics"]) == 2
        assert "electronics_us" in fmap["electronics"]
        assert "electronics_eu" in fmap["electronics"]

    @pytest.mark.asyncio
    async def test_filter_value_none_uses_name(self) -> None:
        """Partition with filter_value=None uses partition name as key."""
        registry = {
            "my_partition": PartitionInfo(
                name="my_partition",
                view_name="svr_test_partition_my_partition",
                index_name="svr_test_idx_my_partition",
                filter_value=None,
                status=PartitionStatus.ACTIVE,
            ),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        fmap = resolver._build_filter_map(registry)

        assert "my_partition" in fmap

    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_map(self) -> None:
        """Empty registry produces empty filter map."""
        config = _make_config({})
        resolver = PartitionResolver(config)

        fmap = resolver._build_filter_map({})

        assert fmap == {}

    @pytest.mark.asyncio
    async def test_nested_split_maps_to_deepest_leaves(
        self, nested_split_registry: dict[str, PartitionInfo],
    ) -> None:
        """Multi-level splits map to the deepest ACTIVE leaves."""
        config = _make_config(nested_split_registry)
        resolver = PartitionResolver(config)

        fmap = resolver._build_filter_map(nested_split_registry)

        assert "electronics_laptops" in fmap["electronics"]
        assert "electronics_desktops" in fmap["electronics"]
        assert len(fmap["electronics"]) == 2


# ===========================================================================
# Test class: _collect_leaves
# ===========================================================================


class TestCollectLeaves:
    """Tests for the _collect_leaves recursive method."""

    def test_active_partition_is_leaf(self) -> None:
        """ACTIVE partition returns itself."""
        p = _make_partition("electronics")
        registry = {"electronics": p}
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        leaves = resolver._collect_leaves(p, registry)

        assert len(leaves) == 1
        assert leaves[0].name == "electronics"

    def test_split_with_active_children(self) -> None:
        """SPLIT partition with ACTIVE children returns those children."""
        parent = _make_partition(
            "electronics",
            status=PartitionStatus.SPLIT,
            child_partitions=["e_laptops", "e_phones"],
        )
        child1 = _make_partition("e_laptops", parent_partition="electronics")
        child2 = _make_partition("e_phones", parent_partition="electronics")
        registry = {
            "electronics": parent,
            "e_laptops": child1,
            "e_phones": child2,
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        leaves = resolver._collect_leaves(parent, registry)

        names = [l.name for l in leaves]
        assert "e_laptops" in names
        assert "e_phones" in names
        assert "electronics" not in names

    def test_skips_disabled_children(self) -> None:
        """DISABLED children are skipped during leaf collection."""
        parent = _make_partition(
            "electronics",
            status=PartitionStatus.SPLIT,
            child_partitions=["e_active", "e_disabled"],
        )
        child_active = _make_partition("e_active", parent_partition="electronics")
        child_disabled = _make_partition(
            "e_disabled",
            parent_partition="electronics",
            status=PartitionStatus.DISABLED,
        )
        registry = {
            "electronics": parent,
            "e_active": child_active,
            "e_disabled": child_disabled,
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        leaves = resolver._collect_leaves(parent, registry)

        assert len(leaves) == 1
        assert leaves[0].name == "e_active"

    def test_nested_splits(
        self, nested_split_registry: dict[str, PartitionInfo],
    ) -> None:
        """Multi-level nested splits collect deepest ACTIVE leaves."""
        config = _make_config(nested_split_registry)
        resolver = PartitionResolver(config)
        root = nested_split_registry["electronics"]

        leaves = resolver._collect_leaves(root, nested_split_registry)

        names = [l.name for l in leaves]
        assert "electronics_laptops" in names
        assert "electronics_desktops" in names
        assert len(names) == 2

    def test_split_with_missing_child(self) -> None:
        """Child referenced but not in registry is silently skipped."""
        parent = _make_partition(
            "electronics",
            status=PartitionStatus.SPLIT,
            child_partitions=["e_exists", "e_missing"],
        )
        child = _make_partition("e_exists", parent_partition="electronics")
        registry = {"electronics": parent, "e_exists": child}
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        leaves = resolver._collect_leaves(parent, registry)

        assert len(leaves) == 1
        assert leaves[0].name == "e_exists"

    def test_split_with_no_children(self) -> None:
        """SPLIT partition with empty child_partitions returns no leaves."""
        parent = _make_partition(
            "electronics",
            status=PartitionStatus.SPLIT,
            child_partitions=[],
        )
        registry = {"electronics": parent}
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        leaves = resolver._collect_leaves(parent, registry)

        assert leaves == []


# ===========================================================================
# Test class: invalidate_filter_map
# ===========================================================================


class TestInvalidateFilterMap:
    """Tests for filter map cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_clears_cache(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """invalidate_filter_map() clears the cached map."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        # Build the cache
        fmap1 = resolver._build_filter_map(three_partition_registry)
        assert fmap1  # non-empty

        resolver.invalidate_filter_map()

        assert resolver._filter_map == {}
        assert resolver._registry_version == 0

    @pytest.mark.asyncio
    async def test_invalidate_forces_rebuild(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """After invalidation, next _build_filter_map rebuilds fresh."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        fmap1 = resolver._build_filter_map(three_partition_registry)
        resolver.invalidate_filter_map()
        fmap2 = resolver._build_filter_map(three_partition_registry)

        # Should be a new dict object (rebuilt)
        assert fmap2 is not fmap1
        # But same contents
        assert fmap2 == fmap1


# ===========================================================================
# Test class: resolve() integration with filter-map
# ===========================================================================


class TestResolveFilterMapIntegration:
    """Tests for resolve() using the filter-map cascParts Distributor step."""

    @pytest.mark.asyncio
    async def test_resolve_all_with_matching_filter(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """resolve(partitions='all', filters={field: val}) uses filter map."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={"category": "electronics"},
        )

        assert len(result) == 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_resolve_all_with_no_filters_fans_out(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """resolve(partitions='all', filters=None) fans out to all."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(partitions="all")

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_resolve_all_with_unmatched_filter_falls_through(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter with non-matching value falls through to 'all' fanout."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={"category": "automotive"},
        )

        # Falls through to "all" since filter value not found
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_resolve_all_with_irrelevant_filter_falls_through(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter key != partition field falls through to 'all'."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={"brand": "Sony"},
        )

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_resolve_explicit_partitions_ignores_filters(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """resolve() with explicit partition list ignores filters entirely."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions=["electronics"],
            filters={"category": "furniture"},
        )

        assert len(result) == 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_resolve_single_string_ignores_filters(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """resolve() with single string partition ignores filters."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="furniture",
            filters={"category": "electronics"},
        )

        assert len(result) == 1
        assert result[0].name == "furniture"

    @pytest.mark.asyncio
    async def test_resolve_none_with_filters_uses_default(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """resolve(None, filters) uses default_partitions (which is 'all')."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions=None,
            filters={"category": "electronics"},
        )

        # Default is "all", filter matches → should resolve to electronics
        assert len(result) == 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_resolve_all_with_split_partition_filter(
        self, split_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter matching SPLIT partition resolves to leaves via resolve()."""
        config = _make_config(split_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={"category": "electronics"},
        )

        names = [p.name for p in result]
        assert "electronics_laptops" in names
        assert "electronics_phones" in names
        assert "electronics" not in names

    @pytest.mark.asyncio
    async def test_resolve_filter_multiple_values(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Filter with list of values resolves all matches."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={"category": ["electronics", "clothing"]},
        )

        names = [p.name for p in result]
        assert len(names) == 2
        assert "electronics" in names
        assert "clothing" in names

    @pytest.mark.asyncio
    async def test_resolve_filter_deduplicates_results(self) -> None:
        """Duplicate partition names from filter list are deduplicated."""
        registry = {
            "electronics_us": _make_partition(
                "electronics_us", filter_value="electronics",
            ),
            "electronics_eu": _make_partition(
                "electronics_eu", filter_value="electronics",
            ),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        # Both "electronics" filter values point to the same partitions
        # The list has "electronics" twice, but dedup should handle it
        result = await resolver.resolve(
            partitions="all",
            filters={"category": ["electronics", "electronics"]},
        )

        names = [p.name for p in result]
        # Should be deduplicated
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_resolve_filter_with_empty_filters_dict(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Empty filters dict falls through to 'all' fan-out."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={},
        )

        # Empty dict is falsy, so filter-map step is skipped
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_resolve_filter_nested_split(
        self, nested_split_registry: dict[str, PartitionInfo],
    ) -> None:
        """Nested splits are resolved to deepest leaves via resolve()."""
        config = _make_config(nested_split_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={"category": "electronics"},
        )

        names = [p.name for p in result]
        assert "electronics_laptops" in names
        assert "electronics_desktops" in names
        assert "electronics" not in names
        assert "electronics_computing" not in names

    @pytest.mark.asyncio
    async def test_resolve_respects_max_partitions_with_filter(self) -> None:
        """Max partitions limit applies to filter-map results too."""
        registry = {
            f"p_{i}": _make_partition(f"p_{i}", filter_value="shared")
            for i in range(10)
        }
        config = _make_config(registry, max_partitions=3)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            filters={"category": "shared"},
        )

        assert len(result) == 3


# ===========================================================================
# Test class: Existing behavior preserved
# ===========================================================================


class TestExistingBehaviorPreserved:
    """Ensure pre-existing resolve() behavior is not broken."""

    @pytest.mark.asyncio
    async def test_resolve_explicit_list(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Explicit list still works."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(["electronics", "furniture"])

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_resolve_single_string(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Single string still works."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve("electronics")

        assert len(result) == 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_resolve_all_without_filters(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """'all' without filters still fans out."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve("all")

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_resolve_none_defaults_to_all(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """None still defaults to config's default_partitions."""
        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve(None)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_raises(
        self, three_partition_registry: dict[str, PartitionInfo],
    ) -> None:
        """Nonexistent partition still raises PartitionNotFoundError."""
        from semantic_vector_router.exceptions import PartitionNotFoundError

        config = _make_config(three_partition_registry)
        resolver = PartitionResolver(config)

        with pytest.raises(PartitionNotFoundError):
            await resolver.resolve(["nonexistent"])

    @pytest.mark.asyncio
    async def test_resolve_disabled_skipped(self) -> None:
        """DISABLED partitions skipped when include_disabled=False."""
        registry = {
            "active": _make_partition("active"),
            "disabled": _make_partition(
                "disabled", status=PartitionStatus.DISABLED,
            ),
        }
        config = _make_config(registry)
        resolver = PartitionResolver(config)

        result = await resolver.resolve("all")

        names = [p.name for p in result]
        assert "active" in names
        assert "disabled" not in names
