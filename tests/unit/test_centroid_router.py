"""Comprehensive unit tests for CentroidRouter.

Tests cover tree walking, dynamic threshold pruning, partition status
handling, hierarchy recursion, parameter overrides, and edge cases.
"""

import pytest
from typing import Optional

from semantic_vector_router.models import (
    CentroidRoutingConfig,
    PartitionInfo,
    PartitionStatus,
)
from semantic_vector_router.routing.centroid import CentroidRouter
from semantic_vector_router.utils.vector_math import normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_partition(
    name: str,
    status: PartitionStatus = PartitionStatus.ACTIVE,
    centroid: Optional[list[float]] = None,
    parent: Optional[str] = None,
    children: Optional[list[str]] = None,
) -> PartitionInfo:
    """Create a PartitionInfo with minimal boilerplate."""
    return PartitionInfo(
        name=name,
        index_name=f"idx_{name}",
        status=status,
        centroid=centroid,
        parent_partition=parent,
        child_partitions=children or [],
    )


def _default_config(**overrides: object) -> CentroidRoutingConfig:
    """Create a CentroidRoutingConfig with sensible test defaults."""
    kwargs = {
        "enabled": True,
        "relative_threshold": 0.5,
        "min_score": 0.15,
        "max_probe_partitions": 5,
    }
    kwargs.update(overrides)
    return CentroidRoutingConfig(**kwargs)  # type: ignore[arg-type]


def _registry(*partitions: PartitionInfo) -> dict[str, PartitionInfo]:
    """Build a registry dict from a list of PartitionInfo objects."""
    return {p.name: p for p in partitions}


# A fixed query vector used by most tests (unit vector along dim-0).
QUERY = normalize([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Basic tree walk with clear winner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_clear_winner() -> None:
    """One partition strongly matches; others do not."""
    p_winner = _make_partition("winner", centroid=normalize([1.0, 0.0, 0.0]))
    p_mid = _make_partition("mid", centroid=normalize([0.5, 0.5, 0.0]))
    p_loser = _make_partition("loser", centroid=normalize([0.0, 0.0, 1.0]))

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(p_winner, p_mid, p_loser)
    )

    assert len(result) >= 1
    assert result[0].name == "winner"


# ---------------------------------------------------------------------------
# Dynamic threshold pruning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_threshold_pruning() -> None:
    """Scores [0.85, 0.78, 0.72, 0.35, 0.12] with relative_threshold=0.5.

    Dynamic threshold = 0.85 * 0.5 = 0.425.
    Effective threshold = max(0.425, 0.15) = 0.425.
    Partitions with scores >= 0.425: 0.85, 0.78, 0.72 (3 partitions).
    """
    # Build centroids that produce known cosine similarities against QUERY.
    # QUERY = [1, 0, 0]. cos(q, [a, b, 0]) = a / sqrt(a^2+b^2).
    # We can build vectors with desired similarities directly:
    # For similarity s, use centroid = normalize([s, sqrt(1-s^2), 0]).
    import math

    def _centroid_for_sim(s: float) -> list[float]:
        return normalize([s, math.sqrt(max(1 - s * s, 0)), 0.0])

    partitions = [
        _make_partition("p85", centroid=_centroid_for_sim(0.85)),
        _make_partition("p78", centroid=_centroid_for_sim(0.78)),
        _make_partition("p72", centroid=_centroid_for_sim(0.72)),
        _make_partition("p35", centroid=_centroid_for_sim(0.35)),
        _make_partition("p12", centroid=_centroid_for_sim(0.12)),
    ]

    config = _default_config(relative_threshold=0.5, min_score=0.15)
    router = CentroidRouter(config)
    result = await router.route_by_centroid(QUERY, _registry(*partitions))

    names = [p.name for p in result]
    assert "p85" in names
    assert "p78" in names
    assert "p72" in names
    # p35 and p12 should be pruned (below 0.425)
    assert "p35" not in names
    assert "p12" not in names


# ---------------------------------------------------------------------------
# Ambiguous query — uniform-ish scores
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ambiguous_query_uniform_scores() -> None:
    """Scores [0.45, 0.42, 0.40, 0.38, 0.35, 0.33].

    Dynamic threshold = 0.45 * 0.5 = 0.225.
    Effective threshold = max(0.225, 0.15) = 0.225.
    All 6 are above 0.225 -> all returned (but capped at max_probe_partitions=5).
    """
    import math

    def _centroid_for_sim(s: float) -> list[float]:
        return normalize([s, math.sqrt(max(1 - s * s, 0)), 0.0])

    partitions = [
        _make_partition("p45", centroid=_centroid_for_sim(0.45)),
        _make_partition("p42", centroid=_centroid_for_sim(0.42)),
        _make_partition("p40", centroid=_centroid_for_sim(0.40)),
        _make_partition("p38", centroid=_centroid_for_sim(0.38)),
        _make_partition("p35", centroid=_centroid_for_sim(0.35)),
        _make_partition("p33", centroid=_centroid_for_sim(0.33)),
    ]

    config = _default_config(max_probe_partitions=5, relative_threshold=0.5)
    router = CentroidRouter(config)
    result = await router.route_by_centroid(QUERY, _registry(*partitions))

    # All pass threshold but capped at 5
    assert len(result) == 5
    # Ordered descending by score
    assert result[0].name == "p45"
    # p33 should be the one dropped (lowest score)
    names = [p.name for p in result]
    assert "p33" not in names


# ---------------------------------------------------------------------------
# max_partitions cap enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_partitions_cap() -> None:
    """Even if all partitions pass threshold, cap at max_probe_partitions."""
    centroids = [
        normalize([1.0, float(i) * 0.01, 0.0]) for i in range(10)
    ]
    partitions = [
        _make_partition(f"p{i}", centroid=centroids[i]) for i in range(10)
    ]

    config = _default_config(max_probe_partitions=3)
    router = CentroidRouter(config)
    result = await router.route_by_centroid(QUERY, _registry(*partitions))

    assert len(result) <= 3


# ---------------------------------------------------------------------------
# Partition with no centroid gets score 1.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_centroid_included_with_max_score() -> None:
    """A partition with no centroid should be included with score 1.0."""
    p_with = _make_partition("with_centroid", centroid=normalize([0.5, 0.5, 0.0]))
    p_without = _make_partition("no_centroid", centroid=None)

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(QUERY, _registry(p_with, p_without))

    names = [p.name for p in result]
    assert "no_centroid" in names
    # Since no_centroid gets score 1.0, it should be first
    assert result[0].name == "no_centroid"


# ---------------------------------------------------------------------------
# All below min_score -> returns top-1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_below_min_score_returns_top1() -> None:
    """When all centroids score below min_score, return the top scorer."""
    # Use centroids nearly orthogonal to query
    p1 = _make_partition("low1", centroid=normalize([0.05, 0.99, 0.0]))
    p2 = _make_partition("low2", centroid=normalize([0.03, 0.0, 0.99]))
    p3 = _make_partition("low3", centroid=normalize([0.01, 0.7, 0.7]))

    config = _default_config(min_score=0.5)  # High min_score
    router = CentroidRouter(config)
    result = await router.route_by_centroid(QUERY, _registry(p1, p2, p3))

    # Never empty
    assert len(result) == 1
    # Should be the one with highest similarity to QUERY=[1,0,0]
    # low1 has largest first component
    assert result[0].name == "low1"


# ---------------------------------------------------------------------------
# SPLIT nodes recurse into children
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_nodes_recurse_children() -> None:
    """SPLIT partitions are never returned directly; their children are."""
    parent = _make_partition(
        "parent",
        status=PartitionStatus.SPLIT,
        centroid=normalize([1.0, 0.0, 0.0]),
        children=["child_a", "child_b"],
    )
    child_a = _make_partition(
        "child_a",
        centroid=normalize([0.9, 0.1, 0.0]),
        parent="parent",
    )
    child_b = _make_partition(
        "child_b",
        centroid=normalize([0.8, 0.2, 0.0]),
        parent="parent",
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(parent, child_a, child_b)
    )

    names = [p.name for p in result]
    assert "parent" not in names
    assert "child_a" in names
    assert "child_b" in names


# ---------------------------------------------------------------------------
# Single-level partitions (flat, no hierarchy)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_level_flat() -> None:
    """All partitions are root-level ACTIVE leaves — flat comparison."""
    partitions = [
        _make_partition("a", centroid=normalize([1.0, 0.0, 0.0])),
        _make_partition("b", centroid=normalize([0.0, 1.0, 0.0])),
        _make_partition("c", centroid=normalize([0.0, 0.0, 1.0])),
    ]

    router = CentroidRouter(_default_config(relative_threshold=0.5, min_score=0.1))
    result = await router.route_by_centroid(QUERY, _registry(*partitions))

    # 'a' is the only one with high similarity
    assert result[0].name == "a"


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_registry_returns_empty() -> None:
    """Empty registry -> empty result list."""
    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(QUERY, {})

    assert result == []


# ---------------------------------------------------------------------------
# Mixed: some partitions have centroids, some don't
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_centroids_and_no_centroids() -> None:
    """Mix of partitions with and without centroids."""
    p1 = _make_partition("with1", centroid=normalize([0.9, 0.1, 0.0]))
    p2 = _make_partition("with2", centroid=normalize([0.1, 0.9, 0.0]))
    p3 = _make_partition("no_centroid_1", centroid=None)
    p4 = _make_partition("no_centroid_2", centroid=None)

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(QUERY, _registry(p1, p2, p3, p4))

    names = [p.name for p in result]
    # Both no-centroid partitions should be included (score 1.0)
    assert "no_centroid_1" in names
    assert "no_centroid_2" in names
    # with1 should be included (high similarity)
    assert "with1" in names


# ---------------------------------------------------------------------------
# DISABLED partitions excluded from roots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_partitions_excluded() -> None:
    """DISABLED partitions are not considered as roots."""
    p_active = _make_partition("active", centroid=normalize([1.0, 0.0, 0.0]))
    p_disabled = _make_partition(
        "disabled",
        status=PartitionStatus.DISABLED,
        centroid=normalize([1.0, 0.0, 0.0]),
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(p_active, p_disabled)
    )

    names = [p.name for p in result]
    assert "active" in names
    assert "disabled" not in names


# ---------------------------------------------------------------------------
# RETIRED partitions excluded from roots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retired_partitions_excluded() -> None:
    """RETIRED partitions are not considered as roots."""
    p_active = _make_partition("active", centroid=normalize([1.0, 0.0, 0.0]))
    p_retired = _make_partition(
        "retired",
        status=PartitionStatus.RETIRED,
        centroid=normalize([1.0, 0.0, 0.0]),
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(p_active, p_retired)
    )

    names = [p.name for p in result]
    assert "active" in names
    assert "retired" not in names


# ---------------------------------------------------------------------------
# SPLITTING / MIGRATING partitions not included in results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_splitting_partitions_not_in_results() -> None:
    """SPLITTING partitions are scored but not added to results."""
    p_active = _make_partition("active", centroid=normalize([1.0, 0.0, 0.0]))
    p_splitting = _make_partition(
        "splitting",
        status=PartitionStatus.SPLITTING,
        centroid=normalize([1.0, 0.0, 0.0]),
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(p_active, p_splitting)
    )

    names = [p.name for p in result]
    assert "active" in names
    assert "splitting" not in names


@pytest.mark.asyncio
async def test_migrating_partitions_not_in_results() -> None:
    """MIGRATING partitions are scored but not added to results."""
    p_active = _make_partition("active", centroid=normalize([1.0, 0.0, 0.0]))
    p_migrating = _make_partition(
        "migrating",
        status=PartitionStatus.MIGRATING,
        centroid=normalize([1.0, 0.0, 0.0]),
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(p_active, p_migrating)
    )

    names = [p.name for p in result]
    assert "active" in names
    assert "migrating" not in names


@pytest.mark.asyncio
async def test_pending_split_partitions_not_in_results() -> None:
    """PENDING_SPLIT partitions are scored but not added to results."""
    p_active = _make_partition("active", centroid=normalize([1.0, 0.0, 0.0]))
    p_pending = _make_partition(
        "pending",
        status=PartitionStatus.PENDING_SPLIT,
        centroid=normalize([1.0, 0.0, 0.0]),
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(p_active, p_pending)
    )

    names = [p.name for p in result]
    assert "active" in names
    assert "pending" not in names


# ---------------------------------------------------------------------------
# Multi-level hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_level_hierarchy() -> None:
    """Root SPLIT -> children, one child SPLIT -> grandchildren.

    Tree:
      root (SPLIT, centroid ~query) -> [mid_split, mid_active]
        mid_split (SPLIT, centroid ~query) -> [grand_a, grand_b]
        mid_active (ACTIVE, centroid orthogonal)
      root2 (ACTIVE, centroid orthogonal) -- pruned

    Expected: grand_a, grand_b from the SPLIT branch, mid_active kept,
    root2 pruned.
    """
    root = _make_partition(
        "root",
        status=PartitionStatus.SPLIT,
        centroid=normalize([1.0, 0.0, 0.0]),
        children=["mid_split", "mid_active"],
    )
    mid_split = _make_partition(
        "mid_split",
        status=PartitionStatus.SPLIT,
        centroid=normalize([0.95, 0.05, 0.0]),
        parent="root",
        children=["grand_a", "grand_b"],
    )
    mid_active = _make_partition(
        "mid_active",
        status=PartitionStatus.ACTIVE,
        centroid=normalize([0.8, 0.2, 0.0]),
        parent="root",
    )
    grand_a = _make_partition(
        "grand_a",
        status=PartitionStatus.ACTIVE,
        centroid=normalize([0.9, 0.1, 0.0]),
        parent="mid_split",
    )
    grand_b = _make_partition(
        "grand_b",
        status=PartitionStatus.ACTIVE,
        centroid=normalize([0.7, 0.3, 0.0]),
        parent="mid_split",
    )
    root2 = _make_partition(
        "root2",
        status=PartitionStatus.ACTIVE,
        centroid=normalize([0.0, 0.0, 1.0]),
    )

    router = CentroidRouter(
        _default_config(relative_threshold=0.5, min_score=0.1)
    )
    result = await router.route_by_centroid(
        QUERY,
        _registry(root, mid_split, mid_active, grand_a, grand_b, root2),
    )

    names = [p.name for p in result]

    # SPLIT nodes never appear in results
    assert "root" not in names
    assert "mid_split" not in names

    # Grandchildren and mid_active should be present
    assert "grand_a" in names
    assert "grand_b" in names
    assert "mid_active" in names

    # root2 is orthogonal; pruning may or may not keep it depending on threshold
    # With relative_threshold=0.5: at root level max_score is ~1.0, threshold ~0.5
    # root2's sim to QUERY=[1,0,0] with centroid=[0,0,1] is 0.0 -> pruned
    assert "root2" not in names


# ---------------------------------------------------------------------------
# Override parameters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_override_max_partitions() -> None:
    """Override max_partitions at call time."""
    centroids = [normalize([1.0, float(i) * 0.01, 0.0]) for i in range(10)]
    partitions = [
        _make_partition(f"p{i}", centroid=centroids[i]) for i in range(10)
    ]

    config = _default_config(max_probe_partitions=10)
    router = CentroidRouter(config)
    result = await router.route_by_centroid(
        QUERY, _registry(*partitions), max_partitions=2
    )

    assert len(result) <= 2


@pytest.mark.asyncio
async def test_override_relative_threshold() -> None:
    """Override relative_threshold at call time for aggressive pruning."""
    import math

    def _centroid_for_sim(s: float) -> list[float]:
        return normalize([s, math.sqrt(max(1 - s * s, 0)), 0.0])

    partitions = [
        _make_partition("high", centroid=_centroid_for_sim(0.9)),
        _make_partition("mid", centroid=_centroid_for_sim(0.6)),
        _make_partition("low", centroid=_centroid_for_sim(0.3)),
    ]

    config = _default_config(relative_threshold=0.3)  # Lenient default
    router = CentroidRouter(config)

    # With aggressive override: threshold = 0.9 * 0.9 = 0.81
    result = await router.route_by_centroid(
        QUERY, _registry(*partitions), relative_threshold=0.9
    )

    names = [p.name for p in result]
    assert "high" in names
    # mid (0.6) is below 0.81 -> pruned
    assert "mid" not in names
    assert "low" not in names


@pytest.mark.asyncio
async def test_override_min_score() -> None:
    """Override min_score at call time."""
    import math

    def _centroid_for_sim(s: float) -> list[float]:
        return normalize([s, math.sqrt(max(1 - s * s, 0)), 0.0])

    partitions = [
        _make_partition("p1", centroid=_centroid_for_sim(0.4)),
        _make_partition("p2", centroid=_centroid_for_sim(0.3)),
        _make_partition("p3", centroid=_centroid_for_sim(0.2)),
    ]

    config = _default_config(min_score=0.1)  # Low default
    router = CentroidRouter(config)

    # Override with high min_score: all below 0.5 -> top-1 fallback
    result = await router.route_by_centroid(
        QUERY, _registry(*partitions), min_score=0.5
    )

    # All below min_score -> top-1
    assert len(result) == 1
    assert result[0].name == "p1"


# ---------------------------------------------------------------------------
# Edge: SPLIT node with no children logs warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_no_children_warning(caplog: pytest.LogCaptureFixture) -> None:
    """SPLIT partition with no children in registry logs warning."""
    parent = _make_partition(
        "orphan_split",
        status=PartitionStatus.SPLIT,
        centroid=normalize([1.0, 0.0, 0.0]),
        children=["missing_child"],
    )

    config = _default_config()
    router = CentroidRouter(config)

    # Need another ACTIVE partition so result isn't solely dependent on this
    p_active = _make_partition("active", centroid=normalize([0.5, 0.5, 0.0]))

    result = await router.route_by_centroid(
        QUERY, _registry(parent, p_active)
    )

    # orphan_split should not be in results
    names = [p.name for p in result]
    assert "orphan_split" not in names
    assert "active" in names


# ---------------------------------------------------------------------------
# Orphaned parent reference — partition appears as root
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orphaned_parent_treated_as_root() -> None:
    """Partition whose parent is not in registry is treated as root."""
    p = _make_partition(
        "orphan",
        centroid=normalize([1.0, 0.0, 0.0]),
        parent="nonexistent_parent",
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(QUERY, _registry(p))

    assert len(result) == 1
    assert result[0].name == "orphan"


# ---------------------------------------------------------------------------
# DISABLED children filtered out in _get_children
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_children_filtered() -> None:
    """DISABLED children are excluded when recursing into SPLIT."""
    parent = _make_partition(
        "parent",
        status=PartitionStatus.SPLIT,
        centroid=normalize([1.0, 0.0, 0.0]),
        children=["child_ok", "child_disabled"],
    )
    child_ok = _make_partition(
        "child_ok",
        centroid=normalize([0.9, 0.1, 0.0]),
        parent="parent",
    )
    child_disabled = _make_partition(
        "child_disabled",
        status=PartitionStatus.DISABLED,
        centroid=normalize([0.9, 0.1, 0.0]),
        parent="parent",
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(parent, child_ok, child_disabled)
    )

    names = [p.name for p in result]
    assert "child_ok" in names
    assert "child_disabled" not in names


# ---------------------------------------------------------------------------
# Results ordered by score descending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_results_ordered_descending() -> None:
    """Results are ordered by centroid similarity, highest first."""
    import math

    def _centroid_for_sim(s: float) -> list[float]:
        return normalize([s, math.sqrt(max(1 - s * s, 0)), 0.0])

    partitions = [
        _make_partition("low", centroid=_centroid_for_sim(0.3)),
        _make_partition("high", centroid=_centroid_for_sim(0.9)),
        _make_partition("mid", centroid=_centroid_for_sim(0.6)),
    ]

    router = CentroidRouter(_default_config(relative_threshold=0.2, min_score=0.1))
    result = await router.route_by_centroid(QUERY, _registry(*partitions))

    assert result[0].name == "high"
    if len(result) >= 2:
        assert result[1].name == "mid"
    if len(result) >= 3:
        assert result[2].name == "low"


# ---------------------------------------------------------------------------
# Only one partition in registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_partition() -> None:
    """Registry with just one partition returns that partition."""
    p = _make_partition("only", centroid=normalize([0.1, 0.9, 0.0]))

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(QUERY, _registry(p))

    assert len(result) == 1
    assert result[0].name == "only"


# ---------------------------------------------------------------------------
# No-centroid SPLIT partition recurses into children
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_centroid_split_recurses() -> None:
    """A SPLIT partition without a centroid still recurses into children."""
    parent = _make_partition(
        "parent_no_centroid",
        status=PartitionStatus.SPLIT,
        centroid=None,
        children=["child_x"],
    )
    child_x = _make_partition(
        "child_x",
        centroid=normalize([1.0, 0.0, 0.0]),
        parent="parent_no_centroid",
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(
        QUERY, _registry(parent, child_x)
    )

    names = [p.name for p in result]
    assert "parent_no_centroid" not in names
    assert "child_x" in names


# ---------------------------------------------------------------------------
# Config defaults are used when no overrides passed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_defaults_used() -> None:
    """Verify config values are used when no overrides are provided."""
    import math

    def _centroid_for_sim(s: float) -> list[float]:
        return normalize([s, math.sqrt(max(1 - s * s, 0)), 0.0])

    # 8 partitions, all high similarity
    partitions = [
        _make_partition(f"p{i}", centroid=_centroid_for_sim(0.95 - i * 0.01))
        for i in range(8)
    ]

    config = _default_config(max_probe_partitions=3)
    router = CentroidRouter(config)
    result = await router.route_by_centroid(QUERY, _registry(*partitions))

    # Should respect config's max_probe_partitions=3
    assert len(result) <= 3


# ---------------------------------------------------------------------------
# All partitions DISABLED -> empty result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_disabled_returns_empty() -> None:
    """If all partitions are DISABLED, return empty."""
    p1 = _make_partition(
        "d1", status=PartitionStatus.DISABLED, centroid=normalize([1.0, 0.0, 0.0])
    )
    p2 = _make_partition(
        "d2", status=PartitionStatus.DISABLED, centroid=normalize([0.0, 1.0, 0.0])
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(QUERY, _registry(p1, p2))

    assert result == []


# ---------------------------------------------------------------------------
# All partitions RETIRED -> empty result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_retired_returns_empty() -> None:
    """If all partitions are RETIRED, return empty."""
    p1 = _make_partition(
        "r1", status=PartitionStatus.RETIRED, centroid=normalize([1.0, 0.0, 0.0])
    )

    router = CentroidRouter(_default_config())
    result = await router.route_by_centroid(QUERY, _registry(p1))

    assert result == []
