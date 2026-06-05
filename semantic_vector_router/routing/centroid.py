"""Hierarchical centroid routing for query-partition assignment.

Routes queries to the most relevant partitions by walking the partition
tree top-down, pruning branches whose centroid embedding is dissimilar
to the query embedding. This reduces fan-out from O(N) to O(log N).
"""

import time
from typing import Any, Optional

from semantic_vector_router.models import (
    CentroidRoutingConfig,
    PartitionInfo,
    PartitionStatus,
)
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.vector_math import cosine_similarity

logger = get_logger(__name__)


class CentroidRouter:
    """Routes queries to relevant partitions using centroid embeddings.

    Walks the partition tree top-down:
    1. Score all siblings at each level against query embedding
    2. Compute dynamic threshold: max_score * relative_threshold
    3. Keep siblings above max(dynamic_threshold, min_score)
    4. For SPLIT nodes: recurse into children
    5. For ACTIVE leaves: add to results
    6. Cap total results at max_probe_partitions (highest scores win)

    Thread-safe and stateless — all state comes from the registry
    passed to route_by_centroid().
    """

    def __init__(self, config: CentroidRoutingConfig) -> None:
        """Initialize the centroid router.

        Args:
            config: Centroid routing configuration.
        """
        self._config = config

    async def route_by_centroid(
        self,
        query_embedding: list[float],
        registry: dict[str, PartitionInfo],
        max_partitions: Optional[int] = None,
        relative_threshold: Optional[float] = None,
        min_score: Optional[float] = None,
    ) -> list[PartitionInfo]:
        """Walk partition tree top-down, pruning by dynamic centroid similarity.

        Args:
            query_embedding: The query's embedding vector.
            registry: Current partition registry (name -> PartitionInfo).
            max_partitions: Override max_probe_partitions from config.
            relative_threshold: Override relative_threshold from config.
            min_score: Override min_score from config.

        Returns:
            List of leaf PartitionInfo objects to search, ordered by
            centroid similarity (highest first). Never empty — returns
            top-1 if all centroids score below min_score.
        """
        effective_max = max_partitions or self._config.max_probe_partitions
        effective_threshold = (
            relative_threshold
            if relative_threshold is not None
            else self._config.relative_threshold
        )
        effective_min_score = (
            min_score if min_score is not None else self._config.min_score
        )

        start_time = time.perf_counter()

        # Find root partitions (no parent, or parent not in registry)
        roots = self._get_root_partitions(registry)

        if not roots:
            logger.warning("No root partitions found in registry")
            return []

        # Walk tree and collect scored leaves
        scored_leaves: list[tuple[float, PartitionInfo]] = []
        self._walk_tree(
            query_embedding=query_embedding,
            partitions=roots,
            registry=registry,
            relative_threshold=effective_threshold,
            min_score=effective_min_score,
            scored_leaves=scored_leaves,
        )

        # Sort by score descending
        scored_leaves.sort(key=lambda x: x[0], reverse=True)

        # If all scored below min_score, still return top-1 (never empty)
        if scored_leaves and all(
            score < effective_min_score for score, _ in scored_leaves
        ):
            result = [scored_leaves[0][1]]
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Centroid routing: all below min_score, returning top-1 "
                f"({scored_leaves[0][1].name}, score={scored_leaves[0][0]:.4f}) "
                f"in {elapsed_ms:.2f}ms"
            )
            return result

        # Cap at max_partitions
        result = [p for _, p in scored_leaves[:effective_max]]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Centroid routing: selected {len(result)} partitions "
            f"from {len(registry)} total in {elapsed_ms:.2f}ms"
        )

        return result

    def _get_root_partitions(
        self, registry: dict[str, PartitionInfo]
    ) -> list[PartitionInfo]:
        """Find root-level partitions (no parent in registry).

        A partition is a root if:
        - parent_partition is None, OR
        - parent_partition is not in the registry (orphaned reference)

        Excludes DISABLED and RETIRED partitions (RETIRED without children
        in registry are skipped).
        """
        roots: list[PartitionInfo] = []
        for p in registry.values():
            if p.status == PartitionStatus.DISABLED:
                continue
            if p.status == PartitionStatus.RETIRED:
                continue
            if p.parent_partition is None or p.parent_partition not in registry:
                roots.append(p)
        return roots

    def _walk_tree(
        self,
        query_embedding: list[float],
        partitions: list[PartitionInfo],
        registry: dict[str, PartitionInfo],
        relative_threshold: float,
        min_score: float,
        scored_leaves: list[tuple[float, PartitionInfo]],
    ) -> None:
        """Recursively walk tree level, scoring and pruning.

        At each level:
        1. Score all partitions at this level
        2. Apply dynamic threshold pruning
        3. Recurse into SPLIT children
        4. Collect ACTIVE leaves
        """
        if not partitions:
            return

        # Score all partitions at this level
        scored: list[tuple[float, PartitionInfo]] = []
        no_centroid: list[PartitionInfo] = []

        for p in partitions:
            if p.centroid is None:
                # No centroid — include it (safer to search extra than miss)
                no_centroid.append(p)
            else:
                score = cosine_similarity(query_embedding, p.centroid)
                scored.append((score, p))

        # Apply dynamic threshold pruning
        if scored:
            max_score = max(s for s, _ in scored)
            dynamic_threshold = max_score * relative_threshold
            effective_threshold = max(dynamic_threshold, min_score)

            surviving: list[tuple[float, PartitionInfo]] = [
                (s, p) for s, p in scored if s >= effective_threshold
            ]

            # If nothing survives the threshold (all below min_score),
            # keep the top scorer at minimum
            if not surviving and scored:
                surviving = [max(scored, key=lambda x: x[0])]
        else:
            surviving = []

        # Process surviving partitions + those without centroids
        for score, p in surviving:
            if p.status == PartitionStatus.SPLIT:
                # Recurse into children
                children = self._get_children(p, registry)
                if children:
                    self._walk_tree(
                        query_embedding=query_embedding,
                        partitions=children,
                        registry=registry,
                        relative_threshold=relative_threshold,
                        min_score=min_score,
                        scored_leaves=scored_leaves,
                    )
                else:
                    # SPLIT with no children — shouldn't happen, skip
                    logger.warning(
                        f"SPLIT partition {p.name} has no children in registry"
                    )
            elif p.status == PartitionStatus.ACTIVE:
                scored_leaves.append((score, p))
            # Skip SPLITTING, MIGRATING, PENDING_SPLIT — not searchable

        # Include partitions without centroids as leaves (if ACTIVE)
        for p in no_centroid:
            if p.status == PartitionStatus.SPLIT:
                children = self._get_children(p, registry)
                if children:
                    self._walk_tree(
                        query_embedding=query_embedding,
                        partitions=children,
                        registry=registry,
                        relative_threshold=relative_threshold,
                        min_score=min_score,
                        scored_leaves=scored_leaves,
                    )
            elif p.status == PartitionStatus.ACTIVE:
                # Give max possible score so it's included (never exclude unknowns)
                scored_leaves.append((1.0, p))

    def _get_children(
        self, parent: PartitionInfo, registry: dict[str, PartitionInfo]
    ) -> list[PartitionInfo]:
        """Get child partitions from registry."""
        children: list[PartitionInfo] = []
        for child_name in parent.child_partitions:
            child = registry.get(child_name)
            if child and child.status != PartitionStatus.DISABLED:
                children.append(child)
        return children


async def compute_partition_centroid(
    collection: Any,
    embedding_field: str,
    partition_filter: Optional[dict[str, Any]] = None,
    sample_size: int = 500,
) -> Optional[list[float]]:
    """Compute centroid embedding for a partition by sampling stored vectors.

    Reads already-stored embedding vectors — zero API calls. Handles both
    array and BinData vector storage formats.

    Args:
        collection: MongoDB AsyncCollection to sample from.
        embedding_field: Field path containing embedding vectors.
        partition_filter: Optional filter to scope to partition documents.
        sample_size: Number of documents to sample.

    Returns:
        Normalized centroid vector, or None if no vectors found.
    """
    from semantic_vector_router.backends.mongodb import bindata_to_vector
    from semantic_vector_router.utils.vector_math import mean_vector, normalize

    match_filter: dict[str, Any] = {
        embedding_field: {"$exists": True, "$ne": None}
    }
    if partition_filter:
        match_filter.update(partition_filter)

    pipeline: list[dict[str, Any]] = [
        {"$match": match_filter},
        {"$sample": {"size": sample_size}},
        {"$project": {embedding_field: 1}},
    ]

    cursor = await collection.aggregate(pipeline)
    docs = await cursor.to_list(length=None)

    vectors: list[list[float]] = []
    for doc in docs:
        vec = doc.get(embedding_field)
        if vec is None:
            continue
        if isinstance(vec, list) and vec:
            vectors.append([float(x) for x in vec])
        else:
            try:
                converted = bindata_to_vector(vec)
                if converted:
                    vectors.append([float(x) for x in converted])
            except Exception:
                continue

    if not vectors:
        return None

    return normalize(mean_vector(vectors))
