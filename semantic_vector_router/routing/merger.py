"""Result merging and score normalization."""

from collections import defaultdict
from typing import Any

from semantic_vector_router.models import SearchHit
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_scores(
    hits: list[SearchHit],
    method: str = "partition_minmax",
) -> list[SearchHit]:
    """Normalize scores across hits.

    Args:
        hits: List of search hits with scores.
        method: Normalization method:
            - "partition_minmax": Normalize within each partition to 0-1
            - "global_minmax": Normalize across all hits to 0-1
            - "none": No normalization

    Returns:
        List of hits with normalized scores.
    """
    if not hits or method == "none":
        return hits

    if method == "partition_minmax":
        return _normalize_by_partition(hits)
    elif method == "global_minmax":
        return _normalize_global(hits)
    else:
        logger.warning(f"Unknown normalization method: {method}, skipping")
        return hits


def _normalize_by_partition(hits: list[SearchHit]) -> list[SearchHit]:
    """Normalize scores within each partition to 0-1 range.

    Args:
        hits: List of search hits.

    Returns:
        Hits with normalized scores.
    """
    # Group by partition
    by_partition: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        by_partition[hit.partition].append(hit)

    # Normalize within each partition
    for partition, partition_hits in by_partition.items():
        scores = [h.score for h in partition_hits]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range > 0:
            for hit in partition_hits:
                hit.score = (hit.score - min_score) / score_range
        else:
            # All scores are the same, set to 1.0
            for hit in partition_hits:
                hit.score = 1.0

    return hits


def _normalize_global(hits: list[SearchHit]) -> list[SearchHit]:
    """Normalize all scores to 0-1 range.

    Args:
        hits: List of search hits.

    Returns:
        Hits with normalized scores.
    """
    scores = [h.score for h in hits]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range > 0:
        for hit in hits:
            hit.score = (hit.score - min_score) / score_range
    else:
        # All scores are the same, set to 1.0
        for hit in hits:
            hit.score = 1.0

    return hits


class ResultMerger:
    """Merges and ranks results from multiple partitions.

    Handles:
    - Score normalization across partitions
    - Result deduplication
    - Final ranking and limiting
    """

    def __init__(
        self,
        normalize_method: str = "partition_minmax",
        deduplicate: bool = True,
        dedupe_field: str = "_id",
    ):
        """Initialize the merger.

        Args:
            normalize_method: Score normalization method.
            deduplicate: Whether to remove duplicate documents.
            dedupe_field: Field to use for deduplication.
        """
        self.normalize_method = normalize_method
        self.deduplicate = deduplicate
        self.dedupe_field = dedupe_field

    def merge(
        self,
        results: list[dict[str, Any]],
        limit: int,
        score_field: str = "_svr_score",
        partition_field: str = "_svr_partition",
    ) -> list[SearchHit]:
        """Merge results from multiple partitions.

        Args:
            results: Raw results from backend searches.
            limit: Maximum results to return.
            score_field: Field containing the score.
            partition_field: Field containing the partition name.

        Returns:
            Merged, ranked, and limited list of SearchHit objects.
        """
        if not results:
            return []

        # Convert to SearchHit objects
        hits = []
        for doc in results:
            hit = SearchHit(
                id=str(doc.get("_id", "")),
                score=float(doc.get(score_field, 0.0)),
                partition=str(doc.get(partition_field, "unknown")),
                document=self._clean_document(doc, score_field, partition_field),
            )
            hits.append(hit)

        # Deduplicate if enabled
        if self.deduplicate:
            hits = self._deduplicate(hits)

        # Normalize scores
        hits = normalize_scores(hits, self.normalize_method)

        # Sort by score (highest first)
        hits.sort(key=lambda h: h.score, reverse=True)

        # Limit results
        return hits[:limit]

    def merge_with_rerank_scores(
        self,
        hits: list[SearchHit],
        rerank_scores: list[float],
        limit: int,
    ) -> list[SearchHit]:
        """Merge hits with reranking scores.

        Args:
            hits: List of search hits.
            rerank_scores: Reranking scores in same order as hits.
            limit: Maximum results to return.

        Returns:
            Reranked and limited list of SearchHit objects.
        """
        if len(hits) != len(rerank_scores):
            raise ValueError(
                f"Hits count ({len(hits)}) doesn't match "
                f"rerank scores count ({len(rerank_scores)})"
            )

        # Apply rerank scores
        for hit, score in zip(hits, rerank_scores):
            hit.rerank_score = score

        # Sort by rerank score (highest first)
        hits.sort(key=lambda h: h.rerank_score or 0, reverse=True)

        # Limit results
        return hits[:limit]

    def _deduplicate(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Remove duplicate documents, keeping highest-scoring version.

        Args:
            hits: List of search hits.

        Returns:
            Deduplicated list of hits.
        """
        seen: dict[str, SearchHit] = {}

        for hit in hits:
            doc_id = hit.document.get(self.dedupe_field, hit.id)
            key = str(doc_id)

            if key not in seen or hit.score > seen[key].score:
                seen[key] = hit

        return list(seen.values())

    def _clean_document(
        self,
        doc: dict[str, Any],
        score_field: str,
        partition_field: str,
    ) -> dict[str, Any]:
        """Remove internal fields from document.

        Args:
            doc: Raw document.
            score_field: Score field to remove.
            partition_field: Partition field to remove.

        Returns:
            Cleaned document.
        """
        cleaned = dict(doc)
        # Remove internal SVR fields
        for field in [score_field, partition_field]:
            cleaned.pop(field, None)
        return cleaned
