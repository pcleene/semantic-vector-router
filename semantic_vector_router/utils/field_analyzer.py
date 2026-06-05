"""Field analyzer for discovering filter-suitable fields.

Analyzes fields in MongoDB collections to determine which are good candidates
for pre-filtering in vector search. Used in SOURCE mode to automatically add
filter fields to the vector search index.

A good filter field has:
- Low-to-moderate cardinality (e.g., category, status, region — not unique IDs)
- High coverage (present in most documents)
- Non-array, non-nested type (string, number, boolean, date)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.models import SVRConfig
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)

# Fields that are never useful as filters
EXCLUDED_FIELDS = frozenset({
    "_id",
    "embedding",
    "embedding_text",
    "_svr_embedding_data",
})

# Maximum cardinality ratio (distinct values / total docs) for a filter field
# Above this, the field is too unique to be useful for filtering
MAX_CARDINALITY_RATIO = 0.3

# Minimum coverage (fraction of docs that have the field) to be useful
MIN_COVERAGE = 0.8

# Maximum distinct values to consider (beyond this, field is too high cardinality)
MAX_DISTINCT_VALUES = 1000


@dataclass
class FieldAnalysis:
    """Analysis result for a single field."""

    name: str
    distinct_count: int
    total_documents: int
    coverage: float  # Fraction of docs that have this field
    cardinality_ratio: float  # distinct_count / total_documents
    sample_values: list[Any] = field(default_factory=list)
    is_suitable: bool = False
    reason: str = ""

    @property
    def suitability_score(self) -> float:
        """Score from 0-1 indicating how suitable this field is for filtering.

        Higher is better. Considers cardinality, coverage, and distinct count.
        """
        if not self.is_suitable:
            return 0.0

        # Prefer lower cardinality
        cardinality_score = 1.0 - self.cardinality_ratio

        # Prefer higher coverage
        coverage_score = self.coverage

        # Prefer fields with a "useful" number of distinct values (2-100)
        if self.distinct_count < 2:
            count_score = 0.1
        elif self.distinct_count <= 20:
            count_score = 1.0
        elif self.distinct_count <= 100:
            count_score = 0.7
        else:
            count_score = 0.4

        return (cardinality_score * 0.4 + coverage_score * 0.3 + count_score * 0.3)


async def analyze_fields(
    backend: BaseBackend,
    config: SVRConfig,
    sample_size: int = 1000,
    exclude_fields: Optional[set[str]] = None,
) -> list[FieldAnalysis]:
    """Analyze collection fields for filter suitability.

    Samples documents from the source collection to discover fields and
    their cardinality characteristics. Returns a ranked list of fields
    suitable for use as filter fields in SOURCE mode vector search.

    Args:
        backend: Database backend.
        config: SVR configuration.
        sample_size: Number of documents to sample for analysis.
        exclude_fields: Additional field names to exclude.

    Returns:
        List of FieldAnalysis objects, sorted by suitability score descending.
    """
    source_collection = config.database.source_collection
    embedding_field = config.vector_search.embedding_field

    # Build exclusion set
    excluded = set(EXCLUDED_FIELDS)
    excluded.add(embedding_field)
    if config.embedding.computed_field:
        excluded.add(config.embedding.computed_field)
    if exclude_fields:
        excluded.update(exclude_fields)

    # Get total document count
    total_docs = await backend.count_documents(source_collection)
    if total_docs == 0:
        logger.warning("Collection is empty, cannot analyze fields")
        return []

    # Sample documents to discover fields and types
    # Field analysis requires MongoDB aggregation pipeline ($sample).
    # Non-MongoDB backends are not yet supported for field analysis.
    if not hasattr(backend, "db"):
        logger.warning(
            "Field analysis requires MongoDB backend (aggregation pipeline). "
            "Skipping for non-MongoDB backend."
        )
        return []
    collection = backend.db[source_collection]  # type: ignore[attr-defined]
    pipeline = [{"$sample": {"size": min(sample_size, total_docs)}}]
    cursor = await collection.aggregate(pipeline)
    samples = await cursor.to_list(length=sample_size)

    if not samples:
        return []

    # Count field occurrences across samples
    field_occurrences: dict[str, int] = {}
    for doc in samples:
        for key in doc:
            if key not in excluded:
                field_occurrences[key] = field_occurrences.get(key, 0) + 1

    # Analyze each field
    results: list[FieldAnalysis] = []
    for field_name, occurrence_count in field_occurrences.items():
        coverage = occurrence_count / len(samples)

        # Skip low coverage fields
        if coverage < MIN_COVERAGE:
            results.append(FieldAnalysis(
                name=field_name,
                distinct_count=0,
                total_documents=total_docs,
                coverage=coverage,
                cardinality_ratio=1.0,
                is_suitable=False,
                reason=f"Low coverage ({coverage:.0%})",
            ))
            continue

        # Check if field values are simple types (not arrays, objects, or embeddings)
        sample_values_raw = [doc.get(field_name) for doc in samples[:10] if field_name in doc]
        if sample_values_raw and isinstance(sample_values_raw[0], (list, dict)):
            results.append(FieldAnalysis(
                name=field_name,
                distinct_count=0,
                total_documents=total_docs,
                coverage=coverage,
                cardinality_ratio=1.0,
                is_suitable=False,
                reason="Complex type (array/object)",
            ))
            continue

        # Get distinct count
        try:
            distinct_values = await backend.get_distinct_values(field_name)
            distinct_count = len(distinct_values)
        except Exception as e:
            logger.warning(f"Failed to get distinct values for '{field_name}': {e}")
            continue

        cardinality_ratio = distinct_count / total_docs if total_docs > 0 else 1.0

        # Determine suitability
        is_suitable = True
        reason = "Good filter candidate"

        if distinct_count < 2:
            is_suitable = False
            reason = "Only 1 distinct value (no filtering benefit)"
        elif distinct_count > MAX_DISTINCT_VALUES:
            is_suitable = False
            reason = f"Too many distinct values ({distinct_count:,})"
        elif cardinality_ratio > MAX_CARDINALITY_RATIO:
            is_suitable = False
            reason = f"High cardinality ratio ({cardinality_ratio:.2%})"

        analysis = FieldAnalysis(
            name=field_name,
            distinct_count=distinct_count,
            total_documents=total_docs,
            coverage=coverage,
            cardinality_ratio=cardinality_ratio,
            sample_values=distinct_values[:10] if distinct_count <= 10 else distinct_values[:5],
            is_suitable=is_suitable,
            reason=reason,
        )
        results.append(analysis)

    # Sort by suitability score descending
    results.sort(key=lambda x: x.suitability_score, reverse=True)

    suitable_count = sum(1 for r in results if r.is_suitable)
    logger.info(
        f"Analyzed {len(results)} fields, {suitable_count} suitable for filtering"
    )

    return results


def get_recommended_filter_fields(
    analyses: list[FieldAnalysis],
    max_fields: int = 5,
) -> list[str]:
    """Get recommended filter field names from analysis results.

    Args:
        analyses: Field analysis results from analyze_fields().
        max_fields: Maximum number of fields to recommend.

    Returns:
        List of field names recommended as filter fields.
    """
    return [
        a.name
        for a in analyses
        if a.is_suitable
    ][:max_fields]
