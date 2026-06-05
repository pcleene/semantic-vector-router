"""Tests for field analyzer utility."""

import pytest

from semantic_vector_router.utils.field_analyzer import (
    FieldAnalysis,
    get_recommended_filter_fields,
)


class TestFieldAnalysis:
    """Tests for FieldAnalysis dataclass."""

    def test_suitable_field_score(self):
        """Test suitability score for a good filter field."""
        analysis = FieldAnalysis(
            name="category",
            distinct_count=15,
            total_documents=100000,
            coverage=0.99,
            cardinality_ratio=0.00015,
            is_suitable=True,
            reason="Good filter candidate",
        )

        score = analysis.suitability_score
        assert score > 0.5
        assert score <= 1.0

    def test_unsuitable_field_score(self):
        """Test suitability score for an unsuitable field."""
        analysis = FieldAnalysis(
            name="unique_id",
            distinct_count=100000,
            total_documents=100000,
            coverage=1.0,
            cardinality_ratio=1.0,
            is_suitable=False,
            reason="Too many distinct values",
        )

        assert analysis.suitability_score == 0.0

    def test_low_cardinality_scores_higher(self):
        """Test that lower cardinality fields score higher."""
        low_card = FieldAnalysis(
            name="status",
            distinct_count=5,
            total_documents=100000,
            coverage=1.0,
            cardinality_ratio=0.00005,
            is_suitable=True,
        )

        medium_card = FieldAnalysis(
            name="region",
            distinct_count=200,
            total_documents=100000,
            coverage=0.95,
            cardinality_ratio=0.002,
            is_suitable=True,
        )

        assert low_card.suitability_score > medium_card.suitability_score


class TestGetRecommendedFilterFields:
    """Tests for get_recommended_filter_fields."""

    def test_returns_suitable_fields(self):
        """Test that only suitable fields are returned."""
        analyses = [
            FieldAnalysis(
                name="category",
                distinct_count=10,
                total_documents=1000,
                coverage=1.0,
                cardinality_ratio=0.01,
                is_suitable=True,
            ),
            FieldAnalysis(
                name="unique_id",
                distinct_count=1000,
                total_documents=1000,
                coverage=1.0,
                cardinality_ratio=1.0,
                is_suitable=False,
            ),
            FieldAnalysis(
                name="status",
                distinct_count=3,
                total_documents=1000,
                coverage=0.99,
                cardinality_ratio=0.003,
                is_suitable=True,
            ),
        ]

        result = get_recommended_filter_fields(analyses)
        assert result == ["category", "status"]

    def test_respects_max_fields(self):
        """Test that max_fields limit is respected."""
        analyses = [
            FieldAnalysis(
                name=f"field_{i}",
                distinct_count=10,
                total_documents=1000,
                coverage=1.0,
                cardinality_ratio=0.01,
                is_suitable=True,
            )
            for i in range(10)
        ]

        result = get_recommended_filter_fields(analyses, max_fields=3)
        assert len(result) == 3

    def test_empty_analyses(self):
        """Test with no analyses."""
        result = get_recommended_filter_fields([])
        assert result == []

    def test_no_suitable_fields(self):
        """Test when no fields are suitable."""
        analyses = [
            FieldAnalysis(
                name="bad_field",
                distinct_count=0,
                total_documents=1000,
                coverage=0.1,
                cardinality_ratio=0.0,
                is_suitable=False,
            ),
        ]

        result = get_recommended_filter_fields(analyses)
        assert result == []
