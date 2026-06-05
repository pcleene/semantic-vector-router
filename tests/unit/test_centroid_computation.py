"""Unit tests for the vector_math module.

Tests cover cosine_similarity, normalize, and mean_vector functions
with all edge cases: identical, orthogonal, opposite, zero vectors,
inconsistent dimensions, and end-to-end pipeline.
"""

import math

import pytest

from semantic_vector_router.utils.vector_math import (
    cosine_similarity,
    mean_vector,
    normalize,
)


# ---------------------------------------------------------------------------
# cosine_similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_identical_unit_vectors(self) -> None:
        """Identical unit vectors have similarity 1.0."""
        v = normalize([0.5, 0.5, 0.5])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors have similarity -1.0."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_first(self) -> None:
        """Zero vector as first argument returns 0.0."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_second(self) -> None:
        """Zero vector as second argument returns 0.0."""
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self) -> None:
        """Both zero vectors returns 0.0."""
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_known_values(self) -> None:
        """Check against manually computed cosine similarity.

        a = [1, 2, 3], b = [4, 5, 6]
        dot = 4 + 10 + 18 = 32
        |a| = sqrt(14), |b| = sqrt(77)
        cos = 32 / sqrt(14 * 77) = 32 / sqrt(1078) ~ 0.97463
        """
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        expected = 32.0 / math.sqrt(14.0 * 77.0)
        assert cosine_similarity(a, b) == pytest.approx(expected, rel=1e-9)

    def test_negative_components(self) -> None:
        """Cosine similarity works correctly with negative components."""
        a = [1.0, -1.0]
        b = [-1.0, 1.0]
        # dot = -1 + -1 = -2; |a| = sqrt(2), |b| = sqrt(2)
        # cos = -2/2 = -1.0
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_single_dimension(self) -> None:
        """Single-dimension vectors."""
        assert cosine_similarity([3.0], [5.0]) == pytest.approx(1.0)
        assert cosine_similarity([3.0], [-5.0]) == pytest.approx(-1.0)

    def test_high_dimensional(self) -> None:
        """Cosine similarity on high-dimensional vectors."""
        dim = 1024
        a = [1.0 / math.sqrt(dim)] * dim
        b = [1.0 / math.sqrt(dim)] * dim
        assert cosine_similarity(a, b) == pytest.approx(1.0, rel=1e-6)

    def test_symmetry(self) -> None:
        """cosine_similarity(a, b) == cosine_similarity(b, a)."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


# ---------------------------------------------------------------------------
# normalize tests
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for normalize."""

    def test_unit_vector_stays_unit(self) -> None:
        """A unit vector remains unchanged after normalization."""
        v = [1.0, 0.0, 0.0]
        result = normalize(v)
        assert result == pytest.approx([1.0, 0.0, 0.0])

    def test_zero_vector_returns_zero(self) -> None:
        """Zero vector returns zero vector of same dimension."""
        v = [0.0, 0.0, 0.0]
        result = normalize(v)
        assert result == [0.0, 0.0, 0.0]

    def test_known_normalization(self) -> None:
        """Normalize [3, 4] -> [0.6, 0.8]."""
        v = [3.0, 4.0]
        result = normalize(v)
        assert result == pytest.approx([0.6, 0.8])

    def test_result_has_unit_length(self) -> None:
        """Normalized vector has L2 norm = 1."""
        v = [1.0, 2.0, 3.0, 4.0]
        result = normalize(v)
        norm = sum(x * x for x in result) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_preserves_direction(self) -> None:
        """Normalized vector points in the same direction."""
        v = [2.0, 0.0, 0.0]
        result = normalize(v)
        assert result == pytest.approx([1.0, 0.0, 0.0])

    def test_negative_components(self) -> None:
        """Normalize works with negative components."""
        v = [-3.0, 4.0]
        result = normalize(v)
        assert result == pytest.approx([-0.6, 0.8])

    def test_single_dimension(self) -> None:
        """Single-dimension normalization."""
        assert normalize([5.0]) == pytest.approx([1.0])
        assert normalize([-5.0]) == pytest.approx([-1.0])

    def test_idempotent(self) -> None:
        """Normalizing an already-normalized vector is idempotent."""
        v = [1.0, 2.0, 3.0]
        once = normalize(v)
        twice = normalize(once)
        assert once == pytest.approx(twice)

    def test_zero_length_list(self) -> None:
        """Empty vector normalizes to empty vector."""
        assert normalize([]) == []


# ---------------------------------------------------------------------------
# mean_vector tests
# ---------------------------------------------------------------------------


class TestMeanVector:
    """Tests for mean_vector."""

    def test_single_vector(self) -> None:
        """Mean of one vector is itself."""
        v = [1.0, 2.0, 3.0]
        assert mean_vector([v]) == pytest.approx(v)

    def test_two_vectors(self) -> None:
        """Mean of two vectors is element-wise average."""
        a = [1.0, 2.0, 3.0]
        b = [3.0, 4.0, 5.0]
        assert mean_vector([a, b]) == pytest.approx([2.0, 3.0, 4.0])

    def test_multiple_vectors(self) -> None:
        """Mean of three vectors."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        c = [0.0, 0.0]
        assert mean_vector([a, b, c]) == pytest.approx(
            [1.0 / 3.0, 1.0 / 3.0]
        )

    def test_empty_list(self) -> None:
        """Mean of empty list is empty list."""
        assert mean_vector([]) == []

    def test_inconsistent_dimensions_raises(self) -> None:
        """Inconsistent vector dimensions raises ValueError."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="Inconsistent vector dimensions"):
            mean_vector([a, b])

    def test_identical_vectors(self) -> None:
        """Mean of identical vectors is that vector."""
        v = [5.0, -3.0, 1.0]
        assert mean_vector([v, v, v]) == pytest.approx(v)

    def test_zero_vectors(self) -> None:
        """Mean of zero vectors is zero vector."""
        z = [0.0, 0.0, 0.0]
        assert mean_vector([z, z]) == pytest.approx(z)

    def test_negative_and_positive(self) -> None:
        """Mean of opposite vectors is zero vector."""
        a = [1.0, 1.0]
        b = [-1.0, -1.0]
        assert mean_vector([a, b]) == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# End-to-end: mean_vector -> normalize -> cosine_similarity
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration-style tests combining vector_math functions."""

    def test_centroid_computation_pipeline(self) -> None:
        """Compute centroid from embeddings, then check query similarity.

        Vectors: [1,0,0], [0.9,0.1,0], [0.8,0.2,0]
        Mean: [0.9, 0.1, 0.0]
        Normalized mean: should be close to [1,0,0] direction
        Query: [1,0,0]
        Similarity should be high.
        """
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ]
        centroid = normalize(mean_vector(embeddings))
        query = [1.0, 0.0, 0.0]
        sim = cosine_similarity(query, centroid)
        assert sim > 0.95

    def test_orthogonal_clusters(self) -> None:
        """Two clusters in orthogonal directions.

        Cluster A: embeddings near [1,0,0]
        Cluster B: embeddings near [0,1,0]
        Query near A should be more similar to centroid A than B.
        """
        cluster_a = [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0]]
        cluster_b = [[0.0, 1.0, 0.0], [0.05, 0.95, 0.0]]

        centroid_a = normalize(mean_vector(cluster_a))
        centroid_b = normalize(mean_vector(cluster_b))

        query = [0.9, 0.1, 0.0]  # Close to cluster A

        sim_a = cosine_similarity(query, centroid_a)
        sim_b = cosine_similarity(query, centroid_b)

        assert sim_a > sim_b
        assert sim_a > 0.9

    def test_centroid_of_single_vector(self) -> None:
        """Centroid of a single vector, normalized, equals that vector normalized."""
        v = [3.0, 4.0, 0.0]
        centroid = normalize(mean_vector([v]))
        expected = normalize(v)
        assert centroid == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Pre-normalized centroids: cosine_similarity reduces to dot product
# ---------------------------------------------------------------------------


class TestPreNormalized:
    """When both vectors are unit length, cosine = dot product."""

    def test_prenormalized_equals_dot(self) -> None:
        """For unit vectors, cosine_similarity equals manual dot product."""
        a = normalize([1.0, 2.0, 3.0])
        b = normalize([4.0, 5.0, 6.0])

        cos_sim = cosine_similarity(a, b)
        dot_product = sum(x * y for x, y in zip(a, b))

        assert cos_sim == pytest.approx(dot_product, rel=1e-9)

    def test_prenormalized_orthogonal(self) -> None:
        """Pre-normalized orthogonal vectors: dot = cos = 0."""
        a = normalize([1.0, 0.0])
        b = normalize([0.0, 1.0])

        cos_sim = cosine_similarity(a, b)
        dot_product = sum(x * y for x, y in zip(a, b))

        assert cos_sim == pytest.approx(0.0)
        assert dot_product == pytest.approx(0.0)

    def test_prenormalized_high_dim(self) -> None:
        """Pre-normalized high-dimensional vectors."""
        dim = 512
        a = normalize([float(i) for i in range(dim)])
        b = normalize([float(dim - i) for i in range(dim)])

        cos_sim = cosine_similarity(a, b)
        dot_product = sum(x * y for x, y in zip(a, b))

        assert cos_sim == pytest.approx(dot_product, rel=1e-6)
