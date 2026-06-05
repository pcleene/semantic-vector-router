"""Vector math utilities for centroid routing.

Pure Python implementation — no numpy dependency.
All functions operate on list[float] vectors.
"""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    If both vectors are pre-normalized to unit length, this reduces
    to a dot product. Handles zero vectors gracefully.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 for zero vectors.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm).

    Args:
        vector: Input vector.

    Returns:
        Unit-length vector. Returns zero vector if input is zero.
    """
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0.0:
        return [0.0] * len(vector)
    return [x / norm for x in vector]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute element-wise mean of a list of vectors.

    Args:
        vectors: List of vectors (all same dimensionality).

    Returns:
        Mean vector. Returns empty list if input is empty.

    Raises:
        ValueError: If vectors have inconsistent dimensions.
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError(
            f"Inconsistent vector dimensions: expected {dim}, "
            f"got {set(len(v) for v in vectors)}"
        )
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(dim)]
