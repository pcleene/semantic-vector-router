"""Tests for vector conversion functions in the MongoDB backend.

Covers vector_to_bindata, bindata_to_vector, and query_vector_for_search
utilities including Phase 1.5 regression tests for correct query vector
format selection.
"""

import pytest
from bson.binary import Binary

from semantic_vector_router.backends.mongodb import (
    bindata_to_vector,
    query_vector_for_search,
    vector_to_bindata,
)
from semantic_vector_router.models import VectorStorageFormat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def float_vector():
    """A short float32-range vector for conversion tests."""
    return [0.1, 0.2, 0.3, -0.4, 0.5]


@pytest.fixture
def int8_vector():
    """A valid int8 vector (values in [-128, 127])."""
    return [-128, -1, 0, 1, 127]


@pytest.fixture
def packed_bit_vector():
    """A valid packed-bit vector (values are 0 or 1, length multiple of 8)."""
    return [0, 1, 1, 0, 1, 0, 0, 1]


# ---------------------------------------------------------------------------
# vector_to_bindata
# ---------------------------------------------------------------------------

class TestVectorToBindata:
    """Tests for vector_to_bindata conversion."""

    def test_vector_to_bindata_array_passthrough(self, float_vector):
        """ARRAY format returns the original list unchanged."""
        result = vector_to_bindata(float_vector, VectorStorageFormat.ARRAY)

        assert result is float_vector
        assert isinstance(result, list)

    def test_vector_to_bindata_float32(self, float_vector):
        """BINDATA_FLOAT32 format returns a Binary object."""
        result = vector_to_bindata(float_vector, VectorStorageFormat.BINDATA_FLOAT32)

        assert isinstance(result, Binary)

    def test_vector_to_bindata_int8(self, int8_vector):
        """BINDATA_INT8 format returns a Binary object with int8 dtype."""
        result = vector_to_bindata(int8_vector, VectorStorageFormat.BINDATA_INT8)

        assert isinstance(result, Binary)

    def test_vector_to_bindata_packed_bit(self, packed_bit_vector):
        """BINDATA_PACKED_BIT format returns a Binary object."""
        result = vector_to_bindata(packed_bit_vector, VectorStorageFormat.BINDATA_PACKED_BIT)

        assert isinstance(result, Binary)

    def test_int8_range_validation(self):
        """Values outside [-128, 127] raise ValueError."""
        out_of_range_vectors = [
            [0, 128],       # above upper bound
            [-129, 0],      # below lower bound
            [0, 0, 256],    # far above upper bound
        ]
        for vec in out_of_range_vectors:
            with pytest.raises(ValueError, match="INT8 vectors must have values in range"):
                vector_to_bindata(vec, VectorStorageFormat.BINDATA_INT8)

    def test_packed_bit_validation(self):
        """Values other than 0 or 1 raise ValueError."""
        invalid_vectors = [
            [0, 1, 2, 0, 0, 0, 0, 0],     # contains 2
            [0, 1, -1, 0, 0, 0, 0, 0],     # contains -1
            [0, 1, 0, 0, 0, 0, 0, 255],    # contains 255
        ]
        for vec in invalid_vectors:
            with pytest.raises(ValueError, match="PACKED_BIT vectors must have values 0 or 1"):
                vector_to_bindata(vec, VectorStorageFormat.BINDATA_PACKED_BIT)


# ---------------------------------------------------------------------------
# bindata_to_vector
# ---------------------------------------------------------------------------

class TestBindataToVector:
    """Tests for bindata_to_vector conversion."""

    def test_bindata_to_vector_float32_roundtrip(self, float_vector):
        """Float32 BinData round-trips back to an equivalent list."""
        bindata = vector_to_bindata(float_vector, VectorStorageFormat.BINDATA_FLOAT32)
        result = bindata_to_vector(bindata)

        assert isinstance(result, list)
        assert len(result) == len(float_vector)
        for original, restored in zip(float_vector, result):
            assert abs(original - restored) < 1e-6

    def test_bindata_to_vector_int8_roundtrip(self, int8_vector):
        """Int8 BinData round-trips back to an equivalent list."""
        bindata = vector_to_bindata(int8_vector, VectorStorageFormat.BINDATA_INT8)
        result = bindata_to_vector(bindata)

        assert isinstance(result, list)
        assert len(result) == len(int8_vector)
        assert result == int8_vector

    def test_bindata_to_vector_list_passthrough(self, float_vector):
        """A plain list (non-Binary) is returned as-is."""
        result = bindata_to_vector(float_vector)

        assert result is float_vector
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# query_vector_for_search — Phase 1.5 regressions
# ---------------------------------------------------------------------------

class TestQueryVectorForSearch:
    """Regression tests for query_vector_for_search (Phase 1.5).

    The key invariant: ARRAY and BINDATA_FLOAT32 must return a plain list so
    the Atlas $vectorSearch driver can handle type conversion. Pre-quantized
    formats (INT8, PACKED_BIT) must return BinData because MongoDB expects the
    query to match the stored format exactly.
    """

    def test_query_vector_for_search_array_returns_list(self, float_vector):
        """ARRAY storage: query vector must be a plain list."""
        result = query_vector_for_search(float_vector, VectorStorageFormat.ARRAY)

        assert isinstance(result, list)
        assert result is float_vector

    def test_query_vector_for_search_float32_returns_list(self, float_vector):
        """BINDATA_FLOAT32 storage: query vector must be a plain list."""
        result = query_vector_for_search(float_vector, VectorStorageFormat.BINDATA_FLOAT32)

        assert isinstance(result, list)
        assert result is float_vector

    def test_query_vector_for_search_int8_returns_bindata(self, int8_vector):
        """BINDATA_INT8 storage: query vector must be BinData."""
        result = query_vector_for_search(int8_vector, VectorStorageFormat.BINDATA_INT8)

        assert isinstance(result, Binary)

    def test_query_vector_for_search_packed_bit_returns_bindata(self, packed_bit_vector):
        """BINDATA_PACKED_BIT storage: query vector must be BinData."""
        result = query_vector_for_search(packed_bit_vector, VectorStorageFormat.BINDATA_PACKED_BIT)

        assert isinstance(result, Binary)
