"""Vector conversion utilities for MongoDB BinData storage.

Handles conversion between Python lists and MongoDB Binary vector types
for optimized vector storage formats (float32, int8, packed_bit).
"""

from typing import Union

from bson.binary import Binary, BinaryVectorDtype

from semantic_vector_router.models import VectorStorageFormat


def vector_to_bindata(
    vector: list[Union[float, int]],
    storage_format: VectorStorageFormat,
) -> Union[Binary, list]:
    """Convert a vector to the appropriate MongoDB storage format.

    Args:
        vector: List of floats or ints representing the embedding.
        storage_format: Target storage format.

    Returns:
        Binary object for BinData formats, or original list for array format.

    Raises:
        ValueError: If storage format and vector type are incompatible.
    """
    if storage_format == VectorStorageFormat.ARRAY:
        return vector  # Store as regular array<double>

    elif storage_format == VectorStorageFormat.BINDATA_FLOAT32:
        # Convert to BinData float32 (3x storage savings vs array<double>)
        return Binary.from_vector(vector, BinaryVectorDtype.FLOAT32)

    elif storage_format == VectorStorageFormat.BINDATA_INT8:
        # Store pre-quantized int8 vectors
        # Validate values are in int8 range
        if not all(-128 <= v <= 127 for v in vector):
            raise ValueError("INT8 vectors must have values in range [-128, 127]")
        return Binary.from_vector(vector, BinaryVectorDtype.INT8)

    elif storage_format == VectorStorageFormat.BINDATA_PACKED_BIT:
        # Store pre-quantized binary vectors
        # Validate values are 0 or 1
        if not all(v in (0, 1) for v in vector):
            raise ValueError("PACKED_BIT vectors must have values 0 or 1")
        return Binary.from_vector(vector, BinaryVectorDtype.PACKED_BIT)

    else:
        raise ValueError(f"Unknown storage format: {storage_format}")


def bindata_to_vector(data: Union[Binary, list]) -> list[Union[float, int]]:
    """Convert MongoDB stored data back to a vector list.

    Args:
        data: Binary object or list from MongoDB.

    Returns:
        List of floats or ints.
    """
    if isinstance(data, Binary):
        bv = data.as_vector()
        return list(bv.data)
    return data


def query_vector_for_search(
    vector: list[Union[float, int]],
    storage_format: VectorStorageFormat,
) -> Union[Binary, list]:
    """Prepare a query vector for $vectorSearch.

    For BinData storage formats, the query vector should also be BinData.
    For array storage, use regular list.

    Args:
        vector: Query embedding vector.
        storage_format: How vectors are stored in the collection.

    Returns:
        Appropriately formatted query vector.
    """
    # For array storage or float32 BinData, we can use regular float list
    # MongoDB will handle the comparison
    if storage_format in (VectorStorageFormat.ARRAY, VectorStorageFormat.BINDATA_FLOAT32):
        return vector

    # For pre-quantized storage, query must match the format
    return vector_to_bindata(vector, storage_format)
