"""Pytest configuration and shared fixtures."""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RerankingConfig,
    RerankerProvider,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)


@pytest.fixture
def sample_config() -> SVRConfig:
    """Create a sample configuration for testing."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(
            field="category",
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


@pytest.fixture
def sample_config_with_partitions(sample_config: SVRConfig) -> SVRConfig:
    """Create config with pre-registered partitions."""
    sample_config.partitions.registry = {
        "electronics": PartitionInfo(
            name="electronics",
            view_name="svr_test_partition_electronics",
            index_name="svr_test_idx_electronics",
            filter_value="electronics",
            document_count=150000,
            status=PartitionStatus.ACTIVE,
        ),
        "furniture": PartitionInfo(
            name="furniture",
            view_name="svr_test_partition_furniture",
            index_name="svr_test_idx_furniture",
            filter_value="furniture",
            document_count=85000,
            status=PartitionStatus.ACTIVE,
        ),
        "clothing": PartitionInfo(
            name="clothing",
            view_name="svr_test_partition_clothing",
            index_name="svr_test_idx_clothing",
            filter_value="clothing",
            document_count=234000,
            status=PartitionStatus.ACTIVE,
        ),
    }
    return sample_config


@pytest.fixture
def sample_config_source() -> SVRConfig:
    """Create a sample configuration with SOURCE index mode."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_test_partition_",
            index_name_prefix="svr_test_idx_",
        ),
        vector_storage=VectorStorageConfig(
            index_on=IndexLocation.SOURCE,
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
        ),
    )


@pytest.fixture
def sample_config_fields() -> SVRConfig:
    """Create a sample configuration with FIELDS index mode."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_test_partition_",
            index_name_prefix="svr_test_idx_",
        ),
        vector_storage=VectorStorageConfig(
            index_on=IndexLocation.FIELDS,
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
        ),
    )


@pytest.fixture
def sample_config_with_fields_partitions(sample_config_fields: SVRConfig) -> SVRConfig:
    """Create FIELDS config with pre-registered partitions that have embedding_field set."""
    sample_config_fields.partitions.registry = {
        "electronics": PartitionInfo(
            name="electronics",
            view_name=None,
            index_name="svr_test_idx_electronics",
            filter_value="electronics",
            document_count=150000,
            status=PartitionStatus.ACTIVE,
            index_location=IndexLocation.FIELDS,
            search_collection="test_collection",
            embedding_field="embedding_electronics",
        ),
        "furniture": PartitionInfo(
            name="furniture",
            view_name=None,
            index_name="svr_test_idx_furniture",
            filter_value="furniture",
            document_count=85000,
            status=PartitionStatus.ACTIVE,
            index_location=IndexLocation.FIELDS,
            search_collection="test_collection",
            embedding_field="embedding_furniture",
        ),
    }
    return sample_config_fields


@pytest.fixture
def sample_config_with_source_partitions(sample_config_source: SVRConfig) -> SVRConfig:
    """Create SOURCE config with pre-registered partitions."""
    sample_config_source.partitions.registry = {
        "electronics": PartitionInfo(
            name="electronics",
            view_name="svr_test_partition_electronics",
            index_name="svr_vector_idx_source",
            filter_value="electronics",
            document_count=150000,
            status=PartitionStatus.ACTIVE,
            index_location=IndexLocation.SOURCE,
            search_collection="test_collection",
        ),
        "furniture": PartitionInfo(
            name="furniture",
            view_name="svr_test_partition_furniture",
            index_name="svr_vector_idx_source",
            filter_value="furniture",
            document_count=85000,
            status=PartitionStatus.ACTIVE,
            index_location=IndexLocation.SOURCE,
            search_collection="test_collection",
        ),
    }
    return sample_config_source


@pytest.fixture
def mock_backend():
    """Create a mock MongoDB backend."""
    backend = AsyncMock()
    backend.connect = AsyncMock()
    backend.disconnect = AsyncMock()
    backend.is_connected = AsyncMock(return_value=True)
    backend.create_partition_view = AsyncMock(
        return_value="svr_test_partition_test"
    )
    backend.create_vector_search_index = AsyncMock()
    backend.count_documents = AsyncMock(return_value=1000)
    # Abstract partition operations (Phase 12.5)
    backend.create_partition_storage = AsyncMock()
    backend.create_partition_index = AsyncMock(return_value="svr_test_idx_test")
    backend.delete_partition_index = AsyncMock()
    backend.delete_partition_storage = AsyncMock()
    backend.partition_storage_exists = AsyncMock(return_value=True)
    backend.get_partition_index_status = AsyncMock()
    backend.ensure_source_index = AsyncMock(return_value="svr_vector_idx_source")
    backend.get_distinct_values = AsyncMock(
        return_value=["electronics", "furniture", "clothing"]
    )
    backend.get_partition_document_counts = AsyncMock(
        return_value={"electronics": 150000, "furniture": 85000, "clothing": 234000}
    )
    backend.watch_collection = AsyncMock()
    return backend


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    embedder.dimensions = 1536
    embedder.model_name = "test-model"
    return embedder


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(return_value=[0.9, 0.8, 0.7])
    reranker.rerank_hits = AsyncMock()
    reranker.model_name = "test-reranker"
    return reranker


@pytest.fixture
def sample_search_results() -> list[dict[str, Any]]:
    """Create sample raw search results."""
    return [
        {
            "_id": "doc1",
            "name": "Wireless Headphones",
            "description": "Premium noise-canceling headphones",
            "price": 299.99,
            "_svr_partition": "electronics",
            "_svr_score": 0.95,
        },
        {
            "_id": "doc2",
            "name": "Bluetooth Speaker",
            "description": "Portable waterproof speaker",
            "price": 79.99,
            "_svr_partition": "electronics",
            "_svr_score": 0.88,
        },
        {
            "_id": "doc3",
            "name": "Office Chair",
            "description": "Ergonomic mesh office chair",
            "price": 449.99,
            "_svr_partition": "furniture",
            "_svr_score": 0.72,
        },
    ]


# Integration test fixtures (require MongoDB connection)


@pytest.fixture
def mongodb_uri() -> str:
    """Get MongoDB URI from environment."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set, skipping integration test")
    return uri


@pytest.fixture
def integration_config(mongodb_uri: str) -> SVRConfig:
    """Create config for integration testing."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="svr_integration_test",
            source_collection="test_products",
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_int_partition_",
            index_name_prefix="svr_int_idx_",
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=384,  # Using smaller dims for testing
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.HUGGINGFACE,
            model="sentence-transformers/all-MiniLM-L6-v2",
            local=True,
            device="cpu",
            dimensions=384,
        ),
        reranking=RerankingConfig(
            enabled=False,  # Disable for integration tests
        ),
    )
