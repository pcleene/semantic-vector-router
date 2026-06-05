"""Integration tests for MongoDB backend."""

import pytest

from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.models import PartitionInfo, PartitionStatus


@pytest.mark.integration
class TestMongoDBBackendIntegration:
    """Integration tests for MongoDB backend (requires real MongoDB connection)."""

    @pytest.fixture
    async def backend(self, integration_config):
        """Create and connect backend."""
        backend = MongoDBBackend(integration_config)
        await backend.connect()
        yield backend
        await backend.disconnect()

    async def test_connect_and_ping(self, backend):
        """Test connection is established."""
        is_connected = await backend.is_connected()
        assert is_connected is True

    async def test_create_and_delete_view(self, backend, integration_config):
        """Test creating and deleting a partition view."""
        # Create view
        view_name = await backend.create_partition_view(
            partition_name="test_partition",
            filter_value="test_value",
        )

        # Verify view exists
        exists = await backend.view_exists(view_name)
        assert exists is True

        # Delete view
        await backend.delete_partition_view(view_name)

        # Verify view is gone
        exists = await backend.view_exists(view_name)
        assert exists is False

    async def test_get_distinct_values(self, backend):
        """Test getting distinct values for a field."""
        # This will return empty if collection is empty, which is fine
        values = await backend.get_distinct_values("category")
        assert isinstance(values, list)

    async def test_count_documents(self, backend):
        """Test counting documents."""
        count = await backend.count_documents()
        assert isinstance(count, int)
        assert count >= 0

    async def test_get_collection_stats(self, backend):
        """Test getting collection statistics."""
        stats = await backend.get_collection_stats()
        assert "name" in stats
        # Stats might have error if collection doesn't exist
        if "error" not in stats:
            assert "count" in stats

    async def test_list_views(self, backend):
        """Test listing views."""
        views = await backend.list_views()
        assert isinstance(views, list)

    async def test_get_sample_document(self, backend):
        """Test getting a sample document."""
        doc = await backend.get_sample_document()
        # May be None if collection is empty
        assert doc is None or isinstance(doc, dict)


@pytest.mark.integration
class TestMongoDBSearchIntegration:
    """Integration tests for vector search (requires data and indexes)."""

    @pytest.fixture
    async def backend_with_data(self, integration_config):
        """Create backend with test data.

        Note: This test requires the collection to have:
        - Documents with a 'category' field
        - Documents with an 'embedding' field (384 dimensions)
        - A vector search index
        """
        backend = MongoDBBackend(integration_config)
        await backend.connect()

        # Check if we have data
        count = await backend.count_documents()
        if count == 0:
            pytest.skip("No test data in collection")

        yield backend
        await backend.disconnect()

    async def test_search_partition(self, backend_with_data, integration_config):
        """Test executing a vector search on a partition."""
        # Create a test partition
        partition = PartitionInfo(
            name="test",
            view_name=integration_config.database.source_collection,  # Use source directly
            index_name="test_index",  # Assumes index exists
            status=PartitionStatus.ACTIVE,
        )

        # This test assumes you have embeddings and an index set up
        # It will fail if the index doesn't exist, which is expected
        try:
            results = await backend_with_data.execute_search(
                partition=partition,
                query="test query",
                query_vector=[0.1] * 384,  # 384 dim for all-MiniLM
                limit=5,
                num_candidates=50,
            )

            assert isinstance(results, list)
            # Results may be empty if no matching data

        except Exception as e:
            # Expected if index doesn't exist
            if "index" in str(e).lower():
                pytest.skip(f"Vector index not set up: {e}")
            raise
