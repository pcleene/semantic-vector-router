"""Unit tests for SVRClient."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.client import SVRClient
from semantic_vector_router.exceptions import ConfigurationError, SearchError
from semantic_vector_router.models import (
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    PartitionInfo,
    PartitionStatus,
    RerankerProvider,
    SearchHit,
    SearchResult,
    SVRConfig,
    WatcherStatus,
)
from semantic_vector_router.routing.merger import ResultMerger
from semantic_vector_router.routing.resolver import PartitionResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_results(partitions=None):
    """Return raw backend result dicts for the given partition names.

    If *partitions* is ``None`` the results span two partitions
    (electronics and furniture) so the caller can exercise multi-partition
    code paths.
    """
    if partitions is None:
        partitions = ["electronics", "furniture"]

    results = []
    if "electronics" in partitions:
        results.extend([
            {
                "_id": "doc1",
                "name": "Wireless Headphones",
                "price": 299.99,
                "_svr_score": 0.95,
                "_svr_partition": "electronics",
            },
            {
                "_id": "doc2",
                "name": "Bluetooth Speaker",
                "price": 79.99,
                "_svr_score": 0.88,
                "_svr_partition": "electronics",
            },
        ])
    if "furniture" in partitions:
        results.append({
            "_id": "doc3",
            "name": "Office Chair",
            "price": 449.99,
            "_svr_score": 0.72,
            "_svr_partition": "furniture",
        })
    if "clothing" in partitions:
        results.append({
            "_id": "doc4",
            "name": "Winter Jacket",
            "price": 189.99,
            "_svr_score": 0.80,
            "_svr_partition": "clothing",
        })
    return results


def _make_client(
    config: SVRConfig,
    mock_backend,
    mock_embedder,
    mock_reranker,
    *,
    connected: bool = True,
    auto_connect_failed: bool = False,
):
    """Build an SVRClient with injected mocks, bypassing real __init__ connect."""
    client = SVRClient(config=config, auto_connect=False)
    client._backend = mock_backend
    client._embedder = mock_embedder
    client._reranker = mock_reranker
    client._resolver = PartitionResolver(config)
    client._merger = ResultMerger()
    client._connected = connected
    client._auto_connect_failed = auto_connect_failed
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchFullFlow:
    """Test the full search flow: resolve -> embed -> search -> merge -> rerank."""

    @pytest.mark.asyncio
    async def test_search_full_flow(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """Full search across multiple partitions with reranking enabled."""
        raw = _raw_results(["electronics", "furniture"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        reranked_hits = [
            SearchHit(
                id="doc1",
                score=0.95,
                rerank_score=0.97,
                partition="electronics",
                document={"name": "Wireless Headphones", "price": 299.99},
            ),
            SearchHit(
                id="doc3",
                score=0.72,
                rerank_score=0.85,
                partition="furniture",
                document={"name": "Office Chair", "price": 449.99},
            ),
            SearchHit(
                id="doc2",
                score=0.88,
                rerank_score=0.60,
                partition="electronics",
                document={"name": "Bluetooth Speaker", "price": 79.99},
            ),
        ]
        mock_reranker.rerank_hits = AsyncMock(return_value=reranked_hits)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        result = await client.search(
            query="wireless headphones",
            partitions=["electronics", "furniture"],
            limit=10,
        )

        # Embedder should be called (BYOM mode, no precomputed vector)
        mock_embedder.embed.assert_awaited_once_with("wireless headphones")

        # Backend search should be called
        mock_backend.search_partitions.assert_awaited_once()

        # Reranker should be called (multi-partition + config enabled)
        mock_reranker.rerank_hits.assert_awaited_once()

        # Result shape
        assert isinstance(result, SearchResult)
        assert result.query == "wireless headphones"
        assert set(result.partitions_searched) == {"electronics", "furniture"}
        assert result.reranked is True
        assert result.latency_ms > 0


class TestSearchPrecomputedVector:
    """Test search with a precomputed query vector."""

    @pytest.mark.asyncio
    async def test_search_with_precomputed_query_vector(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """When query_vector is supplied, embedding step must be skipped."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        precomputed = [0.5] * 1536
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            query_vector=precomputed,
        )

        # Embedder must NOT be called when a precomputed vector is provided
        mock_embedder.embed.assert_not_awaited()

        # Backend should still receive the vector
        call_kwargs = mock_backend.search_partitions.call_args
        assert call_kwargs.kwargs.get("query_vector") == precomputed or \
            (call_kwargs[1].get("query_vector") == precomputed if len(call_kwargs) > 1 else True)

        assert isinstance(result, SearchResult)


class TestSearchAcrossModes:
    """Test correct behaviour for VIEWS, SOURCE, and FIELDS index locations."""

    @pytest.mark.asyncio
    async def test_search_views_mode(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """VIEWS mode (default) -- partitions have view_name set."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )
        assert sample_config_with_partitions.vector_storage.index_on == IndexLocation.VIEWS

        result = await client.search(
            query="laptop",
            partitions=["electronics"],
            limit=5,
        )

        mock_backend.search_partitions.assert_awaited_once()
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_source_mode(
        self,
        sample_config_with_source_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """SOURCE mode -- partitions share a single index on the source collection."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_source_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )
        assert sample_config_with_source_partitions.vector_storage.index_on == IndexLocation.SOURCE

        result = await client.search(
            query="laptop",
            partitions=["electronics"],
            limit=5,
        )

        mock_backend.search_partitions.assert_awaited_once()
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_fields_mode(
        self,
        sample_config_with_fields_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """FIELDS mode -- partitions have per-partition embedding fields."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_fields_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )
        assert sample_config_with_fields_partitions.vector_storage.index_on == IndexLocation.FIELDS

        result = await client.search(
            query="laptop",
            partitions=["electronics"],
            limit=5,
        )

        mock_backend.search_partitions.assert_awaited_once()
        assert isinstance(result, SearchResult)


class TestAutoConnectFailure:
    """Regression: Phase 1.1 -- SearchError when auto-connect failed."""

    @pytest.mark.asyncio
    async def test_auto_connect_failure_raises_search_error(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """If auto_connect_failed is True and client is not connected, search
        must raise a SearchError with a helpful message referencing
        ``await client.connect()``."""
        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
            connected=False,
            auto_connect_failed=True,
        )

        with pytest.raises(SearchError) as exc_info:
            await client.search(query="test query")

        message = str(exc_info.value)
        assert "await client.connect()" in message
        assert "not connected" in message.lower() or "Client not connected" in message

    @pytest.mark.asyncio
    async def test_not_connected_without_auto_connect_failure_calls_connect(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """When _auto_connect_failed is False but client is not connected,
        search should attempt to call connect() rather than raising."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
            connected=False,
            auto_connect_failed=False,
        )

        # Patch connect so it just sets _connected = True without real I/O
        async def fake_connect():
            client._connected = True

        with patch.object(client, "connect", side_effect=fake_connect) as mock_connect:
            result = await client.search(
                query="test query",
                partitions=["electronics"],
                limit=5,
            )

        mock_connect.assert_awaited_once()
        assert isinstance(result, SearchResult)


class TestConnectDisconnectLifecycle:
    """Test connect/disconnect lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_sets_connected(self, sample_config_with_partitions):
        """connect() should initialise components and set _connected = True."""
        with patch("semantic_vector_router.client.create_backend") as MockBackend, \
             patch("semantic_vector_router.client.validate_config", return_value=[]), \
             patch("semantic_vector_router.factories.get_api_key", return_value="mock_key"):
            mock_instance = AsyncMock()
            MockBackend.return_value = mock_instance

            client = SVRClient(config=sample_config_with_partitions, auto_connect=False)
            assert client._connected is False

            await client.connect()

            assert client._connected is True
            assert client._backend is mock_instance
            mock_instance.connect.assert_awaited_once()
            assert client._resolver is not None
            assert client._merger is not None

    @pytest.mark.asyncio
    async def test_disconnect_clears_connected(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """disconnect() should call backend.disconnect() and set _connected = False."""
        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
            connected=True,
        )

        await client.disconnect()

        mock_backend.disconnect.assert_awaited_once()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, sample_config_with_partitions):
        """Calling connect() when already connected should be a no-op."""
        with patch("semantic_vector_router.client.create_backend") as MockBackend, \
             patch("semantic_vector_router.client.validate_config", return_value=[]), \
             patch("semantic_vector_router.factories.get_api_key", return_value="mock_key"):
            mock_instance = AsyncMock()
            MockBackend.return_value = mock_instance

            client = SVRClient(config=sample_config_with_partitions, auto_connect=False)
            await client.connect()
            await client.connect()  # second call should be no-op

            # Backend constructor and connect called only once
            MockBackend.assert_called_once()
            mock_instance.connect.assert_awaited_once()


class TestListPartitions:
    """Test list_partitions delegates to config registry."""

    def test_list_partitions_delegates(self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker):
        """list_partitions should return a list of dicts derived from the
        partition registry in config."""
        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        partitions = client.list_partitions()

        assert isinstance(partitions, list)
        assert len(partitions) == 3

        names = {p["name"] for p in partitions}
        assert names == {"electronics", "furniture", "clothing"}

        # Each dict should have the expected keys
        for p in partitions:
            assert "name" in p
            assert "document_count" in p
            assert "status" in p
            assert "view_name" in p
            assert "index_name" in p

    def test_list_partitions_empty_registry(self, sample_config, mock_backend, mock_embedder, mock_reranker):
        """list_partitions on an empty registry returns an empty list."""
        client = _make_client(
            sample_config,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        partitions = client.list_partitions()
        assert partitions == []


class TestCreatePartition:
    """Test create_partition delegates to provisioner."""

    @pytest.mark.asyncio
    async def test_create_partition_delegates(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """create_partition should obtain a provisioner and call
        provisioner.create_partition with the supplied arguments."""
        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        mock_provisioner = AsyncMock()
        expected_partition = PartitionInfo(
            name="toys",
            view_name="svr_test_partition_toys",
            index_name="svr_test_idx_toys",
            filter_value="toys",
        )
        mock_provisioner.create_partition = AsyncMock(return_value=expected_partition)

        with patch.object(client, "_get_provisioner", return_value=mock_provisioner):
            result = await client.create_partition(name="toys", filter_value="toys")

        mock_provisioner.create_partition.assert_awaited_once_with(
            name="toys",
            filter_value="toys",
        )
        assert result.name == "toys"


class TestDeletePartition:
    """Test delete_partition delegates to provisioner."""

    @pytest.mark.asyncio
    async def test_delete_partition_delegates(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """delete_partition should obtain a provisioner and call
        provisioner.delete_partition with the supplied name."""
        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        mock_provisioner = AsyncMock()
        mock_provisioner.delete_partition = AsyncMock()

        with patch.object(client, "_get_provisioner", return_value=mock_provisioner):
            await client.delete_partition(name="electronics")

        mock_provisioner.delete_partition.assert_awaited_once_with("electronics")


class TestRerankingSinglePartition:
    """Reranking should be skipped for a single-partition search (rerank=None)."""

    @pytest.mark.asyncio
    async def test_reranking_skipped_for_single_partition(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """When rerank is None and only one partition is searched, reranking
        should not be applied even if config.reranking.enabled is True."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )
        # Ensure reranking is enabled in config
        assert client._config.reranking.enabled is True

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=10,
            rerank=None,  # let the client decide
        )

        # Only one partition -> should_rerank is False
        mock_reranker.rerank_hits.assert_not_awaited()
        assert result.reranked is False
        assert result.partitions_searched == ["electronics"]

    @pytest.mark.asyncio
    async def test_reranking_forced_for_single_partition(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """When rerank=True is explicitly passed, reranking should happen even
        for a single partition."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        reranked_hits = [
            SearchHit(id="doc1", score=0.95, rerank_score=0.99, partition="electronics", document={"name": "Wireless Headphones"}),
            SearchHit(id="doc2", score=0.88, rerank_score=0.70, partition="electronics", document={"name": "Bluetooth Speaker"}),
        ]
        mock_reranker.rerank_hits = AsyncMock(return_value=reranked_hits)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=10,
            rerank=True,
        )

        mock_reranker.rerank_hits.assert_awaited_once()
        assert result.reranked is True


class TestRerankingMultiPartition:
    """Reranking should be applied for multi-partition searches (rerank=None)."""

    @pytest.mark.asyncio
    async def test_reranking_applied_for_multi_partition(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """When rerank is None and multiple partitions are searched, reranking
        should be applied because config.reranking.enabled is True."""
        raw = _raw_results(["electronics", "furniture"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        reranked_hits = [
            SearchHit(id="doc1", score=0.95, rerank_score=0.97, partition="electronics", document={"name": "Wireless Headphones"}),
            SearchHit(id="doc3", score=0.72, rerank_score=0.85, partition="furniture", document={"name": "Office Chair"}),
            SearchHit(id="doc2", score=0.88, rerank_score=0.60, partition="electronics", document={"name": "Bluetooth Speaker"}),
        ]
        mock_reranker.rerank_hits = AsyncMock(return_value=reranked_hits)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )
        assert client._config.reranking.enabled is True

        result = await client.search(
            query="headphones",
            partitions=["electronics", "furniture"],
            limit=10,
            rerank=None,
        )

        mock_reranker.rerank_hits.assert_awaited_once()
        assert result.reranked is True
        assert set(result.partitions_searched) == {"electronics", "furniture"}

    @pytest.mark.asyncio
    async def test_reranking_disabled_for_multi_partition(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """When rerank=False is explicitly passed, reranking should be skipped
        even for multiple partitions."""
        raw = _raw_results(["electronics", "furniture"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        result = await client.search(
            query="headphones",
            partitions=["electronics", "furniture"],
            limit=10,
            rerank=False,
        )

        mock_reranker.rerank_hits.assert_not_awaited()
        assert result.reranked is False


class TestSearchEdgeCases:
    """Additional edge-case coverage for search."""

    @pytest.mark.asyncio
    async def test_search_empty_partitions_returns_empty_result(
        self,
        sample_config,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """When the resolver returns no partitions, search should return an
        empty SearchResult immediately."""
        client = _make_client(
            sample_config,  # no partitions registered
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        result = await client.search(query="anything", partitions="all", limit=5)

        assert isinstance(result, SearchResult)
        assert result.hits == []
        assert result.partitions_searched == []
        assert result.total_candidates == 0
        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """Filters dict should be forwarded to the backend."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        filters = {"price": {"$lt": 100}}
        await client.search(
            query="speaker",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        call_kwargs = mock_backend.search_partitions.call_args
        assert call_kwargs.kwargs.get("filters") == filters or \
            call_kwargs[1].get("filters") == filters

    @pytest.mark.asyncio
    async def test_search_reranking_skipped_when_no_reranker(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
    ):
        """Even if should_rerank is True, reranking must be skipped when no
        reranker is available (self._reranker is None)."""
        raw = _raw_results(["electronics", "furniture"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker=None,  # type: ignore[arg-type]
        )
        # Without a reranker, the code path should not attempt reranking
        client._reranker = None

        result = await client.search(
            query="chair",
            partitions=["electronics", "furniture"],
            limit=10,
        )

        assert isinstance(result, SearchResult)
        # Hits should be truncated to limit rather than reranked
        assert len(result.hits) <= 10

    @pytest.mark.asyncio
    async def test_search_candidates_per_partition_override(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """candidates_per_partition should be forwarded to the backend as limit."""
        raw = _raw_results(["electronics"])
        mock_backend.search_partitions = AsyncMock(return_value=raw)

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            candidates_per_partition=50,
        )

        call_kwargs = mock_backend.search_partitions.call_args
        assert call_kwargs.kwargs.get("limit") == 50 or \
            call_kwargs[1].get("limit") == 50


class TestContextManager:
    """Test async context manager protocol."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, sample_config_with_partitions):
        """The client should support async with and call connect/disconnect."""
        with patch("semantic_vector_router.client.create_backend") as MockBackend, \
             patch("semantic_vector_router.client.validate_config", return_value=[]), \
             patch("semantic_vector_router.factories.get_api_key", return_value="mock_key"):
            mock_instance = AsyncMock()
            MockBackend.return_value = mock_instance

            client = SVRClient(config=sample_config_with_partitions, auto_connect=False)

            async with client as c:
                assert c._connected is True
                assert c is client

            mock_instance.disconnect.assert_awaited_once()
            assert client._connected is False


# ===========================================================================
# Initialization variations
# ===========================================================================


class TestInitVariations:
    def test_init_with_dict_config(self):
        config_dict = {
            "database": {"connection_string_env": "MONGODB_URI", "database": "test_db", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=config_dict, auto_connect=False)
        assert client._config.database.database == "test_db"

    def test_init_with_config_path(self, tmp_path, sample_config):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(
            sample_config.model_dump(mode="json", exclude_none=True)
        ))
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config_path=str(config_path), auto_connect=False)
        assert client._config.database.database == sample_config.database.database

    def test_config_property(self, sample_config):
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        assert client.config is client._config

    def test_config_warnings_logged(self, sample_config):
        with patch("semantic_vector_router.client.validate_config", return_value=["warn1"]), \
             patch("semantic_vector_router.client.logger") as mock_logger:
            SVRClient(config=sample_config, auto_connect=False)
        mock_logger.warning.assert_called()


# ===========================================================================
# Auto-connect logic
# ===========================================================================


class TestAutoConnect:
    def test_auto_connect_running_loop(self, sample_config):
        """When event loop is already running, auto_connect_failed is set."""
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        with patch("semantic_vector_router.client.validate_config", return_value=[]), \
             patch("semantic_vector_router.client.logger"), \
             patch("asyncio.get_event_loop", return_value=mock_loop):
            client = SVRClient(config=sample_config, auto_connect=True)

        assert client._auto_connect_failed is True
        assert client._connected is False

    def test_auto_connect_no_event_loop(self, sample_config):
        """When no event loop is available, auto_connect_failed is set."""
        with patch("semantic_vector_router.client.validate_config", return_value=[]), \
             patch("semantic_vector_router.client.logger"), \
             patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
            client = SVRClient(config=sample_config, auto_connect=True)

        assert client._auto_connect_failed is True
        assert client._connected is False


# ===========================================================================
# Embedder / Reranker creation
# ===========================================================================


class TestCreateEmbedder:
    def test_create_voyage_embedder(self, sample_config):
        sample_config.embedding.provider = EmbeddingProvider.VOYAGE
        sample_config.embedding.api_key_env = "VOYAGE_API_KEY"
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        with patch("semantic_vector_router.factories.get_api_key", return_value="test-key"), \
             patch("semantic_vector_router.factories.VoyageEmbedder") as MockVoyage:
            embedder = client._create_embedder()
            MockVoyage.assert_called_once()

    def test_create_cohere_embedder(self, sample_config):
        sample_config.embedding.provider = EmbeddingProvider.COHERE
        sample_config.embedding.api_key_env = "COHERE_API_KEY"
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        with patch("semantic_vector_router.factories.get_api_key", return_value="test-key"), \
             patch("semantic_vector_router.factories.CohereEmbedder") as MockCohere:
            embedder = client._create_embedder()
            MockCohere.assert_called_once()

    def test_create_huggingface_embedder(self, sample_config):
        sample_config.embedding.provider = EmbeddingProvider.HUGGINGFACE
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        with patch("semantic_vector_router.factories.HuggingFaceEmbedder") as MockHF:
            embedder = client._create_embedder()
            MockHF.assert_called_once()

    def test_create_embedder_unknown_provider_raises(self, sample_config):
        sample_config.embedding.provider = "unknown_provider"
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
            client._create_embedder()


class TestCreateReranker:
    def test_create_cohere_reranker(self, sample_config):
        sample_config.reranking.provider = RerankerProvider.COHERE
        sample_config.reranking.api_key_env = "COHERE_API_KEY"
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        with patch("semantic_vector_router.factories.get_api_key", return_value="test-key"), \
             patch("semantic_vector_router.factories.CohereReranker") as MockCohere:
            reranker = client._create_reranker()
            MockCohere.assert_called_once()

    def test_create_reranker_unknown_provider_raises(self, sample_config):
        sample_config.reranking.provider = "unknown_provider"
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        with pytest.raises(ConfigurationError, match="Unknown reranker provider"):
            client._create_reranker()


# ===========================================================================
# Search error path
# ===========================================================================


class TestSearchByomWithoutEmbedder:
    @pytest.mark.asyncio
    async def test_search_byom_without_embedder_raises(
        self, sample_config_with_partitions, mock_backend, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, None, mock_reranker)
        client._embedder = None
        with pytest.raises(SearchError, match="Embedder not initialized"):
            await client.search(query="test", partitions=["electronics"])


# ===========================================================================
# get_partition
# ===========================================================================


class TestGetPartition:
    @pytest.mark.asyncio
    async def test_get_partition_delegates(self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        partition = await client.get_partition("electronics")
        assert partition.name == "electronics"


# ===========================================================================
# Auto-connect in partition methods
# ===========================================================================


class TestAutoConnectPartitionMethods:
    @pytest.mark.asyncio
    async def test_create_partition_auto_connects(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(
            sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
            connected=False,
        )
        mock_provisioner = AsyncMock()
        mock_provisioner.create_partition = AsyncMock(return_value=PartitionInfo(
            name="new", view_name="v", index_name="i", filter_value="new",
        ))

        async def fake_connect():
            client._connected = True

        with patch.object(client, "connect", side_effect=fake_connect) as mock_connect, \
             patch.object(client, "_get_provisioner", return_value=mock_provisioner):
            await client.create_partition("new")
        mock_connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_partition_auto_connects(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(
            sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
            connected=False,
        )
        mock_provisioner = AsyncMock()

        async def fake_connect():
            client._connected = True

        with patch.object(client, "connect", side_effect=fake_connect) as mock_connect, \
             patch.object(client, "_get_provisioner", return_value=mock_provisioner):
            await client.delete_partition("electronics")
        mock_connect.assert_awaited_once()


# ===========================================================================
# refresh_partitions
# ===========================================================================


class TestRefreshPartitions:
    @pytest.mark.asyncio
    async def test_refresh_creates_new(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_scanner = AsyncMock()
        mock_scanner.get_new_partition_values = AsyncMock(return_value=["toys", "books"])
        mock_provisioner = AsyncMock()
        mock_provisioner.create_partitions_batch = AsyncMock(return_value={"toys": MagicMock(), "books": MagicMock()})

        with patch.object(client, "_get_scanner", return_value=mock_scanner), \
             patch.object(client, "_get_provisioner", return_value=mock_provisioner):
            result = await client.refresh_partitions()

        assert set(result) == {"toys", "books"}

    @pytest.mark.asyncio
    async def test_refresh_no_new_values(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_scanner = AsyncMock()
        mock_scanner.get_new_partition_values = AsyncMock(return_value=[])

        with patch.object(client, "_get_scanner", return_value=mock_scanner):
            result = await client.refresh_partitions()

        assert result == []


# ===========================================================================
# detect_new_partitions
# ===========================================================================


class TestDetectNewPartitions:
    @pytest.mark.asyncio
    async def test_detect_returns_strings(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_scanner = AsyncMock()
        mock_scanner.get_new_partition_values = AsyncMock(return_value=["toys", 123])

        with patch.object(client, "_get_scanner", return_value=mock_scanner):
            result = await client.detect_new_partitions()

        assert result == ["toys", "123"]


# ===========================================================================
# Watcher lifecycle
# ===========================================================================


class TestWatcherLifecycle:
    @pytest.mark.asyncio
    async def test_start_watcher(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_watcher = AsyncMock()

        with patch.object(client, "_get_watcher", return_value=mock_watcher):
            await client.start_watcher()

        mock_watcher.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_watcher_exists(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_watcher = AsyncMock()
        client._watcher = mock_watcher

        await client.stop_watcher()
        mock_watcher.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_watcher_none(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        client._watcher = None
        await client.stop_watcher()  # Should not raise

    def test_watcher_status_exists(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_watcher = MagicMock()
        mock_watcher.get_status.return_value = WatcherStatus(running=True)
        client._watcher = mock_watcher

        status = client.watcher_status()
        assert status.running is True

    def test_watcher_status_none(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        client._watcher = None
        status = client.watcher_status()
        assert status.running is False


# ===========================================================================
# Monitoring
# ===========================================================================


class TestMonitoring:
    @pytest.mark.asyncio
    async def test_check_partition_health(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_monitor = AsyncMock()
        mock_monitor.check_all_partitions = AsyncMock(return_value=[])

        with patch.object(client, "_get_monitor", return_value=mock_monitor):
            result = await client.check_partition_health()

        assert result == []
        mock_monitor.check_all_partitions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_health_summary(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_monitor = AsyncMock()
        mock_monitor.get_partition_summary = AsyncMock(return_value={"total": 3})

        with patch.object(client, "_get_monitor", return_value=mock_monitor):
            result = await client.get_health_summary()

        assert result == {"total": 3}


# ===========================================================================
# Lazy initialization helpers
# ===========================================================================


class TestLazyInit:
    def test_get_scanner_creates_once(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        with patch("semantic_vector_router.client.PartitionScanner") as MockScanner:
            s1 = client._get_scanner()
            s2 = client._get_scanner()
        MockScanner.assert_called_once()
        assert s1 is s2

    def test_get_provisioner_creates_once(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        with patch("semantic_vector_router.client.PartitionProvisioner") as MockProv:
            p1 = client._get_provisioner()
            p2 = client._get_provisioner()
        MockProv.assert_called_once()
        assert p1 is p2

    def test_get_watcher_creates_once(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        with patch("semantic_vector_router.client.PartitionWatcher") as MockWatch, \
             patch("semantic_vector_router.client.PartitionProvisioner"):
            w1 = client._get_watcher()
            w2 = client._get_watcher()
        MockWatch.assert_called_once()
        assert w1 is w2

    def test_get_monitor_creates_once(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        with patch("semantic_vector_router.client.PartitionMonitor") as MockMon:
            m1 = client._get_monitor()
            m2 = client._get_monitor()
        MockMon.assert_called_once()
        assert m1 is m2

    def test_get_splitter_creates_once(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        with patch("semantic_vector_router.client.PartitionSplitter") as MockSplit, \
             patch("semantic_vector_router.client.PartitionProvisioner"):
            s1 = client._get_splitter()
            s2 = client._get_splitter()
        MockSplit.assert_called_once()
        assert s1 is s2


# ===========================================================================
# save_config
# ===========================================================================


class TestSaveConfigClient:
    def test_save_config_delegates(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker, tmp_path,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        with patch("semantic_vector_router.client.save_config") as mock_save:
            mock_save.return_value = tmp_path / "config.json"
            result = client.save_config(tmp_path / "config.json")
        mock_save.assert_called_once()


# ===========================================================================
# Phase 6: Metrics, Cache, Correlation ID integration
# ===========================================================================


class TestSearchMetrics:
    """Tests that search() emits metric events correctly."""

    @pytest.mark.asyncio
    async def test_search_emits_search_latency(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        from semantic_vector_router.utils.metrics import MetricType

        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        client._metrics.add_handler(Recorder())

        await client.search("headphones", partitions=["electronics"], limit=5)

        latency_events = [e for e in events if e.metric_type == MetricType.SEARCH_LATENCY]
        assert len(latency_events) == 1
        assert latency_events[0].value > 0

    @pytest.mark.asyncio
    async def test_search_emits_search_results_count(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        from semantic_vector_router.utils.metrics import MetricType

        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        client._metrics.add_handler(Recorder())

        await client.search("headphones", partitions=["electronics"], limit=5)

        result_events = [e for e in events if e.metric_type == MetricType.SEARCH_RESULTS]
        assert len(result_events) == 1

    @pytest.mark.asyncio
    async def test_search_emits_embedding_latency(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        from semantic_vector_router.utils.metrics import MetricType

        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        client._metrics.add_handler(Recorder())

        await client.search("headphones", partitions=["electronics"], limit=5)

        embed_events = [e for e in events if e.metric_type == MetricType.EMBEDDING_LATENCY]
        assert len(embed_events) == 1
        assert embed_events[0].value > 0

    @pytest.mark.asyncio
    async def test_search_emits_error_metric_on_failure(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        from semantic_vector_router.utils.metrics import MetricType

        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(side_effect=RuntimeError("API down"))

        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        client._metrics.add_handler(Recorder())

        with pytest.raises(RuntimeError):
            await client.search("headphones", partitions=["electronics"], limit=5)

        error_events = [e for e in events if e.metric_type == MetricType.ERROR]
        assert len(error_events) == 1


class TestSearchCorrelationId:
    """Tests that search() sets a correlation ID."""

    @pytest.mark.asyncio
    async def test_search_sets_correlation_id(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        from semantic_vector_router.utils.logging import get_correlation_id

        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        await client.search("headphones", partitions=["electronics"], limit=5)

        cid = get_correlation_id()
        assert len(cid) == 12


class TestEmbeddingCacheIntegration:
    """Tests that search() uses the embedding cache correctly."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_embedder(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        # First call: cache miss, calls embedder
        await client.search("headphones", partitions=["electronics"], limit=5)
        assert mock_embedder.embed.call_count == 1

        # Second call: cache hit, skips embedder
        await client.search("headphones", partitions=["electronics"], limit=5)
        assert mock_embedder.embed.call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_cache_miss_stores_embedding(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        await client.search("headphones", partitions=["electronics"], limit=5)

        assert client._embedding_cache.size == 1

    @pytest.mark.asyncio
    async def test_cache_emits_hit_miss_metrics(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        from semantic_vector_router.utils.metrics import MetricType

        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker)
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        client._metrics.add_handler(Recorder())

        # First call: miss
        await client.search("headphones", partitions=["electronics"], limit=5)
        # Second call: hit
        await client.search("headphones", partitions=["electronics"], limit=5)

        miss_events = [e for e in events if e.metric_type == MetricType.CACHE_MISS]
        hit_events = [e for e in events if e.metric_type == MetricType.CACHE_HIT]
        assert len(miss_events) == 1
        assert len(hit_events) == 1

    @pytest.mark.asyncio
    async def test_cache_disabled(
        self, sample_config_with_partitions, mock_backend, mock_embedder, mock_reranker,
    ):
        sample_config_with_partitions.cache.enabled = False
        client = SVRClient(config=sample_config_with_partitions, auto_connect=False)
        client._backend = mock_backend
        client._embedder = mock_embedder
        client._reranker = mock_reranker
        client._resolver = PartitionResolver(sample_config_with_partitions)
        client._merger = ResultMerger()
        client._connected = True

        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results(["electronics"]))

        await client.search("headphones", partitions=["electronics"], limit=5)
        await client.search("headphones", partitions=["electronics"], limit=5)

        # Embedder called twice because cache is disabled
        assert mock_embedder.embed.call_count == 2


class TestMetricsHandlerInit:
    """Tests for metrics_handler parameter on SVRClient.__init__."""

    def test_metrics_handler_registered(self, sample_config_with_partitions):
        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        handler = Recorder()
        client = SVRClient(config=sample_config_with_partitions, auto_connect=False, metrics_handler=handler)

        from semantic_vector_router.utils.metrics import MetricEvent, MetricType
        client._metrics.emit(MetricEvent(metric_type=MetricType.SEARCH_LATENCY, value=1.0))
        assert len(events) == 1

    def test_metrics_disabled(self, sample_config_with_partitions):
        from semantic_vector_router.utils.metrics import NoOpCollector
        sample_config_with_partitions.metrics.enabled = False
        client = SVRClient(config=sample_config_with_partitions, auto_connect=False)
        assert isinstance(client._metrics, NoOpCollector)


# ===========================================================================
# Phase 7: Ingestion integration
# ===========================================================================


class TestClientIngest:
    """Test SVRClient.ingest() delegates to IngestPipeline."""

    @pytest.mark.asyncio
    async def test_ingest_not_connected_auto_connect_failed_raises(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """If auto_connect_failed and not connected, ingest raises IngestionError."""
        from semantic_vector_router.exceptions import IngestionError

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
            connected=False,
            auto_connect_failed=True,
        )

        with pytest.raises(IngestionError, match="not connected"):
            await client.ingest(documents=[{"text": "hello"}])

    @pytest.mark.asyncio
    async def test_ingest_delegates_to_pipeline(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """ingest() should create a pipeline and delegate to it."""
        from semantic_vector_router.models import IngestResult

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        mock_result = IngestResult(inserted=5, failed=0, elapsed_ms=100.0, embed_ms=50.0, write_ms=40.0)

        with patch("semantic_vector_router.client.IngestPipeline") as MockPipeline, \
             patch("semantic_vector_router.factories.get_api_key", return_value="test-key"):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            result = await client.ingest(
                documents=[{"text": "hello"}],
                partition="electronics",
            )

        assert result.inserted == 5
        mock_pipeline.ingest.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingest_emits_metrics(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """ingest() should emit INGEST_LATENCY and INGEST_DOCUMENTS metrics."""
        from semantic_vector_router.models import IngestResult
        from semantic_vector_router.utils.metrics import MetricType

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
        )

        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        client._metrics.add_handler(Recorder())

        mock_result = IngestResult(inserted=3, failed=1, elapsed_ms=200.0, embed_ms=100.0, write_ms=80.0)

        with patch("semantic_vector_router.client.IngestPipeline") as MockPipeline, \
             patch("semantic_vector_router.factories.get_api_key", return_value="test-key"):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            await client.ingest(documents=[{"text": "hello"}])

        latency_events = [e for e in events if e.metric_type == MetricType.INGEST_LATENCY]
        doc_events = [e for e in events if e.metric_type == MetricType.INGEST_DOCUMENTS]
        err_events = [e for e in events if e.metric_type == MetricType.INGEST_ERRORS]

        assert len(latency_events) == 1
        assert len(doc_events) == 1
        assert len(err_events) == 1  # Because failed > 0

    @pytest.mark.asyncio
    async def test_ingest_auto_connects(
        self,
        sample_config_with_partitions,
        mock_backend,
        mock_embedder,
        mock_reranker,
    ):
        """ingest() should auto-connect if not connected."""
        from semantic_vector_router.models import IngestResult

        client = _make_client(
            sample_config_with_partitions,
            mock_backend,
            mock_embedder,
            mock_reranker,
            connected=False,
        )

        mock_result = IngestResult(inserted=1, elapsed_ms=50.0, embed_ms=25.0, write_ms=20.0)

        async def fake_connect():
            client._connected = True

        with patch.object(client, "connect", side_effect=fake_connect) as mock_connect, \
             patch("semantic_vector_router.client.IngestPipeline") as MockPipeline, \
             patch("semantic_vector_router.factories.get_api_key", return_value="test-key"):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            result = await client.ingest(documents=[{"text": "hello"}])

        mock_connect.assert_awaited_once()
        assert result.inserted == 1


class TestRateLimiterRegistryInit:
    """Test that SVRClient initializes the RateLimiterRegistry."""

    def test_rate_limiter_registry_created(self, sample_config):
        """Client should have a RateLimiterRegistry."""
        from semantic_vector_router.utils.rate_limiter import RateLimiterRegistry

        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)
        assert isinstance(client._rate_limiter_registry, RateLimiterRegistry)

    def test_rate_limiter_disabled_creates_unlimited(self, sample_config):
        """When rate_limiting.enabled=False, registry gets None config."""
        sample_config.rate_limiting.enabled = False

        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)

        # The registry should exist but produce unlimited limiters
        limiter = client._rate_limiter_registry.get("openai")
        assert limiter.tokens_per_second >= 10_000


class TestCreateDocumentEmbedder:
    """Test _create_document_embedder for asymmetric embedding support."""

    def test_voyage_document_embedder_uses_document_model(self, sample_config):
        """Voyage document embedder should use effective_document_model."""
        sample_config.embedding.provider = EmbeddingProvider.VOYAGE
        sample_config.embedding.model = "voyage-4-lite"
        sample_config.embedding.document_model = "voyage-4-large"
        sample_config.embedding.api_key_env = "VOYAGE_API_KEY"

        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)

        with patch("semantic_vector_router.factories.get_api_key", return_value="test-key"), \
             patch("semantic_vector_router.factories.VoyageEmbedder") as MockVoyage:
            client._create_document_embedder()
            call_kwargs = MockVoyage.call_args
            assert call_kwargs.kwargs["model"] == "voyage-4-large"
            assert call_kwargs.kwargs["input_type"] == "document"

    def test_cohere_document_embedder_uses_search_document(self, sample_config):
        """Cohere document embedder should use input_type='search_document'."""
        sample_config.embedding.provider = EmbeddingProvider.COHERE
        sample_config.embedding.api_key_env = "COHERE_API_KEY"

        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)

        with patch("semantic_vector_router.factories.get_api_key", return_value="test-key"), \
             patch("semantic_vector_router.factories.CohereEmbedder") as MockCohere:
            client._create_document_embedder()
            call_kwargs = MockCohere.call_args
            assert call_kwargs.kwargs["input_type"] == "search_document"

    def test_openai_document_embedder_reuses_standard(self, sample_config):
        """OpenAI (symmetric) should reuse the standard embedder."""
        sample_config.embedding.provider = EmbeddingProvider.OPENAI
        sample_config.embedding.api_key_env = "OPENAI_API_KEY"

        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = SVRClient(config=sample_config, auto_connect=False)

        with patch("semantic_vector_router.factories.get_api_key", return_value="test-key"), \
             patch("semantic_vector_router.factories.OpenAIEmbedder") as MockOAI:
            client._create_document_embedder()
            MockOAI.assert_called_once()
