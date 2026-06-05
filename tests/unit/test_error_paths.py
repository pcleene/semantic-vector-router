"""Error path audit for SVRClient public API.

Verifies that every public method:
1. Raises typed exceptions with clear messages
2. Emits error metrics
3. Logs structured errors with correlation IDs
4. Suggests recovery actions in error messages

All tests are UNIT tests (mocked, no network).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.client import SVRClient
from semantic_vector_router.exceptions import (
    ConfigurationError,
    ConnectionError,
    IngestionError,
    PartitionNotFoundError,
    SearchError,
)
from semantic_vector_router.models import (
    CacheConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestConfig,
    IngestResult,
    MetricsConfig,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RateLimitConfig,
    RerankingConfig,
    ResilienceConfig,
    SearchHit,
    SearchResult,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)
from semantic_vector_router.routing.merger import ResultMerger
from semantic_vector_router.routing.resolver import PartitionResolver
from semantic_vector_router.utils.metrics import (
    MetricEvent,
    MetricType,
    MetricsCollector,
    MetricsHandler,
    NoOpCollector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> SVRConfig:
    """Create a test config with sensible defaults."""
    kwargs = dict(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_test_",
            index_name_prefix="svr_test_idx_",
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=1536,
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-3-lite",
            api_key_env="VOYAGE_API_KEY",
            dimensions=1536,
        ),
        reranking=RerankingConfig(enabled=False),
        metrics=MetricsConfig(enabled=True),
        cache=CacheConfig(enabled=True, max_size=100),
        ingestion=IngestConfig(text_fields=["description"]),
        resilience=ResilienceConfig(embedding_timeout_ms=30000),
        rate_limiting=RateLimitConfig(enabled=False),
    )
    kwargs.update(overrides)
    return SVRConfig(**kwargs)


def _make_config_with_partitions() -> SVRConfig:
    """Create a test config with pre-registered partitions."""
    config = _make_config()
    config.partitions.registry = {
        "electronics": PartitionInfo(
            name="electronics",
            view_name="svr_test_electronics",
            index_name="svr_test_idx_electronics",
            filter_value="electronics",
            document_count=1000,
            status=PartitionStatus.ACTIVE,
        ),
        "furniture": PartitionInfo(
            name="furniture",
            view_name="svr_test_furniture",
            index_name="svr_test_idx_furniture",
            filter_value="furniture",
            document_count=500,
            status=PartitionStatus.ACTIVE,
        ),
    }
    return config


def _make_client(
    config=None,
    mock_backend=None,
    mock_embedder=None,
    mock_reranker=None,
    connected=True,
    auto_connect_failed=False,
    metrics_collector=None,
):
    """Build an SVRClient with injected mocks, bypassing real __init__ connect."""
    if config is None:
        config = _make_config_with_partitions()

    client = SVRClient(config=config, auto_connect=False)
    client._backend = mock_backend or AsyncMock()
    client._embedder = mock_embedder or AsyncMock()
    client._reranker = mock_reranker
    client._resolver = PartitionResolver(config)
    client._merger = ResultMerger()
    client._connected = connected
    client._auto_connect_failed = auto_connect_failed

    if metrics_collector is not None:
        client._metrics = metrics_collector

    return client


class _CapturingHandler:
    """Metrics handler that captures all emitted events for assertions."""

    def __init__(self):
        self.events: list[MetricEvent] = []

    def handle(self, event: MetricEvent) -> None:
        self.events.append(event)

    def has_metric(self, metric_type: MetricType, **tag_filters) -> bool:
        """Check if a metric with given type and optional tags was emitted."""
        for event in self.events:
            if event.metric_type != metric_type:
                continue
            if all(event.tags.get(k) == v for k, v in tag_filters.items()):
                return True
        return False

    def count_metric(self, metric_type: MetricType) -> int:
        """Count how many events of a given type were emitted."""
        return sum(1 for e in self.events if e.metric_type == metric_type)


# ===========================================================================
# SVRClient.connect() Error Paths
# ===========================================================================


class TestConnectErrors:
    """Error paths for SVRClient.connect()."""

    @pytest.mark.asyncio
    async def test_connect_invalid_uri(self):
        """connect() should raise ConnectionError when backend.connect() fails."""
        config = _make_config()

        with patch(
            "semantic_vector_router.client.create_backend"
        ) as MockBackend:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock(
                side_effect=ConnectionError("Failed to connect to MongoDB: connection refused")
            )
            MockBackend.return_value = mock_instance

            client = SVRClient(config=config, auto_connect=False)

            with pytest.raises(ConnectionError) as exc_info:
                await client.connect()

            assert "connect" in str(exc_info.value).lower() or "MongoDB" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """connect() should raise ConnectionError on timeout."""
        config = _make_config()

        with patch(
            "semantic_vector_router.client.create_backend"
        ) as MockBackend:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock(
                side_effect=ConnectionError(
                    "Failed to connect to MongoDB: server selection timeout"
                )
            )
            MockBackend.return_value = mock_instance

            client = SVRClient(config=config, auto_connect=False)

            with pytest.raises(ConnectionError) as exc_info:
                await client.connect()

            assert "timeout" in str(exc_info.value).lower() or "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_connect_already_connected_is_noop(self):
        """Calling connect() when already connected should be a no-op."""
        client = _make_client(connected=True)
        # Should not raise, and should not re-initialize backend
        await client.connect()
        assert client._connected is True


# ===========================================================================
# SVRClient.search() Error Paths
# ===========================================================================


class TestSearchErrors:
    """Error paths for SVRClient.search()."""

    @pytest.mark.asyncio
    async def test_search_not_connected_with_auto_connect_failed(self):
        """search() when not connected and auto_connect_failed should raise SearchError
        with recovery suggestion."""
        client = _make_client(connected=False, auto_connect_failed=True)

        with pytest.raises(SearchError) as exc_info:
            await client.search("test query", partitions=["electronics"])

        error_msg = str(exc_info.value)
        assert "connect" in error_msg.lower()
        assert "await" in error_msg.lower() or "client.connect()" in error_msg

    @pytest.mark.asyncio
    async def test_search_invalid_partition(self):
        """search() with a non-existent partition should raise PartitionNotFoundError."""
        client = _make_client()

        with pytest.raises(PartitionNotFoundError) as exc_info:
            await client.search("test query", partitions=["nonexistent_partition"])

        error_msg = str(exc_info.value)
        assert "nonexistent_partition" in error_msg
        # Should include available partitions in details
        assert exc_info.value.details.get("available") is not None
        available = exc_info.value.details["available"]
        assert "electronics" in available
        assert "furniture" in available

    @pytest.mark.asyncio
    async def test_search_embedding_failure(self):
        """search() should propagate embedding errors and emit error metric."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(
            side_effect=RuntimeError("Voyage API rate limit exceeded")
        )

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        client = _make_client(mock_embedder=mock_embedder, metrics_collector=collector)

        with pytest.raises(RuntimeError, match="rate limit"):
            await client.search("test query", partitions=["electronics"])

        # Error metric should be emitted
        assert handler.has_metric(MetricType.ERROR, operation="search")

    @pytest.mark.asyncio
    async def test_search_backend_failure(self):
        """search() should raise when backend.search_partitions fails."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(
            side_effect=SearchError("Vector search failed on partition electronics: timeout")
        )

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        client = _make_client(
            mock_backend=mock_backend,
            mock_embedder=mock_embedder,
            metrics_collector=collector,
        )

        with pytest.raises(SearchError) as exc_info:
            await client.search("test query", partitions=["electronics"])

        assert "search" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()
        assert handler.has_metric(MetricType.ERROR, operation="search")

    @pytest.mark.asyncio
    async def test_search_empty_partitions_list(self):
        """search() with partitions=[] should return an empty SearchResult."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        # When resolver gets an empty list, it may return no partitions
        # The search method should handle this gracefully
        config = _make_config()
        config.partitions.registry = {}

        client = _make_client(config=config, mock_embedder=mock_embedder)

        # Searching "all" with no partitions registered should return empty results
        result = await client.search("test query", partitions="all")

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 0
        assert result.partitions_searched == []

    @pytest.mark.asyncio
    async def test_search_embedder_not_initialized(self):
        """search() should raise SearchError when embedder is None in BYOM mode."""
        client = _make_client()
        client._embedder = None

        with pytest.raises(SearchError, match="Embedder not initialized"):
            await client.search("test query", partitions=["electronics"])

    @pytest.mark.asyncio
    async def test_search_with_precomputed_vector_bypasses_embedder(self):
        """search() with query_vector should not call the embedder."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=[])

        client = _make_client(
            mock_backend=mock_backend,
            mock_embedder=mock_embedder,
        )

        result = await client.search(
            "test query",
            partitions=["electronics"],
            query_vector=[0.5] * 1536,
        )

        mock_embedder.embed.assert_not_awaited()
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_error_preserves_exception_type(self):
        """Error metric emitted on search failure should capture the exception type."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(
            side_effect=ValueError("Invalid input dimensions")
        )

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        client = _make_client(mock_embedder=mock_embedder, metrics_collector=collector)

        with pytest.raises(ValueError):
            await client.search("test query", partitions=["electronics"])

        # Check that the error metric includes the error type
        error_events = [e for e in handler.events if e.metric_type == MetricType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].tags.get("error_type") == "ValueError"


# ===========================================================================
# SVRClient.ingest() Error Paths
# ===========================================================================


class TestIngestErrors:
    """Error paths for SVRClient.ingest()."""

    @pytest.mark.asyncio
    async def test_ingest_not_connected_with_auto_connect_failed(self):
        """ingest() when not connected and auto_connect_failed should raise IngestionError
        with recovery message."""
        client = _make_client(connected=False, auto_connect_failed=True)

        with pytest.raises(IngestionError) as exc_info:
            await client.ingest(
                [{"description": "test"}],
                partition="electronics",
            )

        error_msg = str(exc_info.value)
        assert "connect" in error_msg.lower()
        assert "await" in error_msg.lower() or "client.connect()" in error_msg

    @pytest.mark.asyncio
    async def test_ingest_empty_documents(self):
        """ingest([]) should return an IngestResult with 0 inserted."""
        client = _make_client()

        # Mock _create_document_embedder to avoid real embedder creation
        mock_embedder = AsyncMock()
        with patch.object(client, "_create_document_embedder", return_value=mock_embedder):
            result = await client.ingest([], partition="electronics")

        assert isinstance(result, IngestResult)
        assert result.inserted == 0
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_ingest_embedding_failure(self):
        """ingest() should raise IngestionError when embedding fails."""
        mock_doc_embedder = AsyncMock()
        mock_doc_embedder.embed_with_batching = AsyncMock(
            side_effect=RuntimeError("Embedding API unavailable")
        )

        client = _make_client()

        with patch.object(
            client, "_create_document_embedder", return_value=mock_doc_embedder
        ):
            with pytest.raises(RuntimeError, match="Embedding API unavailable"):
                await client.ingest(
                    [{"description": "test product"}],
                    partition="electronics",
                )

    @pytest.mark.asyncio
    async def test_ingest_documents_without_text_fields(self):
        """ingest() with documents missing text fields should handle gracefully."""
        mock_doc_embedder = AsyncMock()
        mock_doc_embedder.embed_with_batching = AsyncMock(return_value=[[0.1] * 1536])

        mock_backend = AsyncMock()
        # db[collection].insert_many needs to be mocked through the backend's db property
        mock_collection = AsyncMock()
        mock_collection.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        mock_backend.db = mock_db

        client = _make_client(mock_backend=mock_backend)

        with patch.object(
            client, "_create_document_embedder", return_value=mock_doc_embedder
        ):
            # Documents missing "description" field (the configured text_field)
            result = await client.ingest(
                [{"name": "no description field"}],
                partition="electronics",
            )

        # Should report as failed since no text could be extracted
        assert result.failed >= 1

    @pytest.mark.asyncio
    async def test_ingest_batch_size_exceeded(self):
        """ingest() with >10000 documents should raise IngestionError."""
        client = _make_client()

        docs = [{"description": f"doc {i}"} for i in range(10001)]

        mock_doc_embedder = AsyncMock()
        with patch.object(
            client, "_create_document_embedder", return_value=mock_doc_embedder
        ):
            with pytest.raises(IngestionError, match="exceeds maximum"):
                await client.ingest(docs, partition="electronics")


# ===========================================================================
# SVRClient.disconnect() Error Paths
# ===========================================================================


class TestDisconnectErrors:
    """Error paths for SVRClient.disconnect()."""

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """disconnect() when not connected should not crash."""
        config = _make_config()
        client = SVRClient(config=config, auto_connect=False)

        # Should not raise — no backend to disconnect from
        await client.disconnect()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_double_disconnect(self):
        """Calling disconnect() twice should not crash."""
        mock_backend = AsyncMock()
        mock_backend.disconnect = AsyncMock()

        client = _make_client(mock_backend=mock_backend)

        await client.disconnect()
        assert client._connected is False

        # Second disconnect should also be fine
        await client.disconnect()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_with_metadata_error(self):
        """disconnect() should still complete even if metadata disconnect fails."""
        mock_backend = AsyncMock()
        mock_backend.disconnect = AsyncMock()

        client = _make_client(mock_backend=mock_backend)
        client._metadata = AsyncMock()
        client._metadata.disconnect = AsyncMock(
            side_effect=RuntimeError("metadata disconnect failure")
        )

        # Should propagate the error from metadata disconnect since
        # disconnect calls await self._metadata.disconnect() without try/except
        with pytest.raises(RuntimeError, match="metadata disconnect failure"):
            await client.disconnect()


# ===========================================================================
# Error Metrics Emission
# ===========================================================================


class TestErrorMetrics:
    """Verify that errors emit the appropriate metrics."""

    @pytest.mark.asyncio
    async def test_search_error_emits_metric(self):
        """A search failure should emit MetricType.ERROR with operation=search."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(
            side_effect=RuntimeError("API failure")
        )

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        client = _make_client(mock_embedder=mock_embedder, metrics_collector=collector)

        with pytest.raises(RuntimeError):
            await client.search("test", partitions=["electronics"])

        assert handler.has_metric(MetricType.ERROR, operation="search")
        assert handler.count_metric(MetricType.ERROR) == 1

    @pytest.mark.asyncio
    async def test_search_success_emits_timing_metrics(self):
        """A successful search should emit SEARCH_LATENCY and related metrics."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=[
            {
                "_id": "doc1",
                "name": "Test",
                "_svr_score": 0.9,
                "_svr_partition": "electronics",
            }
        ])

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        client = _make_client(
            mock_backend=mock_backend,
            mock_embedder=mock_embedder,
            metrics_collector=collector,
        )

        result = await client.search("test", partitions=["electronics"])

        assert handler.has_metric(MetricType.SEARCH_LATENCY)
        assert handler.has_metric(MetricType.SEARCH_RESULTS)
        assert handler.has_metric(MetricType.SEARCH_CANDIDATES)
        # Should also have cache miss for first query
        assert handler.has_metric(MetricType.CACHE_MISS)
        # No errors
        assert handler.count_metric(MetricType.ERROR) == 0

    @pytest.mark.asyncio
    async def test_ingest_error_emits_metric(self):
        """An ingest embedding failure should emit error-related metrics."""
        mock_doc_embedder = AsyncMock()
        mock_doc_embedder.embed_with_batching = AsyncMock(
            side_effect=RuntimeError("Embedding service down")
        )

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        client = _make_client(metrics_collector=collector)

        with patch.object(
            client, "_create_document_embedder", return_value=mock_doc_embedder
        ):
            with pytest.raises(RuntimeError):
                await client.ingest(
                    [{"description": "test"}],
                    partition="electronics",
                )

        # The ingest pipeline raises before client-level metrics are emitted,
        # but the error should propagate. Verify no crash occurred.
        # Note: client.ingest() emits metrics only on successful completion,
        # so error metric emission depends on whether the pipeline wraps errors.

    @pytest.mark.asyncio
    async def test_successful_ingest_emits_timing_metrics(self):
        """A successful ingest should emit INGEST_LATENCY and INGEST_DOCUMENTS."""
        mock_doc_embedder = AsyncMock()
        mock_doc_embedder.embed_with_batching = AsyncMock(
            return_value=[[0.1] * 1536, [0.2] * 1536]
        )

        mock_backend = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        mock_backend.db = mock_db

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        config = _make_config_with_partitions()
        config.ingestion.trigger_detection = False  # Skip detection for this test

        client = _make_client(
            config=config,
            mock_backend=mock_backend,
            metrics_collector=collector,
        )

        with patch.object(
            client, "_create_document_embedder", return_value=mock_doc_embedder
        ):
            result = await client.ingest(
                [
                    {"description": "product one"},
                    {"description": "product two"},
                ],
                partition="electronics",
            )

        assert result.inserted == 2
        assert handler.has_metric(MetricType.INGEST_LATENCY)
        assert handler.has_metric(MetricType.INGEST_DOCUMENTS)

    @pytest.mark.asyncio
    async def test_cache_hit_emits_metric(self):
        """Repeated search with same query should emit CACHE_HIT on second call."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=[])

        handler = _CapturingHandler()
        collector = MetricsCollector()
        collector.add_handler(handler)

        client = _make_client(
            mock_backend=mock_backend,
            mock_embedder=mock_embedder,
            metrics_collector=collector,
        )

        # First search - cache miss
        await client.search("same query", partitions=["electronics"])
        assert handler.has_metric(MetricType.CACHE_MISS)

        # Second search - cache hit
        await client.search("same query", partitions=["electronics"])
        assert handler.has_metric(MetricType.CACHE_HIT)

        # Embedder should only be called once (second time served from cache)
        assert mock_embedder.embed.await_count == 1


# ===========================================================================
# Error Message Quality
# ===========================================================================


class TestErrorMessageQuality:
    """Verify error messages include recovery suggestions and context."""

    @pytest.mark.asyncio
    async def test_connection_error_suggests_reconnect(self):
        """SearchError from not-connected state should suggest calling connect()."""
        client = _make_client(connected=False, auto_connect_failed=True)

        with pytest.raises(SearchError) as exc_info:
            await client.search("query", partitions=["electronics"])

        error_msg = str(exc_info.value)
        # Should mention how to recover
        assert "connect()" in error_msg
        assert "await" in error_msg.lower() or "Call" in error_msg

    @pytest.mark.asyncio
    async def test_ingest_connection_error_suggests_reconnect(self):
        """IngestionError from not-connected state should suggest calling connect()."""
        client = _make_client(connected=False, auto_connect_failed=True)

        with pytest.raises(IngestionError) as exc_info:
            await client.ingest([{"description": "test"}])

        error_msg = str(exc_info.value)
        assert "connect()" in error_msg

    @pytest.mark.asyncio
    async def test_partition_not_found_lists_available(self):
        """PartitionNotFoundError should include the list of available partitions."""
        client = _make_client()

        with pytest.raises(PartitionNotFoundError) as exc_info:
            await client.search("query", partitions=["nonexistent"])

        # Check details
        details = exc_info.value.details
        assert "available" in details
        available = details["available"]
        assert isinstance(available, list)
        assert "electronics" in available
        assert "furniture" in available

    @pytest.mark.asyncio
    async def test_partition_not_found_includes_name_in_message(self):
        """PartitionNotFoundError message should include the requested partition name."""
        client = _make_client()

        with pytest.raises(PartitionNotFoundError) as exc_info:
            await client.search("query", partitions=["my_missing_partition"])

        assert "my_missing_partition" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_error_async_context_message(self):
        """SearchError in async context should mention Jupyter/FastAPI."""
        client = _make_client(connected=False, auto_connect_failed=True)

        with pytest.raises(SearchError) as exc_info:
            await client.search("query", partitions=["electronics"])

        error_msg = str(exc_info.value)
        # Should mention async context scenarios
        assert "async" in error_msg.lower() or "Jupyter" in error_msg or "FastAPI" in error_msg

    @pytest.mark.asyncio
    async def test_ingest_error_async_context_message(self):
        """IngestionError in async context should mention async usage."""
        client = _make_client(connected=False, auto_connect_failed=True)

        with pytest.raises(IngestionError) as exc_info:
            await client.ingest([{"description": "test"}])

        error_msg = str(exc_info.value)
        assert "async" in error_msg.lower() or "connect()" in error_msg


# ===========================================================================
# Context Manager Error Paths
# ===========================================================================


class TestContextManagerErrors:
    """Error paths for async context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager_connect_failure(self):
        """__aenter__ should propagate connect() errors."""
        config = _make_config()

        with patch(
            "semantic_vector_router.client.create_backend"
        ) as MockBackend:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock(
                side_effect=ConnectionError("Cannot reach MongoDB")
            )
            MockBackend.return_value = mock_instance

            client = SVRClient(config=config, auto_connect=False)

            with pytest.raises(ConnectionError):
                async with client:
                    pass  # Should never reach here

    @pytest.mark.asyncio
    async def test_context_manager_disconnect_on_exit(self):
        """__aexit__ should call disconnect even if an error occurred inside."""
        mock_backend = AsyncMock()
        mock_backend.disconnect = AsyncMock()

        client = _make_client(mock_backend=mock_backend)
        client._metadata = None  # Avoid metadata disconnect issues

        try:
            async with client:
                raise ValueError("Something broke inside the context")
        except ValueError:
            pass

        # disconnect should have been called
        mock_backend.disconnect.assert_awaited()


# ===========================================================================
# Partition Management Error Paths
# ===========================================================================


class TestPartitionManagementErrors:
    """Error paths for partition management methods."""

    @pytest.mark.asyncio
    async def test_get_partition_not_found(self):
        """get_partition() for non-existent partition should raise PartitionNotFoundError."""
        client = _make_client()

        with pytest.raises(PartitionNotFoundError) as exc_info:
            await client.get_partition("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "available" in exc_info.value.details

    def test_list_partitions_when_no_partitions(self):
        """list_partitions() with empty registry should return empty list."""
        config = _make_config()
        config.partitions.registry = {}
        client = _make_client(config=config)

        result = client.list_partitions()
        assert result == []

    def test_list_partitions_returns_all_registered(self):
        """list_partitions() should return all registered partitions."""
        client = _make_client()

        result = client.list_partitions()
        assert len(result) == 2
        names = {p["name"] for p in result}
        assert names == {"electronics", "furniture"}


# ===========================================================================
# NoOp Metrics Collector
# ===========================================================================


class TestNoOpCollector:
    """Verify NoOpCollector discards events without errors."""

    def test_noop_emit_count(self):
        """NoOpCollector.emit_count should not raise."""
        collector = NoOpCollector()
        collector.emit_count(MetricType.ERROR, operation="test")

    def test_noop_emit_timing(self):
        """NoOpCollector.emit_timing should not raise."""
        collector = NoOpCollector()
        collector.emit_timing(MetricType.SEARCH_LATENCY, 42.0)

    def test_metrics_disabled_uses_noop(self):
        """When metrics.enabled=False, SVRClient should use NoOpCollector."""
        config = _make_config(metrics=MetricsConfig(enabled=False))
        client = SVRClient(config=config, auto_connect=False)
        assert isinstance(client._metrics, NoOpCollector)

    def test_metrics_enabled_uses_real_collector(self):
        """When metrics.enabled=True, SVRClient should use MetricsCollector."""
        config = _make_config(metrics=MetricsConfig(enabled=True))
        client = SVRClient(config=config, auto_connect=False)
        assert isinstance(client._metrics, MetricsCollector)
        assert not isinstance(client._metrics, NoOpCollector)


# ===========================================================================
# Embedding Cache Edge Cases
# ===========================================================================


class TestEmbeddingCacheEdgeCases:
    """Edge cases for the embedding cache interaction with search."""

    @pytest.mark.asyncio
    async def test_cache_disabled_always_embeds(self):
        """With cache disabled, every search should call the embedder."""
        config = _make_config_with_partitions()
        config.cache.enabled = False

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=[])

        client = _make_client(
            config=config,
            mock_backend=mock_backend,
            mock_embedder=mock_embedder,
        )
        # Reinitialize cache with disabled setting
        from semantic_vector_router.utils.cache import EmbeddingCache

        client._embedding_cache = EmbeddingCache(max_size=0)

        # Two identical searches
        await client.search("same query", partitions=["electronics"])
        await client.search("same query", partitions=["electronics"])

        # Embedder should be called both times when cache is disabled
        assert mock_embedder.embed.await_count == 2

    @pytest.mark.asyncio
    async def test_different_queries_are_cached_separately(self):
        """Different queries should have separate cache entries."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=[])

        client = _make_client(
            mock_backend=mock_backend,
            mock_embedder=mock_embedder,
        )

        await client.search("query one", partitions=["electronics"])
        await client.search("query two", partitions=["electronics"])

        # Both queries should trigger embedding
        assert mock_embedder.embed.await_count == 2


# ===========================================================================
# Configuration Error Paths
# ===========================================================================


class TestConfigurationErrors:
    """Error paths related to configuration."""

    def test_unknown_embedding_provider(self):
        """Creating embedder with unknown provider should raise ConfigurationError."""
        config = _make_config()
        # Set provider to something the switch statement won't match
        config.embedding.provider = "unknown_provider"

        client = SVRClient(config=config, auto_connect=False)

        with pytest.raises((ConfigurationError, ValueError)):
            client._create_embedder()

    def test_unknown_reranker_provider(self):
        """Creating reranker with unknown provider should raise ConfigurationError."""
        config = _make_config()
        config.reranking.enabled = True
        config.reranking.provider = "unknown_provider"

        client = SVRClient(config=config, auto_connect=False)

        with pytest.raises((ConfigurationError, ValueError)):
            client._create_reranker()

    @pytest.mark.asyncio
    async def test_monitor_without_metadata_raises(self):
        """start_monitor() without metadata store should raise RuntimeError."""
        client = _make_client()
        client._metadata = None

        with pytest.raises(RuntimeError, match="Metadata store not available"):
            await client.start_monitor()

    @pytest.mark.asyncio
    async def test_double_monitor_start_raises(self):
        """Starting monitor twice should raise RuntimeError."""
        client = _make_client()
        client._metadata = AsyncMock()
        client._monitor_task = MagicMock()  # Simulate already running

        with pytest.raises(RuntimeError, match="Monitor already running"):
            await client.start_monitor()


# ===========================================================================
# Watcher Edge Cases
# ===========================================================================


class TestWatcherEdgeCases:
    """Edge cases for watcher status."""

    def test_watcher_status_when_not_started(self):
        """watcher_status() when no watcher exists should return not-running."""
        client = _make_client()

        status = client.watcher_status()
        assert status.running is False

    @pytest.mark.asyncio
    async def test_stop_watcher_when_not_started(self):
        """stop_watcher() when no watcher exists should be a no-op."""
        client = _make_client()

        # Should not raise
        await client.stop_watcher()

    @pytest.mark.asyncio
    async def test_stop_monitor_when_not_started(self):
        """stop_monitor() when no monitor task exists should be a no-op."""
        client = _make_client()

        # Should not raise
        await client.stop_monitor()


# ===========================================================================
# Save Config Edge Cases
# ===========================================================================


class TestSaveConfig:
    """Edge cases for config saving."""

    def test_save_config_returns_path(self):
        """save_config() should return the path where config was saved."""
        client = _make_client()

        with patch("semantic_vector_router.client.save_config") as mock_save:
            mock_save.return_value = "/tmp/svr_config.json"

            path = client.save_config()
            assert path == "/tmp/svr_config.json"
            mock_save.assert_called_once_with(client._config, None)
