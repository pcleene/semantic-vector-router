"""Unit tests for SVRClient.search() partition defaulting behavior.

Verifies that the search() method correctly passes partition specifications
through to PartitionResolver.resolve(), and that the resolver's output
drives downstream behavior (backend search, result merging, empty results).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.client import SVRClient
from semantic_vector_router.exceptions import SearchError
from semantic_vector_router.models import (
    EmbeddingMode,
    PartitionInfo,
    PartitionStatus,
    SearchHit,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_partition(name: str, status: PartitionStatus = PartitionStatus.ACTIVE) -> PartitionInfo:
    """Create a PartitionInfo with sensible defaults for testing."""
    return PartitionInfo(
        name=name,
        view_name=f"svr_partition_{name}",
        index_name=f"svr_idx_{name}",
        filter_value=name,
        document_count=1000,
        status=status,
    )


def _raw_results_for(partitions: list[str]) -> list[dict]:
    """Build raw backend result dicts for the given partition names."""
    results = []
    for i, p in enumerate(partitions):
        results.append({
            "_id": f"doc_{p}_{i}",
            "name": f"Item from {p}",
            "price": 10.0 + i,
            "_svr_score": 0.90 - (i * 0.05),
            "_svr_partition": p,
        })
    return results


def _make_client():
    """Create an SVRClient with mocked internals for search testing.

    The client is instantiated with a minimal config dict to satisfy
    load_config, then all internal components are replaced with mocks.
    """
    with patch("semantic_vector_router.client.load_config") as mock_load, \
         patch("semantic_vector_router.client.validate_config", return_value=[]):
        mock_config = MagicMock()
        mock_config.metrics.enabled = False
        mock_config.rate_limiting.enabled = False
        mock_config.cache.enabled = True
        mock_config.cache.max_size = 100
        mock_config.cache.ttl_seconds = 60
        # Use MagicMock for embedding mode - don't assign real enum values
        mock_config.embedding.mode = EmbeddingMode.BYOM
        mock_config.embedding.provider = MagicMock()
        mock_config.embedding.provider.value = "openai"
        mock_config.embedding.model = "text-embedding-3-small"
        mock_config.embedding.dimensions = 1536
        mock_config.reranking.enabled = False
        mock_config.reranking.top_k_per_partition = 20
        mock_config.routing.default_partitions = "all"
        mock_config.routing.centroid_routing.enabled = False
        mock_config.events.enabled = False
        mock_config.scheduler.enabled = False
        mock_load.return_value = mock_config

        client = SVRClient(
            config={
                "database": {"database": "test", "source_collection": "test"},
                "partitioning": {"field": "category"},
            },
            auto_connect=False,
        )

    # Replace internals with mocks
    client._connected = True
    client._config = mock_config

    # Resolver mock
    client._resolver = AsyncMock()
    client._resolver.resolve = AsyncMock(return_value=[])

    # Backend mock
    client._backend = AsyncMock()
    client._backend.search_partitions = AsyncMock(return_value=[])

    # Embedder mock (BYOM mode)
    client._embedder = AsyncMock()
    client._embedder.embed = AsyncMock(return_value=[0.1] * 1536)

    # Merger mock -- use a real ResultMerger so merge() returns SearchHit objects
    from semantic_vector_router.routing.merger import ResultMerger
    client._merger = ResultMerger()

    # Metrics (NoOp-like mock)
    client._metrics = MagicMock()
    client._metrics.emit_timing = MagicMock()
    client._metrics.emit_count = MagicMock()

    # Embedding cache that always misses
    client._embedding_cache = MagicMock()
    client._embedding_cache.get = MagicMock(return_value=None)
    client._embedding_cache.put = MagicMock()

    return client


# ---------------------------------------------------------------------------
# Tests — partition argument passthrough to resolver
# ---------------------------------------------------------------------------


class TestPartitionsPassthrough:
    """Verify that search() passes the partitions argument through to
    resolver.resolve() exactly as provided by the caller."""

    @pytest.mark.asyncio
    async def test_partitions_none_calls_resolver_with_none(self):
        """search(query) with partitions=None passes None to resolver."""
        client = _make_client()
        client._resolver.resolve = AsyncMock(return_value=[])

        await client.search(query="wireless headphones")

        client._resolver.resolve.assert_awaited_once()
        call_args = client._resolver.resolve.call_args
        # First positional arg should be None
        assert call_args.args[0] is None or call_args.kwargs.get("partitions") is None

    @pytest.mark.asyncio
    async def test_partitions_all_calls_resolver_with_all(self):
        """search(query, partitions='all') passes 'all' to resolver."""
        client = _make_client()
        client._resolver.resolve = AsyncMock(return_value=[])

        await client.search(query="wireless headphones", partitions="all")

        client._resolver.resolve.assert_awaited_once()
        call_args = client._resolver.resolve.call_args
        assert call_args.args[0] == "all"

    @pytest.mark.asyncio
    async def test_partitions_single_string_calls_resolver(self):
        """search(query, partitions='electronics') passes string to resolver."""
        client = _make_client()
        partition = _make_partition("electronics")
        raw = _raw_results_for(["electronics"])
        client._resolver.resolve = AsyncMock(return_value=[partition])
        client._backend.search_partitions = AsyncMock(return_value=raw)

        await client.search(query="wireless headphones", partitions="electronics")

        client._resolver.resolve.assert_awaited_once()
        call_args = client._resolver.resolve.call_args
        assert call_args.args[0] == "electronics"

    @pytest.mark.asyncio
    async def test_partitions_list_calls_resolver(self):
        """search(query, partitions=['a', 'b']) passes list to resolver."""
        client = _make_client()
        partitions = [_make_partition("a"), _make_partition("b")]
        raw = _raw_results_for(["a", "b"])
        client._resolver.resolve = AsyncMock(return_value=partitions)
        client._backend.search_partitions = AsyncMock(return_value=raw)

        await client.search(query="test query", partitions=["a", "b"])

        client._resolver.resolve.assert_awaited_once()
        call_args = client._resolver.resolve.call_args
        assert call_args.args[0] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_resolver_receives_filters_kwarg(self):
        """search() passes filters through to resolver.resolve()."""
        client = _make_client()
        client._resolver.resolve = AsyncMock(return_value=[])
        test_filters = {"price": {"$gt": 100}}

        await client.search(query="query", filters=test_filters)

        call_kwargs = client._resolver.resolve.call_args.kwargs
        assert call_kwargs.get("filters") == test_filters

    @pytest.mark.asyncio
    async def test_resolver_receives_query_embedding(self):
        """search() passes the computed query embedding to resolver for
        centroid routing."""
        client = _make_client()
        client._resolver.resolve = AsyncMock(return_value=[])

        await client.search(query="test query")

        call_kwargs = client._resolver.resolve.call_args.kwargs
        # The embedder produces [0.1]*1536
        assert call_kwargs.get("query_embedding") == [0.1] * 1536


# ---------------------------------------------------------------------------
# Tests — empty vs populated resolver results
# ---------------------------------------------------------------------------


class TestEmptyPartitions:
    """Verify correct behavior when no partitions are resolved."""

    @pytest.mark.asyncio
    async def test_no_partitions_returns_empty_search_result(self):
        """When resolver returns empty list, search returns empty SearchResult."""
        client = _make_client()
        client._resolver.resolve = AsyncMock(return_value=[])

        result = await client.search(query="nothing here")

        assert isinstance(result, SearchResult)
        assert result.hits == []
        assert result.partitions_searched == []
        assert result.total_candidates == 0
        assert result.reranked is False
        assert result.query == "nothing here"

    @pytest.mark.asyncio
    async def test_no_partitions_does_not_call_backend(self):
        """When resolver returns empty list, backend.search_partitions
        is never called."""
        client = _make_client()
        client._resolver.resolve = AsyncMock(return_value=[])

        await client.search(query="nothing here")

        client._backend.search_partitions.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests — result flow with resolved partitions
# ---------------------------------------------------------------------------


class TestSearchWithPartitions:
    """Verify end-to-end flow when partitions are resolved."""

    @pytest.mark.asyncio
    async def test_single_partition_returns_results(self):
        """Searching a single partition returns hits from that partition."""
        client = _make_client()
        partition = _make_partition("electronics")
        raw = [
            {
                "_id": "doc1",
                "name": "Wireless Headphones",
                "_svr_score": 0.95,
                "_svr_partition": "electronics",
            },
        ]
        client._resolver.resolve = AsyncMock(return_value=[partition])
        client._backend.search_partitions = AsyncMock(return_value=raw)

        result = await client.search(query="headphones", partitions="electronics")

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1
        assert result.hits[0].partition == "electronics"
        assert result.partitions_searched == ["electronics"]
        assert result.total_candidates == 1
        assert result.query == "headphones"

    @pytest.mark.asyncio
    async def test_multiple_partitions_returns_merged_results(self):
        """Searching multiple partitions returns merged and sorted hits."""
        client = _make_client()
        partitions = [_make_partition("electronics"), _make_partition("furniture")]
        raw = [
            {
                "_id": "doc1",
                "name": "Wireless Headphones",
                "_svr_score": 0.95,
                "_svr_partition": "electronics",
            },
            {
                "_id": "doc2",
                "name": "Office Chair",
                "_svr_score": 0.72,
                "_svr_partition": "furniture",
            },
        ]
        client._resolver.resolve = AsyncMock(return_value=partitions)
        client._backend.search_partitions = AsyncMock(return_value=raw)

        result = await client.search(
            query="office products",
            partitions=["electronics", "furniture"],
            limit=10,
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 2
        assert result.partitions_searched == ["electronics", "furniture"]
        assert result.total_candidates == 2
        # Hits should be sorted by score descending
        assert result.hits[0].score >= result.hits[1].score

    @pytest.mark.asyncio
    async def test_backend_receives_resolved_partitions(self):
        """Backend.search_partitions receives the PartitionInfo objects
        returned by the resolver, not the raw user input."""
        client = _make_client()
        partition_a = _make_partition("alpha")
        partition_b = _make_partition("beta")
        client._resolver.resolve = AsyncMock(return_value=[partition_a, partition_b])
        client._backend.search_partitions = AsyncMock(return_value=[])

        await client.search(query="test", partitions=["alpha", "beta"])

        client._backend.search_partitions.assert_awaited_once()
        call_kwargs = client._backend.search_partitions.call_args.kwargs
        assert call_kwargs["partitions"] == [partition_a, partition_b]

    @pytest.mark.asyncio
    async def test_limit_is_respected(self):
        """The limit parameter caps the number of returned hits."""
        client = _make_client()
        partition = _make_partition("electronics")
        # Backend returns 5 results but limit is 2
        raw = [
            {
                "_id": f"doc{i}",
                "name": f"Item {i}",
                "_svr_score": 0.90 - (i * 0.05),
                "_svr_partition": "electronics",
            }
            for i in range(5)
        ]
        client._resolver.resolve = AsyncMock(return_value=[partition])
        client._backend.search_partitions = AsyncMock(return_value=raw)

        result = await client.search(query="items", partitions="electronics", limit=2)

        assert len(result.hits) <= 2

    @pytest.mark.asyncio
    async def test_candidates_per_partition_override(self):
        """candidates_per_partition is forwarded to backend search."""
        client = _make_client()
        partition = _make_partition("electronics")
        client._resolver.resolve = AsyncMock(return_value=[partition])
        client._backend.search_partitions = AsyncMock(return_value=[])

        await client.search(
            query="test",
            partitions="electronics",
            candidates_per_partition=50,
        )

        call_kwargs = client._backend.search_partitions.call_args.kwargs
        assert call_kwargs["limit"] == 50


# ---------------------------------------------------------------------------
# Tests — query vector passthrough
# ---------------------------------------------------------------------------


class TestQueryVector:
    """Verify behavior when a pre-computed query_vector is provided."""

    @pytest.mark.asyncio
    async def test_precomputed_vector_skips_embedder(self):
        """When query_vector is provided, the embedder is not called."""
        client = _make_client()
        partition = _make_partition("electronics")
        client._resolver.resolve = AsyncMock(return_value=[partition])
        client._backend.search_partitions = AsyncMock(return_value=[])
        precomputed = [0.5] * 1536

        await client.search(
            query="test",
            partitions="electronics",
            query_vector=precomputed,
        )

        client._embedder.embed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_precomputed_vector_sent_to_backend(self):
        """When query_vector is provided, it is forwarded to backend."""
        client = _make_client()
        partition = _make_partition("electronics")
        client._resolver.resolve = AsyncMock(return_value=[partition])
        client._backend.search_partitions = AsyncMock(return_value=[])
        precomputed = [0.5] * 1536

        await client.search(
            query="test",
            partitions="electronics",
            query_vector=precomputed,
        )

        call_kwargs = client._backend.search_partitions.call_args.kwargs
        assert call_kwargs["query_vector"] == precomputed

    @pytest.mark.asyncio
    async def test_precomputed_vector_passed_to_resolver_for_centroid(self):
        """When query_vector is provided, it is passed to resolver as
        query_embedding for centroid routing."""
        client = _make_client()
        client._resolver.resolve = AsyncMock(return_value=[])
        precomputed = [0.5] * 1536

        await client.search(query="test", query_vector=precomputed)

        call_kwargs = client._resolver.resolve.call_args.kwargs
        assert call_kwargs.get("query_embedding") == precomputed


# ---------------------------------------------------------------------------
# Tests — search_sync wrapper
# ---------------------------------------------------------------------------


class TestSearchSync:
    """Verify the synchronous search_sync() wrapper."""

    def test_search_sync_delegates_to_search(self):
        """search_sync() calls the async search() via event loop."""
        client = _make_client()
        # Mock search() to return a known SearchResult
        expected = SearchResult(
            hits=[],
            query="sync query",
            partitions_searched=[],
            total_candidates=0,
            reranked=False,
            latency_ms=1.0,
        )
        # Use a real coroutine function so asyncio.run() can execute it
        async def mock_search(**kwargs):
            return expected

        client.search = mock_search

        result = client.search_sync(query="sync query", limit=5)

        assert isinstance(result, SearchResult)
        assert result.query == "sync query"

    def test_search_sync_passes_partitions(self):
        """search_sync() forwards the partitions argument."""
        client = _make_client()
        captured_kwargs = {}

        async def mock_search(**kwargs):
            captured_kwargs.update(kwargs)
            return SearchResult(
                hits=[],
                query="sync query",
                partitions_searched=["electronics"],
                total_candidates=0,
                reranked=False,
                latency_ms=1.0,
            )

        client.search = mock_search

        result = client.search_sync(
            query="sync query",
            partitions="electronics",
            limit=10,
        )

        assert captured_kwargs["partitions"] == "electronics"
        assert captured_kwargs["query"] == "sync query"
        assert captured_kwargs["limit"] == 10


# ---------------------------------------------------------------------------
# Tests — error conditions
# ---------------------------------------------------------------------------


class TestSearchErrors:
    """Verify error paths in the search method."""

    @pytest.mark.asyncio
    async def test_not_connected_auto_connect_failed_raises(self):
        """When client is not connected and auto_connect_failed is True,
        search() raises SearchError."""
        client = _make_client()
        client._connected = False
        client._auto_connect_failed = True

        with pytest.raises(SearchError, match="Client not connected"):
            await client.search(query="test")

    @pytest.mark.asyncio
    async def test_no_embedder_in_byom_mode_raises(self):
        """When embedding mode is BYOM but embedder is None, raises SearchError."""
        client = _make_client()
        client._embedder = None
        partition = _make_partition("electronics")
        client._resolver.resolve = AsyncMock(return_value=[partition])

        with pytest.raises(SearchError, match="Embedder not initialized"):
            await client.search(query="test", partitions="electronics")

    @pytest.mark.asyncio
    async def test_metrics_emit_on_search_error(self):
        """When search raises, an error metric is emitted."""
        client = _make_client()
        # Force embedder to raise
        client._embedder.embed = AsyncMock(side_effect=RuntimeError("embed fail"))

        with pytest.raises(RuntimeError, match="embed fail"):
            await client.search(query="will fail")

        client._metrics.emit_count.assert_called()
        # Find the error metric call
        error_calls = [
            c for c in client._metrics.emit_count.call_args_list
            if len(c.args) > 0 and hasattr(c.args[0], 'value') and c.args[0].value == "error"
        ]
        assert len(error_calls) > 0


# ---------------------------------------------------------------------------
# Tests — embedding cache interaction
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    """Verify that the embedding cache is used during search."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_embedder(self):
        """When embedding cache returns a cached vector, embedder is not called."""
        client = _make_client()
        cached_vector = [0.2] * 1536
        client._embedding_cache.get = MagicMock(return_value=cached_vector)
        client._resolver.resolve = AsyncMock(return_value=[])

        await client.search(query="cached query")

        client._embedder.embed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_embedder_and_stores(self):
        """When embedding cache misses, embedder is called and result cached."""
        client = _make_client()
        client._embedding_cache.get = MagicMock(return_value=None)
        client._resolver.resolve = AsyncMock(return_value=[])

        await client.search(query="uncached query")

        client._embedder.embed.assert_awaited_once_with("uncached query")
        client._embedding_cache.put.assert_called_once()
