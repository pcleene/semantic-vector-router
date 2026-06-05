"""Unit tests for Phase 10 Unit D centroid routing wiring.

Validates the integration points where centroid routing is wired into
existing components:
1. PartitionResolver centroid cascade
2. SVRClient.search() embedding reorder
3. SVRClient.ingest() post-ingest centroid computation
4. RepartitionEngine compute_centroids step
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.models import (
    CentroidRoutingConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RoutingConfig,
    RoutingMode,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)
from semantic_vector_router.routing.resolver import PartitionResolver
from semantic_vector_router.utils.metrics import MetricsCollector, MetricType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    centroid_enabled: bool = False,
    embedding_mode: EmbeddingMode = EmbeddingMode.BYOM,
    **centroid_kwargs,
) -> SVRConfig:
    """Build a minimal SVRConfig with centroid routing control."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(
            field="category",
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=4,
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=embedding_mode,
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key_env="OPENAI_API_KEY",
            dimensions=4,
        ),
        routing=RoutingConfig(
            mode=RoutingMode.EXPLICIT,
            default_partitions="all",
            max_partitions_per_query=10,
            centroid_routing=CentroidRoutingConfig(
                enabled=centroid_enabled,
                relative_threshold=0.5,
                min_score=0.15,
                max_probe_partitions=5,
                sample_size=100,
                **centroid_kwargs,
            ),
        ),
    )


def _partition(
    name: str,
    *,
    status: PartitionStatus = PartitionStatus.ACTIVE,
    centroid: list[float] | None = None,
    parent: str | None = None,
    children: list[str] | None = None,
    filter_value: str | None = None,
    embedding_field: str | None = None,
) -> PartitionInfo:
    """Build a PartitionInfo with sensible defaults."""
    return PartitionInfo(
        name=name,
        view_name=f"svr_partition_{name}",
        index_name=f"svr_idx_{name}",
        filter_value=filter_value or name,
        status=status,
        centroid=centroid,
        parent_partition=parent,
        child_partitions=children or [],
        embedding_field=embedding_field,
    )


def _register(config: SVRConfig, *parts: PartitionInfo) -> SVRConfig:
    """Register partitions in config and return it."""
    for p in parts:
        config.partitions.registry[p.name] = p
    return config


# Small 4-d vectors for deterministic cosine similarity
VEC_A = [1.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0]
VEC_QUERY = [0.9, 0.1, 0.0, 0.0]  # similar to VEC_A


# ===========================================================================
# 1. PartitionResolver centroid cascade
# ===========================================================================


class TestResolverCentroidCascade:
    """Test the 4-step resolution cascade: explicit -> filter-map -> centroid -> fallback."""

    @pytest.mark.asyncio
    async def test_centroid_routing_attempted_when_enabled_and_embedding_and_all(self):
        """When centroid_routing.enabled=True, query_embedding provided, partitions='all',
        centroid routing should be attempted."""
        config = _make_config(centroid_enabled=True)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            query_embedding=VEC_QUERY,
        )

        # VEC_QUERY is very similar to VEC_A, so electronics should come first
        assert len(result) >= 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_centroid_routing_skipped_when_disabled(self):
        """When centroid_routing.enabled=False, centroid routing is skipped even
        with query_embedding present."""
        config = _make_config(centroid_enabled=False)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            query_embedding=VEC_QUERY,
        )

        # Should fall through to fan-out: both partitions returned
        names = {p.name for p in result}
        assert names == {"electronics", "furniture"}

    @pytest.mark.asyncio
    async def test_centroid_routing_skipped_when_no_embedding(self):
        """When query_embedding is None, centroid routing is skipped."""
        config = _make_config(centroid_enabled=True)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions="all",
            query_embedding=None,
        )

        # Falls through to fan-out
        names = {p.name for p in result}
        assert names == {"electronics", "furniture"}

    @pytest.mark.asyncio
    async def test_centroid_routing_returns_results_no_fallback(self):
        """When centroid routing returns results, they are used (no fallback)."""
        config = _make_config(centroid_enabled=True)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )
        resolver = PartitionResolver(config)

        # Mock _try_centroid_routing to return just one partition
        mock_result = [config.partitions.registry["electronics"]]
        with patch.object(
            resolver, "_try_centroid_routing", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await resolver.resolve(
                partitions="all",
                query_embedding=VEC_QUERY,
            )

        assert len(result) == 1
        assert result[0].name == "electronics"

    @pytest.mark.asyncio
    async def test_centroid_routing_empty_falls_through_to_fanout(self):
        """When centroid routing returns None (no roots), falls through to fan-out."""
        config = _make_config(centroid_enabled=True)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )
        resolver = PartitionResolver(config)

        # Mock centroid routing to return None (fall-through)
        with patch.object(
            resolver, "_try_centroid_routing", new_callable=AsyncMock, return_value=None
        ):
            result = await resolver.resolve(
                partitions="all",
                query_embedding=VEC_QUERY,
            )

        names = {p.name for p in result}
        assert names == {"electronics", "furniture"}

    @pytest.mark.asyncio
    async def test_filter_map_takes_priority_over_centroid(self):
        """Filter-map routing (step 2) runs before centroid routing (step 3)."""
        config = _make_config(centroid_enabled=True)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )
        resolver = PartitionResolver(config)

        # Filters that match the partition field should trigger filter-map
        filters = {"category": "furniture"}

        result = await resolver.resolve(
            partitions="all",
            filters=filters,
            query_embedding=VEC_QUERY,
        )

        # Filter-map should have resolved to furniture only
        assert len(result) == 1
        assert result[0].name == "furniture"

    @pytest.mark.asyncio
    async def test_explicit_partition_list_bypasses_centroid_and_filter(self):
        """Explicit partition list (step 1) bypasses both filter-map and centroid."""
        config = _make_config(centroid_enabled=True)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )
        resolver = PartitionResolver(config)

        result = await resolver.resolve(
            partitions=["furniture"],
            filters={"category": "electronics"},
            query_embedding=VEC_QUERY,
        )

        assert len(result) == 1
        assert result[0].name == "furniture"

    @pytest.mark.asyncio
    async def test_metrics_emitted_when_centroid_routing_used(self):
        """CENTROID_ROUTE_LATENCY and CENTROID_ROUTE_PARTITIONS emitted on success."""
        config = _make_config(centroid_enabled=True)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )

        metrics = MetricsCollector()
        events: list = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        metrics.add_handler(Recorder())

        resolver = PartitionResolver(config, metrics=metrics)

        await resolver.resolve(partitions="all", query_embedding=VEC_QUERY)

        metric_types = {e.metric_type for e in events}
        assert MetricType.CENTROID_ROUTE_LATENCY in metric_types
        assert MetricType.CENTROID_ROUTE_PARTITIONS in metric_types

    @pytest.mark.asyncio
    async def test_metrics_not_emitted_when_centroid_routing_skipped(self):
        """No centroid metrics when centroid routing is disabled."""
        config = _make_config(centroid_enabled=False)
        _register(
            config,
            _partition("electronics", centroid=VEC_A),
            _partition("furniture", centroid=VEC_B),
        )

        metrics = MetricsCollector()
        events: list = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        metrics.add_handler(Recorder())

        resolver = PartitionResolver(config, metrics=metrics)

        await resolver.resolve(partitions="all", query_embedding=VEC_QUERY)

        metric_types = {e.metric_type for e in events}
        assert MetricType.CENTROID_ROUTE_LATENCY not in metric_types
        assert MetricType.CENTROID_ROUTE_PARTITIONS not in metric_types

    def test_invalidate_caches_clears_centroid_router(self):
        """invalidate_caches() sets _centroid_router to None."""
        config = _make_config(centroid_enabled=True)
        resolver = PartitionResolver(config)
        resolver._centroid_router = MagicMock()

        resolver.invalidate_caches()

        assert resolver._centroid_router is None
        assert resolver._filter_map == {}
        assert resolver._registry_version == 0

    def test_resolver_constructor_accepts_optional_metrics(self):
        """PartitionResolver constructor accepts optional metrics parameter."""
        config = _make_config()
        metrics = MetricsCollector()

        resolver = PartitionResolver(config, metrics=metrics)

        assert resolver._metrics is metrics

    def test_resolver_constructor_metrics_default_none(self):
        """PartitionResolver constructor defaults metrics to None."""
        config = _make_config()

        resolver = PartitionResolver(config)

        assert resolver._metrics is None


# ===========================================================================
# 2. SVRClient.search() embedding reorder
# ===========================================================================


class TestSearchEmbeddingReorder:
    """Test that search() embeds the query BEFORE resolving partitions."""

    def _make_client(self, config, *, connected=True):
        """Build an SVRClient with mocks, bypassing real __init__ connect."""
        from semantic_vector_router.client import SVRClient
        from semantic_vector_router.routing.merger import ResultMerger

        client = SVRClient(config=config, auto_connect=False)
        client._backend = AsyncMock()
        client._embedder = AsyncMock()
        client._embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        client._resolver = AsyncMock(spec=PartitionResolver)
        client._resolver.resolve = AsyncMock(
            return_value=[
                _partition("electronics"),
            ]
        )
        client._merger = ResultMerger()
        client._backend.search_partitions = AsyncMock(return_value=[])
        client._reranker = None
        client._connected = connected
        return client

    @pytest.mark.asyncio
    async def test_byom_embedding_before_resolve(self):
        """In BYOM mode, query embedding happens before resolver.resolve()."""
        config = _make_config(centroid_enabled=True, embedding_mode=EmbeddingMode.BYOM)

        client = self._make_client(config)

        call_order = []

        original_embed = client._embedder.embed

        async def track_embed(query):
            call_order.append("embed")
            return await original_embed(query)

        original_resolve = client._resolver.resolve

        async def track_resolve(*args, **kwargs):
            call_order.append("resolve")
            return await original_resolve(*args, **kwargs)

        client._embedder.embed = track_embed
        client._resolver.resolve = track_resolve

        await client.search(query="wireless headphones", partitions="all")

        assert call_order == ["embed", "resolve"]

    @pytest.mark.asyncio
    async def test_resolve_receives_query_embedding_and_filters(self):
        """resolver.resolve() receives query_embedding and filters kwargs."""
        config = _make_config(centroid_enabled=True, embedding_mode=EmbeddingMode.BYOM)

        client = self._make_client(config)

        filters = {"price": {"$gt": 100}}
        await client.search(
            query="test query",
            partitions="all",
            filters=filters,
        )

        resolve_call = client._resolver.resolve
        resolve_call.assert_awaited_once()
        call_kwargs = resolve_call.call_args
        assert call_kwargs.kwargs.get("query_embedding") == [0.1, 0.2, 0.3, 0.4]
        assert call_kwargs.kwargs.get("filters") == filters

    @pytest.mark.asyncio
    async def test_auto_embedding_mode_passes_none_embedding(self):
        """In AUTO mode, resolver.resolve() is called with query_embedding=None."""
        config = _make_config(
            centroid_enabled=True, embedding_mode=EmbeddingMode.AUTO
        )
        # AUTO mode requires atlas_voyage provider to pass validation;
        # patch validate_config to skip that check
        with patch("semantic_vector_router.client.validate_config", return_value=[]):
            client = self._make_client(config)
        # AUTO mode means no embedder used on client side
        client._embedder = None
        # Backend must satisfy AutoEmbeddingCapable for the AUTO query path;
        # use a class with execute_search_with_query so isinstance() passes.
        from semantic_vector_router.backends.mongodb.backend import MongoDBBackend

        client._backend = MagicMock(spec=MongoDBBackend)
        client._backend.search_partitions = AsyncMock(return_value=[])

        await client.search(query="test", partitions="all")

        resolve_call = client._resolver.resolve
        resolve_call.assert_awaited_once()
        call_kwargs = resolve_call.call_args
        # In AUTO mode, no embedding is done client-side, so query_embedding stays None
        assert call_kwargs.kwargs.get("query_embedding") is None

    @pytest.mark.asyncio
    async def test_precomputed_query_vector_passed_to_resolve(self):
        """When user provides query_vector, it's passed to resolver.resolve()."""
        config = _make_config(centroid_enabled=True, embedding_mode=EmbeddingMode.BYOM)

        client = self._make_client(config)
        precomputed = [0.5, 0.5, 0.0, 0.0]

        await client.search(
            query="test",
            partitions="all",
            query_vector=precomputed,
        )

        # Embedder should NOT be called when query_vector is provided
        client._embedder.embed.assert_not_awaited()

        resolve_call = client._resolver.resolve
        call_kwargs = resolve_call.call_args
        assert call_kwargs.kwargs.get("query_embedding") == precomputed

    @pytest.mark.asyncio
    async def test_embedding_cache_hit_vector_passed_to_resolver(self):
        """When embedding cache has a hit, the cached vector is passed to resolver."""
        config = _make_config(centroid_enabled=True, embedding_mode=EmbeddingMode.BYOM)

        client = self._make_client(config)
        cached_vector = [0.9, 0.1, 0.0, 0.0]

        # Seed the cache
        from semantic_vector_router.utils.cache import CacheKey

        cache_key = CacheKey(
            text="test query",
            model="text-embedding-3-small",
            dimensions=4,
            input_type="query",
        )
        client._embedding_cache.put(cache_key, cached_vector)

        await client.search(query="test query", partitions="all")

        # Embedder should NOT be called (cache hit)
        client._embedder.embed.assert_not_awaited()

        resolve_call = client._resolver.resolve
        call_kwargs = resolve_call.call_args
        assert call_kwargs.kwargs.get("query_embedding") == cached_vector


# ===========================================================================
# 3. SVRClient.ingest() post-ingest centroid
# ===========================================================================


class TestIngestPostCentroid:
    """Test post-ingest centroid computation in SVRClient.ingest()."""

    def _make_client(self, config, *, metadata=None):
        """Build an SVRClient with mocks for ingest testing."""
        from semantic_vector_router.client import SVRClient
        from semantic_vector_router.routing.merger import ResultMerger

        client = SVRClient(config=config, auto_connect=False)
        client._backend = AsyncMock()
        client._backend.db = {"test_collection": AsyncMock()}
        client._embedder = AsyncMock()
        client._resolver = PartitionResolver(config)
        client._merger = ResultMerger()
        client._metadata = metadata
        client._connected = True
        return client

    @pytest.mark.asyncio
    async def test_centroid_computed_when_enabled_and_no_centroid(self):
        """When centroid_routing.enabled=True and partition has no centroid,
        compute_partition_centroid is called."""
        config = _make_config(centroid_enabled=True)

        mock_metadata = AsyncMock()
        mock_metadata.get_partition = AsyncMock(
            return_value=_partition("electronics", centroid=None)
        )
        mock_metadata.update_centroid = AsyncMock()

        client = self._make_client(config, metadata=mock_metadata)

        from semantic_vector_router.models import IngestResult

        mock_result = IngestResult(
            inserted=5, failed=0, elapsed_ms=100.0, embed_ms=50.0, write_ms=40.0
        )

        with patch(
            "semantic_vector_router.client.IngestPipeline"
        ) as MockPipeline, patch(
            "semantic_vector_router.factories.get_api_key", return_value="test-key"
        ), patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3, 0.4],
        ) as mock_compute:
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            result = await client.ingest(
                documents=[{"text": "hello"}],
                partition="electronics",
            )

        mock_compute.assert_awaited_once()
        mock_metadata.update_centroid.assert_awaited_once_with(
            "electronics", [0.1, 0.2, 0.3, 0.4]
        )
        assert result.inserted == 5

    @pytest.mark.asyncio
    async def test_centroid_skipped_when_disabled(self):
        """When centroid_routing.enabled=False, centroid computation is skipped."""
        config = _make_config(centroid_enabled=False)

        mock_metadata = AsyncMock()
        client = self._make_client(config, metadata=mock_metadata)

        from semantic_vector_router.models import IngestResult

        mock_result = IngestResult(
            inserted=5, failed=0, elapsed_ms=100.0, embed_ms=50.0, write_ms=40.0
        )

        with patch(
            "semantic_vector_router.client.IngestPipeline"
        ) as MockPipeline, patch(
            "semantic_vector_router.factories.get_api_key", return_value="test-key"
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            await client.ingest(
                documents=[{"text": "hello"}],
                partition="electronics",
            )

        # get_partition should not be called because centroid routing is disabled
        mock_metadata.get_partition.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_centroid_skipped_when_partition_already_has_centroid(self):
        """When partition already has a centroid, computation is skipped."""
        config = _make_config(centroid_enabled=True)

        mock_metadata = AsyncMock()
        mock_metadata.get_partition = AsyncMock(
            return_value=_partition("electronics", centroid=[0.5, 0.5, 0.0, 0.0])
        )

        client = self._make_client(config, metadata=mock_metadata)

        from semantic_vector_router.models import IngestResult

        mock_result = IngestResult(
            inserted=5, failed=0, elapsed_ms=100.0, embed_ms=50.0, write_ms=40.0
        )

        with patch(
            "semantic_vector_router.client.IngestPipeline"
        ) as MockPipeline, patch(
            "semantic_vector_router.factories.get_api_key", return_value="test-key"
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            await client.ingest(
                documents=[{"text": "hello"}],
                partition="electronics",
            )

        # update_centroid should not be called since partition already has centroid
        mock_metadata.update_centroid.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_centroid_skipped_when_no_partition_name(self):
        """When partition=None, centroid computation is skipped."""
        config = _make_config(centroid_enabled=True)

        mock_metadata = AsyncMock()
        client = self._make_client(config, metadata=mock_metadata)

        from semantic_vector_router.models import IngestResult

        mock_result = IngestResult(
            inserted=5, failed=0, elapsed_ms=100.0, embed_ms=50.0, write_ms=40.0
        )

        with patch(
            "semantic_vector_router.client.IngestPipeline"
        ) as MockPipeline, patch(
            "semantic_vector_router.factories.get_api_key", return_value="test-key"
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            await client.ingest(
                documents=[{"text": "hello"}],
                partition=None,
            )

        mock_metadata.get_partition.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_centroid_failure_caught_and_logged(self):
        """Centroid computation failure is caught and logged, doesn't break ingest."""
        config = _make_config(centroid_enabled=True)

        mock_metadata = AsyncMock()
        mock_metadata.get_partition = AsyncMock(
            return_value=_partition("electronics", centroid=None)
        )

        client = self._make_client(config, metadata=mock_metadata)

        from semantic_vector_router.models import IngestResult

        mock_result = IngestResult(
            inserted=5, failed=0, elapsed_ms=100.0, embed_ms=50.0, write_ms=40.0
        )

        with patch(
            "semantic_vector_router.client.IngestPipeline"
        ) as MockPipeline, patch(
            "semantic_vector_router.factories.get_api_key", return_value="test-key"
        ), patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            side_effect=RuntimeError("MongoDB connection failed"),
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            # Should NOT raise
            result = await client.ingest(
                documents=[{"text": "hello"}],
                partition="electronics",
            )

        assert result.inserted == 5

    @pytest.mark.asyncio
    async def test_centroid_skipped_when_metadata_is_none(self):
        """When metadata store is None, centroid computation is skipped."""
        config = _make_config(centroid_enabled=True)

        # metadata=None
        client = self._make_client(config, metadata=None)

        from semantic_vector_router.models import IngestResult

        mock_result = IngestResult(
            inserted=5, failed=0, elapsed_ms=100.0, embed_ms=50.0, write_ms=40.0
        )

        with patch(
            "semantic_vector_router.client.IngestPipeline"
        ) as MockPipeline, patch(
            "semantic_vector_router.factories.get_api_key", return_value="test-key"
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline

            # Should not raise (skipped because metadata is None)
            result = await client.ingest(
                documents=[{"text": "hello"}],
                partition="electronics",
            )

        assert result.inserted == 5


# ===========================================================================
# 4. RepartitionEngine compute_centroids step
# ===========================================================================


class TestRepartitionComputeCentroids:
    """Test the compute_centroids step in the repartition workflow."""

    def _make_engine(self, config=None):
        """Build a RepartitionEngine with mocks."""
        from semantic_vector_router.lifecycle.repartition import RepartitionEngine

        if config is None:
            config = _make_config(centroid_enabled=True)

        backend = AsyncMock()
        backend.db = {"test_collection": AsyncMock()}
        metadata = AsyncMock()

        engine = RepartitionEngine(backend, metadata, config)
        return engine, backend, metadata

    def test_compute_centroids_step_handler_exists(self):
        """compute_centroids is in step_handlers dict."""
        engine, _, _ = self._make_engine()

        # The step handlers are referenced in execute_operation;
        # we verify the step exists by checking the method
        assert hasattr(engine, "_step_compute_centroids")

    @pytest.mark.asyncio
    async def test_compute_centroids_for_active_children(self):
        """Step computes centroids for all ACTIVE children."""
        engine, backend, metadata = self._make_engine()

        parent = _partition(
            "electronics",
            status=PartitionStatus.RETIRED,
            children=["electronics_a", "electronics_b"],
        )
        child_a = _partition("electronics_a", filter_value="a")
        child_b = _partition("electronics_b", filter_value="b")

        metadata.get_partition = AsyncMock(
            side_effect=lambda name: {
                "electronics": parent,
                "electronics_a": child_a,
                "electronics_b": child_b,
            }.get(name)
        )
        metadata.update_centroid = AsyncMock()

        op = {"target_partition": "electronics"}

        with patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3, 0.4],
        ) as mock_compute:
            await engine._step_compute_centroids(op)

        assert mock_compute.await_count == 2
        assert metadata.update_centroid.await_count == 2

    @pytest.mark.asyncio
    async def test_compute_centroids_skips_non_active_children(self):
        """Step skips non-ACTIVE children."""
        engine, backend, metadata = self._make_engine()

        parent = _partition(
            "electronics",
            status=PartitionStatus.RETIRED,
            children=["child_active", "child_disabled"],
        )
        child_active = _partition("child_active")
        child_disabled = _partition(
            "child_disabled", status=PartitionStatus.DISABLED
        )

        metadata.get_partition = AsyncMock(
            side_effect=lambda name: {
                "electronics": parent,
                "child_active": child_active,
                "child_disabled": child_disabled,
            }.get(name)
        )
        metadata.update_centroid = AsyncMock()

        op = {"target_partition": "electronics"}

        with patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3, 0.4],
        ) as mock_compute:
            await engine._step_compute_centroids(op)

        # Only child_active should have centroid computed
        mock_compute.assert_awaited_once()
        metadata.update_centroid.assert_awaited_once_with(
            "child_active", [0.1, 0.2, 0.3, 0.4]
        )

    @pytest.mark.asyncio
    async def test_compute_centroids_handles_empty_vectors(self):
        """Step handles empty vectors gracefully (no centroid stored)."""
        engine, backend, metadata = self._make_engine()

        parent = _partition(
            "electronics",
            status=PartitionStatus.RETIRED,
            children=["child_empty"],
        )
        child_empty = _partition("child_empty")

        metadata.get_partition = AsyncMock(
            side_effect=lambda name: {
                "electronics": parent,
                "child_empty": child_empty,
            }.get(name)
        )
        metadata.update_centroid = AsyncMock()

        op = {"target_partition": "electronics"}

        with patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            return_value=None,  # No vectors found
        ):
            await engine._step_compute_centroids(op)

        # update_centroid should NOT be called since compute returned None
        metadata.update_centroid.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_compute_centroids_uses_partition_embedding_field(self):
        """Step uses partition.embedding_field when available."""
        engine, backend, metadata = self._make_engine()

        parent = _partition(
            "electronics",
            status=PartitionStatus.RETIRED,
            children=["child_custom"],
        )
        child_custom = _partition(
            "child_custom", embedding_field="custom_embedding"
        )

        metadata.get_partition = AsyncMock(
            side_effect=lambda name: {
                "electronics": parent,
                "child_custom": child_custom,
            }.get(name)
        )
        metadata.update_centroid = AsyncMock()

        op = {"target_partition": "electronics"}

        with patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3, 0.4],
        ) as mock_compute:
            await engine._step_compute_centroids(op)

        # Verify custom_embedding was passed as embedding_field
        call_kwargs = mock_compute.call_args
        assert call_kwargs.kwargs.get("embedding_field") == "custom_embedding"

    @pytest.mark.asyncio
    async def test_compute_centroids_uses_default_embedding_field_fallback(self):
        """Step uses default embedding_field when partition has none."""
        engine, backend, metadata = self._make_engine()

        parent = _partition(
            "electronics",
            status=PartitionStatus.RETIRED,
            children=["child_default"],
        )
        # embedding_field=None -> should fallback to config's embedding_field
        child_default = _partition("child_default", embedding_field=None)

        metadata.get_partition = AsyncMock(
            side_effect=lambda name: {
                "electronics": parent,
                "child_default": child_default,
            }.get(name)
        )
        metadata.update_centroid = AsyncMock()

        op = {"target_partition": "electronics"}

        with patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3, 0.4],
        ) as mock_compute:
            await engine._step_compute_centroids(op)

        # Should use config's default embedding_field ("embedding")
        call_kwargs = mock_compute.call_args
        assert call_kwargs.kwargs.get("embedding_field") == "embedding"

    @pytest.mark.asyncio
    async def test_execute_operation_includes_compute_centroids_step(self):
        """execute_operation processes compute_centroids step in the workflow."""
        engine, backend, metadata = self._make_engine()

        op = {
            "_id": "op1",
            "target_partition": "electronics",
            "strategy": "secondary_field",
            "strategy_config": {},
            "steps": [
                {"action": "compute_centroids", "status": "pending"},
            ],
        }
        metadata.get_operation = AsyncMock(return_value=op)
        metadata.update_operation_step = AsyncMock()
        metadata.update_operation_status = AsyncMock()

        # Mock the step handler
        with patch.object(
            engine, "_step_compute_centroids", new_callable=AsyncMock
        ) as mock_step:
            result = await engine.execute_operation("op1")

        assert result is True
        mock_step.assert_awaited_once_with(op)
        # Step should be marked in_progress then done
        metadata.update_operation_step.assert_any_await(
            "op1", "compute_centroids", "in_progress"
        )
        metadata.update_operation_step.assert_any_await(
            "op1", "compute_centroids", "done"
        )
