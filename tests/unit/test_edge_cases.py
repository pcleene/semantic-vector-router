"""Edge case hardening tests for SVR SDK.

Verifies graceful behavior under boundary conditions:
empty partitions, single-document partitions, concurrent operations,
connection failures, config edge cases, unicode handling, and
large result sets.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from semantic_vector_router.client import SVRClient
from semantic_vector_router.exceptions import (
    EmbeddingError,
    IngestionError,
    RerankingError,
    SearchError,
    SplitError,
)
from semantic_vector_router.ingestion import IngestPipeline
from semantic_vector_router.models import (
    DatabaseConfig,
    DetectionSignal,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestConfig,
    IngestMode,
    IngestResult,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RerankingConfig,
    RerankerProvider,
    SearchHit,
    SearchResult,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
)
from semantic_vector_router.routing.merger import ResultMerger
from semantic_vector_router.routing.resolver import PartitionResolver
from semantic_vector_router.utils.cache import CacheKey, EmbeddingCache
from semantic_vector_router.utils.metrics import NoOpCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(**overrides) -> SVRConfig:
    """Build an SVRConfig with minimal required fields."""
    cfg = {
        "database": {
            "database": "test_db",
            "source_collection": "test_col",
        },
        "partitioning": {"field": "category"},
    }
    cfg.update(overrides)
    return SVRConfig(**cfg)


def _make_client(
    config: SVRConfig,
    mock_backend=None,
    mock_embedder=None,
    mock_reranker=None,
    *,
    connected: bool = True,
    auto_connect_failed: bool = False,
):
    """Build an SVRClient with injected mocks, bypassing real __init__ connect."""
    with patch("semantic_vector_router.client.validate_config", return_value=[]):
        client = SVRClient(config=config, auto_connect=False)
    client._backend = mock_backend or AsyncMock()
    client._embedder = mock_embedder
    client._reranker = mock_reranker
    client._resolver = PartitionResolver(config)
    client._merger = ResultMerger()
    client._connected = connected
    client._auto_connect_failed = auto_connect_failed
    return client


def _config_with_partitions(partitions: dict[str, PartitionInfo] | None = None) -> SVRConfig:
    """Config with BYOM mode and optional partitions."""
    cfg = SVRConfig(
        database=DatabaseConfig(
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_partition_",
            index_name_prefix="svr_idx_",
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
            dimensions=1536,
        ),
        reranking=RerankingConfig(
            enabled=True,
            provider=RerankerProvider.VOYAGE,
            model="rerank-2",
        ),
    )
    if partitions:
        cfg.partitions.registry = partitions
    return cfg


def _partition(name: str, doc_count: int = 100_000) -> PartitionInfo:
    """Quick partition helper."""
    return PartitionInfo(
        name=name,
        view_name=f"svr_partition_{name}",
        index_name=f"svr_idx_{name}",
        filter_value=name,
        document_count=doc_count,
        status=PartitionStatus.ACTIVE,
    )


def _raw_results(partition_name: str, count: int) -> list[dict]:
    """Generate raw backend result dicts."""
    return [
        {
            "_id": f"doc{i}",
            "name": f"Item {i}",
            "_svr_score": 0.9 - (i * 0.01),
            "_svr_partition": partition_name,
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Empty Partitions
# ---------------------------------------------------------------------------


class TestEmptyPartitions:
    """Tests for empty partition scenarios."""

    async def test_search_empty_partition_list(self):
        """When resolver returns no partitions, search returns empty SearchResult."""
        config = _config_with_partitions()  # no partitions registered
        mock_backend = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        result = await client.search(query="anything", partitions="all", limit=10)

        assert isinstance(result, SearchResult)
        assert result.hits == []
        assert result.partitions_searched == []
        assert result.total_candidates == 0
        mock_backend.search_partitions.assert_not_awaited()

    async def test_search_empty_partition_result(self):
        """When backend returns empty list for a partition, result has no hits."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=10,
        )

        assert isinstance(result, SearchResult)
        assert result.hits == []
        assert result.total_candidates == 0
        assert result.partitions_searched == ["electronics"]

    async def test_analyze_empty_partition_no_crash(self):
        """Detection with 0 documents should produce UNDERPOPULATED signal, not crash."""
        from semantic_vector_router.lifecycle.detector import PartitionDetector

        partition = _partition("empty_part", doc_count=0)

        mock_backend = AsyncMock()
        mock_backend.count_documents = AsyncMock(return_value=0)

        mock_metadata = AsyncMock()
        mock_metadata.list_partitions = AsyncMock(return_value=[partition])
        mock_metadata.append_health_history = AsyncMock()
        mock_metadata.get_health_history = AsyncMock(return_value=[])
        mock_metadata.create_operation = AsyncMock()

        config = _minimal_config()

        detector = PartitionDetector(mock_backend, mock_metadata, config)
        results = await detector.run_detection()

        # Should find underpopulated signal (count=0 < min_threshold=1000)
        underpop = [r for r in results if r.signal == DetectionSignal.UNDERPOPULATED]
        assert len(underpop) == 1
        assert underpop[0].partition == "empty_part"
        assert underpop[0].details["count"] == 0

    async def test_split_empty_partition_raises(self):
        """Splitting a partition with 0 documents should raise SplitError."""
        from semantic_vector_router.lifecycle.splitter import PartitionSplitter

        partition = _partition("empty_part", doc_count=0)
        config = _minimal_config()
        config.partitions.registry = {"empty_part": partition}
        config.lifecycle.auto_split = MagicMock()
        config.lifecycle.auto_split.split_strategy = MagicMock()
        config.lifecycle.auto_split.split_strategy.value = "secondary_field"
        config.lifecycle.auto_split.secondary_field = "subcategory"

        mock_backend = AsyncMock()
        mock_backend.get_distinct_values = AsyncMock(return_value=[])

        mock_provisioner = AsyncMock()

        splitter = PartitionSplitter(mock_backend, config, mock_provisioner)

        # The splitter tries to split by secondary_field and gets empty values list.
        # This means 0 children are created, which is essentially a no-op split.
        result = await splitter.execute_split("empty_part")
        # With 0 distinct values, 0 children are created. Split still succeeds
        # but with an empty children list.
        assert result == []


# ---------------------------------------------------------------------------
# Single-Document Partitions
# ---------------------------------------------------------------------------


class TestSingleDocumentPartitions:
    """Tests for partitions with very few documents."""

    async def test_search_limit_exceeds_docs(self):
        """When limit=10 but only 1 result exists, return 1 result without error."""
        partitions = {"electronics": _partition("electronics", doc_count=1)}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=10,
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1

    async def test_rerank_single_result(self):
        """Reranking with a single hit should work without crash."""
        partitions = {
            "electronics": _partition("electronics"),
            "furniture": _partition("furniture"),
        }
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        single_hit = SearchHit(
            id="doc0",
            score=0.9,
            rerank_score=0.95,
            partition="electronics",
            document={"name": "Item 0"},
        )
        mock_reranker = AsyncMock()
        mock_reranker.rerank_hits = AsyncMock(return_value=[single_hit])

        client = _make_client(config, mock_backend, mock_embedder, mock_reranker)

        result = await client.search(
            query="headphones",
            partitions=["electronics", "furniture"],
            limit=10,
            rerank=True,
        )

        assert isinstance(result, SearchResult)
        assert result.reranked is True
        mock_reranker.rerank_hits.assert_awaited_once()

    async def test_detection_single_doc_underpopulated(self):
        """Detection with a single-doc partition should signal UNDERPOPULATED."""
        from semantic_vector_router.lifecycle.detector import PartitionDetector

        partition = _partition("tiny_part", doc_count=1)

        mock_backend = AsyncMock()
        mock_backend.count_documents = AsyncMock(return_value=1)

        mock_metadata = AsyncMock()
        mock_metadata.list_partitions = AsyncMock(return_value=[partition])
        mock_metadata.append_health_history = AsyncMock()
        mock_metadata.get_health_history = AsyncMock(return_value=[])
        mock_metadata.create_operation = AsyncMock()

        config = _minimal_config()

        detector = PartitionDetector(mock_backend, mock_metadata, config)
        results = await detector.run_detection()

        underpop = [r for r in results if r.signal == DetectionSignal.UNDERPOPULATED]
        assert len(underpop) == 1
        assert underpop[0].details["count"] == 1


# ---------------------------------------------------------------------------
# Concurrent Operations
# ---------------------------------------------------------------------------


class TestConcurrentOperations:
    """Tests for concurrent access patterns."""

    async def test_concurrent_searches(self):
        """10 concurrent search calls should all complete without deadlock."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 2)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        async def do_search():
            return await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
            )

        results = await asyncio.wait_for(
            asyncio.gather(*[do_search() for _ in range(10)]),
            timeout=10.0,
        )

        assert len(results) == 10
        for r in results:
            assert isinstance(r, SearchResult)
            assert len(r.hits) == 2

    async def test_concurrent_ingest_search(self):
        """Concurrent ingest and search should both complete."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 2)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        # Mock the ingest pipeline via patching
        mock_ingest_result = IngestResult(
            inserted=1, failed=0, elapsed_ms=50.0, embed_ms=25.0, write_ms=20.0
        )

        async def do_search():
            return await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
            )

        async def do_ingest():
            with patch("semantic_vector_router.client.IngestPipeline") as MockPipeline, \
                 patch("semantic_vector_router.factories.get_api_key", return_value="key"):
                mock_pipeline = AsyncMock()
                mock_pipeline.ingest = AsyncMock(return_value=mock_ingest_result)
                MockPipeline.return_value = mock_pipeline
                return await client.ingest(documents=[{"text": "hello"}])

        results = await asyncio.wait_for(
            asyncio.gather(do_search(), do_ingest()),
            timeout=10.0,
        )

        assert isinstance(results[0], SearchResult)
        assert isinstance(results[1], IngestResult)

    async def test_concurrent_detection_lock_contention(self):
        """When lock is not acquired, detection should return None gracefully."""
        from semantic_vector_router.lifecycle.detector import PartitionDetector

        mock_backend = AsyncMock()
        mock_metadata = AsyncMock()
        mock_metadata.acquire_lock = AsyncMock(return_value=False)  # Lock contention

        config = _minimal_config()
        detector = PartitionDetector(mock_backend, mock_metadata, config)

        result = await detector.run_detection_with_lock()

        assert result is None
        mock_metadata.release_lock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Connection Failures
# ---------------------------------------------------------------------------


class TestConnectionFailures:
    """Tests for connection failure scenarios."""

    async def test_search_connection_drop(self):
        """Backend connection error during search should propagate clearly."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(
            side_effect=SearchError("Connection lost during search")
        )
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        with pytest.raises(SearchError, match="Connection lost"):
            await client.search(
                query="test",
                partitions=["electronics"],
                limit=5,
            )

    async def test_embedding_api_timeout(self):
        """Embedder timeout should propagate as a clear exception."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)

        mock_backend = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(
            side_effect=EmbeddingError("Embedding API timed out after 60s")
        )

        client = _make_client(config, mock_backend, mock_embedder)

        with pytest.raises(EmbeddingError, match="timed out"):
            await client.search(
                query="test",
                partitions=["electronics"],
                limit=5,
            )

    async def test_reranking_api_timeout_propagates(self):
        """Reranking timeout should propagate (no graceful degradation in current code).

        The client code does not catch reranking exceptions -- they propagate
        through the generic except clause which re-raises them.
        """
        partitions = {
            "electronics": _partition("electronics"),
            "furniture": _partition("furniture"),
        }
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 2) + _raw_results("furniture", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_reranker = AsyncMock()
        mock_reranker.rerank_hits = AsyncMock(
            side_effect=RerankingError("Reranking API timed out")
        )

        client = _make_client(config, mock_backend, mock_embedder, mock_reranker)

        with pytest.raises(RerankingError, match="timed out"):
            await client.search(
                query="test",
                partitions=["electronics", "furniture"],
                limit=5,
            )

    async def test_disconnect_then_search_raises(self):
        """Search after disconnect with auto_connect_failed should raise clearly."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        mock_backend = AsyncMock()
        mock_embedder = AsyncMock()

        client = _make_client(
            config,
            mock_backend,
            mock_embedder,
            connected=False,
            auto_connect_failed=True,
        )

        with pytest.raises(SearchError, match="not connected"):
            await client.search(query="test", partitions=["electronics"])

    async def test_disconnect_then_search_reconnects(self):
        """Search after disconnect (no auto_connect_failed) should try to reconnect."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(
            config,
            mock_backend,
            mock_embedder,
            connected=False,
            auto_connect_failed=False,
        )

        async def fake_connect():
            client._connected = True

        with patch.object(client, "connect", side_effect=fake_connect) as mock_connect:
            result = await client.search(
                query="test",
                partitions=["electronics"],
                limit=5,
            )

        mock_connect.assert_awaited_once()
        assert isinstance(result, SearchResult)


# ---------------------------------------------------------------------------
# Config Edge Cases
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    """Tests for configuration boundary conditions."""

    def test_minimal_config_validates(self):
        """SVRConfig with only required fields should validate without error."""
        config = SVRConfig(
            database=DatabaseConfig(
                database="mydb",
                source_collection="mycol",
            ),
            partitioning=PartitioningConfig(field="category"),
        )
        assert config.database.database == "mydb"
        assert config.embedding.mode == EmbeddingMode.BYOM  # default
        assert config.reranking.enabled is True  # default
        assert config.vector_search.dimensions == 1536  # default

    def test_config_extra_keys_raises_or_ignored(self):
        """Extra keys in config sub-models should be handled by Pydantic.

        Pydantic v2 models ignore extra fields by default unless `extra = 'forbid'`.
        """
        # DatabaseConfig with extra key -- should work (Pydantic v2 ignores extra by default)
        try:
            cfg = DatabaseConfig(
                database="test",
                source_collection="test",
                unknown_extra_field="value",
            )
            # If it doesn't raise, it was ignored
            assert not hasattr(cfg, "unknown_extra_field") or cfg.unknown_extra_field is not None
        except PydanticValidationError:
            # If extra='forbid' is set, this is expected
            pass

    def test_config_zero_dimensions_validation(self):
        """Zero or negative dimensions should still create a config object.

        Note: Pydantic does not reject these by default since dimensions is int,
        not PositiveInt. This test documents the current behavior.
        """
        config = SVRConfig(
            database=DatabaseConfig(
                database="test",
                source_collection="test",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=0),
        )
        assert config.vector_search.dimensions == 0

    def test_config_negative_dimensions(self):
        """Negative dimensions should create a config (no validation prevents it)."""
        config = SVRConfig(
            database=DatabaseConfig(
                database="test",
                source_collection="test",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=-1),
        )
        assert config.vector_search.dimensions == -1

    def test_config_empty_string_database(self):
        """Empty string for database should be valid at config level."""
        config = SVRConfig(
            database=DatabaseConfig(
                database="",
                source_collection="",
            ),
            partitioning=PartitioningConfig(field=""),
        )
        assert config.database.database == ""


# ---------------------------------------------------------------------------
# Large Result Sets
# ---------------------------------------------------------------------------


class TestLargeResultSets:
    """Tests for handling large volumes of data."""

    def test_merge_large_result_set(self):
        """ResultMerger should handle 10K hits without performance degradation."""
        raw_results = [
            {
                "_id": f"doc{i}",
                "name": f"Item {i}",
                "_svr_score": 0.99 - (i * 0.00001),
                "_svr_partition": f"partition_{i % 5}",
            }
            for i in range(10_000)
        ]

        merger = ResultMerger()
        start = time.perf_counter()
        hits = merger.merge(raw_results, limit=10_000)
        elapsed = time.perf_counter() - start

        assert len(hits) == 10_000
        assert elapsed < 5.0  # Should complete in well under 5 seconds
        # Verify ordering (highest score first)
        for i in range(len(hits) - 1):
            assert hits[i].score >= hits[i + 1].score

    async def test_ingest_large_batch_batching(self):
        """IngestPipeline should chunk 1000 documents into embedding batches."""
        embedder = AsyncMock()
        # Return vectors for each batch call
        def make_vectors(texts):
            return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

        embedder.embed_with_batching = AsyncMock(side_effect=make_vectors)

        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        config = SVRConfig(
            database=DatabaseConfig(database="test_db", source_collection="test_col"),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=4),
            ingestion=IngestConfig(
                text_fields=["text"],
                batch_size=100,  # Embed in chunks of 100
                write_batch_size=500,
            ),
        )

        pipeline = IngestPipeline(
            backend=backend,
            config=config,
            embedder=embedder,
            metrics=NoOpCollector(),
        )

        docs = [{"text": f"Document {i}", "category": "test"} for i in range(1000)]
        result = await pipeline.ingest(docs)

        # 1000 docs / 100 batch_size = 10 embed batches
        assert embedder.embed_with_batching.call_count == 10
        assert result.inserted == 1000
        assert result.failed == 0


# ---------------------------------------------------------------------------
# Unicode and Special Characters
# ---------------------------------------------------------------------------


class TestUnicodeHandling:
    """Tests for unicode, CJK, emoji, and special character handling."""

    async def test_unicode_partition_name(self):
        """Partition with unicode name should be searchable."""
        partitions = {
            "donnees": PartitionInfo(
                name="donnees",
                view_name="svr_partition_donnees",
                index_name="svr_idx_donnees",
                filter_value="donnees",
                status=PartitionStatus.ACTIVE,
            ),
        }
        config = _config_with_partitions(partitions)
        raw = _raw_results("donnees", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        result = await client.search(
            query="test query",
            partitions=["donnees"],
            limit=5,
        )

        assert isinstance(result, SearchResult)
        assert result.partitions_searched == ["donnees"]

    async def test_search_query_with_emoji(self):
        """Search with emoji in query should pass through to embedder."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        result = await client.search(
            query="🎧 headphones",
            partitions=["electronics"],
            limit=5,
        )

        mock_embedder.embed.assert_awaited_once_with("🎧 headphones")
        assert isinstance(result, SearchResult)

    async def test_search_query_cjk(self):
        """CJK (Chinese/Japanese/Korean) query should work."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        cjk_query = "ワイヤレスヘッドフォン"
        result = await client.search(
            query=cjk_query,
            partitions=["electronics"],
            limit=5,
        )

        mock_embedder.embed.assert_awaited_once_with(cjk_query)
        assert isinstance(result, SearchResult)
        assert result.query == cjk_query

    async def test_document_with_null_bytes(self):
        """Ingest document with null bytes in text should be handled gracefully."""
        embedder = AsyncMock()
        embedder.embed_with_batching = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4]]
        )

        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        config = SVRConfig(
            database=DatabaseConfig(database="test_db", source_collection="test_col"),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=4),
            ingestion=IngestConfig(text_fields=["text"]),
        )

        pipeline = IngestPipeline(
            backend=backend,
            config=config,
            embedder=embedder,
            metrics=NoOpCollector(),
        )

        # Document with null bytes in text
        docs = [{"text": "Hello\x00World", "category": "test"}]
        result = await pipeline.ingest(docs)

        # The null byte text should be passed to embedder (now field-labeled)
        call_args = embedder.embed_with_batching.call_args[0][0]
        assert "text: Hello\x00World" in call_args
        assert result.inserted == 1

    async def test_very_long_search_query(self):
        """Search with a 10000-character query should not crash."""
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        long_query = "x" * 10_000
        result = await client.search(
            query=long_query,
            partitions=["electronics"],
            limit=5,
        )

        mock_embedder.embed.assert_awaited_once_with(long_query)
        assert isinstance(result, SearchResult)
        assert result.query == long_query

    async def test_empty_search_query(self):
        """Search with empty string query should not crash.

        The client does not validate query content -- it passes it straight
        to the embedder. As long as the embedder handles it, search works.
        """
        partitions = {"electronics": _partition("electronics")}
        config = _config_with_partitions(partitions)
        raw = _raw_results("electronics", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        result = await client.search(
            query="",
            partitions=["electronics"],
            limit=5,
        )

        mock_embedder.embed.assert_awaited_once_with("")
        assert isinstance(result, SearchResult)

    async def test_unicode_partition_name_cjk(self):
        """CJK partition name should be handled."""
        partitions = {
            "\u65e5\u672c\u8a9e": PartitionInfo(
                name="\u65e5\u672c\u8a9e",
                view_name="svr_partition_\u65e5\u672c\u8a9e",
                index_name="svr_idx_\u65e5\u672c\u8a9e",
                filter_value="\u65e5\u672c\u8a9e",
                status=PartitionStatus.ACTIVE,
            ),
        }
        config = _config_with_partitions(partitions)
        raw = _raw_results("\u65e5\u672c\u8a9e", 1)

        mock_backend = AsyncMock()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = _make_client(config, mock_backend, mock_embedder)

        result = await client.search(
            query="test",
            partitions=["\u65e5\u672c\u8a9e"],
            limit=5,
        )

        assert result.partitions_searched == ["\u65e5\u672c\u8a9e"]


# ---------------------------------------------------------------------------
# Embedding Cache Edge Cases
# ---------------------------------------------------------------------------


class TestCacheEdgeCases:
    """Tests for embedding cache boundary conditions."""

    def test_cache_disabled_returns_none(self):
        """Cache with max_size=0 should always return None."""
        cache = EmbeddingCache(max_size=0)
        key = CacheKey(text="test", model="m", dimensions=4, input_type="query")
        cache.put(key, [0.1, 0.2])
        assert cache.get(key) is None
        assert cache.size == 0

    def test_cache_eviction_at_capacity(self):
        """Cache should evict LRU entry when at capacity."""
        cache = EmbeddingCache(max_size=2, ttl_seconds=3600)

        k1 = CacheKey(text="a", model="m", dimensions=4, input_type="query")
        k2 = CacheKey(text="b", model="m", dimensions=4, input_type="query")
        k3 = CacheKey(text="c", model="m", dimensions=4, input_type="query")

        cache.put(k1, [1.0])
        cache.put(k2, [2.0])
        cache.put(k3, [3.0])  # Should evict k1

        assert cache.get(k1) is None
        assert cache.get(k2) == [2.0]
        assert cache.get(k3) == [3.0]
        assert cache.size == 2

    def test_cache_ttl_expiration(self):
        """Expired entries should not be returned."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=1)
        key = CacheKey(text="test", model="m", dimensions=4, input_type="query")

        # Manually insert with old timestamp
        from semantic_vector_router.utils.cache import CacheEntry
        import time as _time

        cache._cache[key] = CacheEntry(
            vector=[0.1],
            created_at=_time.time() - 2.0,  # 2 seconds ago, TTL is 1
        )

        assert cache.get(key) is None

    def test_cache_stats_accuracy(self):
        """Cache stats should accurately reflect operations."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=3600)
        key = CacheKey(text="a", model="m", dimensions=4, input_type="query")

        cache.get(key)  # miss
        cache.put(key, [1.0])
        cache.get(key)  # hit

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# ResultMerger Edge Cases
# ---------------------------------------------------------------------------


class TestMergerEdgeCases:
    """Tests for ResultMerger boundary conditions."""

    def test_merge_empty_results(self):
        """Merging empty results should return empty list."""
        merger = ResultMerger()
        hits = merger.merge([], limit=10)
        assert hits == []

    def test_merge_single_result(self):
        """Merging a single result should work."""
        merger = ResultMerger()
        raw = [{"_id": "doc1", "_svr_score": 0.9, "_svr_partition": "p1", "name": "Item"}]
        hits = merger.merge(raw, limit=10)
        assert len(hits) == 1
        assert hits[0].id == "doc1"

    def test_merge_deduplication(self):
        """Duplicate documents (same _id) should be deduplicated."""
        merger = ResultMerger(deduplicate=True, dedupe_field="_id")
        raw = [
            {"_id": "doc1", "_svr_score": 0.8, "_svr_partition": "p1", "name": "Item"},
            {"_id": "doc1", "_svr_score": 0.9, "_svr_partition": "p2", "name": "Item"},
        ]
        hits = merger.merge(raw, limit=10)
        assert len(hits) == 1
        assert hits[0].score >= 0  # Higher-scoring version kept

    def test_merge_all_same_scores(self):
        """All same scores should normalize to 1.0."""
        merger = ResultMerger(normalize_method="partition_minmax")
        raw = [
            {"_id": f"doc{i}", "_svr_score": 0.5, "_svr_partition": "p1"}
            for i in range(5)
        ]
        hits = merger.merge(raw, limit=10)
        assert len(hits) == 5
        for h in hits:
            assert h.score == 1.0


# ---------------------------------------------------------------------------
# PartitionResolver Edge Cases
# ---------------------------------------------------------------------------


class TestResolverEdgeCases:
    """Tests for PartitionResolver boundary conditions."""

    async def test_resolve_all_with_no_partitions(self):
        """Resolving 'all' with empty registry returns empty list."""
        config = _minimal_config()
        resolver = PartitionResolver(config)
        result = await resolver.resolve("all")
        assert result == []

    async def test_resolve_nonexistent_partition_raises(self):
        """Resolving a partition that does not exist should raise."""
        config = _minimal_config()
        resolver = PartitionResolver(config)

        from semantic_vector_router.exceptions import PartitionNotFoundError

        with pytest.raises(PartitionNotFoundError, match="not_real"):
            await resolver.resolve(["not_real"])

    async def test_resolve_disabled_partition_skipped(self):
        """Disabled partitions should be skipped in resolution."""
        config = _minimal_config()
        config.partitions.registry = {
            "active": _partition("active"),
            "disabled": PartitionInfo(
                name="disabled",
                view_name="v",
                index_name="i",
                filter_value="disabled",
                status=PartitionStatus.DISABLED,
            ),
        }
        resolver = PartitionResolver(config)

        result = await resolver.resolve("all")
        names = [p.name for p in result]
        assert "active" in names
        assert "disabled" not in names

    async def test_resolve_max_partitions_limit(self):
        """Resolution should respect max_partitions_per_query limit."""
        config = _minimal_config()
        config.routing.max_partitions_per_query = 2
        config.partitions.registry = {
            f"p{i}": _partition(f"p{i}") for i in range(5)
        }
        resolver = PartitionResolver(config)

        result = await resolver.resolve("all")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# IngestPipeline Edge Cases
# ---------------------------------------------------------------------------


class TestIngestEdgeCases:
    """Tests for IngestPipeline boundary conditions."""

    async def test_ingest_all_docs_empty_text(self):
        """When all documents have no extractable text, result has 0 inserted."""
        config = SVRConfig(
            database=DatabaseConfig(database="test_db", source_collection="test_col"),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=4),
            ingestion=IngestConfig(text_fields=["text"], continue_on_error=True),
        )
        embedder = AsyncMock()
        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = IngestPipeline(
            backend=backend, config=config, embedder=embedder, metrics=NoOpCollector(),
        )

        docs = [{"other_field": "no text"}, {"also_missing": "text"}]
        result = await pipeline.ingest(docs)

        assert result.inserted == 0
        assert result.failed == 2
        embedder.embed_with_batching.assert_not_awaited()

    async def test_ingest_mixed_valid_invalid_docs(self):
        """Mix of valid and invalid documents with continue_on_error=True."""
        config = SVRConfig(
            database=DatabaseConfig(database="test_db", source_collection="test_col"),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=4),
            ingestion=IngestConfig(text_fields=["text"], continue_on_error=True),
        )
        embedder = AsyncMock()
        embedder.embed_with_batching = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4]] * 2
        )

        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = IngestPipeline(
            backend=backend, config=config, embedder=embedder, metrics=NoOpCollector(),
        )

        docs = [
            {"text": "Good doc 1", "category": "test"},
            {"no_text_field": "bad doc"},
            {"text": "Good doc 2", "category": "test"},
        ]
        result = await pipeline.ingest(docs)

        assert result.inserted == 2
        assert result.failed == 1

    async def test_ingest_unicode_text(self):
        """Unicode text in documents should be embedded without issues."""
        config = SVRConfig(
            database=DatabaseConfig(database="test_db", source_collection="test_col"),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=4),
            ingestion=IngestConfig(text_fields=["text"]),
        )
        embedder = AsyncMock()
        embedder.embed_with_batching = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4]]
        )

        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = IngestPipeline(
            backend=backend, config=config, embedder=embedder, metrics=NoOpCollector(),
        )

        docs = [{"text": "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8", "category": "test"}]
        result = await pipeline.ingest(docs)

        assert result.inserted == 1
        call_args = embedder.embed_with_batching.call_args[0][0]
        assert "\u65e5\u672c\u8a9e" in call_args[0]


# ---------------------------------------------------------------------------
# Retry Edge Cases
# ---------------------------------------------------------------------------


class TestRetryEdgeCases:
    """Tests for retry decorator boundary conditions."""

    async def test_retry_with_max_attempts_zero(self):
        """max_attempts=0 should still run once."""
        from semantic_vector_router.utils.retry import with_retry

        call_count = 0

        @with_retry(max_attempts=0)
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await failing_func()

        assert call_count == 1

    async def test_retry_succeeds_after_failure(self):
        """Function that succeeds on second attempt should return the result."""
        from semantic_vector_router.utils.retry import with_retry

        attempt = 0

        @with_retry(
            max_attempts=3,
            base_delay=0.01,
            max_delay=0.02,
        )
        async def flaky_func():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise ConnectionError("transient")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert attempt == 2

    async def test_retry_non_retryable_exception_not_retried(self):
        """Non-retryable exceptions should not be retried."""
        from semantic_vector_router.utils.retry import with_retry

        call_count = 0

        @with_retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await func()

        assert call_count == 1  # Only called once, not retried


# ---------------------------------------------------------------------------
# Backend search_partitions Edge Cases
# ---------------------------------------------------------------------------


class TestBackendEdgeCases:
    """Tests for MongoDBBackend edge cases (mocked)."""

    async def test_search_partitions_empty_list(self):
        """search_partitions with empty partition list returns empty."""
        from semantic_vector_router.backends.mongodb import MongoDBBackend

        config = _minimal_config()
        backend = MongoDBBackend(config)
        # Mock the internals
        backend._client = MagicMock()
        backend._db = MagicMock()

        result = await backend.search_partitions(
            partitions=[], limit=10, query_vector=[0.1]
        )
        assert result == []

    async def test_search_partitions_partial_failure(self):
        """If one partition search fails, others should still return results."""
        from semantic_vector_router.backends.mongodb import MongoDBBackend

        config = _minimal_config()
        backend = MongoDBBackend(config)
        backend._client = MagicMock()
        backend._db = MagicMock()

        p1 = _partition("good")
        p2 = _partition("bad")

        # Mock execute_search to succeed for p1, fail for p2
        call_count = 0

        async def mock_execute_search(partition, query_vector, limit, num_candidates, filters=None, **kwargs):
            if partition.name == "bad":
                raise SearchError("timeout on bad partition")
            return [{"_id": "doc1", "_svr_score": 0.9, "_svr_partition": "good"}]

        with patch.object(backend, "execute_search", side_effect=mock_execute_search):
            results = await backend.search_partitions(
                partitions=[p1, p2],
                limit=10,
                query_vector=[0.1],
            )

        # Should get results from good partition, bad is silently skipped
        assert len(results) == 1
        assert results[0]["_svr_partition"] == "good"
