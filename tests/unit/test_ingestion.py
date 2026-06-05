"""Unit tests for document ingestion pipeline (ingestion.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymongo.errors import BulkWriteError

from semantic_vector_router.exceptions import IngestionError
from semantic_vector_router.ingestion import (
    MAX_INGEST_BATCH,
    IngestPipeline,
    _SafeFormatDict,
)
from semantic_vector_router.models import (
    IngestConfig,
    IngestMode,
    IngestProgress,
    IngestResult,
    IndexLocation,
    SVRConfig,
    VectorStorageFormat,
)
from semantic_vector_router.utils.metrics import MetricsCollector, NoOpCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    config_overrides: dict | None = None,
    embedder: AsyncMock | None = None,
    backend: AsyncMock | None = None,
    metrics: MetricsCollector | None = None,
    progress_callback=None,
    index_on: IndexLocation = IndexLocation.VIEWS,
    storage_format: VectorStorageFormat = VectorStorageFormat.ARRAY,
) -> IngestPipeline:
    """Build an IngestPipeline with mocks."""
    from semantic_vector_router.models import (
        DatabaseConfig,
        EmbeddingConfig,
        PartitioningConfig,
        VectorSearchConfig,
        VectorStorageConfig,
    )

    ingest_overrides = config_overrides or {}
    config = SVRConfig(
        database=DatabaseConfig(
            database="test_db", source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(field="category"),
        vector_storage=VectorStorageConfig(
            index_on=index_on,
            storage_format=storage_format,
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=4,
        ),
        ingestion=IngestConfig(**ingest_overrides),
    )

    if embedder is None:
        embedder = AsyncMock()
        embedder.embed_with_batching = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4]]
        )

    if backend is None:
        backend = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.insert_many = AsyncMock()
        mock_collection.bulk_write = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        backend.db = mock_db

    return IngestPipeline(
        backend=backend,
        config=config,
        embedder=embedder,
        metrics=metrics or NoOpCollector(),
        progress_callback=progress_callback,
    )


def _make_docs(n: int = 3, text_field: str = "text") -> list[dict]:
    """Generate n simple documents."""
    return [{text_field: f"Document {i}", "category": "electronics"} for i in range(n)]


# ---------------------------------------------------------------------------
# Basic ingest flow
# ---------------------------------------------------------------------------


class TestIngestBasic:
    """Test basic ingestion flow."""

    @pytest.mark.asyncio
    async def test_ingest_empty_list(self):
        """Ingesting empty list returns empty result."""
        pipeline = _make_pipeline()
        result = await pipeline.ingest([])
        assert result.inserted == 0
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_ingest_exceeds_max_batch_raises(self):
        """Exceeding MAX_INGEST_BATCH raises IngestionError."""
        pipeline = _make_pipeline()
        docs = _make_docs(MAX_INGEST_BATCH + 1)
        with pytest.raises(IngestionError, match="exceeds maximum"):
            await pipeline.ingest(docs)

    @pytest.mark.asyncio
    async def test_ingest_single_document_insert(self):
        """Single document should be embedded and inserted."""
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

        pipeline = _make_pipeline(embedder=embedder, backend=backend)

        result = await pipeline.ingest([{"text": "Hello world", "category": "test"}])

        embedder.embed_with_batching.assert_awaited_once()
        mock_coll.insert_many.assert_awaited_once()
        assert result.inserted == 1
        assert result.failed == 0
        assert result.elapsed_ms > 0
        assert result.embed_ms > 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(self):
        """Multiple documents should all be embedded and inserted."""
        embedder = AsyncMock()
        embedder.embed_with_batching = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4]] * 3
        )

        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = _make_pipeline(embedder=embedder, backend=backend)

        docs = _make_docs(3)
        result = await pipeline.ingest(docs)

        assert result.inserted == 3
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_ingest_returns_ingest_result(self):
        """Result should be an IngestResult with timing."""
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

        pipeline = _make_pipeline(embedder=embedder, backend=backend)

        result = await pipeline.ingest([{"text": "Hello"}])
        assert isinstance(result, IngestResult)
        assert result.elapsed_ms >= 0
        assert result.embed_ms >= 0
        assert result.write_ms >= 0


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestTextExtraction:
    """Test _extract_text method."""

    @pytest.mark.asyncio
    async def test_extract_single_field(self):
        """Extract text from a single field produces field-labeled output."""
        pipeline = _make_pipeline(config_overrides={"text_fields": ["title"]})
        text = pipeline._extract_text({"title": "Hello World"})
        assert text == "title: Hello World"

    @pytest.mark.asyncio
    async def test_extract_multiple_fields(self):
        """Multiple text fields produce field-labeled, newline-separated output."""
        pipeline = _make_pipeline(
            config_overrides={"text_fields": ["title", "body"], "separator": " | "}
        )
        text = pipeline._extract_text({"title": "Title", "body": "Body text"})
        assert text == "title: Title\nbody: Body text"

    @pytest.mark.asyncio
    async def test_extract_missing_field_skipped(self):
        """Missing fields should be skipped in field-labeled output."""
        pipeline = _make_pipeline(
            config_overrides={"text_fields": ["title", "missing"], "separator": " "}
        )
        text = pipeline._extract_text({"title": "Hello"})
        assert text == "title: Hello"

    @pytest.mark.asyncio
    async def test_extract_all_fields_missing_returns_empty(self):
        """If all text fields are missing, return empty string."""
        pipeline = _make_pipeline(
            config_overrides={"text_fields": ["missing1", "missing2"]}
        )
        text = pipeline._extract_text({"other": "value"})
        assert text == ""

    @pytest.mark.asyncio
    async def test_extract_with_template(self):
        """Template-based extraction should work."""
        pipeline = _make_pipeline(
            config_overrides={"template": "{title}\n{description}"}
        )
        text = pipeline._extract_text(
            {"title": "My Title", "description": "My Description"}
        )
        assert text == "My Title\nMy Description"

    @pytest.mark.asyncio
    async def test_extract_template_missing_key_returns_empty(self):
        """Template with missing key should return empty for that key."""
        pipeline = _make_pipeline(
            config_overrides={"template": "{title} - {missing}"}
        )
        text = pipeline._extract_text({"title": "Hello"})
        assert text == "Hello - "


# ---------------------------------------------------------------------------
# Embedding field routing
# ---------------------------------------------------------------------------


class TestEmbeddingFieldRouting:
    """Test _resolve_embedding_field method."""

    @pytest.mark.asyncio
    async def test_views_mode_uses_standard_field(self):
        """VIEWS mode should use the standard embedding field."""
        pipeline = _make_pipeline(index_on=IndexLocation.VIEWS)
        field = pipeline._resolve_embedding_field({"category": "electronics"})
        assert field == "embedding"

    @pytest.mark.asyncio
    async def test_source_mode_uses_standard_field(self):
        """SOURCE mode should use the standard embedding field."""
        pipeline = _make_pipeline(index_on=IndexLocation.SOURCE)
        field = pipeline._resolve_embedding_field({"category": "electronics"})
        assert field == "embedding"

    @pytest.mark.asyncio
    async def test_fields_mode_uses_partition_field(self):
        """FIELDS mode should use 'embedding_{partition}'."""
        pipeline = _make_pipeline(index_on=IndexLocation.FIELDS)
        field = pipeline._resolve_embedding_field(
            {"category": "electronics"}, partition=None
        )
        assert field == "embedding_electronics"

    @pytest.mark.asyncio
    async def test_fields_mode_with_partition_override(self):
        """FIELDS mode with explicit partition should use that partition."""
        pipeline = _make_pipeline(index_on=IndexLocation.FIELDS)
        field = pipeline._resolve_embedding_field(
            {"category": "furniture"}, partition="electronics"
        )
        assert field == "embedding_electronics"

    @pytest.mark.asyncio
    async def test_fields_mode_missing_partition_raises(self):
        """FIELDS mode without partition value should raise IngestionError."""
        pipeline = _make_pipeline(index_on=IndexLocation.FIELDS)
        with pytest.raises(IngestionError, match="FIELDS mode requires"):
            pipeline._resolve_embedding_field({"other": "value"})


# ---------------------------------------------------------------------------
# Vector conversion
# ---------------------------------------------------------------------------


class TestVectorConversion:
    """Test _convert_vector method."""

    @pytest.mark.asyncio
    async def test_array_format_returns_vector_unchanged(self):
        """ARRAY format should return the list unchanged."""
        pipeline = _make_pipeline(storage_format=VectorStorageFormat.ARRAY)
        vector = [0.1, 0.2, 0.3, 0.4]
        result = pipeline._convert_vector(vector)
        assert result == vector

    @pytest.mark.asyncio
    async def test_bindata_float32_returns_binary(self):
        """BINDATA_FLOAT32 format should return a Binary object."""
        pipeline = _make_pipeline(storage_format=VectorStorageFormat.BINDATA_FLOAT32)
        vector = [0.1, 0.2, 0.3, 0.4]
        result = pipeline._convert_vector(vector)
        # Should be a bson Binary object, not a plain list
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# Write modes
# ---------------------------------------------------------------------------


class TestWriteBatch:
    """Test _write_batch method."""

    @pytest.mark.asyncio
    async def test_insert_mode(self):
        """INSERT mode should call insert_many."""
        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = _make_pipeline(backend=backend)
        docs = [{"text": "a", "embedding": [0.1]}, {"text": "b", "embedding": [0.2]}]

        success, errors = await pipeline._write_batch(
            docs, [0, 1], IngestMode.INSERT
        )
        assert success == 2
        assert errors == []
        mock_coll.insert_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_mode_with_ids(self):
        """UPSERT mode with _id should call bulk_write with UpdateOne."""
        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_result = MagicMock()
        mock_result.upserted_count = 2
        mock_result.modified_count = 0
        mock_result.matched_count = 0
        mock_coll.bulk_write = AsyncMock(return_value=mock_result)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = _make_pipeline(backend=backend)
        docs = [
            {"_id": "doc1", "text": "a", "embedding": [0.1]},
            {"_id": "doc2", "text": "b", "embedding": [0.2]},
        ]

        success, errors = await pipeline._write_batch(
            docs, [0, 1], IngestMode.UPSERT
        )
        assert success == 2
        assert errors == []
        mock_coll.bulk_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bulk_write_error_partial_success(self):
        """BulkWriteError should report partial success."""
        backend = AsyncMock()
        mock_coll = AsyncMock()

        bwe = BulkWriteError(
            {"nInserted": 1, "nUpserted": 0, "writeErrors": [
                {"index": 1, "errmsg": "duplicate key"},
            ]}
        )
        mock_coll.insert_many = AsyncMock(side_effect=bwe)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = _make_pipeline(backend=backend)

        success, errors = await pipeline._write_batch(
            [{"a": 1}, {"a": 2}], [0, 1], IngestMode.INSERT
        )
        assert success == 1
        assert len(errors) == 1
        assert errors[0][0] == 1  # global index
        assert "duplicate key" in errors[0][1]

    @pytest.mark.asyncio
    async def test_catastrophic_write_error(self):
        """Non-BulkWriteError should fail all docs in batch."""
        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock(side_effect=Exception("connection lost"))
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = _make_pipeline(backend=backend)

        success, errors = await pipeline._write_batch(
            [{"a": 1}, {"a": 2}], [0, 1], IngestMode.INSERT
        )
        assert success == 0
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Error handling (continue_on_error)
# ---------------------------------------------------------------------------


class TestContinueOnError:
    """Test continue_on_error behavior."""

    @pytest.mark.asyncio
    async def test_empty_text_skipped_with_continue(self):
        """Documents with no text should be skipped when continue_on_error=True."""
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

        pipeline = _make_pipeline(
            config_overrides={"text_fields": ["text"], "continue_on_error": True},
            embedder=embedder,
            backend=backend,
        )

        docs = [
            {"text": "Good doc", "category": "a"},
            {"other": "no text field", "category": "b"},  # will fail text extraction
        ]

        result = await pipeline.ingest(docs)
        assert result.inserted == 1
        assert result.failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_empty_text_raises_with_stop_on_error(self):
        """Documents with no text should raise when continue_on_error=False."""
        pipeline = _make_pipeline(
            config_overrides={"text_fields": ["text"], "continue_on_error": False}
        )

        docs = [{"other": "no text"}]
        with pytest.raises(IngestionError, match="no extractable text"):
            await pipeline.ingest(docs)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """Test progress callback emission."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        """Progress callback should be called during ingestion."""
        progress_events = []

        def on_progress(p):
            progress_events.append(p.phase)

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

        pipeline = _make_pipeline(
            embedder=embedder, backend=backend, progress_callback=on_progress,
        )

        await pipeline.ingest([{"text": "Hello"}])

        assert "embedding" in progress_events
        assert "writing" in progress_events
        assert "complete" in progress_events

    @pytest.mark.asyncio
    async def test_no_callback_no_crash(self):
        """Ingestion should work fine without a callback."""
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

        pipeline = _make_pipeline(
            embedder=embedder, backend=backend, progress_callback=None,
        )

        result = await pipeline.ingest([{"text": "Hello"}])
        assert result.inserted == 1


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    """Test embedding and write batch sizes."""

    @pytest.mark.asyncio
    async def test_embed_batching(self):
        """Documents should be embedded in configured batch sizes."""
        embedder = AsyncMock()
        # Will be called twice: batch of 2, then batch of 1
        embedder.embed_with_batching = AsyncMock(
            side_effect=[
                [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                [[0.1, 0.2, 0.3, 0.4]],
            ]
        )

        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_coll.insert_many = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = _make_pipeline(
            config_overrides={"batch_size": 2},
            embedder=embedder,
            backend=backend,
        )

        docs = _make_docs(3)
        result = await pipeline.ingest(docs)

        assert embedder.embed_with_batching.call_count == 2
        assert result.inserted == 3


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Test metrics emission during ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_emits_embed_latency_metric(self):
        """Ingestion should emit INGEST_EMBED_LATENCY metric."""
        from semantic_vector_router.utils.metrics import MetricType

        events = []

        class Recorder:
            def handle(self, event):
                events.append(event)

        metrics = MetricsCollector()
        metrics.add_handler(Recorder())

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

        pipeline = _make_pipeline(
            embedder=embedder, backend=backend, metrics=metrics,
        )

        await pipeline.ingest([{"text": "Hello"}])

        embed_events = [e for e in events if e.metric_type == MetricType.INGEST_EMBED_LATENCY]
        assert len(embed_events) == 1
        assert embed_events[0].value >= 0


# ---------------------------------------------------------------------------
# _SafeFormatDict
# ---------------------------------------------------------------------------


class TestSafeFormatDict:
    """Test _SafeFormatDict helper."""

    def test_existing_key(self):
        """Existing key should return its value."""
        d = _SafeFormatDict({"name": "Alice"})
        assert d["name"] == "Alice"

    def test_missing_key_returns_empty_string(self):
        """Missing key should return empty string."""
        d = _SafeFormatDict({"name": "Alice"})
        assert d["missing"] == ""

    def test_format_map_with_missing(self):
        """format_map should handle missing keys gracefully."""
        d = _SafeFormatDict({"title": "Hello"})
        result = "{title} - {missing}".format_map(d)
        assert result == "Hello - "


# ---------------------------------------------------------------------------
# Partition routing in full ingest
# ---------------------------------------------------------------------------


class TestIngestPartitionRouting:
    """Test partition-aware field routing during full ingestion."""

    @pytest.mark.asyncio
    async def test_fields_mode_writes_to_partition_field(self):
        """In FIELDS mode, vector should be written to 'embedding_{partition}'."""
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

        pipeline = _make_pipeline(
            index_on=IndexLocation.FIELDS,
            embedder=embedder,
            backend=backend,
        )

        doc = {"text": "Hello", "category": "electronics"}
        result = await pipeline.ingest([doc])

        # Check the document that was written has the right embedding field
        call_args = mock_coll.insert_many.call_args
        written_docs = call_args[0][0]
        assert "embedding_electronics" in written_docs[0]
        assert result.inserted == 1

    @pytest.mark.asyncio
    async def test_views_mode_writes_to_standard_field(self):
        """In VIEWS mode, vector should be written to 'embedding'."""
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

        pipeline = _make_pipeline(
            index_on=IndexLocation.VIEWS,
            embedder=embedder,
            backend=backend,
        )

        doc = {"text": "Hello", "category": "test"}
        result = await pipeline.ingest([doc])

        call_args = mock_coll.insert_many.call_args
        written_docs = call_args[0][0]
        assert "embedding" in written_docs[0]
        assert result.inserted == 1

    @pytest.mark.asyncio
    async def test_partition_override(self):
        """Explicit partition override should be used in FIELDS mode."""
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

        pipeline = _make_pipeline(
            index_on=IndexLocation.FIELDS,
            embedder=embedder,
            backend=backend,
        )

        doc = {"text": "Hello", "category": "furniture"}
        result = await pipeline.ingest([doc], partition="override_partition")

        call_args = mock_coll.insert_many.call_args
        written_docs = call_args[0][0]
        assert "embedding_override_partition" in written_docs[0]


# ---------------------------------------------------------------------------
# Ingest mode override
# ---------------------------------------------------------------------------


class TestIngestModeOverride:
    """Test mode override in ingest()."""

    @pytest.mark.asyncio
    async def test_default_mode_from_config(self):
        """Default mode should come from config."""
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

        pipeline = _make_pipeline(
            config_overrides={"mode": "insert"},
            embedder=embedder,
            backend=backend,
        )

        await pipeline.ingest([{"text": "Hello"}])
        mock_coll.insert_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mode_override_to_upsert(self):
        """Explicit mode=UPSERT should use bulk_write."""
        embedder = AsyncMock()
        embedder.embed_with_batching = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4]]
        )

        backend = AsyncMock()
        mock_coll = AsyncMock()
        mock_result = MagicMock()
        mock_result.upserted_count = 1
        mock_result.modified_count = 0
        mock_result.matched_count = 0
        mock_coll.bulk_write = AsyncMock(return_value=mock_result)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        backend.db = mock_db

        pipeline = _make_pipeline(
            config_overrides={"mode": "insert"},  # config says insert
            embedder=embedder,
            backend=backend,
        )

        await pipeline.ingest([{"text": "Hello", "_id": "doc1"}], mode=IngestMode.UPSERT)
        mock_coll.bulk_write.assert_awaited_once()
