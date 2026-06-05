"""Document ingestion pipeline for Semantic Vector Router.

Accepts raw documents, embeds their text content, converts vectors to the
configured storage format, routes to the correct embedding field based on
index mode, and inserts/upserts into MongoDB.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional, Union

from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from semantic_vector_router.backends.mongodb import vector_to_bindata
from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.exceptions import IngestionError
from semantic_vector_router.models import (
    IndexLocation,
    IngestMode,
    IngestProgress,
    IngestResult,
    SVRConfig,
)
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.metrics import MetricsCollector, MetricType
from semantic_vector_router.utils.text_serializer import serialize_for_embedding

logger = get_logger(__name__)

# Maximum documents in a single ingest() call before requiring chunking
MAX_INGEST_BATCH = 10_000


class IngestPipeline:
    """Document ingestion pipeline.

    Accepts raw documents, embeds their text content, converts vectors
    to the configured storage format, routes to the correct embedding
    field based on index mode, and inserts/upserts into MongoDB.

    The pipeline processes documents in configurable batches:
    1. Extract text from configurable source fields
    2. Embed text in batches via the configured embedding provider
    3. Convert embedding vectors to BinData format (if configured)
    4. Route each document's vector to the correct field:
       - FIELDS mode: embedding_{partition_name}
       - VIEWS/SOURCE mode: standard embedding field
    5. Write documents to MongoDB in bulk (insert or upsert)
    6. Optionally trigger partition detection

    Error handling is per-document: a single document failure does not
    abort the entire batch (when continue_on_error=True).
    """

    def __init__(
        self,
        backend: Any,
        config: SVRConfig,
        embedder: BaseEmbedder,
        metrics: MetricsCollector,
        progress_callback: Optional[Callable[[IngestProgress], None]] = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            backend: Connected MongoDB backend.
            config: SVR configuration.
            embedder: Embedder configured for document embedding (input_type="document").
            metrics: Metrics collector for observability.
            progress_callback: Optional callback invoked with IngestProgress updates.
                Called after each embedding batch and each write batch completes.
        """
        self._backend = backend
        self._config = config
        self._embedder = embedder
        self._metrics = metrics
        self._progress_callback = progress_callback

    async def ingest(
        self,
        documents: list[dict[str, Any]],
        partition: Optional[str] = None,
        mode: Optional[IngestMode] = None,
    ) -> IngestResult:
        """Ingest documents into the collection.

        Args:
            documents: List of document dictionaries. Each document must contain
                the text fields specified in IngestConfig.text_fields, and (for
                FIELDS mode) the partition field so the pipeline knows which
                embedding field to write to.
            partition: Optional partition name override. If provided, all documents
                are routed to this partition. If None, the pipeline reads the
                partition field from each document to determine routing.
            mode: Override ingestion mode (insert/upsert). Defaults to config value.

        Returns:
            IngestResult with counts of inserted/failed documents and timing.

        Raises:
            IngestionError: If ingestion fails catastrophically (e.g., not connected,
                no embedder, all documents fail and continue_on_error=False).
        """
        start_time = time.perf_counter()
        ingest_config = self._config.ingestion
        effective_mode = mode or ingest_config.mode

        if not documents:
            return IngestResult()

        if len(documents) > MAX_INGEST_BATCH:
            raise IngestionError(
                f"Batch size {len(documents)} exceeds maximum {MAX_INGEST_BATCH}. "
                f"Split into smaller batches."
            )

        result = IngestResult()
        progress = IngestProgress(total=len(documents), phase="embedding")

        # Phase 1: Extract text and filter out empty docs
        texts: list[str] = []
        valid_indices: list[int] = []

        for i, doc in enumerate(documents):
            text = self._extract_text(doc)
            if not text:
                logger.warning(
                    f"Document {i} has no extractable text, skipping",
                    extra={"document_index": i},
                )
                result.failed += 1
                result.errors.append((i, "No extractable text from configured text_fields"))
                progress.failed += 1
                if not ingest_config.continue_on_error:
                    raise IngestionError(
                        f"Document {i} has no extractable text and continue_on_error=False"
                    )
                continue
            texts.append(text)
            valid_indices.append(i)

        if not texts:
            result.elapsed_ms = (time.perf_counter() - start_time) * 1000
            return result

        # Phase 2: Embed in batches
        embed_start = time.perf_counter()
        all_vectors: list[list[Union[float, int]]] = []

        batch_size = ingest_config.batch_size
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]
            batch_vectors = await self._embed_batch(batch_texts)
            all_vectors.extend(batch_vectors)
            progress.embedded += len(batch_texts)
            self._emit_progress(progress)

        result.embed_ms = (time.perf_counter() - embed_start) * 1000
        self._metrics.emit_timing(
            MetricType.INGEST_EMBED_LATENCY,
            result.embed_ms,
            documents=str(len(texts)),
        )

        # Phase 3: Prepare documents with vectors
        progress.phase = "writing"
        self._emit_progress(progress)

        prepared_docs: list[dict[str, Any]] = []
        prepared_global_indices: list[int] = []

        for vec_idx, global_idx in enumerate(valid_indices):
            doc = documents[global_idx].copy()
            vector = all_vectors[vec_idx]

            try:
                embed_field = self._resolve_embedding_field(doc, partition)
                converted = self._convert_vector(vector)
                doc[embed_field] = converted
                prepared_docs.append(doc)
                prepared_global_indices.append(global_idx)
            except IngestionError as e:
                result.failed += 1
                result.errors.append((global_idx, str(e)))
                progress.failed += 1
                if not ingest_config.continue_on_error:
                    raise

        # Phase 4: Write in batches
        write_start = time.perf_counter()
        write_batch_size = ingest_config.write_batch_size

        for batch_start in range(0, len(prepared_docs), write_batch_size):
            batch_docs = prepared_docs[batch_start:batch_start + write_batch_size]
            batch_indices = prepared_global_indices[batch_start:batch_start + write_batch_size]

            success_count, batch_errors = await self._write_batch(
                batch_docs, batch_indices, effective_mode
            )
            result.inserted += success_count
            result.failed += len(batch_errors)
            result.errors.extend(batch_errors)
            progress.written += success_count
            progress.failed += len(batch_errors)
            self._emit_progress(progress)

            if batch_errors and not ingest_config.continue_on_error:
                raise IngestionError(
                    f"Write batch failed: {batch_errors[0][1]}",
                    details={"errors": batch_errors},
                )

        result.write_ms = (time.perf_counter() - write_start) * 1000
        result.elapsed_ms = (time.perf_counter() - start_time) * 1000

        progress.phase = "complete"
        self._emit_progress(progress)

        logger.info(
            f"Ingestion complete: {result.inserted} inserted, {result.failed} failed "
            f"in {result.elapsed_ms:.0f}ms",
            extra={
                "inserted": result.inserted,
                "failed": result.failed,
                "elapsed_ms": result.elapsed_ms,
            },
        )

        return result

    def _extract_text(self, document: dict[str, Any]) -> str:
        """Extract text from a document for embedding.

        Three modes:
        1. **Template**: Uses IngestConfig.template (e.g., "{title}\\n{description}")
        2. **Structured** (default): Builds a field-labeled dict from text_fields
           and serializes via ``serialize_for_embedding()`` for richer embeddings.
        3. **Pre-serialized**: If the document already contains a structured
           ``_svr_embedding_text`` dict (e.g., from PostgreSQL content JSONB),
           serializes it directly.

        Returns empty string if no text fields are present.
        """
        ingest_config = self._config.ingestion

        # Check for pre-built structured embedding text (PostgreSQL path)
        pre_built = document.get("_svr_embedding_text")
        if isinstance(pre_built, dict) and pre_built:
            return serialize_for_embedding(pre_built)

        if ingest_config.template:
            try:
                return ingest_config.template.format_map(
                    _SafeFormatDict(document)
                )
            except (KeyError, ValueError):
                return ""

        # Build structured object from text_fields and serialize with labels
        obj: dict[str, Any] = {}
        for field_name in ingest_config.text_fields:
            value = document.get(field_name)
            if value is not None:
                obj[field_name] = value

        if obj:
            return serialize_for_embedding(obj)
        return ""

    def _resolve_embedding_field(
        self,
        document: dict[str, Any],
        partition: Optional[str] = None,
    ) -> str:
        """Determine which embedding field to write the vector to.

        For FIELDS mode: returns "embedding_{partition_name}" where
        partition_name comes from the document's partition field value
        or the explicit partition override.

        For VIEWS/SOURCE mode: returns the standard embedding field
        from config (e.g., "embedding").

        Args:
            document: The document being ingested.
            partition: Optional partition name override.

        Returns:
            The embedding field name to write the vector to.

        Raises:
            IngestionError: If FIELDS mode and partition cannot be determined.
        """
        index_on = self._config.vector_storage.index_on

        if index_on == IndexLocation.FIELDS:
            partition_name = partition
            if partition_name is None:
                partition_field = self._config.partitioning.field
                partition_name = document.get(partition_field)

            if partition_name is None:
                raise IngestionError(
                    f"FIELDS mode requires partition value. Document lacks "
                    f"'{self._config.partitioning.field}' field and no partition override given."
                )
            return f"embedding_{partition_name}"

        # VIEWS or SOURCE mode
        return self._config.vector_search.embedding_field

    def _convert_vector(
        self,
        vector: list[Union[float, int]],
    ) -> Union[Any, list]:
        """Convert embedding vector to configured storage format.

        For MongoDB backends, applies vector_to_bindata() based on storage_format.
        For non-MongoDB backends, returns the raw float array.

        Args:
            vector: Raw embedding vector from the embedder.

        Returns:
            Converted vector ready for storage.
        """
        if not hasattr(self._backend, "db"):
            return vector
        storage_format = self._config.vector_storage.storage_format
        return vector_to_bindata(vector, storage_format)

    async def _embed_batch(
        self,
        texts: list[str],
    ) -> list[list[Union[float, int]]]:
        """Embed a batch of texts using the configured embedder.

        Handles the embedder's max_batch_size by splitting into sub-batches
        if needed. Emits INGEST_EMBED_LATENCY metrics.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        return await self._embedder.embed_with_batching(texts)

    async def _write_batch(
        self,
        operations: list[dict[str, Any]],
        global_indices: list[int],
        mode: IngestMode,
    ) -> tuple[int, list[tuple[int, str]]]:
        """Write a batch of documents to the backend.

        Dispatches to MongoDB-specific path (bulk_write with BulkWriteError
        handling) when the backend has a ``.db`` attribute, otherwise uses
        the backend-agnostic ``insert_documents()`` method.

        Args:
            operations: List of document dicts ready for insertion.
            global_indices: Original document indices for error reporting.
            mode: Insert or upsert.

        Returns:
            Tuple of (success_count, list of (doc_index, error_message) for failures).
        """
        if hasattr(self._backend, "db"):
            return await self._write_batch_mongodb(operations, global_indices, mode)
        return await self._write_batch_generic(operations, global_indices)

    async def _write_batch_generic(
        self,
        operations: list[dict[str, Any]],
        global_indices: list[int],
    ) -> tuple[int, list[tuple[int, str]]]:
        """Backend-agnostic write using insert_documents()."""
        errors: list[tuple[int, str]] = []
        try:
            inserted = await self._backend.insert_documents(operations)
            return inserted, errors
        except Exception as e:
            for global_idx in global_indices:
                errors.append((global_idx, str(e)))
            return 0, errors

    async def _write_batch_mongodb(
        self,
        operations: list[dict[str, Any]],
        global_indices: list[int],
        mode: IngestMode,
    ) -> tuple[int, list[tuple[int, str]]]:
        """MongoDB-specific write with BulkWriteError handling."""
        collection = self._backend.db[self._config.database.source_collection]
        errors: list[tuple[int, str]] = []

        try:
            if mode == IngestMode.INSERT:
                await collection.insert_many(operations, ordered=False)
                return len(operations), errors
            else:
                # UPSERT mode
                ops = []
                for doc in operations:
                    doc_id = doc.get("_id")
                    if doc_id is None:
                        doc_id = str(uuid.uuid4())
                        doc["_id"] = doc_id
                    ops.append(
                        UpdateOne({"_id": doc_id}, {"$set": doc}, upsert=True)
                    )
                result = await collection.bulk_write(ops, ordered=False)
                success = result.matched_count + result.upserted_count
                return success, errors

        except BulkWriteError as e:
            details = e.details
            n_inserted = details.get("nInserted", 0) + details.get("nUpserted", 0)
            write_errors = details.get("writeErrors", [])

            for we in write_errors:
                idx = we.get("index", 0)
                msg = we.get("errmsg", str(we))
                global_idx = global_indices[idx] if idx < len(global_indices) else idx
                errors.append((global_idx, msg))

            return n_inserted, errors

        except Exception as e:
            # Catastrophic failure — all docs in this batch fail
            for global_idx in global_indices:
                errors.append((global_idx, str(e)))
            return 0, errors

    def _emit_progress(self, progress: IngestProgress) -> None:
        """Emit progress update via callback if registered."""
        if self._progress_callback:
            self._progress_callback(progress)


class _SafeFormatDict(dict):
    """Dict subclass that returns empty string for missing keys in format_map."""

    def __missing__(self, key: str) -> str:
        return ""
