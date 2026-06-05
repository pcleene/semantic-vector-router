"""MongoDB Atlas backend implementation.

Uses PyMongo's native async support (pymongo 4.5+) instead of motor.

Supports:
- BinData vector storage (float32, int8, packed_bit) for optimized storage
- Automatic index quantization (scalar, binary) for reduced RAM
- Pre-quantized vector ingestion from embedding providers
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import (
    AutoReconnect,
    BulkWriteError,
    NetworkTimeout,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.backends.mongodb.indexes import MongoDBIndexOps
from semantic_vector_router.backends.mongodb.vectors import (
    query_vector_for_search,
)
from semantic_vector_router.backends.mongodb.views import MongoDBViewOps
from semantic_vector_router.config import get_connection_string
from semantic_vector_router.exceptions import (
    ConnectionError,
    SearchError,
    SVRException,
)
from semantic_vector_router.models import (
    IndexLocation,
    MongoDBIndexQuantization,
    PartitionInfo,
    SVRConfig,
)
from semantic_vector_router.models.backend import IndexStatus, PartitionStorageResult
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.retry import async_retry

logger = get_logger(__name__)

# Transient MongoDB exceptions that are safe to retry
RETRYABLE_MONGODB = (AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError)


class MongoDBBackend(BaseBackend):
    """MongoDB Atlas backend with vector search support.

    Uses PyMongo's native async support (AsyncMongoClient) for async operations.
    """

    def __init__(self, config: SVRConfig):
        """Initialize MongoDB backend.

        Args:
            config: SVR configuration.
        """
        super().__init__(config)
        self._client: Optional[AsyncMongoClient] = None
        self.__db: Optional[AsyncDatabase] = None
        self._server_version: Optional[tuple[int, ...]] = None
        self._last_health_check: Optional[float] = None
        self._views = MongoDBViewOps(config)
        self._indexes = MongoDBIndexOps()
        self._source_index_ensured = False

    @property  # type: ignore[override]
    def _db(self) -> Optional[AsyncDatabase]:
        return self.__db

    @_db.setter
    def _db(self, value: Optional[AsyncDatabase]) -> None:
        self.__db = value
        if hasattr(self, "_views"):
            self._views._db = value
        if hasattr(self, "_indexes"):
            self._indexes._db = value

    async def _with_retry(self, func, *args, **kwargs):
        """Execute an async callable with retry based on resilience config."""
        r = self.config.resilience
        return await async_retry(
            func,
            args=args,
            kwargs=kwargs,
            max_attempts=r.max_retry_attempts,
            base_delay=r.retry_base_delay,
            max_delay=r.retry_max_delay,
            retryable_exceptions=RETRYABLE_MONGODB,
        )

    @property
    def client(self) -> AsyncMongoClient:
        """Get the MongoDB client.

        Raises:
            ConnectionError: If not connected.
        """
        if self._client is None:
            raise ConnectionError("Not connected to MongoDB. Call connect() first.")
        return self._client

    @property
    def db(self) -> AsyncDatabase:
        """Get the database.

        Raises:
            ConnectionError: If not connected.
        """
        if self._db is None:
            raise ConnectionError("Not connected to MongoDB. Call connect() first.")
        return self._db

    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            connection_string = get_connection_string(self.config)
            self._client = AsyncMongoClient(
                connection_string,
                connectTimeoutMS=self.config.resilience.connection_timeout_ms,
                serverSelectionTimeoutMS=self.config.resilience.server_selection_timeout_ms,
                maxPoolSize=self.config.database.max_pool_size,
                minPoolSize=self.config.database.min_pool_size,
                maxIdleTimeMS=self.config.database.max_idle_time_ms or None,
                waitQueueTimeoutMS=self.config.database.wait_queue_timeout_ms or None,
            )
            self._db = self._client[self.config.database.database]

            # Verify connection and detect server version
            await self._client.admin.command("ping")
            server_info = await self._client.server_info()
            version_str = server_info.get("version", "0.0.0")
            self._server_version = tuple(int(p) for p in version_str.split(".")[:3])
            logger.info(
                f"Connected to MongoDB {version_str}, "
                f"database: {self.config.database.database}"
            )
        except PyMongoError as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._db = None
            self._last_health_check = None
            logger.info("Disconnected from MongoDB")

    @property
    def server_version(self) -> tuple[int, ...]:
        """Get the server version tuple, e.g. (8, 0, 19)."""
        return self._server_version or (0, 0, 0)

    @property
    def supports_search_index_on_views(self) -> bool:
        """Whether the server supports creating search indexes on views via driver.

        MongoDB 8.1+ supports driver methods (createSearchIndex, etc.) on views.
        MongoDB 8.0 only supports this via the Atlas UI/Admin API.
        Below 8.0, views cannot have search indexes at all.
        """
        return self.server_version >= (8, 1, 0)

    async def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        if self._client is None:
            return False
        try:
            await self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    async def health_check(self) -> bool:
        """Check connection health, with staleness caching.

        Skips the actual ping if last success was within health_check_interval_s.
        """
        if self._client is None:
            return False

        now = time.monotonic()
        interval = self.config.resilience.health_check_interval_s
        if self._last_health_check is not None and (now - self._last_health_check) < interval:
            return True  # Still fresh

        try:
            await self._client.admin.command("ping")
            self._last_health_check = now
            return True
        except PyMongoError:
            self._last_health_check = None
            return False

    # View Management — delegated to MongoDBViewOps

    def _build_partition_view_pipeline(
        self,
        partition_name: str,
        filter_value: Any,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Build the aggregation pipeline for a partition view."""
        return self._views._build_partition_view_pipeline(
            partition_name, filter_value, filter_expression
        )

    def _build_embedding_field_expression(self) -> Any:
        """Build embedding text expression for view $addFields stage."""
        return self._views._build_embedding_field_expression()

    def _build_concat_expression(self) -> Any:
        """Legacy alias — delegates to _build_embedding_field_expression()."""
        return self._views._build_embedding_field_expression()

    def _build_template_expression(
        self, template: str, fields: list[str]
    ) -> dict[str, Any]:
        """Build $concat expression from template."""
        return self._views._build_template_expression(template, fields)

    async def create_partition_view(
        self,
        partition_name: str,
        filter_value: Any,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a view for a partition. Retries on transient errors."""
        return await self._views.create_partition_view(
            partition_name, filter_value, filter_expression,
            retry_func=self._with_retry,
        )

    async def delete_partition_view(self, view_name: str) -> None:
        """Delete a partition view."""
        await self._views.delete_partition_view(view_name)

    async def view_exists(self, view_name: str) -> bool:
        """Check if a view exists."""
        return await self._views.view_exists(view_name)

    # Index Management — delegated to MongoDBIndexOps

    async def create_vector_search_index(
        self,
        collection_name: str,
        index_name: str,
        embedding_field: str,
        dimensions: int,
        similarity: str,
        filter_fields: Optional[list[str]] = None,
        quantization: Optional[MongoDBIndexQuantization] = None,
    ) -> None:
        """Create a vector search index."""
        await self._indexes.create_vector_search_index(
            collection_name, index_name, embedding_field, dimensions,
            similarity, filter_fields, quantization,
            retry_func=self._with_retry,
        )

    async def delete_vector_search_index(
        self, collection_name: str, index_name: str
    ) -> None:
        """Delete a vector search index."""
        await self._indexes.delete_vector_search_index(collection_name, index_name)

    async def index_exists(self, collection_name: str, index_name: str) -> bool:
        """Check if an index exists."""
        return await self._indexes.index_exists(collection_name, index_name)

    async def get_index_status(
        self, collection_name: str, index_name: str
    ) -> dict[str, Any]:
        """Get the status of an index."""
        return await self._indexes.get_index_status(collection_name, index_name)

    # Search Operations

    def _build_vector_search_pipeline(
        self,
        partition: PartitionInfo,
        limit: int,
        num_candidates: int,
        query_vector: Optional[list[float]] = None,
        query_string: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Build the vector search aggregation pipeline.

        Exactly one of query_vector or query_string must be provided.

        Args:
            partition: Partition to search.
            limit: Max results.
            num_candidates: ANN candidates (ignored when exact=True).
            query_vector: Query vector (BYOM mode).
            query_string: Query string (Atlas auto-embedding mode).
            filters: Additional filters from user.
            exact: If True, use brute-force exact search (no ANN).
                Sets ``$vectorSearch.exact: true`` and omits numCandidates.
            post_native: Raw MongoDB aggregation stages appended AFTER
                ``$vectorSearch`` + ``$addFields``. These run per-partition
                before result merging. Trusted application code only.

        Returns:
            Aggregation pipeline.
        """
        # FIELDS mode: use partition-specific embedding field path
        embedding_path = (
            partition.embedding_field
            if partition.embedding_field
            else self.config.vector_search.embedding_field
        )

        vector_search_stage: dict[str, Any] = {
            "$vectorSearch": {
                "index": partition.index_name,
                "path": embedding_path,
                "limit": limit,
            }
        }

        # ANN vs exact
        if exact:
            vector_search_stage["$vectorSearch"]["exact"] = True
        else:
            vector_search_stage["$vectorSearch"]["numCandidates"] = num_candidates

        # Add query based on which parameter was provided
        if query_string is not None:
            vector_search_stage["$vectorSearch"]["queryString"] = query_string
        elif query_vector is not None:
            # Convert query vector to appropriate format for the configured
            # storage format (e.g., BinData INT8 for pre-quantized storage)
            search_vector = query_vector_for_search(
                query_vector, self.config.vector_storage.storage_format
            )
            vector_search_stage["$vectorSearch"]["queryVector"] = search_vector
        else:
            raise SearchError(
                "Either query_vector or query_string must be provided",
                details={"partition": partition.name}
            )

        # Build combined filter for $vectorSearch
        combined_filter: dict[str, Any] = {}

        # Add partition pre-filter when searching on the source collection.
        # - SOURCE mode always searches on source → needs partition filter
        # - VIEWS mode on <8.1 uses source index → needs partition filter
        # - VIEWS mode on 8.1+ searches on view → view scopes data, no filter needed
        # - FIELDS mode has per-partition index → no filter needed
        source_collection = self.config.database.source_collection
        needs_partition_filter = (
            partition.index_location == IndexLocation.SOURCE
            or (partition.index_location == IndexLocation.VIEWS
                and partition.search_collection == source_collection)
        )
        if needs_partition_filter:
            partition_field = self.config.partitioning.field
            if partition.filter_expression:
                combined_filter.update(partition.filter_expression)
            else:
                combined_filter[partition_field] = partition.filter_value

        # Add user-provided filters
        if filters:
            combined_filter.update(filters)

        # Only add filter if we have conditions
        if combined_filter:
            vector_search_stage["$vectorSearch"]["filter"] = combined_filter

        pipeline = [
            # Stage 1: $vectorSearch (MUST be first — Atlas requirement)
            vector_search_stage,
            # Stage 2: SVR metadata
            {
                "$addFields": {
                    "_svr_partition": partition.name,
                    "_svr_score": {"$meta": "vectorSearchScore"}
                }
            },
            # Stage 3+: User's native pipeline stages (raw passthrough)
            *(post_native or []),
        ]

        return pipeline

    async def _execute_search_pipeline(
        self,
        partition: PartitionInfo,
        pipeline: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Execute a vector search pipeline on the appropriate collection.

        Shared execution logic for both vector and query-based search.
        """
        async def _do_search() -> list[dict[str, Any]]:
            search_collection = partition.search_collection or partition.view_name
            assert search_collection is not None
            collection = self.db[search_collection]

            timeout_ms = self.config.resilience.search_timeout_ms
            cursor = await collection.aggregate(pipeline, maxTimeMS=timeout_ms)
            results = await cursor.to_list(length=limit)

            mode = partition.index_location.value.upper()
            logger.debug(
                f"Search on partition {partition.name} ({mode} mode, "
                f"collection={search_collection}) returned {len(results)} results"
            )
            return results

        try:
            return await self._with_retry(_do_search)
        except OperationFailure as e:
            raise SearchError(
                f"Vector search failed on partition {partition.name}: {e}",
                details={"partition": partition.name, "error_code": e.code}
            )

    async def execute_search(
        self,
        partition: PartitionInfo,
        query_vector: list[float],
        limit: int,
        num_candidates: int,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
        pre_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Execute vector search on a single partition using a query vector.

        Uses the appropriate collection based on index location mode:
        - VIEWS mode: Searches on the partition view
        - SOURCE mode: Searches on source collection with partition filter

        Retries on transient MongoDB errors.

        Note: ``pre_native`` is ignored for MongoDB because ``$vectorSearch``
        doesn't accept arbitrary expressions in its filter.
        """
        pipeline = self._build_vector_search_pipeline(
            partition, limit, num_candidates,
            query_vector=query_vector, filters=filters,
            exact=exact, post_native=post_native,
        )
        return await self._execute_search_pipeline(partition, pipeline, limit)

    async def search_partitions(
        self,
        partitions: list[PartitionInfo],
        limit: int,
        query_vector: Optional[list[float]] = None,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
        pre_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Execute vector search across multiple partitions in parallel.

        Exactly one of query_vector or query must be provided.
        ``post_native`` stages run per-partition before merging.
        ``pre_native`` is ignored for MongoDB.
        """
        if not partitions:
            return []

        num_candidates = limit * self.config.vector_search.num_candidates_multiplier

        # Create search tasks for all partitions
        tasks = []
        for p in partitions:
            if query_vector is not None:
                tasks.append(
                    self.execute_search(
                        partition=p,
                        query_vector=query_vector,
                        limit=limit,
                        num_candidates=num_candidates,
                        filters=filters,
                        exact=exact,
                        post_native=post_native,
                        pre_native=pre_native,
                    )
                )
            elif query is not None:
                tasks.append(
                    self.execute_search_with_query(
                        partition=p,
                        query=query,
                        limit=limit,
                        num_candidates=num_candidates,
                        filters=filters,
                        exact=exact,
                        post_native=post_native,
                    )
                )

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results, handling any errors
        combined: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Search failed on partition {partitions[i].name}: {result}"
                )
                continue
            combined.extend(result)  # type: ignore[arg-type]

        return combined

    # Document Writes

    async def insert_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """Insert documents into a MongoDB collection.

        Uses insert_many with ordered=False to continue on individual
        insert failures (e.g. duplicate _id). On partial failure,
        returns the count of successfully inserted documents.

        Args:
            documents: List of document dicts to insert.
            collection_name: Target collection. Defaults to source collection.

        Returns:
            Number of documents successfully inserted.
        """
        if not documents:
            return 0

        async def _do_insert() -> int:
            coll_name = collection_name or self.config.database.source_collection
            collection = self.db[coll_name]
            result = await collection.insert_many(documents, ordered=False)
            return len(result.inserted_ids)

        try:
            return await self._with_retry(_do_insert)
        except BulkWriteError as e:
            n_inserted = e.details.get("nInserted", 0)
            logger.warning(
                f"Bulk write partially failed: {n_inserted}/{len(documents)} "
                f"documents inserted"
            )
            return n_inserted
        except RETRYABLE_MONGODB:
            raise
        except PyMongoError as e:
            raise SVRException(f"Failed to insert documents: {e}")

    # Collection Operations

    async def get_distinct_values(
        self, field: str, filter_expression: Optional[dict[str, Any]] = None
    ) -> list[Any]:
        """Get distinct values for a field. Retries on transient errors."""
        async def _do_distinct() -> list[Any]:
            collection = self.db[self.config.database.source_collection]
            if filter_expression:
                return await collection.distinct(field, filter_expression)
            return await collection.distinct(field)

        try:
            return await self._with_retry(_do_distinct)
        except RETRYABLE_MONGODB:
            raise
        except PyMongoError as e:
            raise SVRException(f"Failed to get distinct values for {field}: {e}")

    async def count_documents(
        self,
        collection_name: Optional[str] = None,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> int:
        """Count documents in a collection. Retries on transient errors."""
        async def _do_count() -> int:
            coll_name = collection_name or self.config.database.source_collection
            collection = self.db[coll_name]
            filter_expr = filter_expression or {}
            return await collection.count_documents(filter_expr)

        try:
            return await self._with_retry(_do_count)
        except RETRYABLE_MONGODB:
            raise
        except PyMongoError as e:
            raise SVRException(f"Failed to count documents: {e}")

    async def get_partition_document_counts(
        self, field: str
    ) -> dict[str, int]:
        """Get document counts grouped by partition field. Retries on transient errors."""
        async def _do_counts() -> dict[str, int]:
            collection = self.db[self.config.database.source_collection]
            pipeline: list[dict[str, Any]] = [
                {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            cursor = await collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            return {r["_id"]: r["count"] for r in results if r["_id"] is not None}

        try:
            return await self._with_retry(_do_counts)
        except RETRYABLE_MONGODB:
            raise
        except PyMongoError as e:
            raise SVRException(f"Failed to get partition counts: {e}")

    async def get_collection_stats(
        self, collection_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            coll_name = collection_name or self.config.database.source_collection
            stats = await self.db.command("collStats", coll_name)
            return {
                "name": coll_name,
                "count": stats.get("count", 0),
                "size": stats.get("size", 0),
                "avgObjSize": stats.get("avgObjSize", 0),
                "storageSize": stats.get("storageSize", 0),
                "indexes": stats.get("nindexes", 0),
            }
        except PyMongoError as e:
            return {"name": collection_name, "error": str(e)}

    # Change Stream

    async def watch_collection(  # type: ignore[override]
        self,
        pipeline: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Watch collection for changes using change streams."""
        collection = self.db[self.config.database.source_collection]

        try:
            async with await collection.watch(pipeline=pipeline) as stream:
                async for change in stream:
                    yield change
        except PyMongoError as e:
            raise SVRException(f"Change stream error: {e}")

    # ── AutoEmbeddingCapable protocol ──────────────────────────────

    async def execute_search_with_query(
        self,
        partition: PartitionInfo,
        query: str,
        limit: int,
        num_candidates: int,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Execute search using Atlas auto-embedding (query string only)."""
        pipeline = self._build_vector_search_pipeline(
            partition, limit, num_candidates,
            query_string=query, filters=filters,
            exact=exact, post_native=post_native,
        )
        return await self._execute_search_pipeline(partition, pipeline, limit)

    # ── New BaseBackend interface (Phase 12) ─────────────────────────

    async def create_partition_storage(
        self,
        partition: PartitionInfo,
        config: SVRConfig,
    ) -> PartitionStorageResult:
        """Create storage for a partition based on index location mode.

        Returns a PartitionStorageResult enriched with view_name,
        search_collection, and embedding_field so the provisioner can
        build a PartitionInfo without knowing mode-specific details.
        """
        index_location = config.vector_storage.index_on
        source_collection = config.database.source_collection

        if index_location == IndexLocation.VIEWS:
            view_name = await self.create_partition_view(
                partition_name=partition.name,
                filter_value=partition.filter_value,
                filter_expression=partition.filter_expression,
            )
            # Determine search_collection based on MongoDB version
            if self.supports_search_index_on_views:
                search_collection = view_name
            else:
                search_collection = source_collection
            return PartitionStorageResult(
                storage_name=view_name,
                storage_type="view",
                view_name=view_name,
                search_collection=search_collection,
            )
        elif index_location == IndexLocation.FIELDS:
            base_field = config.vector_search.embedding_field
            safe_name = partition.name.replace("-", "_").replace(" ", "_").lower()
            embedding_field = f"{base_field}_{safe_name}"
            return PartitionStorageResult(
                storage_name=partition.name,
                storage_type="field",
                search_collection=source_collection,
                embedding_field=embedding_field,
            )
        else:  # SOURCE
            # Optionally create a browsing view (non-search)
            view_prefix = config.partitioning.view_prefix
            view_name = f"{view_prefix}{partition.name}"
            return PartitionStorageResult(
                storage_name=source_collection,
                storage_type="source",
                view_name=view_name,
                search_collection=source_collection,
            )

    async def delete_partition_storage(
        self,
        partition: PartitionInfo,
    ) -> None:
        """Delete storage for a partition."""
        if partition.view_name:
            await self.delete_partition_view(partition.view_name)

    async def partition_storage_exists(
        self,
        partition: PartitionInfo,
    ) -> bool:
        """Check if partition storage exists."""
        if partition.view_name:
            return await self.view_exists(partition.view_name)
        return True  # SOURCE and FIELDS always "exist"

    async def ensure_source_index(
        self,
        auto_detect_filters: bool = False,
        extra_filter_fields: Optional[list[str]] = None,
    ) -> str:
        """Ensure the source collection has a shared vector search index.

        Used for SOURCE mode and VIEWS mode on pre-8.1 MongoDB. Creates
        the index if it doesn't exist, caches the result.

        Args:
            auto_detect_filters: Auto-detect filterable fields.
            extra_filter_fields: Additional filter fields.

        Returns:
            The source index name.
        """
        from semantic_vector_router.backends.mongodb.index_manager import (
            SOURCE_INDEX_NAME,
        )

        if self._source_index_ensured:
            return SOURCE_INDEX_NAME

        source_collection = self.config.database.source_collection

        # Check if index already exists
        if not await self.index_exists(source_collection, SOURCE_INDEX_NAME):
            # Build filter fields list
            filter_fields = [self.config.partitioning.field]

            if extra_filter_fields:
                for f in extra_filter_fields:
                    if f not in filter_fields:
                        filter_fields.append(f)

            if auto_detect_filters:
                try:
                    from semantic_vector_router.utils.field_analyzer import (
                        analyze_fields,
                        get_recommended_filter_fields,
                    )
                    analyses = await analyze_fields(self, self.config)
                    recommended = get_recommended_filter_fields(analyses)
                    for f in recommended:
                        if f not in filter_fields:
                            filter_fields.append(f)
                    if recommended:
                        logger.info(f"Auto-detected filter fields: {recommended}")
                except Exception as e:
                    logger.warning(f"Filter field auto-detection failed: {e}")

            await self.create_vector_search_index(
                collection_name=source_collection,
                index_name=SOURCE_INDEX_NAME,
                embedding_field=self.config.vector_search.embedding_field,
                dimensions=self.config.vector_search.dimensions,
                similarity=self.config.vector_search.similarity.value,
                filter_fields=filter_fields,
                quantization=self.config.vector_storage.index_quantization,
            )
            logger.info(
                f"Created source index '{SOURCE_INDEX_NAME}' on "
                f"'{source_collection}' with filter fields {filter_fields}"
            )
        else:
            logger.info(f"Source index '{SOURCE_INDEX_NAME}' already exists")

        self._source_index_ensured = True
        return SOURCE_INDEX_NAME

    async def create_partition_index(
        self,
        partition: PartitionInfo,
        config: SVRConfig,
    ) -> str:
        """Create a vector search index for a partition."""
        index_location = config.vector_storage.index_on
        source_collection = config.database.source_collection
        embedding_field = config.vector_search.embedding_field
        dimensions = config.vector_search.dimensions
        similarity = config.vector_search.similarity.value
        quantization = config.vector_storage.index_quantization

        if index_location == IndexLocation.FIELDS:
            assert partition.embedding_field is not None
            index_name = f"{config.partitioning.index_name_prefix}{partition.name}"
            await self.create_vector_search_index(
                collection_name=source_collection,
                index_name=index_name,
                embedding_field=partition.embedding_field,
                dimensions=dimensions,
                similarity=similarity,
                quantization=quantization,
            )
            return index_name
        elif index_location == IndexLocation.VIEWS:
            if self.supports_search_index_on_views and partition.view_name:
                # MongoDB 8.1+: per-view search index
                index_name = f"{config.partitioning.index_name_prefix}{partition.name}"
                await self.create_vector_search_index(
                    collection_name=partition.view_name,
                    index_name=index_name,
                    embedding_field=embedding_field,
                    dimensions=dimensions,
                    similarity=similarity,
                    quantization=quantization,
                )
                return index_name
            else:
                # Pre-8.1: shared source index
                return await self.ensure_source_index()
        else:
            # SOURCE mode: shared source index
            return await self.ensure_source_index()

    async def delete_partition_index(
        self,
        partition: PartitionInfo,
    ) -> None:
        """Delete a partition's vector search index."""
        if partition.index_name:
            collection = (
                partition.search_collection
                or partition.view_name
                or self.config.database.source_collection
            )
            await self.delete_vector_search_index(collection, partition.index_name)

    async def get_partition_index_status(
        self,
        partition: PartitionInfo,
    ) -> IndexStatus:
        """Get the status of a partition's index."""
        if not partition.index_name:
            return IndexStatus.NOT_FOUND
        collection = (
            partition.search_collection
            or partition.view_name
            or self.config.database.source_collection
        )
        raw_status = await self.get_index_status(collection, partition.index_name)
        status_str = raw_status.get("status", "not_found")
        queryable = raw_status.get("queryable", False)
        if queryable:
            return IndexStatus.READY
        if status_str == "not_found":
            return IndexStatus.NOT_FOUND
        if status_str == "error":
            return IndexStatus.ERROR
        return IndexStatus.BUILDING

    # Utility Methods

    async def list_views(self) -> list[str]:
        """List all views in the database."""
        return await self._views.list_views()

    async def list_partition_views(self) -> list[str]:
        """List all SVR partition views."""
        return await self._views.list_partition_views()

    async def get_sample_document(
        self, collection_name: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Get a sample document from a collection.

        Args:
            collection_name: Collection name. Defaults to source collection.

        Returns:
            Sample document or None if collection is empty.
        """
        try:
            coll_name = collection_name or self.config.database.source_collection
            collection = self.db[coll_name]
            doc = await collection.find_one()
            return doc
        except PyMongoError:
            return None

    async def get_field_names(
        self, collection_name: Optional[str] = None
    ) -> list[str]:
        """Get field names from a sample document.

        Args:
            collection_name: Collection name. Defaults to source collection.

        Returns:
            List of field names.
        """
        doc = await self.get_sample_document(collection_name)
        if doc:
            return list(doc.keys())
        return []
