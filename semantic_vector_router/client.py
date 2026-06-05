"""Main SVRClient - primary interface for Semantic Vector Router."""

import os
import time
from pathlib import Path
from typing import Any, Optional, Union

from semantic_vector_router.backends.base import AutoEmbeddingCapable, BaseBackend
from semantic_vector_router.backends.factory import create_backend
from semantic_vector_router.config import (
    _deep_merge,
    detect_embedding_provider,
    load_config,
    resolve_quickstart_params,
    save_config,
    validate_config,
)
from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.exceptions import ConfigurationError, IngestionError, SearchError
from semantic_vector_router.query.filters import validate_filters
from semantic_vector_router.factories import (
    create_document_embedder,
    create_embedder,
    create_reranker,
)
from semantic_vector_router.ingestion import IngestPipeline, IngestResult
from semantic_vector_router.lifecycle.monitor import PartitionMonitor
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.lifecycle.scanner import PartitionScanner
from semantic_vector_router.lifecycle.splitter import PartitionSplitter
from semantic_vector_router.lifecycle.watcher import PartitionWatcher
from semantic_vector_router.models import (
    EmbeddingMode,
    IngestMode,
    PartitionHealthStatus,
    PartitionInfo,
    SearchResult,
    SVRConfig,
    WatcherStatus,
)
from semantic_vector_router.rerankers.base import BaseReranker
from semantic_vector_router.routing.merger import ResultMerger
from semantic_vector_router.routing.resolver import PartitionResolver
from semantic_vector_router.utils.cache import CacheKey, EmbeddingCache
from semantic_vector_router.utils.logging import get_logger, new_correlation_id
from semantic_vector_router.utils.metrics import (
    MetricsCollector,
    MetricsHandler,
    MetricType,
    NoOpCollector,
)
from semantic_vector_router.utils.rate_limiter import RateLimiterRegistry

logger = get_logger(__name__)


class SVRClient:
    """Main client for Semantic Vector Router.

    Provides the primary interface for:
    - Vector search across partitioned indexes
    - Partition management (create, list, delete)
    - Lifecycle management (watcher, monitoring)

    Example:
        >>> svr = SVRClient()
        >>> results = await svr.search(
        ...     query="wireless headphones",
        ...     partitions=["electronics"],
        ...     limit=10
        ... )
        >>> for hit in results.hits:
        ...     print(f"{hit.document['name']} - {hit.score:.3f}")
    """

    @classmethod
    async def quickstart(
        cls,
        database: Optional[str] = None,
        collection: Optional[str] = None,
        partition_field: Optional[str] = None,
        backend: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        connection_string_env: Optional[str] = None,
        preset: Optional[str] = None,
        **kwargs: Any,
    ) -> "SVRClient":
        """Create a connected SVRClient with minimal configuration.

        Auto-detects settings from environment variables and database schema.
        Only requires database connectivity — everything else has smart defaults.

        Resolution order for each parameter:
        1. Explicit argument passed to quickstart()
        2. Environment variable (SVR_DATABASE, SVR_COLLECTION, SVR_PARTITION_FIELD)
        3. Error with helpful message

        Args:
            database: Database name. Falls back to SVR_DATABASE env var.
            collection: Source collection/table. Falls back to SVR_COLLECTION env var.
            partition_field: Field to partition by. Falls back to SVR_PARTITION_FIELD env var.
            backend: Database backend ("mongodb" or "postgres"). Falls back to SVR_BACKEND.
            embedding_provider: Embedding provider name. Auto-detected if not set.
            connection_string_env: Env var name for connection string. Auto-detected from backend.
            preset: Config preset name ("minimal", "production", "development").
            **kwargs: Additional SVRConfig overrides (e.g., dimensions=1024).

        Returns:
            Connected SVRClient ready for search and ingestion.

        Raises:
            ConfigurationError: If required values can't be resolved.

        Example:
            >>> svr = await SVRClient.quickstart(
            ...     database="my_store",
            ...     collection="products",
            ...     partition_field="category",
            ... )
            >>> results = await svr.search("wireless headphones")
        """
        from dotenv import load_dotenv
        load_dotenv()

        # 1. Resolve params from args + env vars
        params = resolve_quickstart_params(
            database=database,
            collection=collection,
            partition_field=partition_field,
            backend=backend,
            embedding_provider=embedding_provider,
        )

        resolved_backend = params["backend"]

        # 2. Resolve connection string env var
        if connection_string_env is None:
            connection_string_env = (
                "POSTGRES_URI" if resolved_backend == "postgres" else "MONGODB_URI"
            )

        # 3. Validate connection string exists
        conn_string = os.environ.get(connection_string_env)
        if not conn_string:
            raise ConfigurationError(
                f"Database connection string not found. "
                f"Set the {connection_string_env} environment variable.\n\n"
                f"Example:\n"
                f"  export {connection_string_env}='your_connection_string_here'"
            )

        # 4. Validate required params
        resolved_db = params["database"]
        resolved_collection = params["collection"]
        resolved_partition_field = params["partition_field"]

        missing = []
        if not resolved_db:
            missing.append("database (or set SVR_DATABASE env var)")
        if not resolved_collection:
            missing.append("collection (or set SVR_COLLECTION env var)")
        if not resolved_partition_field:
            missing.append("partition_field (or set SVR_PARTITION_FIELD env var)")
        if missing:
            raise ConfigurationError(
                "Missing required parameters:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\n\nExample:\n"
                "  svr = await SVRClient.quickstart(\n"
                "      database='my_db',\n"
                "      collection='my_collection',\n"
                "      partition_field='category',\n"
                "  )"
            )

        # 5. Auto-detect embedding provider
        provider, model, dimensions, api_key_env = detect_embedding_provider(
            params["embedding_provider"]
        )

        # Override dimensions from env var if set
        env_dims = params.get("dimensions")
        if env_dims is not None:
            dimensions = env_dims

        # 6. Build config dict
        config_dict: dict[str, Any] = {
            "database": {
                "backend": resolved_backend,
                "connection_string_env": connection_string_env,
                "database": resolved_db,
                "source_collection": resolved_collection,
            },
            "partitioning": {"field": resolved_partition_field},
            "embedding": {
                "provider": provider,
                "model": model,
                "dimensions": dimensions,
                "api_key_env": api_key_env,
            },
            "vector_search": {"dimensions": dimensions},
        }

        # 7. Apply preset if specified
        if preset:
            from semantic_vector_router.presets import get_preset
            preset_config = get_preset(preset)
            config_dict = _deep_merge(preset_config, config_dict)

        # 8. Apply kwargs overrides
        for key, value in kwargs.items():
            if key == "dimensions":
                config_dict["embedding"]["dimensions"] = value
                config_dict["vector_search"]["dimensions"] = value
            elif key == "model":
                config_dict["embedding"]["model"] = value

        # 9. Handle reranking gracefully (disable if no key available)
        reranker_key = os.environ.get("VOYAGE_API_KEY") or os.environ.get(
            "COHERE_API_KEY"
        )
        if not reranker_key:
            config_dict.setdefault("reranking", {})["enabled"] = False

        # 10. Add postgres config if postgres backend
        if resolved_backend == "postgres":
            config_dict["postgres"] = {
                "connection_string_env": connection_string_env,
                "vector_dimensions": config_dict["embedding"]["dimensions"],
            }

        # 11. Create and connect
        instance = cls(config=config_dict, auto_connect=False)
        await instance.connect()

        logger.info(
            f"Quickstart connected: backend={resolved_backend}, "
            f"database={resolved_db}, collection={resolved_collection}, "
            f"partition_field={resolved_partition_field}, "
            f"embedding={provider}/{model}"
        )

        return instance

    @classmethod
    def quickstart_sync(
        cls,
        **kwargs: Any,
    ) -> "SVRClient":
        """Synchronous version of quickstart() for scripts and notebooks.

        Handles asyncio event loop creation automatically.

        Args:
            **kwargs: Same arguments as quickstart().

        Returns:
            Connected SVRClient ready for search and ingestion.

        Raises:
            RuntimeError: If called from an async context.

        Example:
            >>> svr = SVRClient.quickstart_sync(
            ...     database="my_store",
            ...     collection="products",
            ...     partition_field="category",
            ... )
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "quickstart_sync() cannot be used in an async context. "
                    "Use 'svr = await SVRClient.quickstart(...)' instead."
                )
            return loop.run_until_complete(cls.quickstart(**kwargs))
        except RuntimeError as e:
            if "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(cls.quickstart(**kwargs))
            raise

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Union[dict[str, Any], SVRConfig]] = None,
        auto_connect: bool = True,
        metrics_handler: Optional[MetricsHandler] = None,
    ):
        """Initialize SVRClient.

        Args:
            config_path: Path to config file. Searches standard locations if None.
            config: Configuration dict or SVRConfig object. Takes precedence.
            auto_connect: Whether to connect to database on init.
            metrics_handler: Optional metrics handler for observability.
        """
        # Load configuration
        if isinstance(config, SVRConfig):
            self._config = config
        elif isinstance(config, dict):
            self._config = load_config(config_dict=config)
        else:
            self._config = load_config(config_path=config_path)

        # Validate configuration
        warnings = validate_config(self._config)
        for warning in warnings:
            logger.warning(f"Config warning: {warning}")

        # Metrics collector
        if self._config.metrics.enabled:
            self._metrics: MetricsCollector = MetricsCollector()
        else:
            self._metrics = NoOpCollector()
        if metrics_handler:
            self._metrics.add_handler(metrics_handler)

        # Rate limiter registry
        rl_config = self._config.rate_limiting
        self._rate_limiter_registry = RateLimiterRegistry(
            rl_config if rl_config.enabled else None
        )

        # Embedding cache
        self._embedding_cache = EmbeddingCache(
            max_size=self._config.cache.max_size if self._config.cache.enabled else 0,
            ttl_seconds=self._config.cache.ttl_seconds,
        )

        # Initialize components
        self._backend: Optional[BaseBackend] = None
        self._embedder: Optional[BaseEmbedder] = None
        self._reranker: Optional[BaseReranker] = None
        self._resolver: Optional[PartitionResolver] = None
        self._merger: Optional[ResultMerger] = None
        self._metadata: Any = None  # MetadataStore (initialized in connect)

        # Lifecycle components (lazy initialized)
        self._scanner: Optional[PartitionScanner] = None
        self._provisioner: Optional[PartitionProvisioner] = None
        self._watcher: Optional[PartitionWatcher] = None
        self._monitor: Optional[PartitionMonitor] = None
        self._splitter: Optional[PartitionSplitter] = None

        # Event bus and scheduler (Phase 11)
        self._event_bus: Optional[Any] = None
        self._scheduler: Optional[Any] = None

        self._connected = False
        self._auto_connect_failed = False
        self._monitor_task: Optional[Any] = None  # Background detection task

        if auto_connect:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context (Jupyter, FastAPI, etc.)
                    # Can't use run_until_complete — user must await connect()
                    self._auto_connect_failed = True
                    logger.warning(
                        "SVRClient auto-connect skipped: event loop already running. "
                        "Call 'await client.connect()' explicitly."
                    )
                else:
                    loop.run_until_complete(self.connect())
            except RuntimeError:
                # No event loop available
                self._auto_connect_failed = True
                logger.warning(
                    "SVRClient auto-connect skipped: no event loop available. "
                    "Call 'await client.connect()' explicitly."
                )

    @property
    def config(self) -> SVRConfig:
        """Get the current configuration."""
        return self._config

    async def connect(self) -> None:
        """Connect to the database and initialize components."""
        if self._connected:
            return

        # Initialize backend via factory
        self._backend = create_backend(self._config)
        await self._backend.connect()

        # Initialize metadata store
        from semantic_vector_router.backends.metadata import MetadataStore

        self._metadata = MetadataStore(self._config)
        if not self._config.lifecycle.metadata.connection_string_env:
            # Share the main backend's database reference (MongoDB only)
            if hasattr(self._backend, "_db"):
                self._metadata._set_shared_db(self._backend._db)
        try:
            await self._metadata.connect()
            # Migrate partitions from config file on first connect (idempotent)
            if self._config.partitions.registry:
                migrated = await self._metadata.migrate_from_config(self._config)
                if migrated > 0:
                    logger.info(f"Migrated {migrated} partitions to metadata collection")
        except Exception as e:
            logger.warning(f"Metadata store initialization failed, using config fallback: {e}")
            self._metadata = None

        # Initialize embedder (if BYOM mode)
        if self._config.embedding.mode == EmbeddingMode.BYOM:
            self._embedder = self._create_embedder()

        # Initialize reranker (if enabled, with graceful degradation)
        if self._config.reranking.enabled:
            reranking_key_env = self._config.reranking.api_key_env or ""
            reranking_key = os.environ.get(reranking_key_env) if reranking_key_env else None
            if not reranking_key:
                logger.info(
                    "Reranking disabled: %s not set. Using score-based merging.",
                    reranking_key_env,
                )
                self._reranker = None
            else:
                self._reranker = self._create_reranker()

        # Initialize routing components
        self._resolver = PartitionResolver(
            self._config, metadata=self._metadata, metrics=self._metrics
        )
        self._merger = ResultMerger()

        # Initialize event bus (Phase 11)
        if self._config.events.enabled:
            from semantic_vector_router.events.bus import EventBus
            from semantic_vector_router.events.models import SVREventType
            from semantic_vector_router.events.webhook import (
                WebhookConfig as WHConfig,
            )
            from semantic_vector_router.events.webhook import (
                WebhookDispatcher,
            )

            self._event_bus = EventBus()

            # Register webhook handlers from config
            if self._config.events.webhooks:
                wh_configs = []
                for wh in self._config.events.webhooks:
                    event_types = []
                    for ev_str in wh.events:
                        try:
                            event_types.append(SVREventType(ev_str))
                        except ValueError:
                            logger.warning(f"Unknown event type in webhook config: {ev_str}")
                    wh_configs.append(WHConfig(
                        url=wh.url,
                        events=event_types,
                        secret=wh.secret,
                        timeout_seconds=wh.timeout_seconds,
                        retry_count=wh.retry_count,
                        retry_delay_seconds=wh.retry_delay_seconds,
                        headers=wh.headers,
                        enabled=wh.enabled,
                    ))
                if wh_configs:
                    dispatcher = WebhookDispatcher(wh_configs)
                    self._event_bus.subscribe_all(dispatcher)

        # Initialize scheduler (Phase 11)
        if self._config.scheduler.enabled and self._metadata is not None:
            from semantic_vector_router.scheduler.engine import JobScheduler

            self._scheduler = JobScheduler(
                metadata=self._metadata,
                config=self._config,
                event_bus=self._event_bus,
            )
            self._register_default_jobs()
            await self._scheduler.start()

        self._connected = True
        logger.info("SVRClient connected")

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        # Stop scheduler
        if self._scheduler is not None:
            await self._scheduler.stop()
            self._scheduler = None

        await self.stop_monitor()
        if self._metadata:
            await self._metadata.disconnect()
            self._metadata = None
        if self._backend:
            await self._backend.disconnect()
        self._event_bus = None
        self._connected = False
        logger.info("SVRClient disconnected")

    def _create_embedder(self) -> BaseEmbedder:
        """Create the appropriate embedder based on config."""
        return create_embedder(self._config, self._rate_limiter_registry)

    def _create_reranker(self) -> BaseReranker:
        """Create the appropriate reranker based on config."""
        return create_reranker(self._config, self._rate_limiter_registry)

    async def search(
        self,
        query: str,
        partitions: Optional[Union[list[str], str]] = None,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
        candidates_per_partition: Optional[int] = None,
        rerank: Optional[bool] = None,
        query_vector: Optional[list[float]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
        pre_native: Optional[Any] = None,
    ) -> SearchResult:
        """Execute a vector search across specified partitions.

        Args:
            query: Search query string.
            partitions: Partition specification:
                - None: Use default from config
                - "all": Search all partitions
                - List of names: Search specified partitions
            limit: Maximum results to return.
            filters: SVR portable filters (comparison, set, logical, existence
                operators). Validated once here before backend dispatch.
            candidates_per_partition: Override candidates per partition.
            rerank: Override reranking setting.
            query_vector: Pre-computed query embedding (bypasses SDK embedding).
            exact: If True, use brute-force exact search (no ANN
                approximation). MongoDB sets ``$vectorSearch.exact: true``.
                PostgreSQL disables index scan via ``SET LOCAL``.
            post_native: Backend-specific post-processing. Runs
                **per-partition** before merge — not on the final result.
                MongoDB: ``list[dict]`` — aggregation stages appended after
                ``$vectorSearch`` + ``$addFields`` (e.g., ``$lookup``,
                ``$project``).
                PostgreSQL: ``str`` — raw SQL; core query wrapped in
                ``WITH svr_results AS (...) <post_native>``.
                **Never pass user-generated input.** Trusted application
                code only.
            pre_native: Backend-specific pre-filter conditions.
                MongoDB: **Ignored** (``$vectorSearch`` doesn't accept
                arbitrary expressions in its filter).
                PostgreSQL: ``str`` — additional WHERE conditions AND-joined
                with translated filters (e.g., JSONB containment, tsvector).
                **Never pass user-generated input.**

        Returns:
            SearchResult with hits, metadata, and timing info.

        Raises:
            SearchError: If search fails.
            ValueError: If filters contain unsupported or rejected operators.
            PartitionNotFoundError: If specified partition doesn't exist.
        """
        # Validate filters once at the SDK entry point — before any backend call
        if filters:
            validate_filters(filters)

        if not self._connected:
            if self._auto_connect_failed:
                raise SearchError(
                    "Client not connected. Call 'await client.connect()' explicitly "
                    "when using SVRClient inside an existing async context "
                    "(Jupyter notebooks, FastAPI, etc.)."
                )
            await self.connect()

        correlation_id = new_correlation_id()
        start_time = time.perf_counter()

        try:
            # Get query embedding FIRST (needed for centroid routing)
            if self._config.embedding.mode == EmbeddingMode.BYOM and query_vector is None:
                if self._embedder is None:
                    raise SearchError("Embedder not initialized for BYOM mode")

                # Check embedding cache
                cache_key = CacheKey(
                    text=query,
                    model=self._config.embedding.model,
                    dimensions=self._config.embedding.dimensions,
                    input_type="query",
                )
                cached = self._embedding_cache.get(cache_key)
                if cached is not None:
                    query_vector = cached
                    self._metrics.emit_count(
                        MetricType.CACHE_HIT,
                        provider=self._config.embedding.provider.value,
                    )
                else:
                    embed_start = time.perf_counter()
                    query_vector = await self._embedder.embed(query)
                    embed_ms = (time.perf_counter() - embed_start) * 1000
                    self._embedding_cache.put(cache_key, query_vector)
                    self._metrics.emit_count(
                        MetricType.CACHE_MISS,
                        provider=self._config.embedding.provider.value,
                    )
                    self._metrics.emit_timing(
                        MetricType.EMBEDDING_LATENCY,
                        embed_ms,
                        provider=self._config.embedding.provider.value,
                        model=self._config.embedding.model,
                    )

            # Resolve target partitions (with embedding for centroid routing)
            assert self._resolver is not None
            target_partitions = await self._resolver.resolve(
                partitions,
                filters=filters,
                query_embedding=query_vector,
            )

            if not target_partitions:
                return SearchResult(
                    hits=[],
                    query=query,
                    partitions_searched=[],
                    total_candidates=0,
                    reranked=False,
                    latency_ms=0.0,
                )

            # Determine candidates per partition
            if candidates_per_partition is None:
                candidates_per_partition = self._config.reranking.top_k_per_partition

            # Execute searches in parallel across partitions
            assert self._backend is not None
            if query_vector is not None:
                raw_results = await self._backend.search_partitions(
                    partitions=target_partitions,
                    limit=candidates_per_partition,
                    query_vector=query_vector,
                    filters=filters,
                    exact=exact,
                    post_native=post_native,
                    pre_native=pre_native,
                )
            elif isinstance(self._backend, AutoEmbeddingCapable):
                raw_results = await self._backend.search_partitions(
                    partitions=target_partitions,
                    limit=candidates_per_partition,
                    query=query,
                    filters=filters,
                    exact=exact,
                    post_native=post_native,
                    pre_native=pre_native,
                )
            else:
                raise SearchError(
                    "No query vector provided and backend doesn't support "
                    "auto-embedding. Either provide a query_vector or use "
                    "BYOM embedding mode."
                )

            # Merge results
            assert self._merger is not None
            hits = self._merger.merge(raw_results, limit=len(raw_results))
            total_candidates = len(hits)

            # Determine if reranking is needed
            should_rerank = rerank if rerank is not None else (
                self._config.reranking.enabled and len(target_partitions) > 1
            )

            # Rerank if needed
            if should_rerank and self._reranker and hits:
                rerank_start = time.perf_counter()
                hits = await self._reranker.rerank_hits(
                    query=query,
                    hits=hits,
                    top_k=limit,
                )
                rerank_ms = (time.perf_counter() - rerank_start) * 1000
                self._metrics.emit_timing(
                    MetricType.RERANKING_LATENCY,
                    rerank_ms,
                    provider=self._config.reranking.provider.value,
                    model=self._config.reranking.model,
                )
            else:
                # Just limit without reranking
                hits = hits[:limit]

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Emit search metrics
            self._metrics.emit_timing(
                MetricType.SEARCH_LATENCY,
                latency_ms,
                partitions=str(len(target_partitions)),
                reranked=str(should_rerank),
            )
            self._metrics.emit_count(
                MetricType.SEARCH_RESULTS,
                len(hits),
                partitions=str(len(target_partitions)),
            )
            self._metrics.emit_count(
                MetricType.SEARCH_CANDIDATES,
                total_candidates,
                partitions=str(len(target_partitions)),
            )

            return SearchResult(
                hits=hits,
                query=query,
                partitions_searched=[p.name for p in target_partitions],
                total_candidates=total_candidates,
                reranked=should_rerank,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            self._metrics.emit_count(
                MetricType.ERROR,
                operation="search",
                error_type=type(exc).__name__,
            )
            raise

    def search_sync(
        self,
        query: str,
        partitions: Optional[Union[list[str], str]] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> SearchResult:
        """Synchronous search for scripts and notebooks.

        Args:
            query: Search query text.
            partitions: Partition names to search. Defaults to all active partitions.
            limit: Maximum results to return.
            **kwargs: Additional search arguments.

        Returns:
            SearchResult with hits, metadata, and timing info.

        Example:
            >>> results = svr.search_sync("wireless headphones", limit=5)
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "search_sync() cannot be called from an async context. "
                "Use 'await client.search(...)' instead."
            )

        return asyncio.run(
            self.search(query=query, partitions=partitions, limit=limit, **kwargs)
        )

    # Ingestion

    async def ingest(
        self,
        documents: list[dict[str, Any]],
        partition: Optional[str] = None,
        mode: Optional[IngestMode] = None,
        progress_callback: Optional[Any] = None,
    ) -> IngestResult:
        """Ingest documents into the collection with automatic embedding.

        Accepts raw documents, embeds their text content using the configured
        embedding provider, converts vectors to the configured storage format,
        routes to the correct embedding field based on index mode, and
        inserts/upserts into MongoDB.

        For asymmetric embeddings (Voyage 4), automatically uses the document
        model (config.embedding.effective_document_model) instead of the
        query model.

        Args:
            documents: List of document dictionaries.
            partition: Optional partition name. If None, reads from partition field.
            mode: Override insert/upsert mode. Defaults to config.ingestion.mode.
            progress_callback: Optional callback for progress updates.

        Returns:
            IngestResult with inserted/failed counts and timing.

        Raises:
            IngestionError: If ingestion fails.
        """
        if not self._connected:
            if self._auto_connect_failed:
                raise IngestionError(
                    "Client not connected. Call 'await client.connect()' explicitly "
                    "when using SVRClient inside an existing async context."
                )
            await self.connect()

        correlation_id = new_correlation_id()

        # Create document embedder (may differ from query embedder for asymmetric)
        doc_embedder = self._create_document_embedder()

        pipeline = IngestPipeline(
            backend=self._backend,
            config=self._config,
            embedder=doc_embedder,
            metrics=self._metrics,
            progress_callback=progress_callback,
        )

        result = await pipeline.ingest(
            documents=documents,
            partition=partition,
            mode=mode,
        )

        # Emit ingestion metrics
        self._metrics.emit_timing(
            MetricType.INGEST_LATENCY,
            result.elapsed_ms,
            documents=str(len(documents)),
            inserted=str(result.inserted),
            failed=str(result.failed),
        )
        self._metrics.emit_count(
            MetricType.INGEST_DOCUMENTS,
            result.inserted,
        )
        if result.failed > 0:
            self._metrics.emit_count(
                MetricType.INGEST_ERRORS,
                result.failed,
            )

        # Emit ingest event
        if result.inserted > 0:
            await self._emit_event(
                "ingest.completed",
                partition=partition,
                inserted=result.inserted,
                failed=result.failed,
                elapsed_ms=result.elapsed_ms,
            )

        # Trigger partition detection if configured
        if (
            self._config.ingestion.trigger_detection
            and self._metadata is not None
            and result.inserted > 0
        ):
            try:
                from semantic_vector_router.lifecycle.detector import PartitionDetector

                detector = PartitionDetector(
                    self._backend, self._metadata, self._config
                )
                await detector.run_detection()
            except Exception as e:
                logger.warning(f"Post-ingest detection failed: {e}")

        # Compute centroid for partition if missing (zero API calls)
        # Note: centroid computation requires MongoDB .db access (aggregation pipeline).
        # Skipped for non-MongoDB backends until backend-agnostic centroid is implemented.
        if (
            self._config.routing.centroid_routing.enabled
            and self._metadata is not None
            and self._backend is not None
            and result.inserted > 0
            and partition is not None
            and hasattr(self._backend, "db")
        ):
            try:
                p_info = await self._metadata.get_partition(partition)
                if p_info and p_info.centroid is None:
                    from semantic_vector_router.routing.centroid import (
                        compute_partition_centroid,
                    )

                    collection = self._backend.db[
                        self._config.database.source_collection
                    ]
                    field_path = (
                        p_info.embedding_field
                        or self._config.vector_search.embedding_field
                    )
                    partition_filter = None
                    if p_info.filter_value is not None:
                        partition_filter = {
                            self._config.partitioning.field: p_info.filter_value
                        }

                    centroid = await compute_partition_centroid(
                        collection=collection,
                        embedding_field=field_path,
                        partition_filter=partition_filter,
                        sample_size=self._config.routing.centroid_routing.sample_size,
                    )
                    if centroid:
                        await self._metadata.update_centroid(partition, centroid)
                        logger.info(
                            f"Computed initial centroid for partition {partition}"
                        )
            except Exception as e:
                logger.warning(f"Post-ingest centroid computation failed: {e}")

        return result

    def _create_document_embedder(self) -> BaseEmbedder:
        """Create an embedder configured for document embedding."""
        return create_document_embedder(self._config, self._rate_limiter_registry)

    def _register_default_jobs(self) -> None:
        """Register built-in jobs based on scheduler config."""
        from semantic_vector_router.scheduler.models import JobConfig, JobType

        sc = self._config.scheduler

        if sc.detection_interval:
            self._scheduler.register_job("detection", JobConfig(
                job_id="detection",
                job_type=JobType.DETECTION,
                interval=sc.detection_interval,
                maintenance_window=sc.maintenance_window,
                lock_id="scheduler:detection",
            ))

        if sc.centroid_refresh_interval:
            self._scheduler.register_job("centroid_refresh", JobConfig(
                job_id="centroid_refresh",
                job_type=JobType.CENTROID_REFRESH,
                interval=sc.centroid_refresh_interval,
                maintenance_window=sc.maintenance_window,
                lock_id="scheduler:centroid_refresh",
            ))

        if sc.count_update_interval:
            self._scheduler.register_job("count_update", JobConfig(
                job_id="count_update",
                job_type=JobType.PARTITION_COUNT_UPDATE,
                interval=sc.count_update_interval,
                lock_id="scheduler:count_update",
            ))

        if sc.repartition_check_interval:
            self._scheduler.register_job("repartition_check", JobConfig(
                job_id="repartition_check",
                job_type=JobType.REPARTITION,
                interval=sc.repartition_check_interval,
                maintenance_window=sc.maintenance_window,
                lock_id="scheduler:repartition",
            ))

        if sc.index_health_interval:
            self._scheduler.register_job("index_health", JobConfig(
                job_id="index_health",
                job_type=JobType.INDEX_HEALTH_CHECK,
                interval=sc.index_health_interval,
                lock_id="scheduler:index_health",
            ))

        # Set up built-in job handlers
        self._register_job_handlers()

    def _register_job_handlers(self) -> None:
        """Wire built-in job types to their execution logic."""
        from semantic_vector_router.scheduler.models import JobType

        async def _run_detection() -> dict[str, Any]:
            from semantic_vector_router.lifecycle.detector import PartitionDetector
            detector = PartitionDetector(self._backend, self._metadata, self._config)
            results = await detector.run_detection()
            return {"signals": len(results)}

        async def _run_centroid_refresh() -> dict[str, Any]:
            # Centroid refresh requires MongoDB .db access (aggregation pipeline).
            # Skipped for non-MongoDB backends until backend-agnostic centroid is available.
            if not hasattr(self._backend, "db"):
                logger.info(
                    "Centroid refresh skipped: backend does not support "
                    "MongoDB-style aggregation (non-MongoDB backend)"
                )
                return {"partitions_refreshed": 0, "skipped": "non-mongodb-backend"}
            from semantic_vector_router.routing.centroid import compute_partition_centroid
            partitions = await self._metadata.list_partitions(status="active")
            collection = self._backend.db[self._config.database.source_collection]
            refreshed = 0
            for p in partitions:
                field_path = p.embedding_field or self._config.vector_search.embedding_field
                partition_filter = None
                if p.filter_value is not None:
                    partition_filter = {self._config.partitioning.field: p.filter_value}
                centroid = await compute_partition_centroid(
                    collection=collection,
                    embedding_field=field_path,
                    partition_filter=partition_filter,
                    sample_size=self._config.routing.centroid_routing.sample_size,
                )
                if centroid:
                    await self._metadata.update_centroid(p.name, centroid)
                    refreshed += 1
            return {"partitions_refreshed": refreshed}

        async def _run_count_update() -> dict[str, Any]:
            provisioner = self._get_provisioner()
            counts = await provisioner.update_all_partition_counts()
            return {"partitions_updated": len(counts)}

        self._scheduler.set_job_handler(JobType.DETECTION, _run_detection)
        self._scheduler.set_job_handler(JobType.CENTROID_REFRESH, _run_centroid_refresh)
        self._scheduler.set_job_handler(JobType.PARTITION_COUNT_UPDATE, _run_count_update)

    async def _emit_event(
        self,
        event_type_str: str,
        partition: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Emit an SVR event via the event bus."""
        if self._event_bus is None:
            return
        try:
            from semantic_vector_router.events.models import SVREvent, SVREventType
            event = SVREvent(
                event_type=SVREventType(event_type_str),
                partition=partition,
                details=details,
            )
            await self._event_bus.emit(event)
        except Exception as e:
            logger.warning(f"Failed to emit event: {e}")

    # Partition Management

    def list_partitions(self) -> list[dict[str, Any]]:
        """List all registered partitions.

        Returns:
            List of partition info dictionaries.
        """
        return [
            {
                "name": p.name,
                "document_count": p.document_count,
                "status": p.status.value,
                "view_name": p.view_name,
                "index_name": p.index_name,
            }
            for p in self._config.partitions.registry.values()
        ]

    async def get_partition(self, name: str) -> PartitionInfo:
        """Get details for a specific partition.

        Args:
            name: Partition name.

        Returns:
            PartitionInfo for the partition.

        Raises:
            PartitionNotFoundError: If partition doesn't exist.
        """
        assert self._resolver is not None
        return await self._resolver.get_partition(name)

    async def create_partition(
        self,
        name: str,
        filter_value: Optional[Any] = None,
    ) -> PartitionInfo:
        """Create a new partition.

        Args:
            name: Partition name.
            filter_value: Value to filter on (defaults to name).

        Returns:
            Created PartitionInfo.
        """
        if not self._connected:
            await self.connect()

        provisioner = self._get_provisioner()
        return await provisioner.create_partition(
            name=name,
            filter_value=filter_value,
        )

    async def delete_partition(self, name: str) -> None:
        """Delete a partition.

        Args:
            name: Partition name to delete.
        """
        if not self._connected:
            await self.connect()

        provisioner = self._get_provisioner()
        await provisioner.delete_partition(name)

    async def refresh_partitions(self) -> list[str]:
        """Scan for new partition values and create partitions.

        Returns:
            List of newly created partition names.
        """
        if not self._connected:
            await self.connect()

        scanner = self._get_scanner()
        new_values = await scanner.get_new_partition_values()

        if not new_values:
            return []

        provisioner = self._get_provisioner()
        created = await provisioner.create_partitions_batch(new_values)

        return list(created.keys())

    async def detect_new_partitions(self) -> list[str]:
        """Detect partition values that don't have partitions yet.

        Returns:
            List of partition values without partitions.
        """
        if not self._connected:
            await self.connect()

        scanner = self._get_scanner()
        new_values = await scanner.get_new_partition_values()
        return [str(v) for v in new_values]

    # Lifecycle Management

    async def start_watcher(self) -> None:
        """Start the change stream watcher."""
        if not self._connected:
            await self.connect()

        watcher = self._get_watcher()
        await watcher.start()

    async def stop_watcher(self) -> None:
        """Stop the change stream watcher."""
        if self._watcher:
            await self._watcher.stop()

    def watcher_status(self) -> WatcherStatus:
        """Get watcher status.

        Returns:
            Current watcher status.
        """
        if self._watcher:
            return self._watcher.get_status()
        return WatcherStatus(running=False)

    async def check_partition_health(self) -> list[PartitionHealthStatus]:
        """Check health of all partitions.

        Returns:
            List of partition health statuses.
        """
        if not self._connected:
            await self.connect()

        monitor = self._get_monitor()
        return await monitor.check_all_partitions()

    async def get_health_summary(self) -> dict[str, Any]:
        """Get summary of partition health.

        Returns:
            Health summary dictionary.
        """
        if not self._connected:
            await self.connect()

        monitor = self._get_monitor()
        return await monitor.get_partition_summary()

    # Helper methods for lazy initialization

    def _get_scanner(self) -> PartitionScanner:
        if self._scanner is None:
            assert self._backend is not None
            self._scanner = PartitionScanner(self._backend, self._config)
        return self._scanner

    def _get_provisioner(self) -> PartitionProvisioner:
        if self._provisioner is None:
            assert self._backend is not None
            self._provisioner = PartitionProvisioner(
                self._backend, self._config, auto_save_config=True
            )
            if self._event_bus is not None:
                self._provisioner.set_event_bus(self._event_bus)
        return self._provisioner

    def _get_watcher(self) -> PartitionWatcher:
        if self._watcher is None:
            assert self._backend is not None
            self._watcher = PartitionWatcher(
                self._backend,
                self._config,
                provisioner=self._get_provisioner(),
            )
        return self._watcher

    def _get_monitor(self) -> PartitionMonitor:
        if self._monitor is None:
            assert self._backend is not None
            self._monitor = PartitionMonitor(self._backend, self._config)
        return self._monitor

    def _get_splitter(self) -> PartitionSplitter:
        if self._splitter is None:
            assert self._backend is not None
            self._splitter = PartitionSplitter(
                self._backend, self._config, self._get_provisioner()
            )
        return self._splitter

    # Background monitor

    async def start_monitor(self, interval_seconds: int = 3600) -> None:
        """Start background detection task.

        Only one worker runs detection across all instances (lock-based).

        Args:
            interval_seconds: Seconds between detection runs (default 1 hour).
        """
        import asyncio

        if not self._connected:
            await self.connect()

        if self._monitor_task is not None:
            raise RuntimeError("Monitor already running")

        if self._metadata is None:
            raise RuntimeError("Metadata store not available — cannot run monitor")

        from semantic_vector_router.lifecycle.detector import PartitionDetector

        async def _monitor_loop():
            detector = PartitionDetector(self._backend, self._metadata, self._config)
            while True:
                try:
                    results = await detector.run_detection_with_lock()
                    if results:
                        for r in results:
                            logger.info(
                                f"Detection: {r.signal.value} on {r.partition} — "
                                f"{r.suggested_action}"
                            )
                except Exception as e:
                    logger.error(f"Detection run failed: {e}")
                await asyncio.sleep(interval_seconds)

        self._monitor_task = asyncio.create_task(_monitor_loop())
        logger.info(f"Background monitor started (interval={interval_seconds}s)")

    async def stop_monitor(self) -> None:
        """Stop the background detection task."""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except Exception:
                pass
            self._monitor_task = None
            logger.info("Background monitor stopped")

    # Context manager support

    async def __aenter__(self) -> "SVRClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    # Save configuration

    def save_config(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save current configuration to file.

        Args:
            path: Optional path override.

        Returns:
            Path where config was saved.
        """
        return save_config(self._config, path)
