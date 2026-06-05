# Changelog

## [Unreleased] - Phase 11: Operational Scheduling & Event System

### Added
- **Job scheduler engine** (`scheduler/engine.py`) — Background asyncio task that ticks every N seconds, checks which jobs are due based on parsed interval strings, enforces maintenance windows before execution, acquires distributed locks via MongoDB, executes job handlers with configurable timeout. Supports pause/resume, force-run, consecutive failure tracking, and job history persistence.
- **Interval parser** (`scheduler/interval.py`) — Converts human-readable strings (`"1h"`, `"30m"`, `"daily"`, `"weekly"`, `"1h30m"`, `"2d12h"`) to seconds. Supports compound intervals and named presets.
- **Maintenance window enforcement** (`scheduler/window.py`) — `is_within_window()` checks `allowed_days`, `allowed_hours` (start/end), and `timezone` (via `zoneinfo`). Jobs outside the window are skipped with recorded reason.
- **Scheduler models** (`scheduler/models.py`) — `JobType` (6 built-in types), `JobConfig`, `JobStatus`, `JobRun`, `JobState`, `SchedulerStatus`, `MaintenanceWindow`. Full Pydantic v2 models with sensible defaults.
- **Event bus** (`events/bus.py`) — In-process async event dispatch with fire-and-forget semantics. Per-type and global (`*`) subscriptions. Handler exceptions are caught and logged, never propagate. `asyncio.ensure_future` for non-blocking delivery.
- **SVR event types** (`events/models.py`) — `SVREventType` enum with 21 event types across 6 categories: partition lifecycle (`partition.created`, `partition.deleted`), health signals (`health.threshold_breach`, `health.approaching_threshold`, `health.skew_detected`, `health.alert`), operations (`repartition.started`, `repartition.completed`, `repartition.failed`, `repartition.rolled_back`), centroids (`centroid.computed`), ingestion (`ingest.completed`), scheduler (`scheduler.started`, `scheduler.stopped`, `scheduler.job_started`, `scheduler.job_completed`, `scheduler.job_failed`, `scheduler.job_skipped`). `SVREvent` model with `to_dict()` serialization.
- **Webhook dispatcher** (`events/webhook.py`) — HTTP delivery via `httpx` with HMAC-SHA256 request signing (`X-SVR-Signature` header), retry with exponential backoff on 5xx errors (no retry on 4xx), per-webhook event filtering, delivery history tracking, test endpoint support. `WebhookConfig` and `WebhookTestResult` models.
- **Config models** (`models.py`) — `SchedulerConfig` (enabled, tick_interval, worker_id, maintenance_window, per-job intervals, custom_jobs), `EventsConfig` (enabled, webhooks, log_events, store_events, retention), `WebhookConfig` (url, events filter, secret, timeout, retry, headers). Added `scheduler` and `events` fields to `SVRConfig` with backward-compatible defaults.
- **Config validation** (`config.py`) — Validates scheduler intervals via `parse_interval()`, maintenance window hours (0-24), days, timezone. Validates webhook URLs (http/https), timeout (>0), retry_count (>=0), event_retention_days (>0).
- **CLI `svr schedule`** (`cli/schedule.py`) — 7 subcommands: `list` (Rich table of all jobs), `status` (scheduler status), `history <job-id>` (run history), `run <job-id>` (force execute), `pause <job-id>`, `resume <job-id>`, `window` (show maintenance window).
- **CLI `svr webhooks`** (`cli/webhooks.py`) — 3 subcommands: `list` (configured webhooks), `test <url>` (send test event), `history` (delivery history).
- **298 new tests** — `test_interval_parser.py` (45), `test_maintenance_window.py` (50), `test_event_bus.py` (35), `test_webhook_dispatcher.py` (37), `test_scheduler_engine.py` (62), `test_scheduler_cli.py` (25), `test_event_integration.py` (22), `test_scheduler_flow.py` integration (22).
- **Example** — `examples/scheduler_webhooks.py` demonstrating scheduler setup with maintenance windows and webhook event delivery.

### Changed
- **`SVRClient.connect()`** — Initializes `EventBus` (if `events.enabled`), creates `WebhookDispatcher` for configured webhooks, creates and starts `JobScheduler` (if `scheduler.enabled`) with default job registrations (detection, centroid_refresh, count_update, repartition_check, index_health).
- **`SVRClient.disconnect()`** — Stops scheduler before other cleanup.
- **`SVRClient.ingest()`** — Emits `ingest.completed` event after successful ingestion.
- **`PartitionProvisioner`** — Added `set_event_bus()` and `_emit_event()`. Emits `partition.created` and `partition.deleted` events.
- **`RepartitionEngine`** — Accepts optional `event_bus` parameter. Emits `repartition.started`, `repartition.completed`, `repartition.failed`, `repartition.rolled_back`, and `centroid.computed` events.
- **`PartitionDetector`** — Accepts optional `event_bus` parameter. Emits health events (`health.threshold_breach`, `health.approaching_threshold`, `health.skew_detected`, `health.alert`) for each detection result.
- **CLI entry point** (`cli/__init__.py`) — Registers `schedule` and `webhooks` command groups.

## [Unreleased] - Phase 10: Hierarchical Centroid Routing

### Added
- **Hierarchical centroid routing** (`routing/centroid.py`) — `CentroidRouter` that walks the partition tree top-down, scoring partition centroids against query embeddings. Dynamic threshold pruning: `max_score × relative_threshold` adapts to score distribution. Ambiguous queries search more partitions; clear queries prune aggressively. Never returns empty (top-1 fallback). Partitions without centroids are included (never silently excluded).
- **Filter-map routing** — O(1) dict lookup when query filters match the configured partition field. Split-aware: maps filter values directly to leaf descendants. Cached with version-based invalidation.
- **`RoutingMode.AUTO`** — Activates the full routing cascade: explicit → filter-map → centroid → fan-out fallback. `RoutingMode.EXPLICIT` preserves previous behavior exactly (backward compatible).
- **`CentroidRoutingConfig`** model — `enabled`, `relative_threshold` (0.5), `min_score` (0.15), `max_probe_partitions` (5), `sample_size` (500), `centroid_ttl_seconds` (3600), `registry_ttl_seconds` (60).
- **Centroid fields** on `PartitionInfo` — `centroid: Optional[list[float]]`, `centroid_updated_at: Optional[datetime]`.
- **`compute_partition_centroid()`** utility (`routing/centroid.py`) — Samples stored embedding vectors, computes element-wise mean, normalizes to unit length. Zero API calls. Handles both array and BinData vector storage formats.
- **Pure Python vector math** (`utils/vector_math.py`) — `cosine_similarity`, `normalize`, `mean_vector`. No numpy dependency.
- **Centroid routing metrics** — `CENTROID_ROUTE_LATENCY`, `CENTROID_ROUTE_PARTITIONS` added to `MetricType`.
- **Post-repartition centroid computation** — New `compute_centroids` step in repartition workflow computes centroids for newly ACTIVE child partitions.
- **Post-ingest centroid trigger** — After ingestion, computes centroid for partition if none exists and centroid routing is enabled.
- **CLI `svr partitions compute-centroids`** — Compute centroids for all or specific partitions. Options: `--partition`, `--sample-size`.
- **Config validation** for centroid routing — `relative_threshold` in (0, 1], `min_score` in [0, 1), `sample_size > 0`, TTL values > 0.
- **New unit tests** — `test_centroid_router.py`, `test_centroid_computation.py`, `test_filter_map_routing.py`, `test_centroid_wiring.py`.
- **Integration test** — `test_centroid_routing_flow.py`: ingest → compute centroids → search without partition hint → verify correct partition selected.
- **Example** — `examples/centroid_routing.py` demonstrating zero-filter search that routes intelligently.

### Changed
- **`SVRClient.search()`** — Embedding now happens BEFORE partition resolution (needed for centroid routing). Query embedding and filters are passed to resolver for the routing cascade. One embedding per query, cache still applies.
- **`PartitionResolver`** — Accepts optional `metrics` parameter. `resolve()` now takes `query_embedding` parameter. Four-step cascade: explicit → filter-map → centroid → fan-out.
- **`RepartitionEngine`** — Step 6 (`compute_centroids`) added to step handlers for post-repartition centroid computation.
- **`routing/__init__.py`** — Exports `CentroidRouter`.

## [Unreleased] - Phase 8: Developer Experience — Packaging, CI/CD, Examples, Pre-Commit

### Added
- **MIT LICENSE** file with standard MIT text (copyright 2026 Paul C).
- **CONTRIBUTING.md** — comprehensive contributor guide covering development setup,
  running tests (unit and functional), code style (black, ruff, mypy), project
  conventions (structured logging, async PyMongo patterns, mock-at-use-site), and
  PR process with conventional commit prefixes.
- **PEP 561 marker** — `semantic_vector_router/py.typed` for type stub discovery.
- **CI/CD pipeline** (`.github/workflows/ci.yml`) — lint (ruff + black), type check
  (mypy), unit test matrix (Python 3.9–3.13 on ubuntu-latest), functional tests
  (gated to `main` branch, requires Atlas secrets), and PyPI publish on version tags
  via trusted publishing. Codecov integration for coverage tracking.
- **7 example scripts** (`examples/`) — `quickstart.py` (minimal search),
  `fields_mode.py` (FIELDS index mode), `voyage4_asymmetric.py` (asymmetric embeddings),
  `multi_partition_search.py` (cross-partition + reranking),
  `ingest_documents.py` (ingestion pipeline), `fastapi_integration.py` (FastAPI app),
  `custom_metrics.py` (MetricsHandler protocol demo).
- **Pre-commit hooks** (`.pre-commit-config.yaml`) — ruff (lint + format) and mypy
  with project dependencies.
- **`build` and `pre-commit`** added to dev dependencies.

### Changed
- **pyproject.toml** — added Python 3.13 classifier, updated project URLs to
  `pcleene/semantic-vector-router`, added Changelog URL, updated black target-version
  to include py313, moved ruff `select` to `[tool.ruff.lint]` section (fixes
  deprecation warning).
- **Makefile** — replaced venv-prefixed commands with bare `pytest`/`ruff`/`black`
  for CI compatibility. Added `install`, `build`, `clean` targets.
- **mypy configuration** — replaced `strict = true` with pragmatic per-module
  overrides. Core modules with pre-existing type issues are excluded until Phase 9
  hardening. New code is type-checked. CI runs `mypy --ignore-missing-imports`.

### Fixed
- **Python 3.9 compatibility** — removed `slots=True` from `@dataclass` decorators
  in `utils/cache.py` (CacheKey, CacheEntry) and `utils/metrics.py` (MetricEvent).
  `slots` parameter requires Python 3.10+.
- **Type narrowing** in `config.py` — `find_config_file()` return value properly
  narrowed before assignment to typed variable.

## [Unreleased] - Phase 7: Feature Completion — Document Ingestion, Rate Limiting, BinData Wiring

### Added
- **Document ingestion pipeline** (`ingestion.py`) — `IngestPipeline` with text extraction
  (concatenation or template-based), batch embedding, BinData vector conversion, partition-aware
  field routing (FIELDS: `embedding_{partition}`, VIEWS/SOURCE: standard field), and bulk
  MongoDB writes (insert or upsert). Per-document error handling with `continue_on_error`,
  progress callbacks for CLI integration. `IngestConfig` model with `text_fields`, `separator`,
  `template`, `batch_size`, `write_batch_size`, `mode`, `continue_on_error`, `trigger_detection`.
  `IngestMode` enum (INSERT/UPSERT), `IngestResult` and `IngestProgress` models.
- **Token bucket rate limiter** (`utils/rate_limiter.py`) — Async-compatible per-provider rate
  limiting with `TokenBucketRateLimiter` (configurable `tokens_per_second` and `burst` capacity),
  `RateLimiterRegistry` (lazy creation, provider-specific overrides, stats). Wired into
  `BaseEmbedder` and `BaseReranker` via `set_rate_limiter()` / `_acquire_rate_limit()`.
  `RateLimitConfig` and `ProviderRateLimit` models. Default limits for OpenAI (50 rps),
  Voyage (30 rps), Cohere (40 rps), HuggingFace (100 rps).
- **BinData wiring for ingestion** — `IngestPipeline._convert_vector()` calls existing
  `vector_to_bindata()` based on `config.vector_storage.storage_format`. Completes the
  end-to-end path: Voyage 4 int8 output → BinData INT8 storage → BinData INT8 search.
- **CLI `svr ingest` command** (`cli/ingest.py`) — Ingest documents from JSON array, single
  JSON object, or JSONL file. Supports stdin (`-`). Options: `--partition`, `--mode`,
  `--batch-size`. Rich progress bar with embedding/writing phases.
- **`IngestionError` exception** (`exceptions.py`) — Dedicated exception for ingestion failures.
- **6 new `MetricType` values** — `INGEST_LATENCY`, `INGEST_DOCUMENTS`, `INGEST_ERRORS`,
  `INGEST_EMBED_LATENCY`, `RATE_LIMIT_WAIT`, `RATE_LIMIT_ACQUIRED`.
- **Config validation** for ingestion (`batch_size > 0`, `write_batch_size > 0`, text_fields
  warning) and rate limiting (`default_tokens_per_second > 0`, `default_burst > 0`,
  per-provider validation).
- **112 new unit tests** — `test_rate_limiter.py` (15), `test_ingestion.py` (36),
  `test_cli_ingest.py` (35), `test_client.py` Phase 7 extensions (10), `test_config.py`
  Phase 7 extensions (16). Total: 859 unit tests.
- **18 new functional tests** against real MongoDB Atlas — `TestIngestionPipelineReal` (8),
  `TestRateLimiterReal` (4), `TestPhase7ConfigBackwardCompat` (4),
  `TestCLIIngestFunctional` (2). Total: 93 functional tests.

### Changed
- **SVRClient.ingest()** — New write-path method. Creates document-specific embedder
  (Voyage uses document model + `input_type="document"`, Cohere uses `"search_document"`),
  builds `IngestPipeline`, emits metrics (INGEST_LATENCY, INGEST_DOCUMENTS, INGEST_ERRORS),
  triggers partition detection after successful ingest.
- **SVRClient.__init__()** — Initializes `RateLimiterRegistry` from config.
- **BaseEmbedder** — Added `_rate_limiter`, `set_rate_limiter()`, `_acquire_rate_limit()`,
  `embed_with_rate_limit()`, `embed_batch_with_rate_limit()`. All concrete embedders call
  `super().__init__()`.
- **BaseReranker** — Added `_rate_limiter`, `set_rate_limiter()`, `_acquire_rate_limit()`.
  All concrete rerankers call `super().__init__()`.
- **CLI entry point** (`cli/__init__.py`) — Registers `ingest` command group.

## [Unreleased] - Phase 6: Performance, Observability, and Production Readiness

### Added
- **Structured logging** (`utils/logging.py`) — JSON-formatted log entries with
  `SVRLogFormatter`, `ContextVar`-based correlation ID propagation across async awaits,
  `log_operation` decorator for timing, `configure_logging()` for JSON/human-readable
  output. New `LogConfig` model with privacy controls (`log_query_text`, `log_embeddings`).
- **Metrics hooks** (`utils/metrics.py`) — Pluggable event system with `MetricsHandler`
  protocol, `MetricsCollector` (dispatches to handlers), `NoOpCollector` (disabled mode).
  12 metric types covering search, embedding, reranking, cache, detection, and errors.
  `MetricsConfig` model with `include_partition_tags` / `include_query_tags` controls.
- **Embedding cache** (`utils/cache.py`) — Thread-safe in-memory LRU cache with TTL for
  embedding vectors. `CacheKey` (text, model, dimensions, input_type), `EmbeddingCache`
  with get/put/invalidate/clear/stats. `CacheConfig` model (enabled, max_size, ttl_seconds).
  Cache hit skips embedding API call entirely.
- **Connection pool tuning** — `DatabaseConfig` extended with `max_pool_size` (default 100),
  `min_pool_size` (0), `max_idle_time_ms` (0), `wait_queue_timeout_ms` (0). Passed to
  `AsyncMongoClient` constructor. Validated in `validate_config()`.
- **Complete TIME split strategy** (`lifecycle/splitter.py`) — Real aggregation-based
  `_split_by_time()` that discovers min/max dates from MongoDB, generates monthly/quarterly/
  yearly buckets with `_generate_time_buckets()`. Child naming: `{parent}_{label}` (e.g.,
  `articles_2025_Q1`). Filter expressions include parent filter + time range.
- **CLI `svr cache stats`** — Shows cache entries, hit rate, hits, misses, evictions, TTL.
- **93 new unit tests** — `test_logging.py` (20), `test_metrics.py` (16), `test_cache.py`
  (24), updated `test_splitter.py` (13 new), `test_client.py` integration (13),
  `test_config.py` pool/backward-compat (11).
- **21 new functional tests** against real MongoDB Atlas — `TestPoolTuningReal` (2),
  `TestStructuredLoggingReal` (5), `TestMetricsHooksReal` (3), `TestEmbeddingCacheReal` (3),
  `TestPhase6ConfigBackwardCompat` (4), `TestTimeSplitFunctional` (4). Total: 75 functional tests.

### Changed
- **SVRClient.search()** — Sets correlation ID at start (`new_correlation_id()`), checks
  embedding cache before calling embedder, emits metrics for search latency, embedding
  latency, reranking latency, result/candidate counts, cache hit/miss, and errors.
  Uses `time.perf_counter()` for higher precision timing.
- **SVRClient.__init__()** — Accepts optional `metrics_handler` parameter. Initializes
  `MetricsCollector` (or `NoOpCollector` if disabled) and `EmbeddingCache` from config.
- **All 20 modules** now use `get_logger(__name__)` from `utils/logging.py` instead of
  `logging.getLogger(__name__)` for consistent structured logging.
- **MongoDB backend** — `connect()` passes pool tuning params to `AsyncMongoClient`.

## [Unreleased] - Phase 5: Metadata Collection + Detection Pipeline + Repartition Engine

### Added
- **MetadataStore** (`backends/metadata.py`) — MongoDB-backed partition and operation state
  management with `svr_metadata` collection. Three document types: partition
  (`partition:<name>`), operation (`op:<type>-<partition>-<ts>`), lock (`lock:<id>`).
  Supports shared DB mode (same cluster) or separate MongoDB instance via
  `lifecycle.metadata.connection_string_env`.
- **Detection pipeline** (`lifecycle/detector.py`) — `PartitionDetector` with 5 signal types:
  `THRESHOLD_BREACH` (count > threshold), `APPROACHING_THRESHOLD` (linear regression
  predicts breach within trend window), `SEVERE_SKEW` (max/avg ratio exceeds skew_ratio
  among siblings), `UNDERPOPULATED` (count < min_threshold), `STALE` (zero growth in
  last 10+ measurements). Auto-creates operations for auto-executable signals.
- **Repartition engine** (`lifecycle/repartition.py`) — `RepartitionEngine` with 5-step
  workflow: create_children → build_indexes → wait_indexes → switch_routing →
  cleanup_parent. Supports resume (skips completed steps), rollback (delete children,
  reset parent to ACTIVE), and configurable timeout/polling.
- **Distributed lock** — MongoDB find_one_and_update with upsert for multi-worker
  detection coordination. TTL-based expiry (default 5 min).
- **New config models** (`models.py`) — `MetadataConfig`, `DetectionConfig`,
  `RepartitionConfig` with lifecycle defaults. New `PartitionStatus` values:
  `SPLITTING`, `MIGRATING`, `RETIRED`. `DetectionSignal` enum for 5 signal types.
- **New exceptions** (`exceptions.py`) — `MetadataError`, `DetectionError`,
  `RepartitionError`.
- **Config validation** (`config.py`) — validates detection thresholds, skew ratio,
  repartition timeouts/intervals.
- **CLI monitor commands** (`cli/monitor.py`) — `svr monitor check [--auto-execute]`
  runs detection pipeline with Rich table output. `svr monitor history <partition>`
  shows health history.
- **CLI repartition commands** (`cli/repartition.py`) — `svr repartition pending`,
  `execute <op-id>`, `status <op-id>`, `rollback <op-id>`.
- **131 new unit tests** — `test_metadata_store.py` (47), `test_detector.py` (33),
  `test_repartition.py` (31), `test_cli_monitor.py` (8), `test_cli_repartition.py` (12).
- **Functional tests** for MetadataStore CRUD, lock acquire/release/expire, migration,
  detection pipeline, and backward compatibility.

### Changed
- **PartitionResolver** — all methods now `async`. Dual-source: reads from MetadataStore
  if available, falls back to config file registry.
- **SVRClient** — `get_partition()` now async. `connect()` initializes MetadataStore,
  migrates from config, passes metadata to PartitionResolver. `start_monitor()` /
  `stop_monitor()` for background detection.
- **CLI entry point** (`cli/__init__.py`) — registers `monitor` and `repartition`
  command groups.

## [Unreleased] - Phase 4: CLI Completeness

### Added
- **8 new CLI command modules** — full operational CLI for partition management, search
  diagnostics, configuration, lifecycle, and index management. (`cli/partitions.py`,
  `cli/search.py`, `cli/analyze.py`, `cli/watch.py`, `cli/split.py`, `cli/config_cmd.py`,
  `cli/index.py`, `cli/helpers.py`)
- **`svr partitions` commands** — `list`, `status`, `create`, `delete`, `refresh`, `scan`,
  `provision`. Rich tables with colored status, confirmation prompts, progress bars.
- **`svr search` command** — execute vector search queries from the CLI with
  `--partitions`, `--limit`, `--rerank/--no-rerank`, `--format json|table`. Diagnostic
  panel showing partitions searched, candidates, latency, reranking status.
- **`svr analyze` command** — collection field analysis with `--field` and `--filters`
  options. Pure statistics (cardinality, coverage, suitability) using `field_analyzer.py`.
- **`svr watch` commands** — `start` (foreground with Ctrl+C), `status`, `confirm`,
  `reject`, `confirm-all` for change stream watcher and pending partition management.
- **`svr split` commands** — `check` (which partitions need splitting) and `execute`
  (run splits with `--all` flag).
- **`svr config` commands** — `show` (API keys redacted), `validate`, `set` (dot-notation
  keys with JSON type preservation), `path`.
- **`svr index` commands** — `status` (Rich table of all indexes), `wait` (poll until
  queryable with configurable `--timeout`).
- **Non-interactive init** — `svr init --non-interactive` with `--connection-string-env`,
  `--database`, `--collection`, `--partition-field`, `--embedding-provider`,
  `--embedding-model`, `--dimensions`, `--index-location` flags.
- **CLI helper module** (`cli/helpers.py`) — shared `_run_async()`, `_get_backend()`,
  `handle_config_error` decorator for consistent async bridging and error handling.
- **41 new CLI unit tests** (`tests/unit/test_cli.py`) — Click CliRunner tests for all
  command groups with mocked backend and config. Uses correct mock patch-at-use-site pattern.
- **14 new CLI functional tests** (`tests/functional/test_end_to_end.py`) — tests against
  real MongoDB Atlas for partitions, config, analyze, index, and help commands.

### Changed
- **CLI entry point** (`cli/__init__.py`) — registers all 8 command groups (was only `init`).
- **All CLI modules use module-level imports** — required for `unittest.mock.patch` to
  intercept at the call site (Python mock pattern: patch where the name is *used*, not
  where it's *defined*).

## [Unreleased] - Phase 3: Robustness

### Added
- **Retry decorator** (`utils/retry.py`) — `@with_retry(max_attempts, base_delay,
  max_delay, retryable_exceptions)` with exponential backoff + jitter. Works with
  async and sync functions. Respects HTTP 429 `Retry-After` headers.
- **ResilienceConfig model** (`models.py`) — configurable retry attempts, base/max delay,
  MongoDB connection/search timeouts, embedder/reranker API timeouts, health check
  interval, watcher reconnection params. Wired into `SVRConfig.resilience`.
- **Connection health check with staleness** (`mongodb.py`) — `health_check()` caches
  last successful ping, skips re-ping within `health_check_interval_s`.
- **Provisioner rollback** (`provisioner.py`) — `_rollback_partition()` cleans up
  partially-created resources (views, indexes, config entries) on creation failure.
  Logs rollback at WARNING, failed rollback at ERROR.
- **Watcher state persistence** — `pending_partitions` saved to config file via
  `LifecycleConfig.pending_partitions`. Survives restarts.
- **Resilience config validation** (`config.py`) — validates timeout/retry values,
  warns on very low `search_timeout_ms`.
- 68 new unit tests for retry, health check, timeouts, rollback, and reconnection.
- 11 new functional tests for health check, search timeout, connection timeout,
  provisioner rollback, and backward-compatible config loading.

### Changed
- **MongoDB backend** — `AsyncMongoClient` now receives `connectTimeoutMS` and
  `serverSelectionTimeoutMS` from config. `execute_search()` passes `maxTimeMS`
  to aggregation pipeline.
- **Embedder/reranker timeouts** — configurable via `ResilienceConfig` instead of
  hardcoded 60s. Passed through `SVRClient._create_embedder()` and `_create_reranker()`.
- **Retry applied** to MongoDB backend operations (`execute_search`, `get_distinct_values`,
  `count_documents`, `create_vector_search_index`, `create_partition_view`) on transient
  errors (`AutoReconnect`, `NetworkTimeout`, `ServerSelectionTimeoutError`).
- **Retry applied** to embedder/reranker API calls on HTTP 429, 500, 502, 503, 504.
- **Watcher auto-reconnect** — `_watch_loop()` reconnects with exponential backoff
  on change stream errors, up to `watcher_max_retries`. Resets attempt counter on
  successful session.

## [Unreleased] - Phase 2: Testing Foundation

### Added
- **413 unit tests** across 7 core modules with 80%+ coverage:
  models 98%, config 99%, client 97%, provisioner 87%, scanner 100%, splitter 98%,
  mongodb backend 97%.
- **Test infrastructure** — `Makefile` with `test`, `test-unit`, `test-functional`,
  `lint`, `format`, `typecheck` targets. `pytest-timeout` (30s default), `pytest-cov`.
- **Comprehensive test fixtures** (`tests/conftest.py`) — fixtures for all three
  index modes (SOURCE, VIEWS, FIELDS) with mock configs, backends, and partition registries.
- **Unit test files:**
  - `test_scanner.py` — scan/discovery/validation with error handling
  - `test_provisioner.py` — all 3 index modes, config atomicity, batch creation, 50-partition FIELDS limit
  - `test_watcher.py` — start/stop, auto-provisioning, confirmation workflow
  - `test_monitor.py` — health thresholds, needs_attention aggregation
  - `test_splitter.py` — secondary field (scoped), time split, hash deprecation
  - `test_client.py` — full search flow, reranking, connect/disconnect lifecycle
  - `test_voyage_embedder.py` — asymmetric pairs, dimension/quantization switching
  - `test_embedders.py` — OpenAI, Cohere, HuggingFace
  - `test_rerankers.py` — score mapping, hit reranking, top_k limiting
  - `test_vector_conversion.py` — all 4 storage formats, round-trip conversion

## [Unreleased] - Phase 1.5: Functional Test Fixes + VIEWS Redesign

### Added
- **Server version detection** — `MongoDBBackend.server_version` and
  `supports_search_index_on_views` property for version-aware behavior. (`mongodb.py`)
- **18 functional tests** against real MongoDB Atlas — covers all 3 index modes,
  `$vectorSearch`, partition lifecycle, BinData, field analyzer, config validation.
  Uses structured centroid embeddings (orthogonal 32-dim vectors per category).
  (`tests/functional/test_end_to_end.py`)

### Changed
- **VIEWS mode is version-aware** — on MongoDB 8.1+ creates per-view search indexes
  (separate HNSW graphs, driver method). On <8.1 falls back to shared source index
  with partition pre-filter (Atlas doesn't support `createSearchIndex` on views via
  driver below 8.1). Both paths validated by functional tests. (`provisioner.py`, `mongodb.py`)
- **Search pipeline filter logic** — filter decision is now based on
  `partition.search_collection` rather than just `index_location`, correctly handling
  the VIEWS mode 8.1+ vs <8.1 distinction. (`mongodb.py`)
- **Delete partition** — VIEWS mode only deletes per-view index on 8.1+ where
  the partition has its own index. On <8.1 the shared source index is preserved.
  (`provisioner.py`)
- **Project description updated** — SVR is now described as a "vector search
  orchestration layer / management plane" reflecting its full scope.
  (`pyproject.toml`, `README.md`)

### Fixed
- **Async PyMongo `aggregate()`** — `collection.aggregate(pipeline)` returns a
  coroutine in async PyMongo, needs `await` before `.to_list()`. Fixed in search
  pipeline, partition counting, and field analyzer. (`mongodb.py`, `field_analyzer.py`)
- **Async PyMongo `list_search_indexes()`** — same pattern, returns coroutine.
  Split into `await collection.list_search_indexes()` then `await cursor.to_list()`.
  (`mongodb.py`)
- **Async PyMongo `client.close()`** — needs `await` in async PyMongo.
  (`mongodb.py`)

## [Unreleased] - Phase 1: Critical Fixes + FIELDS Mode

### Added
- **FIELDS index location mode** (`index_on: "fields"`) — each partition gets a dedicated
  embedding field on the source collection (e.g., `embedding_electronics`) with its own
  vector search index. Builds separate HNSW graphs without views. Limited to 50 partitions
  (Atlas 64-index cap). (`models.py`, `provisioner.py`, `mongodb.py`)
- **Filter field auto-detection** — new `utils/field_analyzer.py` module that analyzes
  collection fields for filter suitability (cardinality, coverage, type). Integrated into
  `ensure_source_index()` for SOURCE mode. (`utils/field_analyzer.py`, `provisioner.py`)
- **ScanError exception** — dedicated exception for partition scan failures.
  (`exceptions.py`)
- **Quantization compatibility validation** — `validate_config()` now rejects incompatible
  combinations (e.g., pre-quantized BinData + index-level quantization) and warns on
  Voyage quantization type mismatches. (`config.py`)
- **FIELDS mode config validation** — warns at 40+ partitions, errors at 50+. (`config.py`)
- Unit tests for FIELDS mode models, VectorStorageConfig, and field analyzer (10 new tests).

### Changed
- **Config atomicity in provisioner** — `create_partitions_batch()` now suppresses
  individual config saves and writes once at the end. `delete_partition()` deregisters
  from config first, then does best-effort MongoDB cleanup (safe ordering).
  `update_all_partition_counts()` also batches saves. (`provisioner.py`)
- **BinData wired into query path** — `_build_vector_search_pipeline()` now calls
  `query_vector_for_search()` to convert query vectors for pre-quantized storage
  formats. (`mongodb.py`)
- **Search pipeline uses partition-specific embedding field** in FIELDS mode instead
  of the global `embedding_field`. (`mongodb.py`)
- `PartitionInfo.view_name` is now `Optional[str]` (was `str`) to support FIELDS mode
  where no view is created. (`models.py`)
- `PartitionInfo` has new `embedding_field: Optional[str]` for FIELDS mode. (`models.py`)

### Deprecated
- **Hash split strategy** — emits `DeprecationWarning` when used. Hash sharding
  distributes randomly, requiring fan-out to all children (defeats partitioned search
  purpose). Use `secondary_field` instead. (`splitter.py`)

### Fixed
- **Silent auto-connect failure** — `SVRClient` now sets `_auto_connect_failed` flag
  when auto-connect is skipped in async contexts (Jupyter, FastAPI). `search()` raises
  a clear error message instead of silently failing. (`client.py`)
- **Hash split operator** — replaced invalid `$toHashedIndexKey` with
  `$substr` + `$toInt` + `$mod` pipeline. (`splitter.py`)
- **Secondary field split scope** — `_split_by_secondary_field()` now passes the
  parent partition filter to `get_distinct_values()` so it only discovers values
  within the partition being split. (`splitter.py`)
- **Scanner error handling** — all 4 public methods now catch `ConnectionFailure`,
  `ServerSelectionTimeoutError`, and `OperationFailure`, wrapping them in `ScanError`
  with context. (`scanner.py`)

## [0.1.0] - Initial Release

- Core partitioning with VIEWS and SOURCE modes
- Voyage 4 shared embedding space support
- MongoDB vector quantization (BinData float32/int8/packed_bit + index-level scalar/binary)
- Multi-provider embedding (OpenAI, Voyage, Cohere, HuggingFace)
- Multi-provider reranking (Voyage, Cohere)
- Change stream watcher for auto-provisioning
- CLI with `svr init`, `svr status`, `svr provision`
