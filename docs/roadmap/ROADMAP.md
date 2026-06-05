# Semantic Vector Router — Build Plan

**Version:** 0.1.0 → 1.0.0
**Updated:** February 11, 2026
**Build Method:** Phased, with self-assessment checkpoints per phase

---

## Why This SDK Exists

At 20M+ vectors, HNSW graph quality degrades. Pre-filtering on a single large index doesn't fix this — the graph structure was built across the entire vector space, and graph edges connect across semantic boundaries instead of within them. The only way to maintain recall quality at scale is **separate, semantically dense HNSW indexes** per partition.

This SDK manages the full lifecycle: partition creation (via views or dedicated embedding fields), query routing through splits, cross-partition result merging with score normalization, and reranking. Application code says `search("wireless headphones", partitions=["electronics"])` and the SDK handles everything — including when "electronics" was silently split into three child partitions last Tuesday.

---

## Index Location Modes

The SDK supports three modes for creating partitioned HNSW indexes:

| Mode | How It Works | Best For |
|------|-------------|----------|
| **SOURCE** | Single index on source collection, auto-detected filter fields, query-time pre-filtering | Collections <20M vectors where recall isn't degrading |
| **VIEWS** | Filtered views, each with its own index | Large collections, many partitions (>50), data that reclassifies often |
| **FIELDS** | Partition-specific embedding fields (`embedding_{partition}`) on source collection, each with its own index | Maximum query performance, fewer partitions (<50), greenfield schemas |

**VIEWS** and **FIELDS** both produce separate HNSW graphs. VIEWS has zero schema impact but view pipeline overhead on document retrieval and index maintenance. FIELDS has no pipeline overhead but requires partition-specific embedding fields and manual field management on reclassification.

**Auto-escalation rule:** FIELDS mode is capped at 50 indexes (Atlas limit is 64; we reserve headroom). If partitions exceed 50, the SDK recommends migrating to VIEWS mode.

**SOURCE mode auto-filter detection:** In SOURCE mode, the SDK automatically analyzes collection fields to identify good filter candidates (low-to-medium cardinality, high selectivity, appropriate data types) and adds them as `{"type": "filter"}` fields in the vector search index definition. This means pre-filtering works efficiently out of the box without the user manually configuring filter fields. During `svr init` or `svr analyze`, the SDK scans field statistics and recommends which fields to include as filters on the index.

---

## Split Strategy Revision

**Hash split is deprecated.** Hash-based sharding distributes documents randomly across child partitions, which means every query must fan out to all children — defeating the purpose of partitioned search. The HNSW graphs are smaller but not semantically denser.

**All split strategies must produce semantically meaningful child partitions:**

| Strategy | Requires | Produces |
|----------|----------|----------|
| **SECONDARY_FIELD** | A metadata field with meaningful categories | `electronics__phones`, `electronics__laptops` |
| **TIME** | A date/timestamp field | `articles__2024`, `articles__2025` |
| **ALERT_ONLY** | Nothing | No split — just notifies that a partition exceeds threshold |

**If no natural metadata field exists:** The SDK's `svr analyze` command can identify candidate fields (by cardinality, distribution, correlation with partition size). If no fields exist, the recommendation is to generate a categorical metadata field externally (e.g., via LLM classification, clustering, or domain logic) and then use SECONDARY_FIELD. The SDK does not call LLMs itself — it stays pure infrastructure.

---

## Phase 1: Critical Fixes + FIELDS Mode [COMPLETE]

**Goal:** Fix known bugs, add FIELDS index location mode, all existing tests pass.
**Self-assessment:** Run full test suite, manually verify each fix against the described bug.
**Status:** Complete (commit `ebaa2b7`). All items delivered including FIELDS mode, filter auto-detection, async PyMongo fixes, config atomicity, BinData query path, quantization validation, scanner error handling, hash split deprecation, secondary field split scoping, version-aware VIEWS mode.

### 1.1 Fix Silent Auto-Connect Failure
**File:** `client.py:108-120`
Replace silent `pass` on `RuntimeError` with a logged warning and `_connected = False` flag. `search()` checks this flag and raises: `"Client not connected. Call await client.connect() explicitly in async contexts."`

### 1.2 Deprecate Hash Split
**File:** `lifecycle/splitter.py`
- Keep `SplitStrategy.HASH` in the enum for backwards compatibility but mark deprecated
- `execute_split()` with HASH strategy logs a deprecation warning and suggests SECONDARY_FIELD
- Update `AutoSplitConfig` docstring to explain why hash splits don't improve recall
- Fix the `$toHashedIndexKey` operator issue (use `$mod` on `{"$toInt": {"$substr": [{"$toString": "$_id"}, -2, 2]}}`) so it doesn't crash if someone does use it

### 1.3 Add Error Handling to Scanner
**File:** `lifecycle/scanner.py`
Wrap each public method's backend calls in try/except catching `OperationFailure`, `ConnectionFailure`, `ServerSelectionTimeoutError`. Wrap into `ScanError` with context about which operation failed.

### 1.4 Fix Config Atomicity in Provisioner
**File:** `lifecycle/provisioner.py`
- `create_partition()`: Build complete PartitionInfo first, perform all MongoDB operations, then single `save_config()` at the end
- `delete_partition()`: Remove config entry first, then clean up MongoDB resources (orphaned resources > dangling references)
- `create_partitions_batch()`: Single `save_config()` after all partitions created

### 1.5 Wire BinData into Query Path
**File:** `backends/mongodb.py` in `_build_vector_search_pipeline()`
Wrap `query_vector` through `query_vector_for_search(vector, storage_format)` before inserting into the `$vectorSearch` stage. This makes pre-quantized query vectors work (INT8, PACKED_BIT). Ingestion-side BinData wiring deferred to Phase 6 (document ingestion pipeline).

### 1.6 Add Quantization Compatibility Validation
**File:** `config.py` in `validate_config()`
Reject incompatible combinations at config load time:
- `BINDATA_INT8` or `BINDATA_PACKED_BIT` + `index_quantization != NONE` → error (double quantization)
- Voyage `int8` output + `storage_format != BINDATA_INT8` → warning
- Voyage `binary` output + `storage_format != BINDATA_PACKED_BIT` → warning

### 1.7 Add FIELDS Index Location Mode
**New enum value:** `IndexLocation.FIELDS = "fields"`

**Models changes (`models.py`):**
- Add `FIELDS` to `IndexLocation` enum
- `PartitionInfo`: add `embedding_field: Optional[str]` for FIELDS mode (stores `embedding_{partition_name}`)

**Provisioner changes (`provisioner.py`):**
- New path in `create_partition()` for FIELDS mode:
  - Compute field name: `{embedding_field}_{partition_name}` (sanitized)
  - Create vector search index on source collection with `path: embedding_{partition_name}`
  - Set `search_collection = source_collection`
  - Set `embedding_field = embedding_{partition_name}`
- Partition count validation: if FIELDS mode and registry has >=50 partitions, raise error suggesting VIEWS mode

**Backend changes (`mongodb.py`):**
- `_build_vector_search_pipeline()`: use `partition.embedding_field` as path when in FIELDS mode (falls back to config embedding_field)
- `execute_search()`: route to source collection for FIELDS mode (same as SOURCE)

**Config changes (`config.py`):**
- `validate_config()`: warn if FIELDS mode with >40 partitions (approaching limit)

**CLI changes (`cli/init.py`):**
- Add FIELDS as third option in index location selection with explanation

### 1.8 Filter Field Auto-Detection for SOURCE Mode
**New file:** `utils/field_analyzer.py`
Analyze collection fields to identify filter candidates:
- Query `$sample` of documents to detect field types and cardinality
- Score fields: cardinality 2–100 = good, boolean/enum types = good, >50% null = bad, arrays/objects = skip
- Return ranked list of recommended filter fields with reasoning
- Integrate into `ensure_source_index()`: auto-detected filter fields added to the index `filter_fields` param
- Integrate into `svr init`: during SOURCE mode setup, show detected filter fields and let user confirm/override
- Store detected filter fields in config under `vector_storage.auto_filter_fields: list[str]`

### 1.9 Fix Secondary Field Split Scope
**File:** `lifecycle/splitter.py:149`
Pass the partition's filter expression to `get_distinct_values()` so it only scans within the partition being split, not the entire collection.

### Self-Assessment Checklist
- [ ] All existing tests pass (`pytest tests/`)
- [ ] Auto-connect in Jupyter/FastAPI context produces clear error message
- [ ] Hash split logs deprecation warning
- [ ] Scanner methods raise `ScanError` on backend failures (test with mock)
- [ ] Provisioner calls `save_config()` exactly once per operation
- [ ] Pre-quantized query vectors (INT8) are correctly converted in search pipeline
- [ ] Invalid quantization combos rejected at config load
- [ ] FIELDS mode creates index on source collection with partition-specific embedding path
- [ ] FIELDS mode search uses correct embedding field per partition
- [ ] 50-partition limit enforced for FIELDS mode
- [ ] SOURCE mode auto-detects filter fields from collection stats
- [ ] Auto-detected filter fields included in source index definition
- [ ] Secondary field split only scans within parent partition

---

## Phase 2: Testing Foundation [COMPLETE]

**Goal:** 80%+ coverage on core modules. Every bug fix from Phase 1 has a regression test.
**Self-assessment:** Coverage report, all tests green, no skipped tests.
**Status:** Complete (commit `17d1511`). 413 unit tests, 80%+ coverage on all 7 core modules. Makefile, pytest-timeout, conftest fixtures for all 3 index modes. 18 functional tests against real Atlas.

### 2.1 Test Infrastructure
- Add `pytest-timeout` (30s default), `pytest-cov` to dev dependencies
- Create `Makefile` with `test`, `test-unit`, `test-integration`, `lint`, `format`, `typecheck` targets
- Restructure tests into `tests/unit/` and `tests/integration/`
- Create `conftest.py` fixtures for all three index location modes (SOURCE, VIEWS, FIELDS)

### 2.2 Unit Tests for Lifecycle Components
**`tests/unit/test_scanner.py`** — Mock backend, test scan/discovery/validation, test error handling from Phase 1.3
**`tests/unit/test_provisioner.py`** — Test all three index location modes, test config atomicity (single save), test batch creation, test deletion order, test 50-partition FIELDS limit
**`tests/unit/test_watcher.py`** — Test start/stop, auto-provisioning, confirmation workflow, error accumulation
**`tests/unit/test_monitor.py`** — Test health thresholds, needs_attention aggregation, index health
**`tests/unit/test_splitter.py`** — Test secondary field (scoped to partition), test time split, test hash deprecation warning, test schedule enforcement

### 2.3 Unit Tests for Client
**`tests/unit/test_client.py`** — Full search flow with mocked components for all three index modes. Reranking logic. Pre-computed vectors. Connect/disconnect lifecycle. Auto-connect failure message.

### 2.4 Unit Tests for Embedders and Rerankers
**`tests/unit/test_voyage_embedder.py`** — Voyage 4 asymmetric pairs, dimension/quantization switching, API response parsing
**`tests/unit/test_embedders.py`** — OpenAI, Cohere, HuggingFace basic flows
**`tests/unit/test_rerankers.py`** — Score mapping, hit reranking, top_k limiting

### 2.5 Unit Tests for BinData
**`tests/unit/test_vector_conversion.py`** — All four storage formats, round-trip conversion, range validation, query format selection

### 2.6 CLI Tests
**`tests/unit/test_cli.py`** — Click CliRunner tests for init wizard, all provider paths, FIELDS mode selection, error paths

### Self-Assessment Checklist
- [ ] `pytest tests/unit/ --cov=semantic_vector_router` shows >80% on core modules
- [ ] Every Phase 1 fix has at least one regression test
- [ ] All three index modes have test coverage
- [ ] Zero test warnings or skips
- [ ] Tests run in <60s total

---

## Phase 3: Robustness [COMPLETE]

**Goal:** The SDK handles transient failures, timeouts, and connection drops gracefully.
**Self-assessment:** Simulate failures with mocks, verify retry/reconnect/timeout behavior.
**Status:** Complete (commit `f8d4bb0`). Retry decorator, ResilienceConfig, connection/search/API timeouts, health check with staleness, provisioner rollback, watcher auto-reconnect + state persistence. 481 unit tests + 29 functional tests.

### 3.1 Retry Logic with Exponential Backoff
**New file:** `utils/retry.py`
Decorator `@with_retry(max_attempts=3, base_delay=0.5, max_delay=30, retryable_exceptions=(...))`.
Apply to: MongoDB backend methods (retry on `AutoReconnect`, `NetworkTimeout`, `ServerSelectionTimeoutError`), embedding API calls (retry on HTTP 429, 500, 502, 503), reranking API calls.
Make retry params configurable via `ResilienceConfig` in config.

### 3.2 Connection Health and Reconnection
- `health_check()` on MongoDB backend: `db.command("ping")` with staleness check (skip if last success <30s ago)
- Watcher auto-reconnect with exponential backoff (1s → 60s max) and resume token
- Max retry count before giving up (configurable, default 10)

### 3.3 Timeout Configuration
- `search_timeout_ms` → MongoDB `maxTimeMS`
- `embedding_timeout_ms` → httpx timeout
- `reranking_timeout_ms` → httpx timeout
- `connection_timeout_ms` / `server_selection_timeout_ms` → `AsyncMongoClient`

### 3.4 Provisioner Rollback on Failure
Track created resources during `create_partition()`. On failure, reverse operations (delete view/index, remove config entry). Log rollback at WARNING, failed rollback at ERROR.

### 3.5 Watcher State Persistence
Persist `pending_partitions` to config file. On restart, load and resume confirmation workflow.

### Self-Assessment Checklist
- [ ] Retry decorator works with async functions
- [ ] Simulated MongoDB connection drop → auto-reconnect within backoff period
- [ ] Search with `maxTimeMS` times out correctly
- [ ] Provisioner failure mid-creation → resources cleaned up
- [ ] Watcher restart preserves pending partitions
- [ ] All new code has unit tests

---

## Phase 4: CLI Completeness [COMPLETE]

**Goal:** Full operational CLI for partition management, search diagnostics, and configuration.
**Self-assessment:** Run each command manually, verify output formatting, test error paths.
**Status:** Complete (commit `0d98f68`). 8 new CLI modules (partitions, search, analyze, watch, split, config, index, helpers), non-interactive init, 41 CLI unit tests, 14 CLI functional tests. 522 unit tests + 43 functional tests total.

### 4.1 Partition Management Commands
```
svr partitions list                    # Table: name, status, count, index location, mode
svr partitions status [name]           # Health summary (all or one)
svr partitions create <name>           # Manual creation
svr partitions delete <name>           # With confirmation prompt
svr partitions refresh                 # Refresh all document counts
svr partitions scan                    # Scan for new partition values
svr partitions provision               # Auto-provision discovered values
```
Rich tables, colored status, progress bars.

### 4.2 Search Command
```
svr search "query text" [--partitions p1,p2] [--limit 10] [--rerank] [--format json|table]
```
Diagnostic output: partitions queried, per-partition scores, normalization impact, reranking delta, final results.

### 4.3 Analyze Command (Stats-Based)
```
svr analyze                            # Full collection analysis
svr analyze --field category           # Analyze specific field
svr analyze --filters                  # Focus on filter field detection
```
Pure statistics — no LLM calls:

**Partition analysis:**
- Field cardinality (distinct value count)
- Value distribution (min/max/avg documents per value, skew coefficient)
- Recommended partition strategy based on distribution
- Estimated partition sizes
- Warning if any partition would exceed split threshold
- Suggestion to create metadata field if no good candidates exist

**Filter field auto-detection (for SOURCE mode):**
- Scan all fields for filter suitability:
  - Cardinality: 2–100 distinct values = good candidate, >1000 = poor candidate
  - Data type: string, number, boolean, date = suitable; arrays, nested objects = skip
  - Null ratio: >50% nulls = poor candidate
  - Selectivity: fields where filtering reduces result set by >50% = high value
- Rank fields by filter quality score
- Output: recommended filter fields with reasoning
- Auto-apply: during `svr init` in SOURCE mode, detected filter fields are automatically added to the index definition (user can override)

### 4.4 Lifecycle Commands
```
svr watch start [--daemon]             # Start change stream watcher
svr watch stop                         # Stop watcher
svr watch status                       # Show status and pending partitions
svr watch confirm/reject <name>        # Handle pending partitions
svr split check                        # Check which partitions need splitting
svr split execute <name> [--all]       # Execute split
```

### 4.5 Configuration Commands
```
svr config show                        # Display config (API keys redacted)
svr config validate                    # Run validation, show warnings
svr config set <key> <value>           # Update config value
svr config path                        # Show config file path
```

### 4.6 Index Management Commands
```
svr index status                       # All indexes and states
svr index rebuild <partition>          # Rebuild index
svr index wait <partition>             # Wait for queryable state
```

### 4.7 Non-Interactive Init
Implement the stubbed `--non-interactive` flag:
```
svr init --non-interactive \
  --connection-string "$MONGODB_URI" \
  --database mydb --collection products \
  --partition-field category \
  --embedding-provider voyage --embedding-model voyage-4 \
  --dimensions 1024 --index-location fields
```

### Self-Assessment Checklist
- [ ] Every command produces clean output (test with Click CliRunner)
- [ ] `svr analyze` gives correct stats on a test collection
- [ ] `svr search` shows full diagnostic breakdown
- [ ] Non-interactive init generates valid config
- [ ] All commands handle missing config, connection failures gracefully
- [ ] Rich formatting renders correctly in terminal

---

## Phase 5: Metadata Collection + Partition Lifecycle Management [COMPLETE]

**Goal:** Replace local config file as partition source of truth with a shared MongoDB metadata collection. Implement detection pipeline and repartition workflow.
**Self-assessment:** Multi-worker scenario sees consistent partition state. Detection correctly identifies threshold breaches, skew, and growth trends. Repartition workflow executes split with zero search downtime.
**Status:** Complete (commit `cbd1d8e`). MetadataStore, detection pipeline (5 signals), repartition engine (5-step workflow), distributed locking, monitor/repartition CLI, 654 unit tests + 54 functional tests.

### 5.1 Metadata Collection Schema
**New file:** `backends/metadata.py`

Create `svr_metadata` collection in the same database with three document types:

**Partition documents** (replace config file registry):
```javascript
{
  _id: "partition:<name>",
  type: "partition",
  name: "electronics",
  status: "active",           // active | splitting | migrating | retired
  index_location: "fields",   // source | views | fields
  embedding_field: "embedding_electronics",  // FIELDS mode only
  index_name: "svr_vector_idx_electronics",
  view_name: "svr_partition_electronics",    // VIEWS mode only
  search_collection: "products",
  filter_value: "electronics",
  filter_expression: null,
  document_count: 4_200_000,
  last_count_update: ISODate(),
  created_at: ISODate(),
  parent: null,
  children: [],
  health_history: [            // Rolling window, last 30 data points
    { ts: ISODate(), count: 4_100_000 },
    { ts: ISODate(), count: 4_200_000 }
  ]
}
```

**Operation documents** (pending/in-progress repartition ops):
```javascript
{
  _id: "op:<type>-<partition>-<date>",
  type: "operation",
  operation: "split",          // split | merge | migrate | retire
  target_partition: "electronics",
  strategy: "secondary_field",
  config: { secondary_field: "subcategory" },
  status: "pending",           // pending | scheduled | in_progress | completing | done | failed
  scheduled_for: ISODate(),    // null = manual trigger
  steps: [
    { action: "create_child_partitions", status: "pending" },
    { action: "build_indexes", status: "pending" },
    { action: "wait_indexes_queryable", status: "pending" },
    { action: "switch_routing", status: "pending" },
    { action: "cleanup_parent", status: "pending" }
  ],
  created_at: ISODate(),
  started_at: null,
  completed_at: null,
  error: null
}
```

**Lock documents** (distributed lock for detection):
```javascript
{
  _id: "lock:monitor",
  type: "lock",
  holder: "worker-3-pid-12345",
  acquired_at: ISODate(),
  expires_at: ISODate()        // 5 min TTL
}
```

### 5.2 Configurable Metadata Location
The metadata collection can live on a **separate MongoDB instance** from the main data. This allows teams to keep SDK metadata off production clusters.

```json
{
  "lifecycle": {
    "metadata": {
      "connection_string_env": "SVR_METADATA_MONGODB_URI",
      "database": "svr_metadata_db",
      "collection": "svr_metadata"
    }
  }
}
```

If `connection_string_env` is null/omitted, falls back to the main `database.connection_string_env` and `database.database` — same cluster, same DB. If provided, the `MetadataStore` opens a separate `AsyncMongoClient` to the metadata instance.

### 5.3 Migrate Partition Registry to Metadata Collection
- `MetadataStore` class with CRUD operations for partitions, operations, and locks
- On first connect, if `svr_metadata` doesn't exist, migrate from local config file
- `SVRClient` reads partition registry from metadata collection instead of config file
- Config file retains non-partition settings (embedding, routing, reranking config)
- Backward compatibility: if metadata collection doesn't exist, fall back to config file
- `PartitionResolver` reads from metadata store instead of config registry

### 5.3 Detection Pipeline
**New file:** `lifecycle/detector.py`

Detection runs as a pipeline: COLLECT → STORE → ANALYZE → DECIDE

**COLLECT:** Count documents per partition, check index status.

**STORE:** Append to `health_history` array in metadata collection (capped at 30 entries via `$slice`).

**ANALYZE** (all stats-based, no AI):

| Signal | Rule | Auto-executable? |
|--------|------|------------------|
| `THRESHOLD_BREACH` | `count > threshold` | Yes (configurable) |
| `APPROACHING_THRESHOLD` | Linear regression on history predicts breach within 30 days | Yes (schedule split) |
| `SEVERE_SKEW` | `max_count / avg_count > 5x` across siblings | No — suggest only |
| `UNDERPOPULATED` | `count < min_threshold` | No — suggest merge |
| `STALE` | Zero growth in last 30 data points | No — suggest retirement |

**DECIDE:** Map signals to operations. Auto-executable ops get created in metadata. Non-auto ops get written as suggestions.

### 5.4 Distributed Lock for Multi-Worker Detection
Implement MongoDB-based distributed lock using `find_one_and_update` with TTL:
- Worker attempts to acquire `lock:monitor` with 5-minute expiry
- If acquired: run detection pipeline
- If not: skip (another worker is handling it)
- Expired locks are automatically re-acquirable
- No external dependency — uses the same MongoDB connection

### 5.5 Repartition Workflow Engine
**New file:** `lifecycle/repartition.py`

Step-by-step execution with rollback:

**PLAN:** Analyze partition, determine new layout (split children, merge targets), create operation document in metadata.

**PREPARE:** Create new partitions (views/fields/indexes). Old partitions remain active and searchable throughout.

**WAIT:** Poll new indexes until queryable. Configurable timeout (default 30min).

**SWITCH:** Atomic update in metadata collection — new partitions `active`, old partition `retired`. All workers pick up new routing on next query read.

**CLEANUP:** Delete old views/indexes/fields. Remove retired partition.

**ROLLBACK:** If any step fails, old partition is still active. Operation marked `failed` with error. User can retry via CLI or the engine can auto-retry based on config.

### 5.6 Execution Models for Detection
The detection logic is execution-model agnostic. Three ways to run it:

**CLI cron (recommended):**
```bash
# Crontab: check every hour, auto-execute approved operations
0 * * * *  svr monitor check --auto-execute
```

**In-process with distributed lock:**
```python
client = SVRClient(config)
await client.connect()
await client.start_monitor(interval_seconds=3600)  # Background asyncio task
# Only one worker across all instances runs detection (lock-based)
```

**Manual:**
```bash
svr monitor check                      # Run detection, show findings
svr repartition pending                # Show pending operations
svr repartition execute <op-id>        # Execute an operation now
svr repartition schedule <op-id> --at "2026-02-15T02:00:00Z"
svr repartition rollback <op-id>       # Cancel/rollback failed operation
```

### 5.7 Configuration
```json
{
  "lifecycle": {
    "metadata_collection": "svr_metadata",
    "detection": {
      "enabled": true,
      "interval": "1h",
      "threshold_vectors": 10000000,
      "min_threshold_vectors": 1000,
      "skew_ratio": 5.0,
      "trend_window_days": 30,
      "auto_split_on_breach": false,
      "auto_schedule_approaching": true
    },
    "repartition": {
      "schedule": {
        "allowed_days": ["saturday", "sunday"],
        "allowed_hours": { "start": 2, "end": 6 }
      },
      "index_wait_timeout_s": 1800,
      "index_poll_interval_s": 10,
      "auto_cleanup_retired": true
    }
  }
}
```

### Self-Assessment Checklist
- [ ] Metadata collection created on first connect with correct schema
- [ ] Partition registry migrated from config file to metadata collection
- [ ] SVRClient reads partitions from metadata collection
- [ ] Config file fallback works when metadata collection doesn't exist
- [ ] Detection correctly identifies threshold breach, skew, approaching threshold
- [ ] Health history stored and capped at 30 entries
- [ ] Distributed lock prevents concurrent detection across workers
- [ ] Repartition split workflow: create children → build indexes → switch → cleanup
- [ ] Old partition remains searchable during entire repartition
- [ ] Failed repartition doesn't break search (old partition still active)
- [ ] CLI `svr monitor check` and `svr repartition` commands work
- [ ] In-process monitor with distributed lock runs in only one worker

---

## Phase 6: Performance and Observability ✅

**Status:** Complete (commit `d947c28`). 747 unit tests, 54 functional tests, all passing.

### 6.1 Structured Logging ✅
- `utils/logging.py` — `SVRLogFormatter` (JSON), `ContextVar` correlation IDs, `log_operation` decorator
- `LogConfig` model — level, json_format, log_query_text, log_embeddings
- All 20 modules migrated to `get_logger(__name__)`

### 6.2 Metrics Hooks ✅
- `utils/metrics.py` — `MetricsHandler` protocol, `MetricsCollector`, `NoOpCollector`, 12 `MetricType` values
- `MetricsConfig` model — enabled, include_partition_tags, include_query_tags
- `SVRClient(metrics_handler=my_handler)` wires handler at init

### 6.3 Connection Pool Tuning ✅
- `DatabaseConfig` extended: `max_pool_size`, `min_pool_size`, `max_idle_time_ms`, `wait_queue_timeout_ms`
- Passed to `AsyncMongoClient` in `MongoDBBackend.connect()`
- Validated in `validate_config()`

### 6.4 Embedding Cache ✅
- `utils/cache.py` — Thread-safe LRU with TTL, `CacheKey`, `EmbeddingCache`
- `CacheConfig` model — enabled, max_size, ttl_seconds
- Integrated into `SVRClient.search()` — cache hit skips embedder call
- CLI: `svr cache stats`

### 6.5 Complete Time-Based Split ✅
- Real aggregation pipeline (`$min`/`$max`) for time range discovery
- `_generate_time_buckets()` — monthly, quarterly, yearly with UTC boundaries
- Dynamic bucket generation from actual data range
- Child naming: `{parent}_{label}` (e.g., `articles_2025_Q1`)

### Self-Assessment Checklist
- [x] Structured logs in JSON format contain all expected fields
- [x] Custom metrics handler receives all documented events
- [x] Pool settings correctly passed to AsyncMongoClient
- [x] Embedding cache hit rate >0 on repeated queries
- [x] Time split generates correct buckets from real data range

---

## Phase 7: Feature Completion [COMPLETE]

**Goal:** Document ingestion pipeline, rate limiting. The SDK handles both read and write paths.
**Self-assessment:** End-to-end test of ingest → search with all three index modes.
**Status:** Complete (commit `9c9afcb`). IngestPipeline, TokenBucketRateLimiter, RateLimiterRegistry, CLI ingest command, 6 new Pydantic models, 18 metric types. 859 unit tests + 93 functional tests.

### 7.1 Document Ingestion Pipeline ✅
`IngestPipeline` class in `ingestion.py` — accepts raw documents, extracts text from configurable fields (with template support), embeds in batches, converts to BinData, routes to correct field per index mode, writes via insert_many or bulk_write upsert. Progress callbacks, per-document error handling, configurable batch sizes.

### 7.2 Rate Limiting for External APIs ✅
`TokenBucketRateLimiter` in `utils/rate_limiter.py` — async-compatible token bucket with `asyncio.Lock`. `RateLimiterRegistry` for per-provider limits. `RateLimitConfig` and `ProviderRateLimit` Pydantic models. Built-in defaults for OpenAI, Voyage, Cohere, HuggingFace.

### 7.3 BinData Wiring for Ingestion ✅
`_convert_vector()` calls `vector_to_bindata()` during ingestion based on `storage_format` config. End-to-end path: Voyage int8 → BinData INT8 → MongoDB Binary storage.

### 7.4 CLI Ingest Command ✅
`svr ingest <file>` — reads JSON/JSONL files, ingests via pipeline. 12th CLI command group.

### 7.5 New Models ✅
`IngestConfig`, `IngestMode`, `IngestResult`, `IngestProgress`, `RateLimitConfig`, `ProviderRateLimit` — 6 new Pydantic v2 models with sensible defaults and backward-compatible config loading.

### Self-Assessment Checklist
- [x] Ingest documents → search returns them (all three index modes)
- [x] FIELDS mode ingestion writes to correct `embedding_{partition}` field
- [x] BinData INT8 storage round-trips correctly (ingest as int8, search with int8 query)
- [x] Rate limiter queues requests under load without dropping
- [x] Ingestion progress callbacks fire for all phases

---

## Phase 8: Developer Experience [COMPLETE]

**Goal:** The SDK is publishable: docs, examples, CI, packaging.
**Self-assessment:** `pip install` + `svr init` works from scratch. CI green on all Python versions.
**Status:** Complete. MIT LICENSE, CONTRIBUTING.md, py.typed, CI/CD pipeline (GitHub Actions, Python 3.9–3.13 matrix), 7 example scripts, pre-commit hooks (ruff + mypy), Makefile update, pyproject.toml updates, pragmatic mypy config with per-module overrides. Package builds as wheel+sdist with py.typed included. 859 unit tests + 93 functional tests still passing.

### 8.1 Package Files ✅
- **LICENSE** — MIT license (copyright 2026 Paul C)
- **CONTRIBUTING.md** — Dev setup, testing, code style, project conventions, PR process
- **py.typed** — PEP 561 marker in `semantic_vector_router/`
- **pyproject.toml** — Python 3.13 classifier, correct GitHub URLs, Changelog URL

### 8.2 CI/CD Pipeline ✅
`.github/workflows/ci.yml` with 5 jobs:
- **lint** — ruff + black (Python 3.12)
- **typecheck** — mypy with `--ignore-missing-imports` (Python 3.12)
- **test** — unit tests on Python 3.9–3.13 matrix with Codecov on 3.12
- **functional** — gated to `main` branch, requires Atlas secrets
- **publish** — PyPI trusted publishing on version tags

### 8.3 Examples ✅
7 example scripts in `examples/`:
- `quickstart.py` — Minimal 10-line search example
- `fields_mode.py` — FIELDS index mode with config dict
- `voyage4_asymmetric.py` — Asymmetric embedding with document_model
- `multi_partition_search.py` — Cross-partition + reranking comparison
- `ingest_documents.py` — IngestPipeline with progress callbacks
- `fastapi_integration.py` — FastAPI lifespan + search endpoint
- `custom_metrics.py` — InMemoryMetrics MetricsHandler demo

### 8.4 Pre-Commit Hooks ✅
`.pre-commit-config.yaml` with ruff-pre-commit (lint + format) and mirrors-mypy.

### 8.5 Type Checking ✅
Pragmatic mypy config: `check_untyped_defs = true` globally, `ignore_errors = true` per-module overrides for 15 modules with pre-existing issues. Fixed Python 3.9 compat (`slots=True` removal), type narrowing in config.py. Clean CI run.

### Self-Assessment Checklist
- [x] `pip install -e ".[dev]"` succeeds
- [x] Package builds as wheel and sdist
- [x] py.typed and LICENSE included in wheel
- [x] CI pipeline defined for Python 3.9–3.13
- [x] All 7 examples parse without syntax errors
- [x] LICENSE, CHANGELOG, CONTRIBUTING exist and are accurate
- [x] `mypy semantic_vector_router/ --ignore-missing-imports` clean
- [x] `ruff check examples/` passes
- [x] All 859 unit tests pass
- [x] All 93 functional tests pass

---

## Phase 9: Integration Testing + Hardening [COMPLETE]

**Goal:** End-to-end tests with real MongoDB Atlas. Edge cases. Production readiness.
**Self-assessment:** Integration suite passes against Atlas cluster. No silent failures.
**Status:** Complete (commit `aa053ec`). 7 integration test files, 9 performance benchmarks, edge case tests (1318 lines), error path tests (1066 lines), mypy strict migration (0 errors across 52 source files). 1376 unit + 35 integration + 93 functional + 9 performance tests.

### 9.1 Integration Tests ✅
- `test_search_flow.py` — Create collection, insert docs, create partitions (all three modes), search, verify results
- `test_fields_mode_flow.py` — FIELDS mode: ingest with partition-specific fields, search, verify correct index used
- `test_lifecycle_flow.py` — Scanner → provisioner → monitor → splitter flow
- `test_quantization_flow.py` — BinData round-trip with pre-quantized vectors

### 9.2 Edge Case Hardening ✅
- Empty partitions, single-document partitions, identical vectors, large result sets (10K+)
- Concurrent searches, config corruption recovery, connection drops mid-search
- Unicode/special chars, cache boundaries

### 9.3 Mypy Strict Migration ✅
- Removed all 15 `ignore_errors` overrides, fixed 87 type errors across 17 files
- Zero errors across all 52 source files

### Self-Assessment Checklist
- [x] All integration tests pass against real Atlas cluster
- [x] Edge cases handled gracefully with clear error messages
- [x] Zero silent failures: every error path produces actionable output
- [x] Stress test: 20 concurrent searches complete without deadlocks
- [x] Mypy strict: zero errors

---

## Phase 10: Hierarchical Centroid Routing [COMPLETE]

**Goal:** Route queries to the most relevant partitions using precomputed centroid embeddings, reducing fan-out from O(N) to O(log N).
**Self-assessment:** Centroid routing correctly prunes partitions. Filter-map routing resolves in O(1). Backward compatibility preserved.
**Status:** Complete (commit `68c9724`). CentroidRouter, filter-map routing, RoutingMode.AUTO cascade, centroid metrics, CLI compute-centroids.

### 10.1 Centroid Router ✅
- `routing/centroid.py` — Hierarchical tree walk with dynamic threshold pruning
- `utils/vector_math.py` — Pure Python cosine_similarity, normalize, mean_vector (no numpy)
- `compute_partition_centroid()` — Sample stored embeddings, compute mean, normalize

### 10.2 Filter-Map Routing ✅
- O(1) dict lookup when query filters match partition field
- Split-aware: maps filter values to leaf descendants

### 10.3 Routing CascParts Distributor ✅
- `RoutingMode.AUTO`: explicit → filter-map → centroid → fan-out
- `RoutingMode.EXPLICIT` preserved for backward compatibility

### 10.4 Integration ✅
- Embed-before-resolve reorder in `search()` (centroid routing needs the query vector)
- Post-repartition and post-ingest centroid triggers
- CLI `svr partitions compute-centroids`

### Self-Assessment Checklist
- [x] RoutingMode.AUTO activates the full cascade
- [x] Filter-map resolves in O(1), split-aware
- [x] Centroid routing walks tree with dynamic threshold pruning
- [x] Zero-API-call centroid computation
- [x] All integration tests pass against real Atlas + Voyage

---

## Phase 11: Operational Scheduling & Event System [COMPLETE]

**Goal:** Job scheduler with maintenance windows that queues operations until safe execution windows, plus event bus with webhook dispatch for external notifications.
**Self-assessment:** Scheduler enforces maintenance windows, acquires distributed locks, persists job state. Event bus delivers events to handlers and webhooks with HMAC signing and retry. All existing tests pass.
**Status:** Complete. JobScheduler engine, interval parser, maintenance window enforcement, EventBus, WebhookDispatcher with HMAC-SHA256, 21 SVREventType values, SchedulerConfig + EventsConfig + WebhookConfig models, config validation, CLI schedule + webhooks commands, event wiring into client/provisioner/repartition/detector. 298 new tests (1376 total unit). Example script.

### 11.1 Job Scheduler Engine
- `scheduler/engine.py` — `JobScheduler` with background asyncio tick loop, maintenance window enforcement, distributed lock acquisition, job handler dispatch, pause/resume, force-run, failure tracking, state persistence
- `scheduler/interval.py` — `parse_interval()` for human-readable durations
- `scheduler/window.py` — `is_within_window()` with timezone support
- `scheduler/models.py` — `JobType`, `JobConfig`, `JobStatus`, `JobRun`, `JobState`, `SchedulerStatus`

### 11.2 Event Bus + Webhook Dispatch
- `events/bus.py` — `EventBus` with per-type and global subscriptions, fire-and-forget async dispatch
- `events/models.py` — `SVREventType` (21 types), `SVREvent` with `to_dict()` serialization
- `events/webhook.py` — `WebhookDispatcher` with HMAC-SHA256 signing, retry on 5xx, event filtering, delivery history

### 11.3 Integration
- `SVRClient` — initializes event bus and scheduler on connect, registers default jobs, wires handlers
- `PartitionProvisioner` — emits partition.created/deleted events
- `RepartitionEngine` — emits repartition lifecycle events
- `PartitionDetector` — emits health signal events
- `SVRClient.ingest()` — emits ingest.completed event

### 11.4 CLI
- `svr schedule list|status|history|run|pause|resume|window` — 7 subcommands
- `svr webhooks list|test|history` — 3 subcommands

### Self-Assessment Checklist
- [x] Scheduler enforces maintenance windows (jobs outside window are skipped)
- [x] Interval parser handles all formats (1h, 30m, daily, weekly, compound)
- [x] Distributed lock prevents concurrent job execution across workers
- [x] Event bus delivers to per-type and global subscribers
- [x] Webhook dispatcher signs payloads with HMAC-SHA256
- [x] Webhook retries on 5xx, no retry on 4xx
- [x] All existing 1100+ unit tests still pass (1376 total)
- [x] Config validation catches invalid intervals, window hours, webhook URLs
- [x] Backward compatible: scheduler.enabled=False by default

---

## Phase 11.5: Structural Refactoring [COMPLETE]

**Goal:** Zero behavioral changes — purely structural cleanup of 5 oversized modules to prepare for multi-phase growth.
**Self-assessment:** All 1,376 unit tests pass after each of the 5 extraction steps. All public imports preserved.
**Status:** Complete. 10 new files, 72 total modules. config.py (636→245), provisioner.py (665→549), models.py (700+→subpackage), mongodb.py (938→689), client.py (1131→988).

### 11.5.1 Config Validators Extraction ✅
- `config_validators.py` — 13 focused validators extracted from `config.py`

### 11.5.2 Provisioner Split ✅
- `lifecycle/index_manager.py` + `lifecycle/view_manager.py` — index and view CRUD extracted from provisioner
- Provisioner delegates via composition, retry_func passed as parameter

### 11.5.3 Models Subpackage ✅
- `models.py` → `models/` subpackage with 7 files: `enums.py`, `config.py`, `partition.py`, `search.py`, `results.py`, `scheduler.py`, `svr_config.py`
- `__init__.py` re-exports all 49 classes with explicit `__all__`
- SVRConfig isolated in `svr_config.py` to prevent circular imports

### 11.5.4 MongoDB Backend Split ✅
- `backends/mongodb_views.py` (MongoDBViewOps) + `backends/mongodb_indexes.py` (MongoDBIndexOps)
- Composition with auto-propagating `_db` property
- Test fixture updated for `__new__`-based backend construction

### 11.5.5 Factory Extraction ✅
- `factories.py` — `create_embedder()`, `create_reranker()`, `create_document_embedder()`
- 30 test patches updated across 4 test files

### Self-Assessment Checklist
- [x] All 1,376 unit tests pass
- [x] All public imports preserved (backward compatible)
- [x] Models subpackage re-exports all 49 classes
- [x] MongoDB ops classes auto-receive db reference via property
- [x] Ruff clean (only pre-existing issues remain)

---

## Effort Estimate

| Phase | Description | Status | Sessions |
|-------|-------------|--------|----------|
| 1 | Critical Fixes + FIELDS Mode | Complete (`ebaa2b7`) | 1 |
| 2 | Testing Foundation | Complete (`17d1511`) | 1 |
| 3 | Robustness | Complete (`f8d4bb0`) | 1 |
| 4 | CLI Completeness | Complete (`0d98f68`) | 2 |
| 5 | Metadata Collection + Partition Lifecycle | Complete (`cbd1d8e`) | 3 |
| 6 | Performance + Observability | Complete (`d947c28`) | 1 |
| 7 | Feature Completion (Ingestion + Rate Limiting) | Complete (`9c9afcb`) | 1 |
| 8 | Developer Experience | Complete (`e41ceff`) | 1 |
| 9 | Integration + Hardening | Complete (`aa053ec`) | 2 |
| 10 | Hierarchical Centroid Routing | Complete (`68c9724`) | 1 |
| 11 | Operational Scheduling & Event System | Complete (`1dbebdc`) | 1 |
| 11.5 | Structural Refactoring | Complete | 1 |

**One "session" = one focused Claude Code conversation.**

**Total: 16 sessions** across 12 phases to reach current state (alpha, publishable to PyPI).

See [FUTURE_PHASES.md](./FUTURE_PHASES.md) for the forward-looking roadmap.

---

## Build Protocol

Each phase follows this protocol:

1. **Read** — Re-read all files being modified (context refresh)
2. **Implement** — Make changes, run linter (`ruff check`)
3. **Test** — Run relevant test suite
4. **Self-assess** — Check every item on the phase's checklist
5. **Fix** — Address any failures found in self-assessment
6. **Report** — Summarize what was done, what passed, what issues remain

If self-assessment reveals issues that can't be resolved within the phase, they become Phase N+1 blockers documented in the report.

---

## Architecture Decisions (Resolved)

| Question | Decision |
|----------|----------|
| Hash split | Deprecated. Splits must produce semantically meaningful partitions. |
| FIELDS vs VIEWS | Both supported. FIELDS for <50 partitions + max perf. VIEWS for >50 or legacy schemas. |
| Partition suggestions | Stats-based analysis only. No LLM dependency. SDK stays pure infrastructure. |
| BinData wiring | Query path in Phase 1 (simple). Ingestion path in Phase 6 (with ingestion pipeline). |
| Index limit (FIELDS) | 50 partition cap. Auto-recommend VIEWS beyond that. |
| MongoDB version compat | Resolved: `MongoDBBackend` detects server version on connect. VIEWS mode creates per-view indexes on 8.1+, falls back to shared source index on <8.1. |

## Architecture Decisions (All Resolved)

| Question | Decision | Phase |
|----------|----------|-------|
| Hash split | Deprecated. Splits must produce semantically meaningful partitions. | 1 |
| FIELDS vs VIEWS | Both supported. FIELDS for <50 partitions + max perf. VIEWS for >50. | 1 |
| Partition suggestions | Stats-based analysis only. No LLM dependency. SDK stays pure infrastructure. | 1 |
| Index limit (FIELDS) | 50 partition cap. Auto-recommend VIEWS beyond that. | 1 |
| MongoDB version compat | `MongoDBBackend` detects server version. Per-view indexes on 8.1+, shared source on <8.1. | 1 |
| BinData wiring | Query path in Phase 1. Ingestion path in Phase 7 (`_convert_vector()`). | 1→7 |
| Watcher deployment | In-process `asyncio.Task` with distributed lock for multi-worker. | 5 |
| Cache invalidation | TTL-only (simple, no watcher dependency). Configurable `ttl_seconds`. | 6 |
| Ingestion API scope | Full pipeline: embed → convert → route → write. Not helper-only. | 7 |
| Centroid routing | Hierarchical tree walk with dynamic threshold. Pre-normalized centroids. No numpy. | 10 |
| Auto-embedding centroid | When Atlas embeds server-side, centroid routing gracefully skipped. BYOM fully supported. | 10 |
| Scheduler design | Background asyncio task (not cron). Distributed lock reuse from Phase 5. | 11 |
| Event bus design | In-process fire-and-forget (not message queue). External delivery via webhooks. | 11 |
| Module decomposition | Composition over inheritance. Auto-propagating properties. Full re-exports. | 11.5 |
