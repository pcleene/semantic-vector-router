# Semantic Vector Router — Executive CTO Summary

**Date:** February 2026
**Status:** Alpha (v0.1.0) — 14 phases complete, publishable to PyPI, multi-backend production-hardened

---

## What It Is

Semantic Vector Router (SVR) is a **semantic search workload orchestrator** — a Python SDK that manages the full lifecycle of partitioned vector search across database backends. It solves a problem that every team hits when vector search moves from prototype to production: **how do you manage thousands to millions of embeddings across logical partitions without drowning in infrastructure complexity?**

MongoDB Atlas and PostgreSQL/pgvector are both fully supported backends. Phase 12 introduced a formal backend abstraction layer with capability-based Protocol mixins, and Phase 13 delivered a production-grParts Distributor PostgreSQL backend that proves the abstraction works across fundamentally different database paradigms (document store vs. relational). Users can switch backends by changing one config field. The orchestration intelligence — routing cascade, centroid routing, detection pipeline, repartition engine, job scheduling, event-driven notifications — is genuinely backend-agnostic, not just by design but by validated implementation. MongoDB is always present as the metadata/control plane store regardless of where vectors live.

Instead of making developers manage indexes, views, filters, embeddings, and partition lifecycle manually, SVR provides a single abstraction:

```python
svr = SVRClient()
await svr.connect()

result = await svr.search("wireless noise-cancelling headphones", partitions=["electronics"])
```

Behind that call, SVR handles partition resolution, embedding generation, vector search execution, optional reranking, result merging, caching, metrics emission, retry logic, and structured logging — all async, all configurable.

---

## The Problem It Solves

Teams building production vector search — regardless of database — face compounding orchestration complexity:

| Challenge | What goes wrong |
|-----------|----------------|
| **Partition management** | Manually creating and maintaining indexes per category, keeping them in sync |
| **Routing intelligence** | Deciding which partitions to search without scanning everything |
| **Lifecycle operations** | Splitting overgrown partitions, rebalancing, retiring old indexes — without downtime |
| **Operational safety** | Operations that should only run during off-peak hours fire immediately |
| **Embedding orchestration** | Wiring up provider SDKs, handling rate limits, caching repeated queries |
| **Observability** | No structured way to trace a search through embedding → route → search → rerank pipeline |
| **Notifications** | When a partition auto-splits or breaches a threshold, nobody knows unless they're watching logs |
| **Resilience** | One failed API call or dropped connection breaks the entire search path |

These are orchestration problems, not database problems. SVR solves them through a management plane with a single configuration file, consistent API, and 14-command CLI.

---

## Core Capabilities

### 1. Partitioned Vector Search

Three index modes, each with distinct tradeoffs:

| Mode | How it works | Best for |
|------|-------------|----------|
| **SOURCE** | Single index on source collection, partition filter at query time | Few partitions, simple setup |
| **VIEWS** | MongoDB views per partition, shared source index | Many partitions, browsability |
| **FIELDS** | Per-partition embedding fields, separate indexes | Maximum isolation (up to 50 partitions) |

Partition resolution is fully async with dual-source lookup (metadata collection or config fallback). Queries can target a single partition, multiple partitions (fan-out with result merging), or discover partitions automatically.

### 2. Multi-Provider Embedding & Reranking

Pluggable provider architecture with a common interface:

- **Embedders:** OpenAI, Voyage AI, Cohere, HuggingFace (local)
- **Rerankers:** Voyage AI, Cohere
- Embedding cache (LRU + TTL) eliminates redundant API calls — configurable max size and expiry
- BinData vector storage support for MongoDB's native binary format
- **Token-bucket rate limiting** — per-provider, async-compatible, with configurable burst and queuing

### 3. Document Ingestion Pipeline

Full write-path support — the SDK handles both read and write:

- **`IngestPipeline`** — accepts raw documents, embeds text, converts vectors, routes to correct field, writes to MongoDB
- Automatic text extraction from configurable source fields (with template support)
- Partition-aware field routing: FIELDS mode writes to `embedding_{partition}`, VIEWS/SOURCE writes to standard field
- BinData conversion during ingestion — Voyage 4 int8 → BinData INT8 storage works end-to-end
- INSERT or UPSERT mode with configurable batch sizes for embedding and write operations
- Progress callbacks for monitoring long-running ingestion jobs
- Per-document error handling: one failure doesn't abort the batch (configurable)

### 4. Partition Lifecycle Management

This is the differentiator. SVR treats partitions as living entities:

- **Detection pipeline** — 5 signals monitor partition health continuously:
  - Threshold breach (too many vectors)
  - Approaching threshold (trend-based prediction)
  - Severe skew (imbalanced sibling partitions)
  - Underpopulated (below minimum threshold)
  - Stale (zero growth detected)
- **Auto-split** — When partitions outgrow thresholds, SVR splits them by secondary field, time buckets (monthly/quarterly/yearly), or alerts for manual intervention
- **Repartition engine** — 5-step zero-downtime workflow: create children, build indexes, wait for readiness, switch routing, cleanup parent. Supports resume on failure and full rollback
- **Distributed locking** — MongoDB-based locks with TTL for safe multi-worker coordination
- **Metadata store** — Dedicated `svr_metadata` collection tracks partition state, operations, and locks across distributed deployments

### 5. Production Observability

- **Structured logging** — JSON-formatted log output with correlation IDs that propagate across async awaits. Every search request gets a unique trace ID. Privacy controls for query text and embeddings
- **Metrics hooks** — Pluggable `MetricsHandler` protocol. Wire SVR into Datadog, Prometheus, CloudWatch, or any monitoring system. 18 metric types covering search latency, embedding latency, cache hit rates, ingestion throughput, rate limiter waits, detection events, and errors
- **Embedding cache stats** — Hit rate, eviction count, and memory usage via CLI or programmatic access

### 6. Resilience

- **Retry with exponential backoff** — Configurable max retries, base delay, jitter, and retry-eligible exception types
- **Connection health checks** — Automatic reconnection on dropped MongoDB connections
- **Provisioner rollback** — If index or view creation fails mid-operation, cleanup runs automatically
- **Watcher auto-reconnect** — Change stream watchers recover from transient failures

### 7. Full CLI

14 command groups for operational management without writing code:

| Command | Purpose |
|---------|---------|
| `svr init` | Interactive or non-interactive project setup |
| `svr partitions` | List, create, delete, inspect partitions |
| `svr search` | Execute vector searches from terminal |
| `svr analyze` | Partition health analysis and statistics |
| `svr index` | Create, list, delete vector search indexes |
| `svr config` | View and validate configuration |
| `svr watch` | Start change stream watchers |
| `svr split` | Manual partition splitting |
| `svr monitor` | Real-time partition monitoring |
| `svr repartition` | Run/resume/rollback repartition workflows |
| `svr cache` | Embedding cache statistics |
| `svr ingest` | Document ingestion from JSON/JSONL files |
| `svr schedule` | Job scheduler management (list, run, pause, resume) |
| `svr webhooks` | Webhook configuration and testing |

### 8. Multi-Backend Architecture (Phases 12 + 13)

The backend abstraction layer that makes SVR genuinely portable across vector databases — **proven by two production implementations**:

- **`BaseBackend` interface** — abstract base class defining 16 universal operations every backend must support: partition storage, index lifecycle, vector search, document CRUD, ingestion, and cross-partition search
- **Capability-based Protocol mixins** — `ChangeStreamCapable` and `AutoEmbeddingCapable` allow backends to opt into features that not all databases support. The orchestration layer checks capabilities at runtime rather than requiring stub implementations
- **Universal models** — `IndexStatus` enum (`READY`, `BUILDING`, `ERROR`, `NOT_FOUND`) and `PartitionStorageResult` provide a common vocabulary for cross-backend status reporting
- **`BackendFactory`** — config-driven backend instantiation (`backend: "mongodb"` resolves to `MongoDBBackend`). Extensible registration for new backends
- **Filter translation DSL** — MongoDB query syntax as the universal filter format. Pass-through for MongoDB (zero overhead), translatable for PostgreSQL and future backends
- **PostgreSQL 17 + pgvector 0.8.1** — full production backend (Phases 13–14). 4 production modules, hardened config validators, SQL injection protection via `psycopg.sql`, normalized score expressions, and backend-agnostic ingestion

### 9. Developer Experience (Phase 8)

The SDK is now publishable:

- **`pip install semantic-vector-router`** — builds cleanly as wheel and sdist via hatchling
- **CI/CD pipeline** — GitHub Actions with lint (ruff + black), type check (mypy), test matrix (Python 3.9–3.13), functional tests on `main`, and PyPI publish on tagged releases via trusted publishing
- **7 example scripts** — quickstart, FIELDS mode, Voyage 4 asymmetric embeddings, cross-partition search with reranking, document ingestion, FastAPI integration, custom metrics handler
- **CONTRIBUTING.md** — full contributor guide with dev setup, testing, code style, project conventions, and PR process
- **Pre-commit hooks** — ruff (lint + format) and mypy
- **PEP 561 type marker** — `py.typed` included in wheel for downstream type checking
- **MIT License** — clean open-source licensing

### 10. Testing Excellence & Type Safety (Phase 9)

Production-grParts Distributor test coverage and strict type checking:

- **Integration tests** (7 files) — end-to-end tests for all 3 index modes (SOURCE, FIELDS, VIEWS), ingest-search roundtrip, lifecycle detection pipeline, and CLI commands. All run against real MongoDB Atlas with real Voyage AI embeddings
- **Performance benchmarks** (9 tests) — search latency, multi-partition fan-out, cache speedup (3x+), 20 concurrent searches without deadlock, ingest throughput
- **Edge case unit tests** (1318 lines) — empty/single-doc partitions, concurrent operations, connection failures, unicode/special chars, cache boundaries, large result merging (10K hits)
- **Error path unit tests** (1066 lines) — every public `SVRClient` method's failure modes with recovery suggestions and metric emission verification
- **Mypy strict migration** — removed all 15 `ignore_errors` overrides, fixed 87 type errors across 17 source files. Zero errors across all 52 source files
- **CI integration job** — integration tests run on `main` branch with secrets-based Atlas/Voyage credentials

### 11. Hierarchical Centroid Routing (Phase 10)

The core differentiating feature — intelligent query-to-partition routing that reduces fan-out from O(N) to O(log N):

- **Routing cascade** — four-step resolution with first-match-wins semantics:
  1. **Explicit partitions** → resolve directly (backward compatible, zero overhead)
  2. **Filter-map routing** → O(1) dict lookup when query filters match the partition field
  3. **Centroid routing** → hierarchical tree walk scoring partition centroids against query embedding, pruning branches below a dynamic threshold (`max_score × relative_threshold`)
  4. **Fallback** → fan-out to all partitions (preserves existing behavior)
- **CentroidRouter** — walks the partition tree top-down using precomputed centroid embeddings. Pure Python vector math (cosine similarity, normalize, mean vector) — no numpy dependency. Never returns empty results (top-1 fallback guarantee)
- **Zero-API-call centroid computation** — samples stored embedding vectors from MongoDB, computes normalized mean. Integrated into repartition workflow (step 6), post-ingest trigger, and CLI (`svr partitions compute-centroids`)
- **Embedding reorder** — `search()` now embeds the query BEFORE partition resolution (centroid routing needs the vector). Embedding still happens exactly once, cache still applies. In auto-embedding mode (Atlas server-side), centroid routing is gracefully skipped
- **Filter-map routing** — split-aware leaf resolution with O(1) dict lookup. When a query includes `filters={"category": "electronics"}` and that matches the partition field, SVR resolves directly to the matching leaf partitions without any centroid or fan-out overhead
- **Full observability** — `CENTROID_ROUTE_LATENCY` and `CENTROID_ROUTE_PARTITIONS` metrics emitted through the existing hooks system
- **Configuration** — `CentroidRoutingConfig` with `enabled`, `relative_threshold` (0.5), `min_score` (0.15), `max_probe_partitions` (5), `sample_size` (500). Validated at config load time

### 12. Operational Scheduling & Event System (Phase 11)

**Production operations need timing control and external notifications.** Before Phase 11, SVR operations fired immediately when triggered — a repartition during peak traffic or an auto-split on Black Friday degrades your SLA. Now, SVR queues operations until safe maintenance windows and notifies external systems when things happen.

- **Job scheduler engine** — Background asyncio task that manages recurring jobs (detection, centroid refresh, count updates, repartition checks, index health). Configurable tick interval, distributed locking via MongoDB (only one worker executes per job), pause/resume, force-run, consecutive failure tracking, and job state persistence across restarts
- **Maintenance windows** — `"repartitions only run 2-5am UTC on weekends"` — reuses the existing `SplitScheduleConfig` model (`allowed_days`, `allowed_hours`, `timezone`). Jobs scheduled outside the window are skipped with recorded reason
- **Interval parser** — Human-readable duration strings (`"1h"`, `"30m"`, `"daily"`, `"weekly"`, `"2d12h"`) parsed to seconds. Existing interval strings in config models are now actually enforced
- **Event bus** — In-process async event dispatch with fire-and-forget semantics. 21 event types across 6 categories (partition lifecycle, health signals, repartition operations, centroid computation, ingestion, scheduler). Per-type and global subscriptions. Handler exceptions caught and logged, never propagate
- **Webhook dispatcher** — HTTP delivery via httpx with HMAC-SHA256 request signing, retry with exponential backoff on 5xx (no retry on 4xx), per-webhook event filtering, delivery history. A Slack/PagerDuty/custom endpoint receives: `"partition electronics_phones was created"`, `"health threshold breach on electronics: 12M vectors"`, `"repartition completed for articles"`
- **Wired into existing modules** — `PartitionProvisioner`, `RepartitionEngine`, `PartitionDetector`, and `SVRClient.ingest()` all emit events automatically. Zero manual instrumentation needed

**Key design decisions:**
1. **Scheduler disabled by default** (`scheduler.enabled=False`) — zero impact on existing deployments. Events enabled by default (cheap with no handlers)
2. **Reuses existing infrastructure** — distributed locks from Phase 5, `SplitScheduleConfig` from Phase 5, metrics hooks pattern from Phase 6, background task pattern from `_monitor_loop`
3. **Event bus is in-process, not a message queue** — suitable for single-process and multi-worker (with lock coordination). External delivery via webhooks, not Kafka/Redis

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                        SVRClient                             │
│  search() ── ingest() ── connect() ── get_partition() ──close│
├──────────────┬──────────────┬────────────────┬───────────────┤
│  Embedders   │   Routing    │   Lifecycle    │   Utils       │
│  ──────────  │   ───────    │   ─────────    │   ─────       │
│  OpenAI      │  Resolver    │  Provisioner   │  Logging      │
│  Voyage      │  CentroidRtr │   IndexMgr     │  Metrics      │
│  Cohere      │  FilterMap   │   ViewMgr      │  Cache        │
│  HuggingFace │  Merger      │  Watcher       │  Retry        │
│              │              │  Splitter      │  RateLimiter  │
│  Rerankers   │  Ingestion   │  Detector      │  VectorMath   │
│  ──────────  │  ─────────   │  Repartition   │               │
│  Voyage      │  Pipeline    │  Monitor       │  Scheduler    │
│  Cohere      │  BinData     │  Scanner       │  ──────────   │
│              │              │                │  Engine       │
│  Factories   │              │  Events        │  Interval     │
│  ──────────  │              │  ────────      │  Window       │
│  Embedder    │              │  EventBus      │               │
│  Reranker    │              │  Webhooks      │               │
├──────────────┴──────────────┴────────────────┴───────────────┤
│            BaseBackend + BackendFactory (Phase 12)            │
│  Protocol Mixins: ChangeStreamCapable, AutoEmbeddingCapable  │
│  Universal Models: IndexStatus, PartitionStorageResult       │
│  Filter Translation DSL (MongoDB syntax → any backend)       │
├──────────────────────────────────────────────────────────────┤
│  MongoDB Backend (async PyMongo)  │  PostgreSQL/pgvector      │
│  backends/mongodb/                │  backends/postgres/       │
│  backend.py │ views.py │ indexes │  backend.py │ config.py   │
│  vectors.py │ index_manager.py   │  filters.py (SQL WHERE)   │
│  view_manager.py │ svr_metadata  │  HNSW + IVFFlat indexes   │
└──────────────────────────────────────────────────────────────┘
```

---

## By the Numbers

| Metric | Value |
|--------|-------|
| Production code | ~21,500 lines across 90+ modules |
| Unit tests | 1,710 (all mocked, no network, <18s) |
| Integration tests | 66+ (real MongoDB Atlas + Voyage AI + PostgreSQL 17) |
| Functional tests | 93 (against real MongoDB Atlas) |
| Performance benchmarks | 9 (latency, throughput, concurrency) |
| Total tests | 1,878+ |
| mypy strict | 0 errors across 90+ source files |
| CLI commands | 14 command groups (with multi-backend support) |
| Embedding providers | 4 (OpenAI, Voyage, Cohere, HuggingFace) |
| Reranking providers | 2 (Voyage, Cohere) |
| Custom exceptions | 20 typed error classes |
| Config models | 35+ Pydantic v2 models |
| Detection signals | 5 |
| Event types | 21 (across 6 categories) |
| Metric types | 20 |
| Routing strategies | 4 (explicit, filter-map, centroid, fan-out) |
| Supported backends | 2 production (MongoDB Atlas, PostgreSQL/pgvector) |
| Example scripts | 9 |
| CI Python versions | 5 (3.9–3.13) |
| Development phases | 14 completed |

---

## Development Phases Completed

| Phase | Focus | Commit |
|-------|-------|--------|
| 1 | FIELDS mode, filter auto-detection, async fixes, BinData | `ebaa2b7` |
| 2 | Testing foundation, 80%+ coverage | `17d1511` |
| 3 | Retry/backoff, resilience, rollback, reconnect | `f8d4bb0` |
| 4 | Full CLI (12 command groups), non-interactive init | `0d98f68` |
| 5 | Metadata store, detection pipeline, repartition engine | `cbd1d8e` |
| 6 | Structured logging, metrics hooks, embedding cache, pool tuning | `d947c28` |
| 7 | Document ingestion pipeline, rate limiting, BinData wiring | `9c9afcb` |
| 8 | Developer experience: packaging, CI/CD, examples, pre-commit | `e41ceff` |
| 9 | Testing excellence, mypy strict migration, performance benchmarks | `aa053ec` |
| 10 | Hierarchical centroid routing, filter-map routing, auto partition resolution | `68c9724` |
| 11 | Operational scheduling, maintenance windows, event bus, webhook dispatch | `1dbebdc` |
| 11.5 | Structural refactoring: 5 oversized modules split into focused submodules | `3a82197` |
| 12 | Backend abstraction layer, capability Protocols, filter DSL, PostgreSQL PoC | `8cdba22` |
| 12.5 | Backend interface completion — clean BaseBackend for multi-backend readiness | `36b4b3c` |
| 13 | PostgreSQL/pgvector production backend — 222 tests, full BaseBackend implementation | `7a82bca` |
| 14 | Audit hardening, backend-agnostic ingestion, CLI multi-backend, 102 new tests | *pending* |

---

## What's Next

See FUTURE_PHASES.md for the full forward-looking roadmap.
See MULTI_BACKEND_ROADMAP.md for the multi-backend architecture vision.

**Phase 15: Adoption Experience** *(next — P0)*

- Guided onboarding (`svr onboard`), existing index attachment, pre-embedded ingestion
- `CREATE INDEX CONCURRENTLY` for non-blocking Postgres index builds on large tables
- LISTEN/NOTIFY for PostgreSQL change detection, automated repartition

**Phase 16+: Production Platform**

- Multi-tenant isolation, query analytics, API gateway with auth, streaming ingestion, dashboard UI
- Cold partition mode, per-partition index parameters, recall monitoring

---

## Technical Stack

- **Language:** Python 3.9+ (tested on 3.9–3.13)
- **Database:** MongoDB Atlas (async PyMongo) + PostgreSQL 17/pgvector 0.8.1 (psycopg3 async) — both production backends
- **Metadata store:** MongoDB (always present as control plane — locks, partition state, job history)
- **Search:** MongoDB Atlas Vector Search + pgvector (IVFFlat, HNSW) — abstracted behind BaseBackend interface
- **Validation:** Pydantic v2
- **CLI:** Click + Rich
- **HTTP:** httpx (async)
- **Testing:** pytest + pytest-asyncio (1,878+ tests)
- **Packaging:** hatchling (modern pyproject.toml)
- **CI/CD:** GitHub Actions + Codecov + PyPI trusted publishing
- **Linting:** ruff + black + mypy

---

## Why This Matters

Every company building AI-powered search or retrieval hits the same wall: vector search works great in a demo, but managing it at scale across dozens or hundreds of logical partitions is an infrastructure nightmare. SVR turns that into a configuration problem instead of an engineering problem.

**The commercial moat is the orchestration intelligence, not the backends.** Centroid routing, detection pipeline (5 signals), auto-repartitioning, maintenance windows, webhook notifications — these algorithms are backend-agnostic and hard to replicate. The more backends SVR supports, the more valuable the routing layer becomes. A user with data in both Postgres and MongoDB gets unified partition routing across both.

The lifecycle management layer — detection, splitting, repartition with zero downtime — is something no existing vector search SDK offers. It's the difference between "we have vector search" and "we have vector search that operates itself."

The SDK is now pip-installable with CI running across 5 Python versions, 9 example scripts, comprehensive documentation, and 1,878+ tests (1,710 unit, 66+ integration, 93 functional, 9 performance). Strict mypy with zero errors across all 90+ modules. With Phase 14's audit hardening and backend-agnostic ingestion, both MongoDB and PostgreSQL backends are production-grParts Distributor — the multi-backend vision is proven by two hardened implementations across fundamentally different database paradigms. It's ready for alpha users and early adopters.

This is the foundation for a managed service where customers configure their partitioning strategy, choose their backend, and SVR handles the rest — routing, lifecycle, scheduling, notifications, and observability across any vector database.

---

## Notable Changes by Phase

### Phase 12: Backend Abstraction Layer

**The prerequisite for multi-backend support — and the proof that the orchestration layer is genuinely portable.** Before Phase 12, SVR's architecture claimed backend-agnosticism but was deeply coupled to MongoDB at every interface boundary. Phase 12 delivers a formal abstraction layer with capability-based Protocols, universal models, and a config-driven factory — then validates it by running 6 proof-of-concept integration tests against a real PostgreSQL 17 + pgvector 0.8.1 instance.

**Key architectural decisions:**

1. **Capability-based Protocol mixins over monolithic interfaces** — Not every backend supports change streams or server-side auto-embedding. Rather than forcing all backends to implement MongoDB-shaped methods (or raise `NotImplementedError`), Phase 12 introduces `ChangeStreamCapable` and `AutoEmbeddingCapable` as Protocol mixins. The orchestration layer checks `isinstance(backend, ChangeStreamCapable)` before enabling change stream features. This means PostgreSQL can implement `BaseBackend` without faking MongoDB-specific capabilities.

2. **Universal models decouple status reporting** — `IndexStatus` enum (`READY`, `BUILDING`, `ERROR`, `NOT_FOUND`) and `PartitionStorageResult` provide a common vocabulary for index lifecycle across backends. MongoDB's `queryable: true/false` maps to `READY/BUILDING`; pgvector's synchronous index creation maps to immediate `READY`. The provisioner and lifecycle manager work against these universal models, not backend-specific status shapes.

3. **MongoDB backend reorganized into a package** — `backends/mongodb.py` (monolith) became `backends/mongodb/` with 6 focused modules: `backend.py` (connection + search), `views.py` (view CRUD), `indexes.py` (index CRUD), `vectors.py` (vector utilities), `index_manager.py` (lifecycle), and `view_manager.py` (lifecycle). All existing import paths preserved through `__init__.py` re-exports. This mirrors the structure that `backends/postgres/` will follow in Phase 13.

4. **Filter translation DSL** — MongoDB query syntax (`{"$gte": 5, "$in": ["a", "b"]}`) serves as the universal filter format. For MongoDB, filters pass through unchanged (zero overhead). For other backends, a translation layer converts to native syntax (e.g., PostgreSQL WHERE clauses). This avoids inventing a custom DSL while still supporting multi-backend portability.

5. **BackendFactory for config-driven instantiation** — `backend: "mongodb"` in config resolves to `MongoDBBackend` via factory lookup. Extensible for future backends (`backend: "postgres"` will resolve to `PostgresBackend`). This eliminates hardcoded backend construction in `SVRClient.connect()`.

6. **Full backward compatibility** — zero behavioral changes for existing MongoDB users. All 1,405 unit tests pass, all existing import paths work, all CLI commands unchanged. The abstraction layer is additive, not disruptive.

**PostgreSQL validation:** 6 PoC integration tests verified against PostgreSQL 17 with pgvector 0.8.1 — IVFFlat and HNSW index creation, vector insert, cosine similarity search, and index status queries all pass. This proves the abstraction interface is sufficient for a real second backend, not just a theoretical exercise.

### Phase 11: Operational Scheduling & Event System

**The bridge between "it can do operations" and "it does operations safely."** Before Phase 11, triggering a repartition or detection run was immediate and uncontrolled. Now SVR queues operations into configurable maintenance windows and notifies external systems when events occur.

**Key architectural decisions:**

1. **Scheduler is a background asyncio task, not a cron replacement** — it ticks every N seconds, checks what's due, and executes within the same process. Distributed locking (reusing Phase 5 MongoDB locks) ensures only one worker runs each job across a multi-worker deployment.

2. **Event bus is in-process, not a message queue** — `asyncio.ensure_future` for fire-and-forget delivery to handlers. This is intentional: SVR doesn't want to require Kafka, Redis, or RabbitMQ as a dependency. For external delivery, webhooks bridge the gap with HMAC signing and retry logic.

3. **Minimal changes to existing modules** — provisioner, repartition engine, and detector each gained ~15 lines of code (an optional `event_bus` parameter and `_emit_event()` calls). Zero behavioral changes when events are disabled. This validates the original architecture: the module boundaries were clean enough that cross-cutting concerns like events could be added without refactoring.

4. **Reuses SplitScheduleConfig as MaintenanceWindow** — the `allowed_days`, `allowed_hours`, `timezone` model was declared in Phase 5 but never enforced. Phase 11 wires it into the scheduler engine via a type alias (`MaintenanceWindow = SplitScheduleConfig`), avoiding model duplication.

### Phase 10: Hierarchical Centroid Routing

**The single most important feature for production viability.** Before Phase 10, `partitions="all"` meant fan-out to every partition — N vector searches for N partitions. After Phase 10, SVR intelligently routes queries to 2-3 relevant partitions using centroid similarity, reducing search cost by 90%+ in large deployments.

**Key architectural decisions:**

1. **Embed-before-resolve reorder** — `search()` previously resolved partitions then embedded the query. Phase 10 reverses this: embed first, then use the query vector for centroid routing, then search. The embedding still happens exactly once and the cache still applies. This was a non-trivial change that touched the core search flow.

2. **Graceful degradation for auto-embedding mode** — When Atlas handles embedding server-side (`EmbeddingMode.AUTO`), SVR has no client-side query vector. Rather than forcing a client-side embed just for routing, centroid routing is gracefully skipped and the cascParts Distributor falls through to fan-out. This preserves zero-config simplicity for auto-embedding users while delivering full routing power for BYOM users.

3. **Zero-API-call centroid computation** — Centroids are computed by sampling stored embedding vectors from MongoDB, not by calling the embedding API. This means centroid computation is free, fast, and works regardless of embedding provider availability.

4. **Filter-map routing as cascParts Distributor step 2** — When query filters include the partition field (e.g., `filters={"category": "electronics"}`), SVR resolves directly to matching leaf partitions via O(1) dict lookup. This is faster than centroid routing and takes priority in the cascade. Split-aware: maps through SPLIT/RETIRED parents to their ACTIVE leaf descendants.

5. **Post-ingest and post-repartition centroid triggers** — Centroids are computed automatically when needed: after the first ingest into a partition (if centroid is missing), and as step 6 of the repartition workflow (for newly ACTIVE child partitions). No manual intervention required.

**Integration test validation:** All 6 integration tests pass against real MongoDB Atlas + Voyage AI — "headphones" queries route to electronics, "desk/chair" queries route to furniture, explicit partitions bypass routing, filter-map takes priority over centroid routing.

### Phase 11.5: Structural Refactoring

**Zero behavioral changes, pure structural improvement.** Five oversized modules were split into focused, testable submodules. Every public import path, CLI command, and mock patch target preserved through re-exports and composition.

**What was refactored:**

| Module | Before | After | Extracted to |
|--------|--------|-------|-------------|
| `config.py` | 636 lines | 245 lines | `config_validators.py` (13 validators) |
| `lifecycle/provisioner.py` | 665 lines | 549 lines | `lifecycle/index_manager.py`, `lifecycle/view_manager.py` |
| `models.py` | 700+ lines (monolith) | 7-file subpackage | `models/enums.py`, `models/config.py`, `models/partition.py`, `models/search.py`, `models/results.py`, `models/scheduler.py`, `models/svr_config.py` |
| `backends/mongodb.py` | 938 lines | 689 lines | `backends/mongodb_views.py`, `backends/mongodb_indexes.py` |
| `client.py` | 1131 lines | 988 lines | `factories.py` (embedder/reranker creation) |

**Key architectural decisions:**

1. **Composition over inheritance** — `MongoDBBackend` delegates to `MongoDBIndexOps` and `MongoDBViewOps` via composition. A `_db` property auto-propagates the database reference to ops classes, so tests that set `backend._db = mock_db` work without changes.

2. **Atomic `models.py` → `models/` swap** — The monolithic `models.py` was replaced with a 7-file subpackage using `mv models.py models_old.py && mv models_pkg models`. The `__init__.py` re-exports all 49 classes with explicit `__all__`, so every existing `from semantic_vector_router.models import X` statement works unchanged.

3. **SVRConfig isolated in `svr_config.py`** — `SVRConfig` imports from both `config.py` (all `*Config` models) and `scheduler.py` (`SchedulerConfig`, `EventsConfig`), while `scheduler.py` imports `MaintenanceWindow` from `config.py`. Separating `SVRConfig` into its own file eliminates circular dependency risk within the subpackage.

4. **Factory extraction preserves test patterns** — Embedder/reranker factory logic moved from `client.py` to `factories.py`. Test patches updated from `semantic_vector_router.client.VoyageEmbedder` to `semantic_vector_router.factories.VoyageEmbedder` (30 patches across 4 test files). This is the one case where patch targets moved — necessary because the constructors are now called from `factories.py`'s namespace.

**Verification:** 1,376 unit tests pass after each of the 5 extraction steps. Zero behavioral changes confirmed.

### Phase 13: PostgreSQL/pgvector Production Backend

**The proof that SVR's backend abstraction actually works.** Phase 13 implemented a complete `PostgresBackend` that passes through the same `BaseBackend` interface as `MongoDBBackend`, proving the abstraction handles both document-store and relational paradigms without leaking. Users can switch from MongoDB to Postgres by changing `backend: "postgres"` in config — no application code changes.

**What was built:**

| Component | Details |
|-----------|---------|
| `backends/postgres/config.py` | `PostgresBackendConfig`, `PgIndexType` (HNSW/IVFFlat), `PgDistanceMetric` (cosine/L2/IP), `HnswConfig`, `IvfflatConfig` |
| `backends/postgres/filters.py` | SVR MongoDB-syntax filter DSL → parameterized SQL WHERE clauses. 11 operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $and, $or). All values parameterized — zero SQL injection surface |
| `backends/postgres/backend.py` | Full `PostgresBackend` implementing all 14 `BaseBackend` abstract methods + `search_partitions()`. Single table with `partition_name` column, JSONB `content` for flexible document storage, pgvector `embedding` column |
| `client.py` | `_backend` type changed from `Optional[MongoDBBackend]` to `Optional[BaseBackend]`. MetadataStore `_set_shared_db()` guarded with `hasattr` check |
| 222 new tests | 52 backend unit (mocked psycopg), 45 filter tests, 95 config tests, 5 factory tests, 25 integration (real PG17 + pgvector 0.8.1) |

**Key architectural decisions:**

1. **Single table with partition_name column, not per-partition tables** — Simpler cross-partition queries, single shared HNSW index, mirrors MongoDB SOURCE mode. Postgres native `PARTITION BY LIST` is a future optimization that can be added without changing the backend interface.

2. **HNSW as default index** — Better recall with no training data needed. IVFFlat available for write-heavy workloads where build speed matters more than query latency.

3. **JSONB `content` column** — PostgreSQL's JSONB type stores arbitrary document fields (like MongoDB's schemaless documents) while supporting efficient querying. The filter translator auto-detects top-level columns (id, partition_name, timestamps) vs JSONB content fields (`content->>'field'`), with `::numeric` cast for numeric comparisons.

4. **Synchronous index builds** — `CREATE INDEX` (not `CONCURRENTLY`) for Phase 13 simplicity. `wait_for_index_ready()` returns `True` immediately — the key difference from MongoDB Atlas's async index builds. `CREATE INDEX CONCURRENTLY` is a Phase 14 optimization.

5. **Score normalization** — pgvector's `<=>` returns cosine distance (0 = identical). SVR expects similarity scores (1 = identical). Backend converts via `1 - distance`. Each distance metric has its own operator and normalization.

6. **Lazy imports to break circular deps** — `backends/postgres/__init__.py` uses `__getattr__` to defer `PostgresBackend` import, preventing circular dependency chains through models package.

**Abstraction validation:** Every Phase 12/12.5 abstraction element was validated — 14 abstract methods implemented without stubs, capability Protocols correctly exclude PostgresBackend from ChangeStream/AutoEmbedding, universal models (`IndexStatus`, `PartitionStorageResult`) work naturally, BackendFactory dispatches correctly, filter translation DSL maps cleanly to SQL.

**Verification:** 1,608 unit tests + 31 Postgres integration tests pass. Zero regressions on all pre-existing tests.

### Phase 14: Audit Hardening & Backend-Agnostic Ingestion

**Production hardening driven by independent audit + completing the backend abstraction for writes.** Phase 13's audit identified 6 issues ranging from SQL injection vectors to incorrect score calculations. Phase 14 resolves all 6 and extends the backend abstraction to cover document ingestion — meaning SVR can now both read and write through any backend without backend-specific code in the application layer.

**What was built:**

| Component | Details |
|-----------|---------|
| `backends/base.py` | 2 new `@abstractmethod`s: `insert_documents()` and `search_partitions()` — BaseBackend now has 16 abstract methods |
| `backends/postgres/config.py` | Range validators for HNSW (`m` 2-100, `ef_construction` 10-2000, `ef_search` 10-1000), IVFFlat (`lists` 1-10000, `probes` 1-500), pool sizes, schema/table regex, cross-field pool validation |
| `backends/postgres/filters.py` | Field name validation regex (`^[a-zA-Z_][a-zA-Z0-9_.]*$`), `validate_field_name()` — blocks SQL injection in `_column_ref()` |
| `backends/postgres/backend.py` | `_get_score_expression()` for correct L2/IP score normalization, `insert_documents()` with upsert, DDL hardened with `psycopg.sql.Identifier()`, `get_partition_document_counts` field parameter fix |
| `backends/mongodb/backend.py` | `insert_documents()` with `_with_retry` pattern and `BulkWriteError` partial-failure recovery |
| `ingestion.py` | `_write_batch()` refactored: dispatches to `_write_batch_mongodb()` or `_write_batch_generic()` (uses `backend.insert_documents()`) |
| `client.py`, `field_analyzer.py`, `repartition.py` | 5 `hasattr(backend, "db")` guards — non-MongoDB backends skip MongoDB-specific operations gracefully |
| `cli/init.py` | `svr init --backend postgres` with `--schema`, `--table-prefix`, `--index-type`, `--distance-metric` options. Both interactive and non-interactive modes |
| `pyproject.toml` | Suppressed `datetime.utcnow()` and pytest-asyncio sync test warnings |
| 102 new tests | Score expressions, field validation, config validators, insert_documents (both backends), BaseBackend contract, ingestion dispatch, .db access guards, partition doc counts |

**Key architectural decisions:**

1. **`psycopg.sql` for all DDL** — All table/index/schema names in dynamic SQL now use `sql.Identifier()`, `sql.SQL()`, and `sql.Literal()`. This is the PostgreSQL-standard approach to preventing SQL injection in DDL statements where parameterized queries don't apply.

2. **Backend-agnostic ingestion via dispatch** — `_write_batch()` checks `hasattr(backend, "db")` to route between the MongoDB-specific path (with `BulkWriteError` handling) and the generic path (using `backend.insert_documents()`). This means the ingestion pipeline works for any backend that implements `BaseBackend`.

3. **Score normalization per distance metric** — `_get_score_expression()` returns the correct SQL expression for each metric: `1 - (embedding <=> %s)` for cosine, `1.0 / (1.0 + (embedding <-> %s))` for L2, `-(embedding <#> %s)` for inner product. Previously all metrics used the cosine formula.

4. **Field-aware partition counts** — `get_partition_document_counts(field=...)` now correctly uses the `field` parameter: direct column reference for `partition_name`, JSONB expression (`content->>'field'`) for other fields. Previously hardcoded to `partition_name`.

5. **CLI backend selection** — `svr init --backend postgres` generates a complete `svr_config.yaml` with PostgreSQL-specific settings. Auto-embedding mode (Atlas-only) is excluded for Postgres. Partition provisioning is skipped for Postgres (tables are created on-demand).

**Verification:** 1,710 unit tests pass (0 failures, 22 pre-existing RuntimeWarnings). 102 new Phase 14 tests all pass.
