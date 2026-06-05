# Semantic Vector Router — Roadmap

**Last updated:** February 2026
**Current state:** Alpha (v0.1.0) — 13 phases complete, 85+ modules, 1,776+ tests
**Build plan:** See [ROADMAP.md](./ROADMAP.md) for completed phase specifications
**Multi-backend vision:** See [MULTI_BACKEND_ROADMAP.md](./MULTI_BACKEND_ROADMAP.md) for strategic architecture

This document captures the forward-looking roadmap. Phases are ordered by priority — P0 items are required for the open-source product to be adoption-ready, P1 for production users, P2+ for managed service and differentiation.

---

## Completed Phases (Summary)

| Phase | Focus | Commit |
|-------|-------|--------|
| 1 | FIELDS mode, filter auto-detection, async fixes | `ebaa2b7` |
| 2 | Testing foundation, 80%+ coverage | `17d1511` |
| 3 | Retry/backoff, resilience, rollback, reconnect | `f8d4bb0` |
| 4 | Full CLI (14 command groups), non-interactive init | `0d98f68` |
| 5 | Metadata store, detection pipeline, repartition engine | `cbd1d8e` |
| 6 | Structured logging, metrics hooks, embedding cache, pool tuning | `d947c28` |
| 7 | Document ingestion pipeline, rate limiting, BinData wiring | `9c9afcb` |
| 8 | Developer experience: packaging, CI/CD, examples, pre-commit | `e41ceff` |
| 9 | Testing excellence, mypy strict, integration/perf tests | `aa053ec` |
| 10 | Hierarchical centroid routing, filter-map routing | `68c9724` |
| 11 | Job scheduler, maintenance windows, event bus, webhooks | `1dbebdc` |
| 11.5 | Structural refactoring: 5 modules split into focused submodules | `3a82197` |
| 12 | Backend abstraction layer, capability Protocols, BackendFactory, PostgreSQL PoC | `8cdba22` |
| 12.5 | Backend interface completion — clean BaseBackend for multi-backend readiness | `36b4b3c` |
| 13 | PostgreSQL/pgvector production backend — 222 tests, full BaseBackend implementation | `7a82bca` |

---

## Phase 14: Production Completeness (P0) — **Next**

**Goal:** Close every gap that prevents a real user from running SVR with PostgreSQL in production. After Phase 14, the Postgres experience is functionally equivalent to MongoDB for the core workflow: configure → connect → ingest → search → monitor.

### Key deliverables
- **Postgres ingestion pipeline** — `insert_documents()` on `BaseBackend`, backend-agnostic `IngestPipeline`
- **Phase 13 audit fixes** — score normalization per metric, field name validation, `sql.Identifier()` for DDL, config range validators
- **`search_partitions()` formalized** on `BaseBackend` as `@abstractmethod`
- **MongoDB-specific `.db` access guarded** — `hasattr` checks on 5 code paths
- **CLI Postgres awareness** — `svr init --backend postgres` generates correct config

### Build prompt
See PHASE14_PROMPT.md

---

## Phase 15: Adoption Experience & BYOM (P0)

**Goal:** Make SVR frictionless for teams adopting it on existing workloads — whether they have existing embeddings, existing indexes, or want to evaluate before committing.

### 15.1 Guided Onboarding Analysis

- **`svr onboard`** — CLI command that scans existing collections/tables, detects embedding fields and indexes, recommends partitioning strategies with cost/benefit estimates
- **Dry-run provisioning** — `svr onboard --dry-run` shows exactly what SVR would create without creating anything
- **Backend-aware** — works for both MongoDB and Postgres (scans tables/collections accordingly)
- **Report export** — `svr onboard --format json` for programmatic consumption

### 15.2 Existing Index Attachment

- **`existing_index_name` config option** — SVR uses an existing vector index instead of creating its own
- **Auto-detection during onboard** — discovers existing vector search indexes and offers to map them
- **No ownership assumption** — SVR never modifies/deletes indexes it didn't create

### 15.3 Pre-Embedded Document Ingestion

- **`IngestDocument.vector: Optional[list[float]]`** — skip embedding for documents with pre-computed vectors
- **Mixed-mode batches** — documents with and without pre-computed vectors in a single `ingest()` call
- **Dimension validation** — validate vector dimensions match config before writing

### 15.4 Automated Repartition Execution

- **`REPARTITION` job handler** — scheduler auto-executes detected splits within maintenance windows
- **Approval modes** — auto-execute: false (default), auto-execute: true, or auto-execute with size threshold
- **Safety guardrails** — max concurrent repartitions, cooldown period, maintenance window enforcement

### 15.5 Backend-Agnostic Centroid Computation

- **`sample_embeddings()` on BaseBackend** — abstract method for sampling vectors (replaces MongoDB `.db` access)
- **Postgres implementation** — `SELECT embedding FROM svr_vectors WHERE partition_name = %s ORDER BY RANDOM() LIMIT N`
- **Removes all remaining `hasattr(backend, "db")` guards** from client.py, repartition.py

### Open questions
- Should guided onboard recommend centroid routing config based on analyzed data shape?
- Should `existing_index_name` support per-partition overrides for VIEWS/FIELDS mode?
- Auto-repartition: should scheduler wait for index warmup or fire-and-forget?

---

## Phase 16: Multi-Tenant Isolation (P0)

**Goal:** Allow a single SVR deployment to serve multiple tenants with configuration, connection, and partition isolation. Required for the managed service.

### Key features
- **Per-tenant SVRConfig** — tenant ID in config, separate partition namespaces
- **Connection pool isolation** — per-tenant connection pools with configurable sizes
- **Tenant-scoped metadata** — `svr_metadata` collection includes `tenant_id` field
- **Rate limiting per tenant** — token-bucket rate limiter scoped to tenant
- **Tenant middleware** — `TenantContext` propagated through async context vars
- **CLI tenant management** — `svr tenants list/create/delete/config`

### Open questions
- Should tenants share embedding provider clients or get isolated instances?
- Tenant isolation at database level vs collection level vs document level?
- How to handle tenant-specific embedding models (different dimensions per tenant)?

---

## Phase 17: Query Analytics Pipeline (P1)

**Goal:** Measure search quality, detect degradation, and surface insights for partition tuning.

### Key features
- **Search quality metrics** — MRR, nDCG, precision@k
- **Slow query detection** — queries exceeding P95 latency threshold trigger events
- **Partition hit distribution** — track hot/cold partitions
- **Query clustering** — group similar queries by embedding proximity
- **Embedding drift detection** — compare recent query embeddings against historical centroids
- **Analytics store** — dedicated `svr_analytics` collection with TTL indexes

### Dependencies
- Phase 11 event bus (analytics events)
- Phase 16 tenant scoping (per-tenant analytics)

---

## Phase 18: Auth & API Gateway Layer (P1)

**Goal:** Foundation for managed service — HTTP API layer with authentication, authorization, and usage metering.

### Key features
- **FastAPI gateway** — RESTful API wrapping SVRClient methods
- **Authentication** — API key auth, JWT validation, OAuth2 client credentials
- **Authorization** — role-based access control with per-tenant scoping
- **Usage metering** — API calls, embedding tokens, documents ingested per tenant
- **Health endpoints** — `/health`, `/ready`, `/metrics` (Prometheus format)
- **OpenAPI spec** — auto-generated, versioned API documentation

### Open questions
- Self-hosted gateway vs cloud-only?
- Billing integration (Stripe, custom)?
- API versioning strategy (URL path vs header)?

---

## Phase 19: Streaming & Real-Time Ingestion (P2)

**Goal:** Support continuous document streams rather than batch-only ingestion.

### Key features
- **Change stream ingestion** — watch for inserts/updates, auto-embed and route
- **Kafka/Pub-Sub connector** — consume from message queues
- **Incremental centroid updates** — moving average instead of full recomputation
- **Backpressure handling** — buffer when rate limits are hit
- **Dead letter queue** — failed documents routed to DLQ for inspection

### Dependencies
- Phase 7 ingestion pipeline (batch foundation)
- Phase 11 scheduler (stream worker lifecycle)

---

## Phase 20: Dashboard UI (P2)

**Goal:** Real-time web dashboard for partition health, search analytics, and ingestion monitoring.

### Key features
- **React + FastAPI** — SPA frontend with WebSocket real-time updates
- **Partition health view** — tree visualization, color-coded by health status
- **Search analytics** — query volume, latency percentiles, cache hit rate charts
- **Centroid routing visualization** — show routing decisions, partition scores, pruned branches
- **Configuration editor** — edit SVR config through UI with validation
- **Multi-tenant switcher** — switch between tenants in dashboard

### Dependencies
- Phase 17 analytics (data source)
- Phase 18 API gateway (backend)
- Phase 16 multi-tenant (tenant switching)

---

## Phase 21: Cross-Region Replication (P3)

**Goal:** Support geographically distributed deployments with partition-level replication.

### Key features
- **Region-aware routing** — route queries to nearest region's partitions
- **Partition replication** — replicate hot partitions across regions
- **Conflict resolution** — handle concurrent writes (last-write-wins or custom)
- **Failover** — automatic failover to secondary region

### Dependencies
- Phase 16 multi-tenant (region as isolation dimension)
- Phase 18 API gateway (region-aware routing at API layer)

---

## Backlog Ideas (Unprioritized)

These are ideas that don't yet warrant a full phase but may be incorporated into future work:

### HNSW Lifecycle Intelligence
*Derived from analysis of HNSW production challenges (memory, build time, parameter tuning, graph fragmentation). See PHASE12_REVIEW.md for full gap analysis.*

- **Cold partition mode** — partitions that exist in metadata but skip vector index creation. Re-index on demand when queried. Saves memory for archive/rarely-queried data
- **Index rebuild without split** — rebuild a partition's vector index to fix graph fragmentation from soft deletes, without splitting. CLI: `svr index rebuild <partition>`. Scheduler: `JobType.INDEX_REBUILD`
- **Churn rate detection signal** — track insert/update/delete ratio per partition. High churn triggers rebuild recommendation (graph fragmentation), not split recommendation (size growth). New `DetectionSignal.HIGH_CHURN`
- **Diagnostic intelligence** — three-way diagnostic flow: partition too large → recommend SPLIT, high churn → recommend REBUILD, embedding distribution scattered → recommend SPLIT BY NEW DIMENSION. The "index lifecycle manager" positioning
- **Per-partition index parameters** — allow `PartitionInfo` to override global `M`, `ef_construction`, quantization settings. Heavily-queried partitions get higher accuracy, archive partitions get aggressive quantization
- **Per-partition quantization** — different quantization modes per partition. High-traffic partitions use no quantization (max recall), archive partitions use binary (min memory). Quantization + partitioning compound benefit
- **Build time benchmarks** — benchmark partitioned vs. monolithic HNSW build times for README/landing page. Test with 1M, 10M, 100M vectors across 1, 5, 10, 20 partitions

### Multi-Backend & Platform
- **Cross-backend search** — single `svr.search()` fans out across MongoDB + Postgres partitions simultaneously
- **Additional backends** — Pinecone (namespaces), Qdrant (collections), Weaviate (classes/tenants)
- **MetadataBackend abstraction** — pluggable metadata store (Postgres-based, SQLite for single-node) for MongoDB-free deployments

### Search & Discovery
- **Hybrid search** — combine vector search with full-text search (BM25 + vector fusion)
- **Auto-embedding model selection** — benchmark multiple models against a labeled dataset, recommend best model per partition
- **Multi-modal search** — image/audio embedding support alongside text
- **Federated search** — search across multiple SVR deployments with unified result merging

### Operations & Infrastructure
- **Partition archival** — move cold partitions to cheaper storage (S3/GCS) with on-demand rehydration
- **A/B testing framework** — route a percentage of queries through different search configurations, compare quality metrics
- **GraphQL API** — alternative to REST for clients that prefer GraphQL
- **Terraform/Pulumi provider** — infrastructure-as-code for SVR deployments

---

## Priority Legend

| Priority | Meaning |
|----------|---------|
| **P0** | Must-have for open-source adoption and managed service MVP |
| **P1** | Important for production users, needed within 2-3 phases |
| **P2** | Valuable differentiation, can follow core features |
| **P3** | Future capability, low urgency |

---

## Adoption Scenarios

| Scenario | Today (Post-Phase 14) | After Phase 15 |
|----------|----------------------|----------------|
| **Greenfield** — new collection, no data | Full pipeline works (MongoDB + Postgres) | No change needed |
| **Existing data, no embeddings** | `ingest()` embeds + writes (both backends) | No change needed |
| **Existing data WITH embeddings** | Must know partitioning strategy upfront | `svr onboard` analyzes and recommends |
| **Existing data, embeddings, AND index** | Must use SVR-named index | `existing_index_name` reuses their index |
| **Own embedding pipeline** | `ingest()` re-embeds | `IngestDocument.vector` skips re-embedding |
| **Want to evaluate first** | `svr analyze` gives stats only | `svr onboard --dry-run` shows full plan |
| **Auto-split on threshold breach** | Detection + manual execution | Scheduler auto-executes in maintenance windows |
| **Postgres vector search** | Full support — ingest + search + CLI init | Centroid routing works for Postgres too |
