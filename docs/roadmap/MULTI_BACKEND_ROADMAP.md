# Multi-Backend Roadmap — From MongoDB SDK to Semantic Search Orchestrator

**Last updated:** February 2026
**Status:** Planning — prerequisite phases (11.5) in progress

---

## The Shift

SVR started as a vector search management layer for MongoDB Atlas. After 11 phases, the core IP has become **backend-agnostic algorithms**:

- **Routing cascade** — explicit, filter-map, centroid, fan-out (no MongoDB in any of those)
- **Detection pipeline** — 5 health signals computed from counts and statistics (any database can report counts)
- **Repartition engine** — 6-step workflow with resume and rollback (the steps are abstract: "create partition storage", "build index", "switch routing")
- **Centroid routing** — pure Python vector math, no database dependency
- **Job scheduler** — maintenance windows, distributed locks, interval parsing (backend-agnostic)
- **Event bus** — in-process dispatch, webhook delivery (completely backend-agnostic)

The MongoDB-specific parts are concentrated in two places: `backends/mongodb.py` (937 lines) and `lifecycle/provisioner.py` (665 lines). Everything else — client, routing, detection, scheduler, events, metrics, logging, cache, rate limiting — works regardless of database.

**The product is the orchestration intelligence, not the backend.**

---

## Architecture Vision

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SVRClient (facade)                           │
│   search() ─ ingest() ─ connect() ─ monitor() ─ repartition()      │
├─────────┬───────────┬───────────────┬──────────────┬────────────────┤
│ Routing │ Embedding │   Lifecycle   │  Scheduling  │  Observability │
│ ─────── │ ───────── │   ─────────   │  ──────────  │  ──────────── │
│ CascParts Distributor │ OpenAI    │  Provisioner  │  Scheduler   │  Logging       │
│ Centroid│ Voyage    │  Detector     │  EventBus    │  Metrics       │
│ Filter  │ Cohere    │  Repartition  │  Webhooks    │  Cache         │
│ Merger  │ HuggingFace│ Scanner     │  Windows     │  RateLimiter   │
├─────────┴───────────┴──────┬────────┴──────────────┴────────────────┤
│                            │                                        │
│       Backend Interface    │    Metadata Store (always MongoDB)      │
│       (pluggable)          │    ─────────────────────────────        │
│                            │    Partition state, centroids,          │
│  ┌──────────┐ ┌──────────┐ │    job history, locks, operations      │
│  │ MongoDB  │ │ Postgres │ │                                        │
│  │  Atlas   │ │ pgvector │ │                                        │
│  └──────────┘ └──────────┘ │                                        │
│  ┌──────────┐ ┌──────────┐ │                                        │
│  │ Pinecone │ │ Qdrant   │ │                                        │
│  │ (future) │ │ (future) │ │                                        │
│  └──────────┘ └──────────┘ │                                        │
└────────────────────────────┴────────────────────────────────────────┘
```

**Key principle:** MongoDB is always present as the metadata/control plane store, even when the vector data lives in another database. The `svr_metadata` collection stores partition state, centroids, operation history, distributed locks, and job state — regardless of where vectors are searched.

---

## Why MongoDB as Universal Metadata Store

| Requirement | Why MongoDB fits |
|------------|-----------------|
| Flexible schema | Metadata evolves every phase — documents handle schema changes without migrations |
| Distributed locks | `findOneAndUpdate` with TTL upsert — proven across 11 phases, race-safe |
| Change streams | Event-driven reactions to partition state changes |
| Operation tracking | Document-shaped operation records with nested step arrays |
| Centroid storage | Variable-length float arrays stored natively |
| Job state persistence | Scheduler state survives process restarts |
| Free tier available | MongoDB Community is free, widely deployed, well-understood |
| Already proven | 1,376 tests validate the metadata patterns |

Alternative considered: SQLite for local metadata. Rejected because distributed locking and change streams don't work, and the managed service needs shared state across workers.

---

## What `BaseBackend` Needs to Become

### Current interface (MongoDB-shaped, 303 lines)

The current `BaseBackend` leaks MongoDB concepts:

| Method | MongoDB concept leaked |
|--------|----------------------|
| `create_partition_view()` | Views are MongoDB-specific. Postgres uses schemas/tables |
| `delete_partition_view()` | Same |
| `view_exists()` | Same |
| `create_vector_search_index()` with `filter_fields` | Atlas Search filter fields |
| `watch_collection()` returning `AsyncIterator` | MongoDB change streams |
| `get_index_status()` returning `dict` with `queryable` | Atlas Search index lifecycle |

### Target interface (backend-agnostic)

The abstract interface should speak in terms of **capabilities**, not MongoDB operations:

```python
class BaseBackend(ABC):
    """Abstract backend for vector search operations."""

    # Connection lifecycle
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: ...
    async def health_check(self) -> bool: ...

    # Partition storage (replaces create_partition_view / FIELDS logic)
    async def create_partition_storage(
        self, partition: PartitionInfo, config: SVRConfig
    ) -> PartitionStorageResult: ...

    async def delete_partition_storage(
        self, partition: PartitionInfo
    ) -> None: ...

    # Index lifecycle (replaces create_vector_search_index)
    async def create_partition_index(
        self, partition: PartitionInfo, config: SVRConfig
    ) -> None: ...

    async def delete_partition_index(
        self, partition: PartitionInfo
    ) -> None: ...

    async def get_index_status(
        self, partition: PartitionInfo
    ) -> IndexStatus: ...
    # IndexStatus is a universal model: ready, building, error, not_found

    async def wait_for_index_ready(
        self, partition: PartitionInfo, timeout_s: float
    ) -> bool: ...

    # Search (stays mostly the same — already fairly abstract)
    async def execute_search(
        self, partition: PartitionInfo,
        query_vector: list[float],
        limit: int, num_candidates: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]: ...

    async def search_partitions(
        self, partitions: list[PartitionInfo],
        query_vector: list[float],
        limit: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]: ...

    # Data operations
    async def get_distinct_values(self, field: str, ...) -> list[Any]: ...
    async def count_documents(self, ...) -> int: ...
    async def get_partition_document_counts(self, field: str) -> dict[str, int]: ...

    # Optional capabilities (not all backends support these)
    def supports_change_streams(self) -> bool: return False
    def supports_auto_embedding(self) -> bool: return False
    def supports_views(self) -> bool: return False
```

**Key changes:**
1. No `create_partition_view()` — replaced by `create_partition_storage()` which each backend implements differently (MongoDB creates views, Postgres creates schemas/tables)
2. No `filter_fields` parameter on index creation — each backend knows how to build its own index type
3. `execute_search()` takes `query_vector` not `query` string — auto-embedding is a MongoDB-specific feature, handled by a capability flag
4. Return types are universal models (`IndexStatus`, `SearchHit`) not raw dicts
5. Capability flags let the orchestrator adapt behavior per backend

---

## What Changes Per Backend

| Concern | MongoDB Atlas | Postgres/pgvector | Pinecone (future) |
|---------|--------------|-------------------|-------------------|
| **Partition storage** | Views (VIEWS mode) or embedding fields (FIELDS mode) or pre-filter (SOURCE mode) | Separate tables per partition, or schema-per-partition, or single table with partition column | Namespaces within an index |
| **Index creation** | Atlas Search index with HNSW config | `CREATE INDEX ... USING hnsw` on vector column | Managed — no explicit index creation |
| **Index readiness** | Poll `queryable` status | Immediate (synchronous index build) | Poll index status API |
| **Vector search** | `$vectorSearch` aggregation stage | `ORDER BY embedding <=> $1 LIMIT $2` | Query API with namespace filter |
| **Filter fields** | Atlas Search `filter` field type | SQL `WHERE` clause | Metadata filters |
| **Change streams** | Native MongoDB change streams | `LISTEN`/`NOTIFY` or logical replication | Not supported |
| **BinData vectors** | MongoDB Binary subtype | Not applicable — native `vector` type | Not applicable — native float arrays |
| **Auto-embedding** | Atlas auto-embedding via Voyage | Not supported | Not supported |
| **Connection** | `AsyncMongoClient` | `asyncpg` pool | `pinecone.Pinecone()` |

---

## Config Model Separation

Currently `models.py` mixes universal and backend-specific config:

**Universal (keep in core models):**
- `SVRConfig`, `PartitionInfo`, `SearchHit`, `SearchResult`
- `EmbeddingConfig`, `RerankingConfig`, `CacheConfig`, `RateLimitConfig`
- `SchedulerConfig`, `EventsConfig`, `WebhookConfig`
- `DetectionConfig`, `RepartitionConfig`, `AutoSplitConfig`
- `CentroidRoutingConfig`, `RoutingMode`
- `PartitionStatus`, `DetectionSignal`

**MongoDB-specific (move to backend config):**
- `IndexLocation` (SOURCE/VIEWS/FIELDS — Postgres has different modes)
- `MongoDBIndexQuantization` (scalar/binary — Postgres uses different quantization)
- `VectorStorageConfig.storage_format` (BinData types — MongoDB-only)
- `VoyageQuantization` (int8/binary output → BinData — MongoDB pipeline)
- `DatabaseConfig.max_pool_size`, etc. (MongoDB-specific pool params)

**Target structure:**
```python
# models/config.py — universal config
class SVRConfig:
    backend: str = "mongodb"  # "mongodb", "postgres", "pinecone"
    backend_config: dict[str, Any] = {}  # Backend-specific config blob

# backends/mongodb/config.py — MongoDB-specific
class MongoDBConfig:
    connection_string_env: str
    database: str
    source_collection: str
    index_location: IndexLocation  # SOURCE/VIEWS/FIELDS
    max_pool_size: int = 100
    storage_format: StorageFormat = StorageFormat.FLOAT32

# backends/postgres/config.py — Postgres-specific
class PostgresConfig:
    connection_string_env: str
    schema: str = "public"
    partition_strategy: PgPartitionStrategy  # TABLE_PER_PARTITION/SINGLE_TABLE
    index_type: PgIndexType  # HNSW/IVFFLAT
```

---

## Provisioner Refactoring for Multi-Backend

The provisioner currently hardcodes three MongoDB index modes. For multi-backend, each backend provides its own provisioning strategy:

```python
# lifecycle/provisioner.py — becomes the orchestrator
class PartitionProvisioner:
    """Backend-agnostic partition lifecycle orchestrator."""

    async def create_partition(self, name, filter_value=None, ...):
        # 1. Create partition storage (delegates to backend)
        await self.backend.create_partition_storage(partition, config)
        # 2. Create index (delegates to backend)
        await self.backend.create_partition_index(partition, config)
        # 3. Save metadata (always MongoDB)
        await self.metadata.save_partition(partition)
        # 4. Emit event
        await self._emit_event("partition.created", name)
```

The provisioner no longer knows about views, FIELDS mode, or Atlas Search indexes. It speaks in abstract operations. Each backend implements those operations in its own way.

---

## Implementation Phases

### Phase 11.5: Structural Refactoring (prerequisite)
- Split provisioner into orchestrator + index/view managers
- Split models into subpackage
- Split config validators
- **This is valuable on its own** — it also happens to prepare the seams for multi-backend

### Phase 12: Backend Abstraction Layer
- Redesign `BaseBackend` interface (capability-based, not MongoDB-shaped)
- Separate universal config models from backend-specific config
- Create `backends/mongodb/` package (move current mongodb.py + config into it)
- Refactor provisioner to use abstract backend operations
- Create `BackendFactory` — instantiate backend from config `backend: "mongodb"`
- **Validation:** all 1,500+ existing tests pass with the refactored MongoDB backend fitting the new interface. No new backends yet — just prove the abstraction works.

### Phase 13: Postgres/pgvector Backend
- `backends/postgres/` package — connection, search, index lifecycle
- `backends/postgres/config.py` — Postgres-specific config models
- Partition strategy: table-per-partition (HNSW index per table) or single table with partition column
- Async driver: `asyncpg`
- Vector type: pgvector `vector` column
- Integration tests against real Postgres instance
- **This is the proof** that the abstraction works — if Postgres fits cleanly into `BaseBackend`, the design is right

### Future: Additional backends
- Pinecone (namespaces as partitions, managed indexes)
- Qdrant (collections as partitions)
- Weaviate (classes/tenants as partitions)
- Each backend is a self-contained package under `backends/`

---

## What Stays the Same

These components are already backend-agnostic and won't change:

| Component | Why it's backend-agnostic |
|-----------|-------------------------|
| Routing cascParts Distributor | Pure algorithms on partition metadata |
| Centroid router | Pure Python vector math |
| Detection pipeline | Operates on counts and statistics |
| Repartition engine | Abstract steps, delegates to backend |
| Job scheduler | Manages timing, not data |
| Event bus + webhooks | In-process dispatch + HTTP delivery |
| Embedding providers | Backend-independent — embed before search |
| Reranking | Backend-independent — rerank after search |
| CLI | Calls SVRClient, doesn't touch backend directly |
| Metrics + logging | Cross-cutting, backend-independent |
| Embedding cache | Caches vectors regardless of backend |
| Rate limiter | Limits embedding API calls, not backend calls |

**This is ~80% of the codebase.** The multi-backend refactoring touches ~20% of the code (backend, provisioner, config models) while preserving everything else.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Breaking existing MongoDB users | Backward-compatible config loading. Old config files work unchanged |
| Leaky abstraction (Postgres doesn't fit) | Design the interface from Postgres requirements first, then verify MongoDB fits |
| Metadata store coupling | MongoDB metadata store is explicitly always-present, not a coupling concern |
| Test coverage gaps | 1,500+ existing tests validate MongoDB path. New backends get their own integration tests |
| Over-abstraction | Only abstract what's needed for the second backend. Don't anticipate Pinecone/Qdrant specifics |

---

## The Commercial Pitch

> SVR manages partitioned vector search across any database. Your data lives in Postgres? MongoDB? Both? SVR handles partition routing, health detection, auto-repartitioning, and lifecycle management — with a unified API, unified CLI, and a control plane that works across all your vector stores.
>
> The orchestration intelligence — centroid routing, detection signals, maintenance windows, webhook notifications — works the same regardless of where your vectors live. The complexity of multi-backend vector search becomes a configuration choice, not an engineering project.
