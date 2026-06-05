# SVR Lifecycle Subsystem — Architecture Documentation

> **Audience**: Engineering Manager / Architect reviewing the SVR codebase
> **Scope**: `semantic_vector_router/lifecycle/` — 10 files, ~2,800 lines
> **Last updated**: February 2026 (Phase 15)

---

## 1. Executive Summary

The lifecycle subsystem is the operational brain of SVR. It handles everything that happens to partitions *after* initial setup: discovering new partition values in live data, provisioning storage and indexes, monitoring health, detecting problems, splitting overgrown partitions, and orchestrating multi-step repartitioning workflows with rollback.

Think of it as the "Kubernetes for vector partitions" — it watches, detects, acts, and self-heals.

---

## 2. File Inventory

| File | Lines | Role | Key Class |
|------|-------|------|-----------|
| `__init__.py` | 16 | Package facParts Distributor | — |
| `provisioner.py` | 504 | Partition CRUD orchestrator | `PartitionProvisioner` |
| `detector.py` | 581 | Health detection pipeline (5 signals) | `PartitionDetector` |
| `scanner.py` | 216 | Discovers partition values from data | `PartitionScanner` |
| `monitor.py` | 231 | Real-time health & threshold checking | `PartitionMonitor` |
| `watcher.py` | 317 | Change stream listener for new values | `PartitionWatcher` |
| `splitter.py` | 503 | Splits overgrown partitions (3 strategies) | `PartitionSplitter` |
| `repartition.py` | 453 | Multi-step repartition workflow engine | `RepartitionEngine` |
| `index_manager.py` | 7 | Backward-compat shim | — |
| `view_manager.py` | 4 | Backward-compat shim | — |

---

## 3. Architecture Diagram

```
                    ┌──────────────┐
                    │  SVRClient   │  (client.py)
                    │  Lazy init   │
                    └──────┬───────┘
                           │ creates on demand
          ┌────────────────┼────────────────────────┐
          │                │                         │
          ▼                ▼                         ▼
   ┌─────────────┐  ┌──────────────┐        ┌──────────────┐
   │   Scanner   │  │ Provisioner  │◄───────│   Watcher    │
   │ (discover)  │  │  (create/    │        │ (change      │
   │             │  │   delete)    │◄──┐    │  streams)    │
   └─────────────┘  └──────┬───────┘   │    └──────────────┘
                           │           │
                    ┌──────┴───────┐   │
                    │              │   │
                    ▼              ▼   │
             ┌──────────┐  ┌──────────┐
             │ Splitter │  │Repartition│
             │(split    │  │ Engine   │
             │ overgrown)│  │(workflow)│
             └──────────┘  └──────────┘

          ┌────────────────────────────┐
          │  Monitor    │   Detector   │
          │ (health     │  (5-signal   │
          │  status)    │   pipeline)  │
          └────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ MetadataStore│  (backends/metadata.py)
                    │ (persistence │
                    │  + locking)  │
                    └──────────────┘
```

---

## 4. File-by-File Deep Dive

### 4.1 `provisioner.py` — The Central Orchestrator

**What it does**: Creates, deletes, verifies, and counts partitions. Every other lifecycle component that needs to create a partition delegates here.

**Constructor**:
```python
class PartitionProvisioner:
    def __init__(
        self,
        backend: BaseBackend,
        config: SVRConfig,
        auto_save_config: bool = True,
    ):
```

**Key methods**:

| Method | Purpose |
|--------|---------|
| `create_partition(name, filter_value, ...)` | Full creation with rollback on failure |
| `create_partitions_batch(values, ...)` | Batch creation, single config save at end |
| `delete_partition(name, ...)` | Deregister + best-effort backend cleanup |
| `verify_partition(name)` | Check storage + index existence/health |
| `update_all_partition_counts()` | Refresh document counts for all partitions |
| `set_event_bus(bus)` | Inject EventBus for lifecycle event emission |

**Creation workflow** (the core of the provisioner):

```
create_partition(name, filter_value)
  1. Check registry for duplicates           → PartitionAlreadyExistsError
  2. FIELDS mode cap check (max 50)          → PartitionProvisioningError
  3. backend.create_partition_storage(...)    → PartitionStorageResult
  4. backend.create_partition_index(...)      → index_name
  5. backend.count_documents(...)             → doc_count
  6. Register PartitionInfo in config
  7. save_config(config)
  8. Emit "partition.created" event
  ── On failure at any step ──
  _rollback_partition()                      → best-effort cleanup
```

**Why it matters**: This is the only module that writes to both the backend (storage/indexes) and the config file. All other modules go through it.

---

### 4.2 `detector.py` — The Health Detection Pipeline

**What it does**: Runs a 4-phase detection pipeline with 5 independent health signals. This is the most algorithmically sophisticated module in the lifecycle subsystem.

**The COLLECT-STORE-ANALYZE-DECIDE pipeline**:

```python
async def run_detection(self) -> list[DetectionResult]:
    # COLLECT — gather current partition counts
    counts = await self._collect_counts(partitions)

    # STORE — persist health snapshots for trend analysis
    await self._store_health_data(counts)

    # ANALYZE — run 5 independent checks
    results = []
    results += await self._check_threshold_breaches(counts)
    results += await self._check_approaching_thresholds(counts)
    results += await self._check_skew(counts)
    results += await self._check_underpopulated(counts)
    results += await self._check_stale(counts)

    # DECIDE — create auto-executable operations
    await self._create_operations(results)

    return results
```

**The 5 detection signals**:

| Signal | Trigger | Auto-executable? | Suggested Action |
|--------|---------|-------------------|------------------|
| `THRESHOLD_BREACH` | count > `threshold_vectors` | Yes | Split partition |
| `APPROACHING_THRESHOLD` | Linear regression predicts breach within `trend_window_days` | No | Alert |
| `SKEW` | max/avg ratio > `skew_ratio` | No | Rebalance |
| `UNDERPOPULATED` | count < `min_threshold_vectors` | No | Merge |
| `STALE` | Last 10 measurements identical | No | Archive |

**The trend prediction algorithm** (`_calculate_trend_slope`):

```python
# Linear regression: vectors/day growth rate
# Uses numpy when available, manual least-squares fallback
slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
vectors_per_day = slope * 86400  # Convert from per-second to per-day
days_to_threshold = (threshold - current_count) / vectors_per_day
```

**Distributed locking**: `run_detection_with_lock()` acquires a distributed lock via MetadataStore before running detection, ensuring only one instance runs at a time in a multi-node deployment.

---

### 4.3 `scanner.py` — Partition Value Discovery

**What it does**: Queries the source collection/table to discover all unique partition field values and their document counts. This is the "what partitions *should* exist?" question.

**Key methods**:

```python
class PartitionScanner:
    async def scan_partition_values(self, limit=None) -> dict[str, int]:
        """Returns {value: count} sorted by count descending."""

    async def get_new_partition_values(self) -> list[Any]:
        """Values in data but not in registry."""

    async def validate_partitions(self) -> dict[str, list[str]]:
        """Returns {'missing': [...], 'orphaned': [...], 'valid': [...]}"""
```

**Usage pattern**: The scanner is typically the first thing called during `svr init` and `svr partitions scan` — it tells you what's in your data before you create partitions.

---

### 4.4 `monitor.py` — Real-Time Health Status

**What it does**: Lightweight health checking — is each partition healthy, warning, or critical? Unlike the detector (which runs periodically and persists state), the monitor is a point-in-time snapshot.

**Health status logic**:

```python
if count > threshold:
    status = "critical"       # Over capacity
elif count / threshold > 0.8:
    status = "warning"        # Approaching capacity
else:
    status = "healthy"
```

**Key methods**:

| Method | Returns |
|--------|---------|
| `check_partition_health(name)` | Single `PartitionHealthStatus` |
| `check_all_partitions()` | All statuses, sorted by utilization desc |
| `get_critical_partitions()` | Only partitions over threshold |
| `needs_attention()` | `{critical: [...], warning: [...], unhealthy_indexes: [...]}` |
| `check_index_health(name)` | Index status + queryability |

---

### 4.5 `watcher.py` — Change Stream Listener

**What it does**: Watches MongoDB change streams for inserts/updates that introduce new partition field values. When a new value appears, it either auto-provisions a partition or queues it for manual confirmation.

**The watch loop**:

```python
async def _watch_loop(self):
    while self._running:
        try:
            # Requires ChangeStreamCapable backend
            async with await self.backend.watch_collection(...) as stream:
                async for change in stream:
                    await self._handle_change(change, partition_field)
        except Exception:
            # Exponential backoff with jitter
            delay = min(base * (2 ** attempt), max_delay)
            delay *= (0.5 + random.random())  # Jitter
            await asyncio.sleep(delay)
```

**Two-phase provision**:

```
New value detected in change stream
  ├─ auto_provision=True && !confirmation_required
  │    └─ _auto_provision() → provisioner.create_partition()
  └─ confirmation_required=True
       └─ Add to _pending_partitions → _persist_pending()
            └─ Later: confirm_partition() → _auto_provision()
            └─ Or:    reject_partition() → remove from pending
```

**Capability check**: At `start()`, the watcher checks `isinstance(self.backend, ChangeStreamCapable)`. PostgreSQL backends will fail this check (change streams are MongoDB-specific). This is the Protocol-based capability pattern from Phase 12.

---

### 4.6 `splitter.py` — Partition Splitting Engine

**What it does**: When a partition grows beyond `threshold_vectors`, the splitter breaks it into smaller child partitions using one of three strategies.

**Three split strategies**:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `SECONDARY_FIELD` | Splits by distinct values of a secondary field (e.g., split "electronics" by "brand") | Hierarchical data |
| `TIME` | Splits by time buckets (monthly/quarterly/yearly) on a date field | Time-series data |
| `HASH` | Modulus-based sharding by `_id` | **DEPRECATED** — defeats semantic routing |

**Secondary field split example**:

```
Partition "electronics" (500K docs) → split by "brand"
  ├─ electronics__apple     (150K docs)
  ├─ electronics__samsung   (120K docs)
  ├─ electronics__sony      (90K docs)
  └─ electronics__other     (140K docs)
```

**Schedule enforcement**: Splits can be restricted to maintenance windows:

```python
def _is_within_schedule(self) -> bool:
    schedule = self._config.lifecycle.auto_split.schedule
    now = datetime.now()
    # Check day_of_week (0=Monday) and hour range
    if now.weekday() not in schedule.days_of_week:
        return False
    if not (schedule.start_hour <= now.hour < schedule.end_hour):
        return False
    return True
```

When outside the schedule window, the partition is marked `PENDING_SPLIT` for later execution.

---

### 4.7 `repartition.py` — The Workflow/Saga Engine

**What it does**: Orchestrates complex multi-step repartitioning operations with persistent state, resume capability, and rollback. This is the most operationally sophisticated module.

**The 6-step workflow**:

```
execute_operation(op_id)
  Step 1: create_children     → Create child partitions, mark parent SPLITTING
  Step 2: build_indexes       → Verify all child indexes exist
  Step 3: wait_indexes        → Poll until all indexes are queryable (with timeout)
  Step 4: switch_routing      → Retire parent, activate children
  Step 5: compute_centroids   → Sample embeddings, compute normalized mean vectors
  Step 6: cleanup_parent      → Best-effort delete parent index/view
```

**Resume capability**: Each step's completion is persisted to MetadataStore. If the process crashes mid-operation, calling `execute_operation(op_id)` again skips completed steps:

```python
for step_name, handler in self.step_handlers.items():
    step = operation["steps"].get(step_name, {})
    if step.get("status") == "done":
        continue  # Skip already-completed steps
    await handler(operation)
    await self._mark_step_done(op_id, step_name)
```

**Rollback**: `rollback_operation(op_id)` reverses the operation by deleting child partitions and resetting the parent to ACTIVE status.

**Centroid computation** (Step 5):

```python
# Sample up to 1000 documents from each child partition
# Extract embedding vectors, compute normalized mean
centroid = sum(vectors) / len(vectors)
norm = sqrt(sum(c**2 for c in centroid))
normalized = [c / norm for c in centroid]
# Store as partition.centroid for centroid-based routing
```

---

### 4.8 Backward-Compatibility Shims

`index_manager.py` and `view_manager.py` are pure re-export shims created during Phase 11.5:

```python
# index_manager.py (7 lines)
from semantic_vector_router.backends.mongodb.index_manager import (
    MAX_FIELDS_PARTITIONS, SOURCE_INDEX_NAME, IndexManager,
)

# view_manager.py (4 lines)
from semantic_vector_router.backends.mongodb.view_manager import ViewManager
```

These preserve the original import paths after the MongoDB backend was refactored from a single file into a package.

---

## 5. Dependency Graph

### Intra-Lifecycle Dependencies

```
provisioner.py  ◄──── watcher.py      (auto-provision delegates to Provisioner)
provisioner.py  ◄──── splitter.py     (child partition creation via Provisioner)
provisioner.py  ◄──── repartition.py  (creates internal Provisioner instance)
```

**Key insight**: `PartitionProvisioner` is the dependency hub — three modules depend on it. `PartitionDetector`, `PartitionScanner`, and `PartitionMonitor` have zero intra-lifecycle dependencies.

### External Dependencies

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  lifecycle/  │────▶│  backends/       │     │  models/     │
│              │     │  base.py         │     │  (all enums, │
│  provisioner │────▶│  metadata.py     │     │   configs,   │
│  detector    │────▶│  mongodb/        │     │   results)   │
│  scanner     │     │    index_manager │     └──────────────┘
│  monitor     │     └──────────────────┘
│  watcher     │     ┌──────────────────┐     ┌──────────────┐
│  splitter    │────▶│  config.py       │     │  events/     │
│  repartition │     │  save_config()   │     │  bus, models │
└─────────────┘     └──────────────────┘     └──────────────┘
```

### Who Uses Lifecycle (External Consumers)

| Consumer | Classes Used |
|----------|-------------|
| `client.py` | All 8 classes (lazy initialization) |
| `cli/init.py` | Scanner, Provisioner |
| `cli/partitions.py` | Scanner, Provisioner |
| `cli/watch.py` | Watcher, Provisioner |
| `cli/split.py` | Splitter, Provisioner |
| `cli/monitor.py` | Detector |
| `cli/repartition.py` | RepartitionEngine |

---

## 6. Design Patterns

### 6.1 Strategy Pattern

Used in two places:

- **`PartitionSplitter`**: `SplitStrategy` enum selects between `SECONDARY_FIELD`, `TIME`, `HASH`, and `ALERT_ONLY` behaviors
- **`PartitionProvisioner`**: The 3 index modes (`SOURCE`, `VIEWS`, `FIELDS`) are handled by `BaseBackend` delegation — the provisioner doesn't know which mode is active

### 6.2 Observer Pattern

Three modules emit events through an optional `EventBus`:

```python
# In provisioner.py
def _emit_event(self, event_type_str, partition, **details):
    if self._event_bus is None:
        return
    from semantic_vector_router.events.models import SVREvent, SVREventType
    event = SVREvent(type=SVREventType(event_type_str), ...)
    self._event_bus.emit(event)
```

Events emitted: `partition.created`, `partition.deleted`, `health.threshold_breach`, `health.approaching_threshold`, `health.skew_detected`, `repartition.started`, `repartition.completed`, `repartition.failed`, `centroid.computed`.

### 6.3 Template Method Pattern

`PartitionProvisioner.create_partition()` follows a rigid sequence with rollback — the steps are fixed but the backend implementations vary per database.

### 6.4 Workflow/Saga Pattern

`RepartitionEngine` implements a persistent workflow with:
- **Step tracking**: Each step's status is persisted to MetadataStore
- **Resumability**: Crashed operations can be resumed from the last completed step
- **Compensation (rollback)**: Explicit rollback deletes children and resets parent

### 6.5 Pipeline Pattern

`PartitionDetector.run_detection()` follows the COLLECT → STORE → ANALYZE → DECIDE pipeline, where each phase feeds the next.

### 6.6 Capability-Based Protocols

`PartitionWatcher` checks `isinstance(backend, ChangeStreamCapable)` at runtime. This is the Protocol-based capability pattern from the backend abstraction layer — backends declare capabilities, and lifecycle modules check before using them.

### 6.7 Distributed Lock Pattern

`PartitionDetector.run_detection_with_lock()` uses `MetadataStore.acquire_lock()` / `release_lock()` for leader election in multi-node deployments. Only one node runs detection at a time.

### 6.8 Exponential Backoff with Jitter

`PartitionWatcher._watch_loop()` reconnects to change streams with exponential backoff + jitter:

```python
delay = min(base_delay * (2 ** attempt), max_delay)
delay *= (0.5 + random.random())  # Add jitter to prevent thundering herd
```

---

## 7. Data Flow Examples

### Flow 1: New partition discovered in live data

```
User inserts document with category="gaming"
  │
  ▼
PartitionWatcher._watch_loop()
  │ detects new value "gaming" in change stream
  ▼
watcher._handle_change()
  ├── auto_provision=True
  │     └── watcher._auto_provision("gaming")
  │           └── provisioner.create_partition("gaming", "gaming")
  │                 ├── backend.create_partition_storage()
  │                 ├── backend.create_partition_index()
  │                 ├── config.save_config()
  │                 └── event_bus.emit("partition.created")
  └── confirmation_required=True
        └── Add to pending → user confirms later via CLI
```

### Flow 2: Partition grows too large

```
PartitionDetector.run_detection()
  │ COLLECT: counts = {electronics: 15M, books: 2M, ...}
  │ STORE:   metadata.append_health_history(counts)
  │ ANALYZE: _check_threshold_breaches() → electronics > 10M threshold
  │ DECIDE:  metadata.create_operation(split electronics)
  ▼
PartitionSplitter.execute_split("electronics")
  │ Strategy: SECONDARY_FIELD on "brand"
  │ Scans distinct brands: apple, samsung, sony, lg
  ▼
PartitionProvisioner.create_partition("electronics__apple", ...)
PartitionProvisioner.create_partition("electronics__samsung", ...)
  ...
  │ Parent "electronics" marked SPLIT
  │ save_config()
  ▼
Done — queries now route to child partitions
```

### Flow 3: Repartition with rollback safety

```
RepartitionEngine.execute_operation("op-123")
  Step 1: create_children    ✓ (persisted to MetadataStore)
  Step 2: build_indexes      ✓ (persisted)
  Step 3: wait_indexes       ✓ (polled until queryable)
  Step 4: switch_routing     ✗ CRASH!

  ─── Process restarts ───

RepartitionEngine.execute_operation("op-123")  ← resume
  Step 1: create_children    SKIP (already done)
  Step 2: build_indexes      SKIP (already done)
  Step 3: wait_indexes       SKIP (already done)
  Step 4: switch_routing     ✓ (retires parent, activates children)
  Step 5: compute_centroids  ✓ (sample embeddings, compute mean)
  Step 6: cleanup_parent     ✓ (delete old index/view)

  ─── Or if something goes wrong ───

RepartitionEngine.rollback_operation("op-123")
  → Delete all children
  → Reset parent to ACTIVE
  → Mark operation as rolled_back
```

---

## 8. Integration Points Summary

| Integration | Direction | Mechanism |
|------------|-----------|-----------|
| Backend abstraction | Lifecycle → Backend | `BaseBackend` interface (backend-agnostic) |
| Metadata persistence | Lifecycle → MetadataStore | Health history, operations, locks |
| Config persistence | Lifecycle → config.py | `save_config()` for partition registry |
| Event emission | Lifecycle → EventBus | Late-imported `SVREvent` + `SVREventType` |
| Client lazy init | Client → Lifecycle | `_get_provisioner()`, `_get_scanner()`, etc. |
| CLI commands | CLI → Lifecycle | Direct instantiation in command handlers |
| Centroid routing | Lifecycle → Routing | `compute_partition_centroid()` in repartition |

---

## 9. Improvement Opportunities

### 9.1 MongoDB Leakage in Backend-Agnostic Code

**Issue**: `provisioner.py` imports `MAX_FIELDS_PARTITIONS` and `SOURCE_INDEX_NAME` from `backends.mongodb.index_manager`. `scanner.py` catches `pymongo.errors.*` exceptions. These create hard dependencies on MongoDB in what should be backend-agnostic modules.

**Fix**: Move constants to `backends/base.py` or expose them via the `BaseBackend` interface. Wrap backend-specific exceptions in generic SVR exceptions at the backend layer.

### 9.2 MongoDB-Specific Code in Repartition Engine

**Issue**: `repartition.py`'s `_step_compute_centroids` checks `hasattr(self.backend, "db")` and directly accesses `self.backend.db[source_collection]`. `splitter.py`'s `_split_by_time` calls `self.backend.get_collection()` with a `# type: ignore[attr-defined]`.

**Fix**: Define `sample_embeddings(partition, limit)` and `aggregate_time_range(field)` methods on `BaseBackend` so strategies work across backends.

### 9.3 Monitor vs. Detector Overlap

**Issue**: Both `PartitionMonitor` and `PartitionDetector` check partition counts against thresholds. The monitor is simpler (point-in-time), while the detector has persistence, trend analysis, and operation creation.

**Fix**: Consider making the monitor a thin wrapper around the detector's data, or explicitly document their distinct roles (monitor = user-facing CLI status, detector = automated pipeline).

### 9.4 Missing Re-exports in `__init__.py`

**Issue**: `PartitionDetector`, `RepartitionEngine`, and `DetectionResult` are not re-exported from `lifecycle/__init__.py`. Consumers import them directly from their modules.

**Fix**: Add these to `__init__.py` exports for API consistency.

### 9.5 Sequential Health Checks

**Issue**: `PartitionMonitor.check_all_partitions()` and `PartitionDetector._collect_counts()` iterate sequentially over partitions. For 100+ partitions, this adds latency.

**Fix**: Use `asyncio.gather()` with bounded concurrency (`asyncio.Semaphore`) to parallelize count queries.

### 9.6 Hardcoded Magic Numbers

**Issue**: Default threshold of 10,000,000 vectors in `monitor.py`. Stale detection requires exactly 10 data points in `detector.py`.

**Fix**: Make these configurable in `SVRConfig.lifecycle` with sensible defaults.

### 9.7 PostgreSQL Change Stream Alternative

**Issue**: `PartitionWatcher` only works with `ChangeStreamCapable` backends (MongoDB). PostgreSQL has no change streams.

**Fix**: Implement a polling-based watcher alternative for PostgreSQL using `LISTEN/NOTIFY` or periodic `scan_partition_values()` polling. Register the strategy based on backend capabilities.

---

## 10. Test Coverage

The lifecycle subsystem is extensively tested:

| Module | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| Provisioner | 45+ | 12 (real Atlas) |
| Detector | 30+ | 4 |
| Scanner | 20+ | 6 |
| Monitor | 25+ | 3 |
| Watcher | 20+ | 4 |
| Splitter | 35+ | 5 |
| Repartition | 25+ | 3 |
| **Total** | **200+** | **37** |

Key testing patterns:
- All unit tests use mocked backends (no network)
- Integration tests run against real MongoDB Atlas
- `_create_embedder` and `_create_reranker` are mocked to avoid API key requirements
- Distributed lock tests verify single-instance behavior

---

## 11. Summary

The lifecycle subsystem demonstrates several production-grParts Distributor engineering practices:

1. **Backend abstraction** — lifecycle logic doesn't know which database it's operating on
2. **Operational safety** — rollback on failure, resume on crash, distributed locking
3. **Observability** — event emission at every lifecycle state change
4. **Scheduling** — maintenance windows for disruptive operations like splits
5. **Trend analysis** — linear regression for proactive threshold breach prediction

It's the subsystem that transforms SVR from "a vector search library" into "an operational platform for vector search at scale."
